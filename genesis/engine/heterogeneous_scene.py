import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
from genesis.constants import link_ref_frame
from genesis.engine.entities.rigid_entity import KinematicEntity
from genesis.engine.entities.rigid_entity.description import _balanced_variant_mapping
from genesis.engine.materials.base import Material
from genesis.engine.scene import Scene
from genesis.options import (
    BaseCouplerOptions,
    FEMOptions,
    KinematicOptions,
    MPMOptions,
    PBDOptions,
    ProfilingOptions,
    RigidOptions,
    SceneOptions,
    SimOptions,
    SPHOptions,
    ViewerOptions,
    VisOptions,
)
from genesis.options.entities import Spec, get_entity_spec
from genesis.options.morphs import Morph
from genesis.options.renderers import RendererOptions
from genesis.options.surfaces import Surface
from genesis.repr_base import RBC
from genesis.utils.misc import BuildableMixin, indices_to_mask, tensor_to_cpu

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint, RigidLink


@dataclasses.dataclass(frozen=True)
class EnvironmentPartition:
    """The partition of the environments of a heterogeneous scene into groups, each group simulated by one scene.

    The environments of a group keep their order in the scene of the group: the environment of local index i in group g
    is 'groups_envs_idx[g][i]'.
    """

    groups_variants_idx: tuple[tuple[int, ...], ...]
    """The variant of each entity that each group carries, the entities in the order they were added."""
    groups_envs_idx: tuple[torch.Tensor, ...]
    """The environments of each group, in increasing order."""
    envs_group_idx: torch.Tensor
    """The group of each environment."""
    envs_local_idx: torch.Tensor
    """The index of each environment in the scene of its group."""

    def select(self, envs_idx=None) -> list[tuple[int, torch.Tensor, torch.Tensor | None]]:
        """Split a selection of environments by group.

        'envs_idx' selects environments of the heterogeneous scene by index, like the 'envs_idx' argument of the
        accessors of a scene, None selecting them all. The result holds one entry per group with at least one selected
        environment: the index of the group, the positions of its selected environments in the selection, and their
        indices in the scene of the group, None if 'envs_idx' is None.
        """
        if envs_idx is None:
            return [(i_group, rows, None) for i_group, rows in enumerate(self.groups_envs_idx)]
        n_envs = len(self.envs_group_idx)
        if isinstance(envs_idx, (int, np.integer)) and not -n_envs <= envs_idx < n_envs:
            gs.raise_exception(f"`envs_idx` out of range: {envs_idx} not in [{-n_envs}, {n_envs}).")
        (envs_mask,) = indices_to_mask(envs_idx, boolean_mask=False)
        envs_group_idx = self.envs_group_idx[envs_mask]
        envs_local_idx = self.envs_local_idx[envs_mask]
        # Sorting by group gathers the selected environments of each group in one block, so a single read back of the
        # counts splits the selection without synchronizing once per group.
        envs_pos = torch.argsort(envs_group_idx, stable=True)
        groups_n_envs = torch.bincount(envs_group_idx, minlength=len(self.groups_envs_idx)).tolist()
        selection = []
        i_start = 0
        for i_group, n_envs in enumerate(groups_n_envs):
            if n_envs > 0:
                rows = envs_pos[i_start : i_start + n_envs]
                selection.append((i_group, rows, envs_local_idx[rows]))
            i_start += n_envs
        return selection


class HeterogeneousEntity(BuildableMixin, RBC):
    """An entity of a heterogeneous scene, simulated in each group scene by an entity of the variant the group carries.

    Its accessors read and write all the environments at once, or those 'envs_idx' selects by index in the heterogeneous
    scene, each group scene receiving the environments it holds. The quantities whose size depends on the variant (the
    qpos, the degrees of freedom and the links) follow a padded layout spanning the largest variant, in which each
    variant holds its own entries, in its own order, as the leading ones. A getter reads zero where the variant of an
    environment has no entry, and a setter leaves such entries out. The local indices ('qs_idx_local', 'dofs_idx_local',
    'links_idx_local') index this padded layout: each group applies the indices its variant has and ignores the others,
    a getter returning one column per index and a setter taking one value per index. The keyword options of the
    accessors are those of 'RigidEntity', with the same defaults. Anything else is reached through the entity of each
    group, 'entities'.
    """

    def __init__(self, spec: Spec, morphs: tuple[Morph, ...]):
        self._spec = spec
        self._morphs = morphs
        self._entities: tuple["RigidEntity", ...] = ()
        self._partition: EnvironmentPartition | None = None
        self._groups_n_qs: tuple[int, ...] = ()
        self._groups_n_dofs: tuple[int, ...] = ()
        self._groups_n_links: tuple[int, ...] = ()
        self._is_dofs_info_batched = False

    @gs.assert_unbuilt
    def build(self, entities: Sequence["RigidEntity"], partition: EnvironmentPartition) -> None:
        """Record the entity of each group scene, 'entities' following the order of the groups in 'partition'."""
        self._entities = tuple(entities)
        self._partition = partition
        # Only the kinematic entities have degrees of freedom, and the accessors wrapped here are theirs alone
        if isinstance(self._entities[0], KinematicEntity):
            self._groups_n_qs = tuple(entity.n_qs for entity in self._entities)
            self._groups_n_dofs = tuple(entity.n_dofs for entity in self._entities)
            self._groups_n_links = tuple(entity.n_links for entity in self._entities)
            # Every group scene receives a copy of the same options
            self._is_dofs_info_batched = self._entities[0].scene.options.rigid.batch_dofs_info
        self._is_built = True

    def _split_columns(self, idx_local, groups_size: tuple[int, ...] | None) -> tuple[tuple[int, ...], list]:
        """Split local indices of the padded layout by group.

        Returns the shape of the column axis, empty for an accessor whose values have none ('groups_size' None), and
        for each group None if it holds no selected column, or the columns of its values in the result and the local
        indices it applies, None for all of them.
        """
        if groups_size is None:
            return (), [(None, None)] * len(self._entities)
        if idx_local is None:
            return (max(groups_size),), [(slice(0, size), None) if size > 0 else None for size in groups_size]
        # The columns are split on the host, which the indices given as a list or an array never leave
        (cols_mask,) = indices_to_mask(idx_local, to_torch=False, boolean_mask=False)
        cols_idx = np.arange(max(groups_size))[tensor_to_cpu(cols_mask)]
        groups_cols = []
        for size in groups_size:
            is_col = cols_idx < size
            cols = torch.as_tensor(np.flatnonzero(is_col), device=gs.device)
            groups_cols.append((cols, cols_idx[is_col]) if is_col.any() else None)
        return (len(cols_idx),), groups_cols

    def _gather(
        self,
        getters: Sequence[Callable],
        envs_idx,
        idx_local=None,
        groups_size: tuple[int, ...] | None = None,
        element_shape: tuple[int, ...] = (),
        n_outputs: int = 1,
        **kwargs,
    ):
        selection = self._partition.select(envs_idx)
        cols_shape, groups_cols = self._split_columns(idx_local, groups_size)
        n_envs = sum(rows.shape[0] for _, rows, _ in selection)
        shape = (n_envs, *cols_shape, *element_shape)
        tensors = tuple(torch.zeros(shape, dtype=gs.tc_float, device=gs.device) for _ in range(n_outputs))
        for i_group, rows, envs_idx_local in selection:
            if groups_cols[i_group] is None:
                continue
            cols, idx_local_group = groups_cols[i_group]
            if cols is None:
                values = getters[i_group](envs_idx=envs_idx_local, **kwargs)
                index = rows
            else:
                values = getters[i_group](idx_local_group, envs_idx=envs_idx_local, **kwargs)
                index = (rows, cols) if isinstance(cols, slice) else (rows[:, None], cols)
            # The values of the parameters shared by the environments of a group have no environment axis, and
            # broadcast over its rows.
            for tensor, value in zip(tensors, values if n_outputs > 1 else (values,)):
                tensor[index] = value
        return tensors if n_outputs > 1 else tensors[0]

    def _scatter(
        self,
        setters: Sequence[Callable],
        values: tuple,
        envs_idx,
        idx_local=None,
        groups_size: tuple[int, ...] | None = None,
        is_batched: bool = True,
        **kwargs,
    ) -> None:
        if not is_batched and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        values = tuple(torch.as_tensor(value, dtype=gs.tc_float, device=gs.device) for value in values)
        selection = self._partition.select(envs_idx)
        cols_shape, groups_cols = self._split_columns(idx_local, groups_size)
        n_envs = sum(rows.shape[0] for _, rows, _ in selection)
        for value in values:
            if cols_shape and value.ndim > 0 and value.shape[-1] != cols_shape[0]:
                gs.raise_exception(f"Expected {cols_shape[0]} entries along the last axis, got {value.shape[-1]}.")
            if value.ndim > 1 and value.shape[0] != n_envs:
                gs.raise_exception(f"Expected one row per selected environment ({n_envs}), got {value.shape[0]}.")
        for i_group, rows, envs_idx_local in selection:
            if groups_cols[i_group] is None:
                continue
            cols, idx_local_group = groups_cols[i_group]
            values_group = [value[rows] if value.ndim > 1 else value for value in values]
            if cols is None:
                setters[i_group](*values_group, envs_idx=envs_idx_local, **kwargs)
            else:
                values_group = [value[..., cols] if value.ndim > 0 else value for value in values_group]
                setters[i_group](*values_group, idx_local_group, envs_idx=envs_idx_local, **kwargs)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ getters ---------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_qpos(self, qs_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Get the qpos of every environment over the padded layout of 'n_qs' entries."""
        getters = [entity.get_qpos for entity in self._entities]
        return self._gather(getters, envs_idx, qs_idx_local, self._groups_n_qs)

    @gs.assert_built
    def get_dofs_position(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Get the position of the degrees of freedom of every environment over the padded layout of 'n_dofs' entries.

        The other getters of degrees of freedom of this entity follow the same layout.
        """
        getters = [entity.get_dofs_position for entity in self._entities]
        return self._gather(getters, envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def get_dofs_velocity(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Get the velocity of the degrees of freedom of every environment."""
        getters = [entity.get_dofs_velocity for entity in self._entities]
        return self._gather(getters, envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Get the smooth force on the degrees of freedom over the last step (see 'RigidEntity.get_dofs_force')."""
        getters = [entity.get_dofs_force for entity in self._entities]
        return self._gather(getters, envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def get_dofs_control_force(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Get the control force the last step applied to the degrees of freedom of every environment."""
        getters = [entity.get_dofs_control_force for entity in self._entities]
        return self._gather(getters, envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def get_dofs_kp(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Get the positional gains of the controller of the degrees of freedom (see 'set_dofs_kp')."""
        getters = [entity.get_dofs_kp for entity in self._entities]
        return self._gather(getters, envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def get_dofs_kv(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Get the velocity gains of the controller of the degrees of freedom (see 'set_dofs_kp')."""
        getters = [entity.get_dofs_kv for entity in self._entities]
        return self._gather(getters, envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def get_dofs_armature(self, dofs_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Get the armature of the degrees of freedom (see 'set_dofs_kp')."""
        getters = [entity.get_dofs_armature for entity in self._entities]
        return self._gather(getters, envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the lower and upper position limits of the degrees of freedom (see 'set_dofs_kp')."""
        getters = [entity.get_dofs_limit for entity in self._entities]
        return self._gather(getters, envs_idx, dofs_idx_local, self._groups_n_dofs, n_outputs=2)

    @gs.assert_built
    def get_dofs_force_range(self, dofs_idx_local=None, envs_idx=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the lower and upper force limits of the degrees of freedom (see 'set_dofs_kp')."""
        getters = [entity.get_dofs_force_range for entity in self._entities]
        return self._gather(getters, envs_idx, dofs_idx_local, self._groups_n_dofs, n_outputs=2)

    @gs.assert_built
    def get_pos(self, envs_idx=None, *, relative=True) -> torch.Tensor:
        """Get the position of the base link of every environment (see 'RigidEntity.get_pos' for 'relative')."""
        getters = [entity.get_pos for entity in self._entities]
        return self._gather(getters, envs_idx, element_shape=(3,), relative=relative)

    @gs.assert_built
    def get_quat(self, envs_idx=None, *, relative=True) -> torch.Tensor:
        """Get the quaternion of the base link of every environment (see 'RigidEntity.get_quat' for 'relative')."""
        getters = [entity.get_quat for entity in self._entities]
        return self._gather(getters, envs_idx, element_shape=(4,), relative=relative)

    @gs.assert_built
    def get_vel(self, envs_idx=None, *, relative=True) -> torch.Tensor:
        """Get the linear velocity of the base link of every environment (see 'RigidEntity.get_vel' for 'relative')."""
        getters = [entity.get_vel for entity in self._entities]
        return self._gather(getters, envs_idx, element_shape=(3,), relative=relative)

    @gs.assert_built
    def get_ang(self, envs_idx=None) -> torch.Tensor:
        """Get the angular velocity of the base link of every environment."""
        getters = [entity.get_ang for entity in self._entities]
        return self._gather(getters, envs_idx, element_shape=(3,))

    @gs.assert_built
    def get_links_pos(
        self, links_idx_local=None, envs_idx=None, *, ref: link_ref_frame = link_ref_frame.link_origin, relative=True
    ) -> torch.Tensor:
        """Get the position of the links of every environment over the padded layout of 'n_links' entries.

        See 'RigidEntity.get_links_pos' for 'ref' and 'relative'.
        """
        getters = [entity.get_links_pos for entity in self._entities]
        return self._gather(
            getters, envs_idx, links_idx_local, self._groups_n_links, element_shape=(3,), ref=ref, relative=relative
        )

    @gs.assert_built
    def get_links_quat(self, links_idx_local=None, envs_idx=None, *, relative=True) -> torch.Tensor:
        """Get the quaternion of the links of every environment (see 'RigidEntity.get_links_quat' for 'relative')."""
        getters = [entity.get_links_quat for entity in self._entities]
        return self._gather(
            getters, envs_idx, links_idx_local, self._groups_n_links, element_shape=(4,), relative=relative
        )

    @gs.assert_built
    def get_links_vel(
        self, links_idx_local=None, envs_idx=None, *, ref: link_ref_frame = link_ref_frame.link_origin, relative=True
    ) -> torch.Tensor:
        """Get the linear velocity of the links (see 'RigidEntity.get_links_vel' for 'ref' and 'relative')."""
        getters = [entity.get_links_vel for entity in self._entities]
        return self._gather(
            getters, envs_idx, links_idx_local, self._groups_n_links, element_shape=(3,), ref=ref, relative=relative
        )

    @gs.assert_built
    def get_links_ang(self, links_idx_local=None, envs_idx=None) -> torch.Tensor:
        """Get the angular velocity of the links of every environment."""
        getters = [entity.get_links_ang for entity in self._entities]
        return self._gather(getters, envs_idx, links_idx_local, self._groups_n_links, element_shape=(3,))

    @gs.assert_built
    def get_mass(self, envs_idx=None) -> torch.Tensor:
        """Get the total mass of the entity in every environment, in kg, over its links not fixed to the world."""
        return self._gather([entity.get_mass for entity in self._entities], envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- setters ----------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_qpos(self, qpos, qs_idx_local=None, envs_idx=None, *, zero_velocity=True, skip_forward=False) -> None:
        """Set the qpos (see 'RigidEntity.set_qpos' for the options).

        'qpos' is a scalar, a vector holding one entry per selected index of the padded layout (all of them if
        'qs_idx_local' is None) shared by every selected environment, or one such vector per selected environment. Every
        setter of this entity takes its values this way.
        """
        setters = [entity.set_qpos for entity in self._entities]
        self._scatter(
            setters,
            (qpos,),
            envs_idx,
            qs_idx_local,
            self._groups_n_qs,
            zero_velocity=zero_velocity,
            skip_forward=skip_forward,
        )

    @gs.assert_built
    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None, *, zero_velocity=True) -> None:
        """Set the position of the degrees of freedom (see 'RigidEntity.set_dofs_position' for 'zero_velocity')."""
        setters = [entity.set_dofs_position for entity in self._entities]
        self._scatter(setters, (position,), envs_idx, dofs_idx_local, self._groups_n_dofs, zero_velocity=zero_velocity)

    @gs.assert_built
    def set_dofs_velocity(self, velocity=None, dofs_idx_local=None, envs_idx=None, *, skip_forward=False) -> None:
        """Set the velocity of the degrees of freedom, zero if 'velocity' is None."""
        setters = [entity.set_dofs_velocity for entity in self._entities]
        velocity = 0.0 if velocity is None else velocity
        self._scatter(setters, (velocity,), envs_idx, dofs_idx_local, self._groups_n_dofs, skip_forward=skip_forward)

    @gs.assert_built
    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the positional gains of the controller of the degrees of freedom.

        Selecting environments requires 'RigidOptions.batch_dofs_info', as for the other parameters of the degrees of
        freedom, the environments of a group sharing them otherwise.
        """
        setters = [entity.set_dofs_kp for entity in self._entities]
        self._scatter(
            setters, (kp,), envs_idx, dofs_idx_local, self._groups_n_dofs, is_batched=self._is_dofs_info_batched
        )

    @gs.assert_built
    def set_dofs_kv(self, kv, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the velocity gains of the controller of the degrees of freedom (see 'set_dofs_kp')."""
        setters = [entity.set_dofs_kv for entity in self._entities]
        self._scatter(
            setters, (kv,), envs_idx, dofs_idx_local, self._groups_n_dofs, is_batched=self._is_dofs_info_batched
        )

    @gs.assert_built
    def set_dofs_armature(self, armature, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the armature of the degrees of freedom (see 'set_dofs_kp')."""
        setters = [entity.set_dofs_armature for entity in self._entities]
        self._scatter(
            setters, (armature,), envs_idx, dofs_idx_local, self._groups_n_dofs, is_batched=self._is_dofs_info_batched
        )

    @gs.assert_built
    def set_dofs_force_range(self, lower, upper, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the lower and upper force limits of the degrees of freedom (see 'set_dofs_kp')."""
        setters = [entity.set_dofs_force_range for entity in self._entities]
        self._scatter(
            setters,
            (lower, upper),
            envs_idx,
            dofs_idx_local,
            self._groups_n_dofs,
            is_batched=self._is_dofs_info_batched,
        )

    @gs.assert_built
    def control_dofs_position(self, position, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the target position of the controller of the degrees of freedom."""
        setters = [entity.control_dofs_position for entity in self._entities]
        self._scatter(setters, (position,), envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def control_dofs_velocity(self, velocity, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the target velocity of the controller of the degrees of freedom."""
        setters = [entity.control_dofs_velocity for entity in self._entities]
        self._scatter(setters, (velocity,), envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def control_dofs_force(self, force, dofs_idx_local=None, envs_idx=None) -> None:
        """Set the motor force of the degrees of freedom."""
        setters = [entity.control_dofs_force for entity in self._entities]
        self._scatter(setters, (force,), envs_idx, dofs_idx_local, self._groups_n_dofs)

    @gs.assert_built
    def set_pos(self, pos, envs_idx=None, *, zero_velocity=True, relative=True, skip_forward=False) -> None:
        """Set the position of the base link (see 'RigidEntity.set_pos' for the options)."""
        setters = [entity.set_pos for entity in self._entities]
        self._scatter(
            setters, (pos,), envs_idx, zero_velocity=zero_velocity, relative=relative, skip_forward=skip_forward
        )

    @gs.assert_built
    def set_quat(self, quat, envs_idx=None, *, zero_velocity=True, relative=True, skip_forward=False) -> None:
        """Set the quaternion of the base link (see 'RigidEntity.set_quat' for the options)."""
        setters = [entity.set_quat for entity in self._entities]
        self._scatter(
            setters, (quat,), envs_idx, zero_velocity=zero_velocity, relative=relative, skip_forward=skip_forward
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------- parts ----------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_link(self, name: str) -> tuple["RigidLink | None", ...]:
        """Get the link of a given name in the entity of each group, None in the groups whose variant has none.

        Raises if no variant has a link of that name.
        """
        links = tuple(next((link for link in entity.links if link.name == name), None) for entity in self._entities)
        if all(link is None for link in links):
            gs.raise_exception(f"No variant of the entity has a link named '{name}'.")
        return links

    @gs.assert_built
    def get_joint(self, name: str) -> tuple["RigidJoint | None", ...]:
        """Get the joint of a given name in the entity of each group, None in the groups whose variant has none.

        Raises if no variant has a joint of that name.
        """
        joints = tuple(
            next((joint for joint in entity.joints if joint.name == name), None) for entity in self._entities
        )
        if all(joint is None for joint in joints):
            gs.raise_exception(f"No variant of the entity has a joint named '{name}'.")
        return joints

    # ------------------------------------------------------------------------------------
    # ---------------------------------- properties --------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def spec(self) -> Spec:
        """The spec of the entity, each group scene replacing its morph with that of the variant the group carries."""
        return self._spec

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant."""
        return self._morphs

    @gs.assert_built
    @property
    def entities(self) -> tuple["RigidEntity", ...]:
        """The entity of each group scene, in the order of the groups of the partition of the scene."""
        return self._entities

    @gs.assert_built
    @property
    def n_qs(self) -> int:
        """The number of entries of the qpos of the largest variant."""
        return max(self._groups_n_qs)

    @gs.assert_built
    @property
    def n_dofs(self) -> int:
        """The number of degrees of freedom of the largest variant."""
        return max(self._groups_n_dofs)

    @gs.assert_built
    @property
    def n_links(self) -> int:
        """The number of links of the largest variant."""
        return max(self._groups_n_links)

    @gs.assert_built
    @property
    def envs_n_qs(self) -> torch.Tensor:
        """The number of entries of the qpos of the variant each environment carries."""
        return torch.tensor(self._groups_n_qs, dtype=gs.tc_int, device=gs.device)[self._partition.envs_group_idx]

    @gs.assert_built
    @property
    def envs_n_dofs(self) -> torch.Tensor:
        """The number of degrees of freedom of the variant each environment carries."""
        return torch.tensor(self._groups_n_dofs, dtype=gs.tc_int, device=gs.device)[self._partition.envs_group_idx]

    @gs.assert_built
    @property
    def envs_n_links(self) -> torch.Tensor:
        """The number of links of the variant each environment carries."""
        return torch.tensor(self._groups_n_links, dtype=gs.tc_int, device=gs.device)[self._partition.envs_group_idx]

    @gs.assert_built
    @property
    def active_dofs_mask(self) -> torch.Tensor:
        """The mask of the degrees of freedom each environment carries, over the 'n_dofs' of the padded layout."""
        return torch.arange(self.n_dofs, device=gs.device) < self.envs_n_dofs[:, None]


@gs.assert_initialized
class HeterogeneousScene(BuildableMixin, RBC):
    """A batch of parallel environments whose entities may differ in structure from one environment to another.

    An entity added with several morphs carries one of them per environment, the environments being split into
    contiguous blocks of nearly equal size, as for a heterogeneous entity of a scene. The environments are partitioned
    into groups by the combination of variants they carry, and each group is an ordinary scene holding one variant of
    every entity, built with as many environments as the group holds. The variants may thus differ in anything a scene
    accepts: the number of links and degrees of freedom, the joint types, a fixed or a free base, the equality
    constraints. Each group scene is simulated at its own size, so an environment takes the memory and the time of the
    variants it carries. Each group scene launches its own kernels at every step, so the time of a step also grows with
    the number of groups, at most one plus the sum over the entities of their number of variants minus one.

    The options are those of 'gs.Scene', and each group scene receives a copy of them. The interactive viewer shows the
    environments of the first group alone. Cameras, sensors, recorders, emitters and force fields are added to a scene
    before it is built, so a heterogeneous scene, whose group scenes come to exist as it builds, offers none of them.
    After build, the scene of each group ('scenes') and the entity of each group ('HeterogeneousEntity.entities') give
    access to everything a built scene offers.
    """

    def __init__(
        self,
        sim_options: SimOptions | None = None,
        rigid_options: RigidOptions | None = None,
        kinematic_options: KinematicOptions | None = None,
        mpm_options: MPMOptions | None = None,
        sph_options: SPHOptions | None = None,
        fem_options: FEMOptions | None = None,
        pbd_options: PBDOptions | None = None,
        coupler_options: BaseCouplerOptions | None = None,
        vis_options: VisOptions | None = None,
        viewer_options: ViewerOptions | None = None,
        profiling_options: ProfilingOptions | None = None,
        renderer: RendererOptions | None = None,
        show_viewer: bool | None = None,
        options: SceneOptions | None = None,
    ):
        individual_options = (
            sim_options,
            rigid_options,
            kinematic_options,
            mpm_options,
            sph_options,
            fem_options,
            pbd_options,
            coupler_options,
            vis_options,
            viewer_options,
            profiling_options,
            renderer,
        )
        if options is not None:
            if any(option is not None for option in individual_options):
                gs.raise_exception("`options` holds every option a scene is created with, so it is given alone.")
            self._options = options
        else:
            self._options = SceneOptions(
                sim=sim_options,
                rigid=rigid_options,
                kinematic=kinematic_options,
                mpm=mpm_options,
                sph=sph_options,
                fem=fem_options,
                pbd=pbd_options,
                coupler=coupler_options,
                vis=vis_options,
                viewer=viewer_options,
                profiling=profiling_options,
                renderer=renderer,
            )
        self._show_viewer = show_viewer
        self._entities: list[HeterogeneousEntity] = []
        self._scenes: tuple[Scene, ...] = ()
        self._partition: EnvironmentPartition | None = None

    @gs.assert_unbuilt
    def add_entity(
        self,
        morph: Morph | Iterable[Morph],
        material: Material | None = None,
        surface: Surface | None = None,
        visualize_contact: bool | None = None,
        name: str | None = None,
    ) -> HeterogeneousEntity:
        """Add an entity to every group scene, as 'gs.Scene.add_entity' does.

        Parameters
        ----------
        morph : gs.morphs.Morph | list[gs.morphs.Morph]
            The morph of the entity, or one morph per variant. Each environment carries one variant, the environments
            being split into contiguous blocks of nearly equal size, and the variants may differ in structure.
        material : gs.materials.Material | None, optional
            The material of the entity. If None, use ``gs.materials.Rigid()``.
        surface : gs.surfaces.Surface | None, optional
            The surface of the entity. If None, use ``gs.surfaces.Default()``.
        visualize_contact : bool | None, optional
            Whether the contacts of the entity are drawn as markers (see 'gs.Scene.add_entity'). If None, defaults to
            False.
        name : str | None, optional
            The name of the entity in every group scene. If None, each group scene generates one.

        Returns
        -------
        entity : HeterogeneousEntity
            The entity, whose accessors are available once the scene is built.
        """
        morphs = (morph,) if isinstance(morph, Morph) else tuple(morph)
        spec = get_entity_spec(morphs[0], material, surface, visualize_contact, name=name)
        entity = HeterogeneousEntity(spec, morphs)
        self._entities.append(entity)
        return entity

    @gs.assert_unbuilt
    def build(self, n_envs: int, env_spacing=(0.0, 0.0), n_envs_per_row: int | None = None, center_envs_at_origin=True):
        """Partition the environments into groups, then create and build the scene of each group.

        Parameters
        ----------
        n_envs : int
            The number of parallel environments, at least one.
        env_spacing : tuple of float, shape (2,)
            The spacing between adjacent environments of each group scene, for visualization only. The environments of
            different groups overlap.
        n_envs_per_row : int | None
            The number of environments per row of each group scene, for visualization only. If None, it is set to the
            square root of the number of environments of the group.
        center_envs_at_origin : bool
            Whether to put the center of the environments of each group scene at the origin, for visualization only.
        """
        if n_envs < 1:
            gs.raise_exception(f"A heterogeneous scene has at least one environment, got 'n_envs={n_envs}'.")

        envs_variants_idx = np.empty((n_envs, len(self._entities)), dtype=gs.np_int)
        for i_entity, entity in enumerate(self._entities):
            envs_variants_idx[:, i_entity] = _balanced_variant_mapping(len(entity.morphs), n_envs)
        groups_variants_idx, envs_group_idx = np.unique(envs_variants_idx, axis=0, return_inverse=True)

        envs_local_idx = np.empty((n_envs,), dtype=gs.np_int)
        groups_envs_idx, scenes, groups_entities = [], [], []
        for i_group, variants_idx in enumerate(groups_variants_idx):
            envs_idx = np.flatnonzero(envs_group_idx == i_group)
            envs_local_idx[envs_idx] = np.arange(len(envs_idx))
            groups_envs_idx.append(torch.as_tensor(envs_idx, device=gs.device))

            scene = Scene(
                options=self._options.model_copy(deep=True), show_viewer=self._show_viewer if i_group == 0 else False
            )
            groups_entities.append(
                [
                    scene.add_entity(spec=entity.spec.model_copy(update={"morph": entity.morphs[i_variant]}))
                    for entity, i_variant in zip(self._entities, variants_idx)
                ]
            )
            scene.build(
                n_envs=len(envs_idx),
                env_spacing=env_spacing,
                n_envs_per_row=n_envs_per_row,
                center_envs_at_origin=center_envs_at_origin,
            )
            scenes.append(scene)

        self._scenes = tuple(scenes)
        self._partition = EnvironmentPartition(
            groups_variants_idx=tuple(map(tuple, groups_variants_idx.tolist())),
            groups_envs_idx=tuple(groups_envs_idx),
            envs_group_idx=torch.as_tensor(envs_group_idx, device=gs.device),
            envs_local_idx=torch.as_tensor(envs_local_idx, device=gs.device),
        )
        for entity, entities in zip(self._entities, zip(*groups_entities)):
            entity.build(entities, self._partition)
        self._is_built = True

    @gs.assert_built
    def step(self) -> None:
        """Run one simulation step of every group scene."""
        for scene in self._scenes:
            scene.step()

    @gs.assert_built
    def reset(self, envs_idx=None) -> None:
        """Reset the environments 'envs_idx' selects by index to their initial state, all of them if None."""
        for i_group, _, envs_idx_local in self._partition.select(envs_idx):
            self._scenes[i_group].reset(envs_idx=envs_idx_local)

    def destroy(self) -> None:
        """Destroy every group scene, releasing its memory."""
        for scene in self._scenes:
            scene.destroy()
        self._scenes = ()
        self._is_built = False

    @property
    def options(self) -> SceneOptions:
        """The options each group scene receives a copy of."""
        return self._options

    @property
    def entities(self) -> tuple[HeterogeneousEntity, ...]:
        """The entities, in the order they were added."""
        return tuple(self._entities)

    @property
    def is_built(self) -> bool:
        """Whether the scene has been built."""
        return self._is_built

    @gs.assert_built
    @property
    def n_envs(self) -> int:
        """The number of parallel environments."""
        return len(self._partition.envs_group_idx)

    @gs.assert_built
    @property
    def scenes(self) -> tuple[Scene, ...]:
        """The scene of each group, in the order of the groups of 'partition'."""
        return self._scenes

    @gs.assert_built
    @property
    def partition(self) -> EnvironmentPartition:
        """The partition of the environments into groups."""
        return self._partition
