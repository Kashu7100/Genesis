import dataclasses
import functools
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.constants import link_ref_frame
from genesis.engine.entities.rigid_entity import KinematicEntity
from genesis.engine.entities.rigid_entity.description import _balanced_variant_mapping
from genesis.engine.materials.base import Material
from genesis.engine.scene import Scene, resolve_scene_options
from genesis.engine.states.solvers import SimState
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
from genesis.options.sensors import SensorOptions
from genesis.options.surfaces import Surface
from genesis.repr_base import RBC
from genesis.utils.misc import BuildableMixin, indices_to_mask, tensor_to_cpu
from genesis.vis.camera import Camera

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint, RigidLink
    from genesis.engine.sensors.base_sensor import Sensor


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


class HeterogeneousSensor(BuildableMixin, RBC):
    """A sensor of a heterogeneous scene, made of one sensor per group scene created from the options of its variant.

    Its readings cover every environment, or those 'envs_idx' selects by index in the heterogeneous scene, one row per
    selected environment in the order of the selection, and take the type the sensor of a scene returns. Anything else,
    such as the setters of a sensor, is reached through the sensor of each group, 'sensors'.
    """

    def __init__(self, variants_options: tuple[SensorOptions, ...]):
        self._variants_options = variants_options
        self._sensors: tuple["Sensor", ...] = ()
        self._partition: EnvironmentPartition | None = None

    @gs.assert_unbuilt
    def build(self, sensors: Sequence["Sensor"], partition: EnvironmentPartition) -> None:
        """Record the sensor of each group scene, raising if its readings differ in shape from one group to another.

        'sensors' follows the order of the groups in 'partition'.
        """
        # The entries of the readings of different variants share no meaning, which padding them to a common shape
        # would pretend, so a reading takes one shape in every group
        readings_shape = set()
        for sensor in sensors:
            reading = sensor.read()
            fields = reading if isinstance(reading, tuple) else (reading,)
            readings_shape.add(tuple(field.shape[1:] for field in fields))
        if len(readings_shape) > 1:
            gs.raise_exception(
                f"The readings of {type(self._variants_options[0]).__name__} differ in shape from one group of "
                f"environments to another, {sorted(readings_shape)}. Give it options reading as many entries in every "
                "variant."
            )
        self._sensors = tuple(sensors)
        self._partition = partition
        self._is_built = True

    def _gather(self, readers: Sequence[Callable], envs_idx):
        selection = self._partition.select(envs_idx)
        groups_rows, groups_fields = [], []
        for i_group, rows, envs_idx_local in selection:
            reading = readers[i_group](envs_idx=envs_idx_local)
            groups_rows.append(rows)
            groups_fields.append(reading if isinstance(reading, tuple) else (reading,))
        n_envs = sum(rows.shape[0] for rows in groups_rows)
        fields = []
        for field_groups in zip(*groups_fields):
            # Every selected environment is the row of exactly one group, so each entry is written
            field = field_groups[0].new_empty((n_envs, *field_groups[0].shape[1:]))
            for rows, field_group in zip(groups_rows, field_groups):
                field[rows] = field_group
            fields.append(field)
        if isinstance(reading, tuple):
            return type(reading)(*fields)
        return fields[0]

    @gs.assert_built
    def read(self, envs_idx=None):
        """Read the sensor in the selected environments, with the imperfections of its options (see 'Sensor.read')."""
        return self._gather([sensor.read for sensor in self._sensors], envs_idx)

    @gs.assert_built
    def read_ground_truth(self, envs_idx=None):
        """Read the sensor in the selected environments, without imperfections (see 'Sensor.read_ground_truth')."""
        return self._gather([sensor.read_ground_truth for sensor in self._sensors], envs_idx)

    @property
    def variants_options(self) -> tuple[SensorOptions, ...]:
        """The options of the sensor, a single one or one per variant of the entity it is attached to."""
        return self._variants_options

    @gs.assert_built
    @property
    def sensors(self) -> tuple["Sensor", ...]:
        """The sensor of each group scene, in the order of the groups of the partition of the scene."""
        return self._sensors


class HeterogeneousRender(NamedTuple):
    """The images a camera of a heterogeneous scene renders, in the order of the result of 'Camera.render'.

    Each one holds the image of every group scene, in the order of the groups of the partition of the scene, as the
    camera of that group scene renders it (see 'Camera.render'), or None if it was not asked for.
    """

    rgb: tuple | None
    """The RGB image of each group scene."""
    depth: tuple | None
    """The depth image of each group scene."""
    segmentation: tuple | None
    """The segmentation mask of each group scene."""
    normal: tuple | None
    """The surface normal image of each group scene."""


class HeterogeneousCamera(BuildableMixin, RBC):
    """A camera of a heterogeneous scene, created at the same pose in every group scene.

    Each group scene renders its own environments alone, so a render holds the images of each group scene apart.
    Anything beyond rendering is reached through the camera of each group scene, 'cameras'.
    """

    def __init__(self):
        self._cameras: tuple[Camera, ...] = ()

    @gs.assert_unbuilt
    def build(self, cameras: Sequence[Camera]) -> None:
        """Record the camera of each group scene, in the order of the groups of the partition of the scene."""
        self._cameras = tuple(cameras)
        self._is_built = True

    @gs.assert_built
    def render(
        self,
        rgb=True,
        depth=False,
        segmentation=False,
        colorize_seg=False,
        normal=False,
        antialiasing=False,
        force_render=False,
    ) -> HeterogeneousRender:
        """Render the camera of every group scene, with the arguments of 'Camera.render'.

        Returns
        -------
        render : HeterogeneousRender
            The images of each group scene, as 'Camera.render' returns them for a scene.
        """
        groups_images = [
            camera.render(rgb, depth, segmentation, colorize_seg, normal, antialiasing, force_render)
            for camera in self._cameras
        ]
        return HeterogeneousRender(*(None if images[0] is None else images for images in zip(*groups_images)))

    @gs.assert_built
    @property
    def cameras(self) -> tuple[Camera, ...]:
        """The camera of each group scene, in the order of the groups of the partition of the scene."""
        return self._cameras


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

    The options are those of 'gs.Scene', and each group scene receives a copy of them. Each group scene lays out and
    renders its own environments, so the interactive viewer shows a single group scene, chosen at build, and a camera
    renders each group scene apart. Sensors, cameras and lights are added before the build, as for a scene, and each
    group scene receives them as it is created. Recorders, emitters, force fields and USD stages are unavailable. After
    build, the scene of each group ('scenes') and the entity of each group ('HeterogeneousEntity.entities') give access
    to everything a built scene offers.
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
        self._options = resolve_scene_options(
            options,
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
        # Each group scene numbers its own environments, where an index of the heterogeneous scene selects another one
        if self._options.vis.rendered_envs_idx is not None:
            gs.raise_exception("A heterogeneous scene renders all its environments. Leave 'rendered_envs_idx' None.")
        self._show_viewer = show_viewer
        self._scenes: tuple[Scene, ...] = ()
        self._entities: list[HeterogeneousEntity] = []
        self._sensors: list[HeterogeneousSensor] = []
        # The request adding each camera and each light to a group scene, called on every group scene as it is created
        self._cameras: list[tuple[functools.partial, HeterogeneousCamera]] = []
        self._lights: list[functools.partial] = []
        self._partition: EnvironmentPartition | None = None

    @gs.assert_unbuilt
    def add_entity(
        self,
        morph: Morph | Iterable[Morph] | None = None,
        material: Material | None = None,
        surface: Surface | None = None,
        visualize_contact: bool | None = None,
        name: str | None = None,
        *,
        spec: Spec | None = None,
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
        spec : gs.entities.Spec | None, optional
            The spec to construct the entity from, in place of the individual arguments. Its morph may be one morph per
            variant, as 'morph' may.

        Returns
        -------
        entity : HeterogeneousEntity
            The entity, whose accessors are available once the scene is built.
        """
        if spec is None:
            if morph is None:
                gs.raise_exception("Entity construction requires a morph.")
            morphs = (morph,) if isinstance(morph, Morph) else tuple(morph)
            # The spec takes the first variant alone, since every kind of entity takes the one morph of a group scene
            spec = get_entity_spec(morphs[0], material, surface, visualize_contact, name=name)
        elif any(value is not None for value in (morph, material, surface, visualize_contact, name)):
            gs.raise_exception("Pass either an entity spec or individual entity arguments, never both.")
        else:
            morphs = spec.morphs
            if not morphs:
                gs.raise_exception("Entity construction requires a morph.")
        entity = HeterogeneousEntity(spec, morphs)
        self._entities.append(entity)
        return entity

    @gs.assert_unbuilt
    def add_sensor(self, sensor_options: SensorOptions | Sequence[SensorOptions]) -> HeterogeneousSensor:
        """Add a sensor to every group scene, as 'gs.Scene.add_sensor' does.

        The entities are added to every group scene in the same order, so 'entity_idx' names the same entity in all of
        them, while a local index (of a link, of a degree of freedom) may name another part in each variant of that
        entity. One options object applies to every group scene, and the build raises if it is invalid for a variant.
        One options object per variant of the entity that 'entity_idx' names, all of the same type and naming that
        entity, gives each group scene the options of the variant it carries, as 'add_entity' takes one morph per
        variant. Indices of links in the solver, such as 'filter_link_idx', index the links of each group scene.

        Parameters
        ----------
        sensor_options : SensorOptions | list[SensorOptions]
            The options of the sensor, or one options object per variant of the entity it is attached to.

        Returns
        -------
        sensor : HeterogeneousSensor
            The sensor, which reads every environment once the scene is built. The build raises if its readings differ
            in shape from one group scene to another.
        """
        variants_options = (sensor_options,) if isinstance(sensor_options, SensorOptions) else tuple(sensor_options)
        if len(variants_options) > 1:
            entity_idx = variants_options[0].entity_idx
            if not 0 <= entity_idx < len(self._entities):
                gs.raise_exception(f"Options per variant name an entity of the scene, got 'entity_idx={entity_idx}'.")
            if any(
                type(options) is not type(variants_options[0]) or options.entity_idx != entity_idx
                for options in variants_options
            ):
                gs.raise_exception("The options of the variants of a sensor share their type and their 'entity_idx'.")
            n_variants = len(self._entities[entity_idx].morphs)
            if len(variants_options) != n_variants:
                gs.raise_exception(
                    f"Expected one options object per variant of entity {entity_idx} ({n_variants}), got "
                    f"{len(variants_options)}."
                )
        sensor = HeterogeneousSensor(variants_options)
        self._sensors.append(sensor)
        return sensor

    @gs.assert_unbuilt
    def add_camera(
        self,
        model="pinhole",
        res=(320, 320),
        pos=(0.5, 2.5, 3.5),
        lookat=(0.5, 0.5, 0.5),
        up=(0.0, 0.0, 1.0),
        fov=30,
        GUI=False,
        spp=256,
        denoise=None,
        near=0.1,
        far=20.0,
        debug=False,
        lens=None,
    ) -> HeterogeneousCamera:
        """Add a camera at the same pose to every group scene, with the arguments of 'gs.Scene.add_camera'.

        The camera of each group scene takes the environment its scene binds a camera to by default, since an index of
        an environment of the heterogeneous scene names an environment of one group scene alone.

        Returns
        -------
        camera : HeterogeneousCamera
            The camera, which renders every group scene once the scene is built.
        """
        add_camera = functools.partial(
            Scene.add_camera,
            model=model,
            res=res,
            pos=pos,
            lookat=lookat,
            up=up,
            fov=fov,
            GUI=GUI,
            spp=spp,
            denoise=denoise,
            near=near,
            far=far,
            debug=debug,
            lens=lens,
        )
        camera = HeterogeneousCamera()
        self._cameras.append((add_camera, camera))
        return camera

    @gs.assert_unbuilt
    def add_light(
        self,
        pos,
        dir,
        color=(1.0, 1.0, 1.0),
        intensity=1.0,
        directional=False,
        castshadow=True,
        cutoff=45.0,
        attenuation=0.0,
    ) -> None:
        """Add a light of the batch renderer to every group scene, with the arguments of 'gs.Scene.add_light'."""
        self._lights.append(
            functools.partial(
                Scene.add_light,
                pos=pos,
                dir=dir,
                color=color,
                intensity=intensity,
                directional=directional,
                castshadow=castshadow,
                cutoff=cutoff,
                attenuation=attenuation,
            )
        )

    @gs.assert_unbuilt
    def add_mesh_light(
        self,
        morph=None,
        color=(1.0, 1.0, 1.0, 1.0),
        intensity=20.0,
        revert_dir=False,
        double_sided=False,
        cutoff=180.0,
    ) -> None:
        """Add a mesh light of the ray tracer to every group scene, with the arguments of 'gs.Scene.add_mesh_light'."""
        self._lights.append(
            functools.partial(
                Scene.add_mesh_light,
                morph=morph,
                color=color,
                intensity=intensity,
                revert_dir=revert_dir,
                double_sided=double_sided,
                cutoff=cutoff,
            )
        )

    @gs.assert_unbuilt
    def build(
        self,
        n_envs: int,
        env_spacing=(0.0, 0.0),
        n_envs_per_row: int | None = None,
        center_envs_at_origin=True,
        viewer_env_idx: int = 0,
    ):
        """Partition the environments into groups, then create and build the scene of each group.

        If the build of a group scene raises, the group scenes created so far are destroyed and the scene stays unbuilt.

        Parameters
        ----------
        n_envs : int
            The number of parallel environments, at least one.
        env_spacing : tuple of float, shape (2,)
            The spacing between adjacent environments of each group scene, for visualization only. Each group scene
            lays out its own environments.
        n_envs_per_row : int | None
            The number of environments per row of each group scene, for visualization only. If None, it is set to the
            square root of the number of environments of the group.
        center_envs_at_origin : bool
            Whether to put the center of the environments of each group scene at the origin, for visualization only.
        viewer_env_idx : int
            The environment whose group scene the interactive viewer shows, if 'show_viewer' asks for it. The viewer
            shows the environments of that group scene alone. Defaults to the first environment.
        """
        if n_envs < 1:
            gs.raise_exception(f"A heterogeneous scene has at least one environment, got 'n_envs={n_envs}'.")
        if not 0 <= viewer_env_idx < n_envs:
            gs.raise_exception(f"'viewer_env_idx' names one of the {n_envs} environments, got {viewer_env_idx}.")

        envs_variants_idx = np.empty((n_envs, len(self._entities)), dtype=gs.np_int)
        for i_entity, entity in enumerate(self._entities):
            envs_variants_idx[:, i_entity] = _balanced_variant_mapping(len(entity.morphs), n_envs)
        groups_variants_idx, envs_group_idx = np.unique(envs_variants_idx, axis=0, return_inverse=True)
        i_viewer_group = envs_group_idx[viewer_env_idx]

        envs_local_idx = np.empty((n_envs,), dtype=gs.np_int)
        groups_envs_idx, scenes, groups_entities, groups_sensors, groups_cameras = [], [], [], [], []
        # A failed build destroys every group scene created so far, so that none outlives the unbuilt scene
        is_done = False
        try:
            for i_group, variants_idx in enumerate(groups_variants_idx):
                envs_idx = np.flatnonzero(envs_group_idx == i_group)
                envs_local_idx[envs_idx] = np.arange(len(envs_idx))
                groups_envs_idx.append(torch.as_tensor(envs_idx, device=gs.device))

                show_viewer = self._show_viewer if i_group == i_viewer_group else False
                scene = Scene(options=self._options.model_copy(deep=True), show_viewer=show_viewer)
                scenes.append(scene)
                groups_entities.append(
                    [
                        scene.add_entity(spec=entity.spec.model_copy(update={"morph": entity.morphs[i_variant]}))
                        for entity, i_variant in zip(self._entities, variants_idx)
                    ]
                )
                sensors = []
                for i_sensor, sensor in enumerate(self._sensors):
                    variants_options = sensor.variants_options
                    i_variant = variants_idx[variants_options[0].entity_idx] if len(variants_options) > 1 else 0
                    # Each group scene receives its own copy of the options, which a sensor may resolve in place
                    try:
                        sensors.append(scene.add_sensor(variants_options[i_variant].model_copy(deep=True)))
                    except gs.GenesisException as e:
                        gs.raise_exception_from(
                            f"The options of sensor {i_sensor} are invalid for the variants "
                            f"{tuple(variants_idx.tolist())} of the entities, in the order they were added. Give one "
                            "options object per variant of the entity it is attached to (see "
                            "'HeterogeneousScene.add_sensor').",
                            e,
                        )
                groups_sensors.append(sensors)
                groups_cameras.append([add_camera(scene) for add_camera, _ in self._cameras])
                for add_light in self._lights:
                    add_light(scene)
                scene.build(
                    n_envs=len(envs_idx),
                    env_spacing=env_spacing,
                    n_envs_per_row=n_envs_per_row,
                    center_envs_at_origin=center_envs_at_origin,
                )

            partition = EnvironmentPartition(
                groups_variants_idx=tuple(map(tuple, groups_variants_idx.tolist())),
                groups_envs_idx=tuple(groups_envs_idx),
                envs_group_idx=torch.as_tensor(envs_group_idx, device=gs.device),
                envs_local_idx=torch.as_tensor(envs_local_idx, device=gs.device),
            )
            # The sensors build first, since their build alone may raise, which would leave the entities built before it
            for sensor, sensors in zip(self._sensors, zip(*groups_sensors)):
                sensor.build(sensors, partition)
            for entity, entities in zip(self._entities, zip(*groups_entities)):
                entity.build(entities, partition)
            for (_, camera), cameras in zip(self._cameras, zip(*groups_cameras)):
                camera.build(cameras)
            is_done = True
        finally:
            if not is_done:
                for scene in scenes:
                    scene.destroy()

        self._scenes = tuple(scenes)
        self._partition = partition
        self._is_built = True

    @gs.assert_built
    def step(self) -> None:
        """Run one simulation step of every group scene."""
        for scene in self._scenes:
            scene.step()

    @gs.assert_built
    def get_state(self) -> tuple[SimState, ...]:
        """Return the current state of every group scene, which 'reset' restores.

        The state of each group scene ('gs.Scene.get_state') comes in the order of the groups of 'partition'.
        """
        return tuple(scene.get_state() for scene in self._scenes)

    @gs.assert_built
    def reset(self, state: tuple[SimState, ...] | None = None, envs_idx=None) -> None:
        """Reset the selected environments to the initial state of the scene.

        'state', as 'get_state' returns it, becomes the initial state of the scene before the reset, as for
        'gs.Scene.reset', so a later reset restores it in every environment. 'envs_idx' selects environments by index,
        all of them if None.
        """
        if state is not None and (not isinstance(state, tuple) or len(state) != len(self._scenes)):
            gs.raise_exception(f"Expected the state of each of the {len(self._scenes)} group scenes ('get_state').")
        # Every group scene takes the state, with an empty selection where no environment of the group is selected, so
        # that the state becomes the initial state of all of them
        groups_envs_idx_local = [self._partition.envs_local_idx[:0]] * len(self._scenes)
        for i_group, _, envs_idx_local in self._partition.select(envs_idx):
            groups_envs_idx_local[i_group] = envs_idx_local
        groups_state = (None,) * len(self._scenes) if state is None else state
        for scene, scene_state, envs_idx_local in zip(self._scenes, groups_state, groups_envs_idx_local):
            scene.reset(scene_state, envs_idx=envs_idx_local)

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
    def sensors(self) -> tuple[HeterogeneousSensor, ...]:
        """The sensors, in the order they were added."""
        return tuple(self._sensors)

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
