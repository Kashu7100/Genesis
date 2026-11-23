from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
from genesis.constants import link_ref_frame
from genesis.options.entities import Spec
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface
from genesis.repr_base import RBC
from genesis.utils.misc import BuildableMixin, indices_to_mask, tensor_to_cpu

from .rigid_entity import KinematicEntity

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.grouped_solver import EnvironmentPartition

    from .rigid_entity import RigidEntity
    from .rigid_joint import RigidJoint
    from .rigid_link import RigidLink


class HeterogeneousEntity(BuildableMixin, RBC):
    """A rigid entity of a scene whose rigid simulation is split into groups of environments (see GroupedRigidSolver).

    The rigid solver of each group simulates the entity by an entity of the variant the group carries, the same entity
    in every group for an entity created from one morph. Its accessors read and write all the environments at once, or
    those 'envs_idx' selects by index in the scene, each group receiving the environments it holds. The quantities whose
    size depends on the variant (the qpos, the degrees of freedom and the links) follow a padded layout spanning the
    largest variant, in which each variant holds its own entries, in its own order, as the leading ones. A getter reads
    zero where the variant of an environment has no entry, and a setter leaves such entries out. The local indices
    ('qs_idx_local', 'dofs_idx_local', 'links_idx_local') index this padded layout: each group applies the indices its
    variant has and ignores the others, a getter returning one column per index and a setter taking one value per index.
    The keyword options of the accessors are those of 'RigidEntity', with the same defaults. Anything else is reached
    through the entity of each group, 'entities'.
    """

    def __init__(self, idx: int, surface: Surface, spec: Spec):
        """Record an entity of the scene of index 'idx', made from 'spec' with the surface the scene resolved for it."""
        self._uid = gs.UID()
        self._idx = idx
        self._surface = surface
        self._name = f"heterogeneous_{self._uid.short()}" if spec.name is None else spec.name
        self._spec = spec._frozen_copy(update={"name": self._name})
        self._entities: tuple["RigidEntity", ...] = ()
        self._partition: "EnvironmentPartition | None" = None
        self._groups_n_qs: tuple[int, ...] = ()
        self._groups_n_dofs: tuple[int, ...] = ()
        self._groups_n_links: tuple[int, ...] = ()
        self._is_dofs_info_batched = False
        self._is_batched = True

    @gs.assert_unbuilt
    def build(self, entities: Sequence["RigidEntity"], partition: "EnvironmentPartition", is_batched: bool) -> None:
        """Record the entity of each group, 'entities' following the order of the groups in 'partition'.

        'is_batched' states that the scene has parallel environments. A scene without them forms a single group, whose
        entity the accessors call as they are given.
        """
        self._entities = tuple(entities)
        self._partition = partition
        self._is_batched = is_batched
        # Only the kinematic entities have degrees of freedom, and the accessors wrapped here are theirs alone
        if isinstance(self._entities[0], KinematicEntity):
            self._groups_n_qs = tuple(entity.n_qs for entity in self._entities)
            self._groups_n_dofs = tuple(entity.n_dofs for entity in self._entities)
            self._groups_n_links = tuple(entity.n_links for entity in self._entities)
            # Every group solver receives a copy of the same options
            self._is_dofs_info_batched = self._entities[0].solver.options.batch_dofs_info
        self._is_built = True

    def _destroy(self) -> None:
        self._is_built = False

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
        if not self._is_batched:
            if groups_size is None:
                return getters[0](envs_idx=envs_idx, **kwargs)
            return getters[0](idx_local, envs_idx=envs_idx, **kwargs)
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
        if not self._is_batched:
            if groups_size is None:
                setters[0](*values, envs_idx=envs_idx, **kwargs)
            else:
                setters[0](*values, idx_local, envs_idx=envs_idx, **kwargs)
            return
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
    def uid(self) -> gs.UID:
        """The identifier of the entity in the scene."""
        return self._uid

    @property
    def idx(self) -> int:
        """The index of the entity in the scene, which the entity of each group shares."""
        return self._idx

    @property
    def name(self) -> str:
        """The name of the entity, the one given to the scene or one generated from its identifier."""
        return self._name

    @property
    def surface(self) -> Surface:
        """The surface the entity of each group is drawn with."""
        return self._surface

    @property
    def desc(self) -> None:
        """The description of the entity, which none carries, so that 'Scene.export' rejects the scene."""
        return None

    @property
    def spec(self) -> Spec:
        """The spec of the entity, each group replacing its morphs with the morph of the variant the group carries."""
        return self._spec

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant."""
        return self._spec.morphs

    @gs.assert_built
    @property
    def entities(self) -> tuple["RigidEntity", ...]:
        """The entity of each group, in the order of the groups of the partition of the scene."""
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
