import dataclasses
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

import genesis as gs
from genesis.engine.entities.rigid_entity.description import RigidEntityDescription, _balanced_variant_mapping
from genesis.engine.entities.rigid_entity.heterogeneous_entity import HeterogeneousEntity
from genesis.engine.state_change import Subscriber
from genesis.engine.states import SolverCheckpoint
from genesis.utils.data import ConfigItem, DataItem, namespace
from genesis.utils.misc import indices_to_mask

from ..base_solver import GravityMixin, Solver, TimeBasedMixin
from .rigid_solver import RigidSolver

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator
    from genesis.options.solvers import RigidOptions
    from genesis.options.surfaces import Surface


@dataclasses.dataclass(frozen=True)
class EnvironmentPartition:
    """The partition of the environments of a scene into groups, each group simulated by a rigid solver of its own.

    The environments of a group keep their order in the solver of the group: the environment of local index i in group g
    is 'groups_envs_idx[g][i]'.
    """

    groups_variants_idx: tuple[tuple[int, ...], ...]
    """The variant of each rigid entity that each group carries, the entities in the order they were added."""
    groups_envs_idx: tuple[torch.Tensor, ...]
    """The environments of each group, in increasing order."""
    envs_group_idx: torch.Tensor
    """The group of each environment."""
    envs_local_idx: torch.Tensor
    """The index of each environment in the solver of its group."""

    def select(self, envs_idx=None) -> list[tuple[int, torch.Tensor, torch.Tensor | None]]:
        """Split a selection of environments by group.

        'envs_idx' selects environments of the scene by index, like the 'envs_idx' argument of the accessors of a scene,
        None selecting them all. The result holds one entry per group with at least one selected environment: the index
        of the group, the positions of its selected environments in the selection, and their indices in the solver of
        the group, None if 'envs_idx' is None.
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


def merge_groups_checkpoints(checkpoints: list[SolverCheckpoint]) -> SolverCheckpoint:
    """Merge the checkpoints of the solvers of the groups into one, the entries of each named under 'groups.<index>'."""
    return SolverCheckpoint(
        arrays={
            f"groups.{i_group}.{name}": array
            for i_group, checkpoint in enumerate(checkpoints)
            for name, array in checkpoint.arrays.items()
        },
        configs={
            f"groups.{i_group}.{name}": config
            for i_group, checkpoint in enumerate(checkpoints)
            for name, config in checkpoint.configs.items()
        },
        kinds=checkpoints[0].kinds,
    )


def split_groups_checkpoint(checkpoint: SolverCheckpoint, i_group: int) -> SolverCheckpoint:
    """Return the checkpoint of the solver of one group out of those 'merge_groups_checkpoints' merged."""
    prefix = f"groups.{i_group}."
    return SolverCheckpoint(
        arrays={name[len(prefix) :]: array for name, array in checkpoint.arrays.items() if name.startswith(prefix)},
        configs={name[len(prefix) :]: config for name, config in checkpoint.configs.items() if name.startswith(prefix)},
        kinds=checkpoint.kinds,
    )


class GroupedRigidSolver(GravityMixin, TimeBasedMixin, Solver):
    """The rigid solver of a scene holding a rigid entity whose variants differ in structure, simulating groups of
    environments apart.

    The environments are partitioned into groups by the combination of variants of the rigid entities they carry, each
    entity added with several morphs carrying one of them per environment, the environments being split into contiguous
    blocks of nearly equal size. A rigid solver of its own simulates each group at the size of the group, holding one
    variant of every rigid entity, so the variants may differ in anything a rigid solver accepts: the number of links and
    degrees of freedom, the joint types, a fixed or a free base, the equality constraints. An environment takes the
    memory and the time of the variants it carries, while the time of a step grows with the number of groups too, each
    group launching its own kernels.

    The rigid entities of the scene are HeterogeneousEntity, which gather and scatter their accessors over the groups.
    The solver takes part in the simulation as one rigid solver: it steps, resets, reports errors and saves its state for
    every group, its state holding that of each group.
    """

    entity_spec_cls: ClassVar[type[gs.entities.Spec]] = gs.entities.Rigid

    # Its whole state lives in the solvers of its groups, whose data it yields (see 'data')
    is_checkpointable = True

    # override typing
    _entities: list[HeterogeneousEntity] = gs.List()

    def __init__(self, sim: "Simulator", options: "RigidOptions") -> None:
        super().__init__(sim, options)
        self._solvers: tuple[RigidSolver, ...] = ()
        self._partition: EnvironmentPartition | None = None

    def add_entity(
        self,
        idx: int,
        surface: "Surface | None",
        spec: "gs.entities.Rigid | None",
        desc: RigidEntityDescription | None = None,
    ) -> HeterogeneousEntity:
        """Record a rigid entity of index 'idx' in the scene, which the solver of each group creates as the solver builds.

        The entity is created from the spec, each group taking the morph of the variant it carries, since the groups are
        only known once the number of environments is.
        """
        if desc is not None:
            gs.raise_exception("A scene whose rigid entities differ in structure cannot be created from descriptions.")
        entity = HeterogeneousEntity(idx, surface, spec)
        self._entities.append(entity)
        return entity

    @gs.assert_unbuilt
    def build(self, n_envs: int) -> None:
        """Partition the environments into groups, then create the solver of each group with its entities and build it.

        The descriptions resolved for the first group holding a variant of an entity are shared by the next ones, so
        that every variant reads its asset once.
        """
        if n_envs < 1:
            gs.raise_exception(
                f"A scene whose rigid entities differ in structure has parallel environments, got 'n_envs={n_envs}'."
            )
        super().build(n_envs)

        envs_variants_idx = np.empty((n_envs, len(self._entities)), dtype=gs.np_int)
        for i_entity, entity in enumerate(self._entities):
            envs_variants_idx[:, i_entity] = _balanced_variant_mapping(len(entity.morphs), n_envs)
        groups_variants_idx, envs_group_idx = np.unique(envs_variants_idx, axis=0, return_inverse=True)

        envs_local_idx = np.empty((n_envs,), dtype=gs.np_int)
        variants_desc: dict[tuple[int, int], RigidEntityDescription] = {}
        groups_envs_idx, solvers, groups_entities = [], [], []
        for i_group, variants_idx in enumerate(groups_variants_idx):
            envs_idx = np.flatnonzero(envs_group_idx == i_group)
            envs_local_idx[envs_idx] = np.arange(len(envs_idx))
            scene_envs_idx = torch.as_tensor(envs_idx, dtype=gs.tc_int, device=gs.device)
            groups_envs_idx.append(scene_envs_idx)
            # Each group takes a copy of the options, which a rigid solver resolves in place as it builds
            solver = RigidSolver(self._sim, self._options.model_copy(deep=True), scene_envs_idx)
            entities = []
            for i_entity, (entity, i_variant) in enumerate(zip(self._entities, variants_idx)):
                desc = variants_desc.get((i_entity, i_variant))
                if desc is None:
                    # The entity of a group holds no name of its own, the scene naming the heterogeneous entity alone
                    spec = entity.spec.model_copy(update={"morph": entity.morphs[i_variant], "name": None})
                    entities.append(solver.add_entity(entity.idx, entity.surface, spec))
                    variants_desc[(i_entity, i_variant)] = entities[-1].desc
                else:
                    entities.append(solver.add_entity(entity.idx, entity.surface, None, desc))
            groups_entities.append(entities)
            # The simulator settles the substep of every solver before building them (see 'Simulator.build')
            solver._substeps, solver._substep_dt = self._substeps, self._substep_dt
            # The structs of each group are opted in and sized in a namespace of their own (see 'opt_in')
            with namespace(f"rigid_group_{i_group}"):
                solver.build(len(envs_idx))
            solvers.append(solver)
        self._solvers = tuple(solvers)

        self._partition = EnvironmentPartition(
            groups_variants_idx=tuple(map(tuple, groups_variants_idx.tolist())),
            groups_envs_idx=tuple(groups_envs_idx),
            envs_group_idx=torch.as_tensor(envs_group_idx, dtype=gs.tc_int, device=gs.device),
            envs_local_idx=torch.as_tensor(envs_local_idx, dtype=gs.tc_int, device=gs.device),
        )
        for entity, entities in zip(self._entities, zip(*groups_entities)):
            entity.build(entities, self._partition)
        self._is_built = True

    def destroy(self) -> None:
        for solver in self._solvers:
            solver.destroy()
        super().destroy()

    def subscribe(self, subscriber: Subscriber) -> None:
        """Register a Subscriber with the solver of every group (see 'Solver.subscribe')."""
        for solver in self._solvers:
            solver.subscribe(subscriber)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep(self) -> None:
        """Run one substep of the solver of every group, for a scene the rigid solver alone advances."""
        for solver in self._solvers:
            solver.substep()

    def substep_pre_coupling(self) -> None:
        for solver in self._solvers:
            solver.substep_pre_coupling()

    def substep_post_coupling(self) -> None:
        for solver in self._solvers:
            solver.substep_post_coupling()

    def clear_external_force(self) -> None:
        """Clear the external forces of the solver of every group, which a step applies once."""
        for solver in self._solvers:
            solver.clear_external_force()

    # ------------------------------------------------------------------------------------
    # ------------------------------------- gravity --------------------------------------
    # ------------------------------------------------------------------------------------

    def set_gravity(self, gravity, envs_idx=None) -> None:
        """Set the gravity of the selected environments of the scene, in m/s^2 (see 'GravityMixin.set_gravity')."""
        gravity = torch.as_tensor(gravity, dtype=gs.tc_float, device=gs.device)
        for i_group, rows, envs_idx_local in self._partition.select(envs_idx):
            self._solvers[i_group].set_gravity(gravity[rows] if gravity.ndim > 1 else gravity, envs_idx_local)

    def get_gravity(self, envs_idx=None) -> torch.Tensor:
        """Get the gravity of the selected environments of the scene, in m/s^2 (see 'GravityMixin.get_gravity')."""
        selection = self._partition.select(envs_idx)
        gravity = torch.empty((sum(len(rows) for _, rows, _ in selection), 3), dtype=gs.tc_float, device=gs.device)
        for i_group, rows, envs_idx_local in selection:
            gravity[rows] = self._solvers[i_group].get_gravity(envs_idx_local)
        return gravity

    # ------------------------------------------------------------------------------------
    # -------------------------------------- errors --------------------------------------
    # ------------------------------------------------------------------------------------

    def update_errno(self) -> None:
        for solver in self._solvers:
            solver.update_errno()

    def check_errno(self, blocking: bool = True, *, context: str | None = None) -> None:
        """Raise on the first error the solver of a group flagged, naming the environments in the scene."""
        for solver in self._solvers:
            solver.check_errno(blocking, context=context)

    def get_error_envs_mask(self) -> torch.Tensor:
        envs_mask = torch.zeros((self.n_envs_,), dtype=torch.bool, device=gs.device)
        for solver in self._solvers:
            envs_mask[solver.scene_envs_idx] = solver.get_error_envs_mask()
        return envs_mask

    def clear_errno(self, envs_idx=None) -> None:
        for i_group, _, envs_idx_local in self._partition.select(envs_idx):
            self._solvers[i_group].clear_errno(envs_idx_local)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ checkpoints -----------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        """Yield the data of the solver of every group, named under 'groups.<index of the group>'."""
        for i_group, solver in enumerate(self._solvers):
            for item in solver.data:
                yield item._replace(name=f"groups.{i_group}.{item.name}")

    def get_state(self) -> SolverCheckpoint:
        """Return the state of the solver of every group, its arrays named under 'groups.<index of the group>'."""
        return merge_groups_checkpoints([solver.get_state() for solver in self._solvers])

    def set_state(self, state: SolverCheckpoint, envs_idx=None) -> None:
        """Write back the state 'get_state' returned in the selected environments of the scene, all of them if None."""
        for i_group, _, envs_idx_local in self._partition.select(envs_idx):
            self._solvers[i_group].set_state(split_groups_checkpoint(state, i_group), envs_idx_local)

    def __getstate__(self) -> SolverCheckpoint:
        return merge_groups_checkpoints([solver.__getstate__() for solver in self._solvers])

    def __setstate__(self, state: SolverCheckpoint) -> None:
        for i_group, solver in enumerate(self._solvers):
            solver.__setstate__(split_groups_checkpoint(state, i_group))

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return bool(self._entities)

    @property
    def solvers(self) -> tuple[RigidSolver, ...]:
        """The rigid solver of each group, in the order of the groups of 'partition'."""
        return self._solvers

    @gs.assert_built
    @property
    def partition(self) -> EnvironmentPartition:
        """The partition of the environments of the scene into groups."""
        return self._partition
