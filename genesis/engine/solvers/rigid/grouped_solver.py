import dataclasses
import inspect
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

import genesis as gs
from genesis.engine.entities.rigid_entity import HeterogeneousEntity, RigidEntity
from genesis.engine.entities.rigid_entity.description import (
    EnvironmentsLayout,
    RigidEntityDescription,
    VariantStructureError,
    _balanced_variant_mapping,
)
from genesis.engine.render_context import ScopedRenderContext
from genesis.engine.state_change import Subscriber
from genesis.engine.states import SolverCheckpoint
from genesis.utils.data import ConfigItem, DataItem, namespace
from genesis.utils.misc import indices_to_mask

from ..base_solver import GravityMixin, Solver, TimeBasedMixin
from ..kinematic_solver import KinematicSolver
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


def is_dofs_info_batched(solver: RigidSolver, arguments: dict[str, Any]) -> bool:
    """Whether the parameters of the degrees of freedom hold one value per environment."""
    return solver.options.batch_dofs_info


def is_links_info_batched(solver: RigidSolver, arguments: dict[str, Any]) -> bool:
    """Whether the inertial parameters of the links hold one value per environment."""
    return solver.options.batch_links_info


def is_sol_params_batched(solver: RigidSolver, arguments: dict[str, Any]) -> bool:
    """Whether the solver parameters of the constraints an accessor selects hold one value per environment.

    Those of the equality constraints always do, those of the joints as the joint parameters do, those of the geoms
    never do.
    """
    if arguments["eqs_idx"] is not None:
        return True
    if arguments["joints_idx"] is not None:
        return solver.options.batch_joints_info
    return False


def dispatch_over_groups(
    method: Callable,
    values_ndim: dict[str, int] | None = None,
    is_info_batched: Callable[[RigidSolver, dict[str, Any]], bool] | None = None,
) -> Callable:
    """Return the accessor of GroupedRigidSolver calling 'method', an accessor of RigidSolver, on the solver of each group.

    Every group lays out the rigid entities of the layout solver first, in the same order, so the indices of their links,
    degrees of freedom, geoms and every other item stand for the same items in every group (see
    'GroupedRigidSolver.build'). The environments that 'envs_idx' selects are split by group, each group receiving its own
    in its numbering, and a getter gathers the rows of every group in the order of the selection. 'values_ndim' marks a
    setter, giving for each argument holding values the number of axes of the value of one item in one environment: an
    argument holding the axes of the environments and of the items, or the axis of the environments alone for a single
    item, holds one row per selected environment, split by group, and so does an argument of indices holding one row
    per selected environment. 'is_info_batched' tells from the solver of a group and the arguments
    whether the data accessed holds one entry per environment, always if None. A getter of data shared by every
    environment, or of a scene without parallel environments, returns the reading of the first group, while a setter
    writes every group.
    """
    signature = inspect.signature(method)
    is_setter = values_ndim is not None

    def accessor(self: "GroupedRigidSolver", *args, **kwargs):
        bound = signature.bind(self._solvers[0], *args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        is_batched = (
            self.n_envs > 0
            and "envs_idx" in arguments
            and (is_info_batched is None or is_info_batched(self._solvers[0], arguments))
        )
        if not is_batched:
            outputs = []
            for solver in self._solvers if is_setter else self._solvers[:1]:
                arguments["self"] = solver
                outputs.append(method(*bound.args, **bound.kwargs))
            return outputs[0]

        selection = self._partition.select(arguments["envs_idx"])
        n_selected = sum(len(rows) for _, rows, _ in selection)
        # A value holding the axis of the environments and the axis of the items holds one row per selected environment,
        # as does a value holding a single extra axis when a single item is selected and that axis spans the selection
        items_idx = [value for name, value in arguments.items() if name.endswith("_idx") and name != "envs_idx"]
        items_idx = [value for value in items_idx if value is not None]
        is_single_item = not items_idx or isinstance(items_idx[0], (int, np.integer)) or np.size(items_idx[0]) == 1
        rows_values = {}
        for name, value in tuple(arguments.items()):
            if value is None:
                continue
            ndim = np.ndim(value)
            if name in (values_ndim or {}):
                is_per_env = ndim == values_ndim[name] + 2 or (
                    is_single_item and ndim == values_ndim[name] + 1 and np.shape(value)[0] == n_selected
                )
            else:
                is_per_env = name.endswith("_idx") and name != "envs_idx" and ndim == 2
            if is_per_env:
                rows_values[name] = torch.as_tensor(value, device=gs.device)
        groups_output = []
        for i_group, rows, envs_idx_local in selection:
            arguments["self"] = self._solvers[i_group]
            arguments["envs_idx"] = envs_idx_local
            for name, value in rows_values.items():
                arguments[name] = value[rows]
            groups_output.append((rows, method(*bound.args, **bound.kwargs)))
        if is_setter:
            return None
        is_tuple = isinstance(groups_output[0][1], tuple)
        outputs = []
        for i_output in range(len(groups_output[0][1]) if is_tuple else 1):
            group_output = groups_output[0][1][i_output] if is_tuple else groups_output[0][1]
            output = group_output.new_empty((n_selected, *group_output.shape[1:]))
            for rows, group_outputs in groups_output:
                output[rows] = group_outputs[i_output] if is_tuple else group_outputs
            outputs.append(output)
        return tuple(outputs) if is_tuple else outputs[0]

    accessor.__name__ = method.__name__
    accessor.__doc__ = f"Call '{method.__qualname__}' on the solver of each group (see 'dispatch_over_groups')."
    return accessor


def refuse_over_groups(name: str) -> Callable:
    """Return the accessor of GroupedRigidSolver raising that a scene simulated by groups offers no 'name'."""

    def accessor(self: "GroupedRigidSolver", *args, **kwargs):
        gs.raise_exception(
            f"'{name}' is not supported for rigid entities whose scene simulates its environments by groups, the rigid "
            "entities of the scene differing in structure from one environment to another."
        )

    accessor.__name__ = name
    return accessor


class GroupedRigidSolver(GravityMixin, TimeBasedMixin, Solver):
    """The rigid solver of a scene holding a rigid entity whose variants differ in structure, simulating groups of
    environments apart.

    The environments are partitioned into groups by the combination of variants of those entities they carry, each
    carrying one variant per environment, the environments being split into contiguous blocks of nearly equal size. A
    rigid solver of its own simulates each group at the size of the group, so the variants may differ in anything a rigid
    solver accepts: the number of links and degrees of freedom, the joint types, a fixed or a free base, the equality
    constraints. An environment takes the memory and the time of the variants it carries, while the time of a step grows
    with the number of groups too, each group launching its own kernels.

    Every other rigid entity stays a RigidEntity, simulated in every group with the same indices, whose accessors reach
    the solver of each group through this one. An entity whose variants differ in structure is a HeterogeneousEntity,
    which gathers and scatters its accessors over the groups. The solver takes part in the simulation as one rigid
    solver: it steps, resets, reports errors and saves its state for every group, its state holding that of each group.
    """

    entity_spec_cls: ClassVar[type[gs.entities.Spec]] = gs.entities.Rigid

    # Its whole state lives in the solvers of its groups, whose data it yields (see 'data')
    is_checkpointable = True

    def __init__(self, sim: "Simulator", options: "RigidOptions", layout_solver: RigidSolver) -> None:
        """Create the solver of the given options, taking over the rigid entities of 'layout_solver'.

        'layout_solver' is the rigid solver the rigid entities of the scene were added to so far, never built, which
        keeps the entities whose variants share one structure and their layout (see 'build').
        """
        super().__init__(sim, options)
        self._layout_solver = layout_solver
        self._heterogeneous_entities: list[HeterogeneousEntity] = []
        self._entities.extend(layout_solver.entities)
        self._solvers: tuple[RigidSolver, ...] = ()
        self._partition: EnvironmentPartition | None = None

    def add_entity(
        self,
        idx: int,
        surface: "Surface | None",
        spec: "gs.entities.Rigid | None",
        desc: RigidEntityDescription | None = None,
    ) -> RigidEntity | HeterogeneousEntity:
        """Add a rigid entity of index 'idx' in the scene, a RigidEntity unless its variants differ in structure.

        A RigidEntity joins the layout solver, a HeterogeneousEntity is created in the solver of each group as the
        solver builds, from the morph of the variant the group carries (see 'add_heterogeneous_entity').
        """
        try:
            entity = self._layout_solver.add_entity(idx, surface, spec, desc)
        except VariantStructureError:
            return self.add_heterogeneous_entity(idx, surface, spec)
        self._entities.append(entity)
        return entity

    def add_heterogeneous_entity(self, idx: int, surface: "Surface", spec: "gs.entities.Rigid") -> HeterogeneousEntity:
        """Record a rigid entity of index 'idx' whose variants differ in structure, which the groups are split by."""
        entity = HeterogeneousEntity(idx, surface, spec)
        self._heterogeneous_entities.append(entity)
        self._entities.append(entity)
        return entity

    @gs.assert_unbuilt
    def build(self, n_envs: int) -> None:
        """Partition the environments into groups, then create the solver of each group with its entities and build it.

        Every group adds the entities of the layout solver first, in their order and from their descriptions, so that
        they take the same indices in every group as in the layout solver, then the variant it carries of each entity
        whose variants differ in structure. Those of the entities of the layout solver that hold variants spread them
        over the environments of the scene in every group (see EnvironmentsLayout). A scene without parallel
        environments forms one group carrying the first variant of every entity. The descriptions resolved for the first
        group holding a variant are shared by the next ones, so that every variant reads its asset once.
        """
        super().build(n_envs)
        layout_entities = self._layout_solver.entities
        for entity in layout_entities:
            if entity.desc.attachment is not None:
                gs.raise_exception(
                    f"Entity '{entity.name}' is attached to another one, which a scene whose rigid entities differ in "
                    "structure from one environment to another does not support."
                )

        envs_variants_idx = np.empty((self.n_envs_, len(self._heterogeneous_entities)), dtype=gs.np_int)
        for i_entity, entity in enumerate(self._heterogeneous_entities):
            envs_variants_idx[:, i_entity] = _balanced_variant_mapping(len(entity.morphs), self.n_envs_)
        groups_variants_idx, envs_group_idx = np.unique(envs_variants_idx, axis=0, return_inverse=True)

        envs_local_idx = np.empty((self.n_envs_,), dtype=gs.np_int)
        # The entities of a group hold no name of their own, the scene naming those it holds alone
        layout_descs = [
            dataclasses.replace(entity.desc, spec=entity.desc.spec.model_copy(update={"name": None}))
            for entity in layout_entities
        ]
        variants_desc: dict[tuple[int, int], RigidEntityDescription] = {}
        groups_envs_idx, solvers, groups_entities = [], [], []
        for i_group, variants_idx in enumerate(groups_variants_idx):
            envs_idx = np.flatnonzero(envs_group_idx == i_group)
            envs_local_idx[envs_idx] = np.arange(len(envs_idx))
            groups_envs_idx.append(torch.as_tensor(envs_idx, dtype=gs.tc_int, device=gs.device))
            # Each group takes a copy of the options, which a rigid solver resolves in place as it builds
            envs_layout = EnvironmentsLayout(self.n_envs_, envs_idx)
            solver = RigidSolver(self._sim, self._options.model_copy(deep=True), envs_layout)
            for entity, desc in zip(layout_entities, layout_descs):
                solver.add_entity(entity.idx, entity.surface, None, desc)
            entities = []
            for i_entity, (entity, i_variant) in enumerate(zip(self._heterogeneous_entities, variants_idx)):
                desc = variants_desc.get((i_entity, i_variant))
                if desc is None:
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
                solver.build(len(envs_idx) if self.n_envs > 0 else 0)
            solvers.append(solver)
        self._solvers = tuple(solvers)

        self._partition = EnvironmentPartition(
            groups_variants_idx=tuple(map(tuple, groups_variants_idx.tolist())),
            groups_envs_idx=tuple(groups_envs_idx),
            envs_group_idx=torch.as_tensor(envs_group_idx, dtype=gs.tc_int, device=gs.device),
            envs_local_idx=torch.as_tensor(envs_local_idx, dtype=gs.tc_int, device=gs.device),
        )
        # The entities of the layout solver reach the solver of each group through this one, at the same indices
        for entity in layout_entities:
            entity._bind_solver(self)
            entity._build()
        for entity, entities in zip(self._heterogeneous_entities, zip(*groups_entities)):
            entity.build(entities, self._partition, is_batched=self.n_envs > 0)
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
    # ------------------------------------ accessors -------------------------------------
    # ------------------------------------------------------------------------------------

    # The accessors that the entities of the layout solver, their links, joints, geoms and equality constraints call on
    # their solver, which reach every group (see 'dispatch_over_groups')
    set_gravity = dispatch_over_groups(RigidSolver.set_gravity, values_ndim={"gravity": 1})
    get_gravity = dispatch_over_groups(RigidSolver.get_gravity)
    set_qpos = dispatch_over_groups(RigidSolver.set_qpos, values_ndim={"qpos": 0})
    get_qpos = dispatch_over_groups(RigidSolver.get_qpos)
    set_dofs_position = dispatch_over_groups(RigidSolver.set_dofs_position, values_ndim={"position": 0})
    set_dofs_velocity = dispatch_over_groups(RigidSolver.set_dofs_velocity, values_ndim={"velocity": 0})
    get_dofs_position = dispatch_over_groups(RigidSolver.get_dofs_position)
    get_dofs_velocity = dispatch_over_groups(RigidSolver.get_dofs_velocity)
    get_dofs_acc = dispatch_over_groups(RigidSolver.get_dofs_acc)
    get_dofs_force = dispatch_over_groups(RigidSolver.get_dofs_force)
    get_dofs_control_force = dispatch_over_groups(RigidSolver.get_dofs_control_force)
    get_dofs_actuator_force = dispatch_over_groups(RigidSolver.get_dofs_actuator_force)
    control_dofs_force = dispatch_over_groups(RigidSolver.control_dofs_force, values_ndim={"force": 0})
    control_dofs_velocity = dispatch_over_groups(RigidSolver.control_dofs_velocity, values_ndim={"velocity": 0})
    control_dofs_position = dispatch_over_groups(RigidSolver.control_dofs_position, values_ndim={"position": 0})
    control_dofs_position_velocity = dispatch_over_groups(
        RigidSolver.control_dofs_position_velocity, values_ndim={"position": 0, "velocity": 0}
    )
    set_dofs_kp = dispatch_over_groups(RigidSolver.set_dofs_kp, {"kp": 0}, is_dofs_info_batched)
    set_dofs_kv = dispatch_over_groups(RigidSolver.set_dofs_kv, {"kv": 0}, is_dofs_info_batched)
    set_dofs_act_gain = dispatch_over_groups(RigidSolver.set_dofs_act_gain, {"act_gain": 0}, is_dofs_info_batched)
    set_dofs_act_bias = dispatch_over_groups(
        RigidSolver.set_dofs_act_bias, {"bias0": 0, "bias1": 0, "bias2": 0}, is_dofs_info_batched
    )
    set_dofs_force_range = dispatch_over_groups(
        RigidSolver.set_dofs_force_range, {"lower": 0, "upper": 0}, is_dofs_info_batched
    )
    set_dofs_limit = dispatch_over_groups(RigidSolver.set_dofs_limit, {"lower": 0, "upper": 0}, is_dofs_info_batched)
    set_dofs_stiffness = dispatch_over_groups(RigidSolver.set_dofs_stiffness, {"stiffness": 0}, is_dofs_info_batched)
    set_dofs_armature = dispatch_over_groups(RigidSolver.set_dofs_armature, {"armature": 0}, is_dofs_info_batched)
    set_dofs_damping = dispatch_over_groups(RigidSolver.set_dofs_damping, {"damping": 0}, is_dofs_info_batched)
    set_dofs_frictionloss = dispatch_over_groups(
        RigidSolver.set_dofs_frictionloss, {"frictionloss": 0}, is_dofs_info_batched
    )
    get_dofs_kp = dispatch_over_groups(RigidSolver.get_dofs_kp, is_info_batched=is_dofs_info_batched)
    get_dofs_kv = dispatch_over_groups(RigidSolver.get_dofs_kv, is_info_batched=is_dofs_info_batched)
    get_dofs_act_gain = dispatch_over_groups(RigidSolver.get_dofs_act_gain, is_info_batched=is_dofs_info_batched)
    get_dofs_act_bias = dispatch_over_groups(RigidSolver.get_dofs_act_bias, is_info_batched=is_dofs_info_batched)
    get_dofs_force_range = dispatch_over_groups(RigidSolver.get_dofs_force_range, is_info_batched=is_dofs_info_batched)
    get_dofs_limit = dispatch_over_groups(RigidSolver.get_dofs_limit, is_info_batched=is_dofs_info_batched)
    get_dofs_stiffness = dispatch_over_groups(RigidSolver.get_dofs_stiffness, is_info_batched=is_dofs_info_batched)
    get_dofs_armature = dispatch_over_groups(RigidSolver.get_dofs_armature, is_info_batched=is_dofs_info_batched)
    get_dofs_damping = dispatch_over_groups(RigidSolver.get_dofs_damping, is_info_batched=is_dofs_info_batched)
    get_dofs_frictionloss = dispatch_over_groups(
        RigidSolver.get_dofs_frictionloss, is_info_batched=is_dofs_info_batched
    )
    get_dofs_invweight = dispatch_over_groups(RigidSolver.get_dofs_invweight, is_info_batched=is_dofs_info_batched)
    set_base_links_pos = dispatch_over_groups(RigidSolver.set_base_links_pos, values_ndim={"pos": 1})
    set_base_links_quat = dispatch_over_groups(RigidSolver.set_base_links_quat, values_ndim={"quat": 1})
    get_links_pos = dispatch_over_groups(RigidSolver.get_links_pos)
    get_links_quat = dispatch_over_groups(RigidSolver.get_links_quat)
    get_links_vel = dispatch_over_groups(RigidSolver.get_links_vel)
    get_links_ang = dispatch_over_groups(RigidSolver.get_links_ang)
    get_links_acc = dispatch_over_groups(RigidSolver.get_links_acc)
    get_links_acc_ang = dispatch_over_groups(RigidSolver.get_links_acc_ang)
    get_links_COM = dispatch_over_groups(RigidSolver.get_links_COM)
    set_links_mass = dispatch_over_groups(RigidSolver.set_links_mass, {"mass": 0}, is_links_info_batched)
    set_links_inertia = dispatch_over_groups(RigidSolver.set_links_inertia, {"inertia": 2}, is_links_info_batched)
    set_links_COM = dispatch_over_groups(RigidSolver.set_links_COM, {"com": 1}, is_links_info_batched)
    get_links_mass = dispatch_over_groups(RigidSolver.get_links_mass, is_info_batched=is_links_info_batched)
    get_links_inertia = dispatch_over_groups(RigidSolver.get_links_inertia, is_info_batched=is_links_info_batched)
    get_links_invweight = dispatch_over_groups(RigidSolver.get_links_invweight, is_info_batched=is_links_info_batched)
    apply_links_external_wrench = dispatch_over_groups(
        RigidSolver.apply_links_external_wrench, values_ndim={"force": 1, "torque": 1, "pos": 1}
    )
    get_joints_anchor_pos = dispatch_over_groups(RigidSolver.get_joints_anchor_pos)
    get_joints_anchor_axis = dispatch_over_groups(RigidSolver.get_joints_anchor_axis)
    get_geoms_pos = dispatch_over_groups(RigidSolver.get_geoms_pos)
    get_geoms_quat = dispatch_over_groups(RigidSolver.get_geoms_quat)
    get_vgeoms_pos = dispatch_over_groups(RigidSolver.get_vgeoms_pos)
    get_vgeoms_quat = dispatch_over_groups(RigidSolver.get_vgeoms_quat)
    set_geoms_friction_ratio = dispatch_over_groups(
        RigidSolver.set_geoms_friction_ratio, values_ndim={"friction_ratio": 0}
    )
    get_geoms_friction_ratio = dispatch_over_groups(RigidSolver.get_geoms_friction_ratio)
    set_geom_friction = dispatch_over_groups(RigidSolver.set_geom_friction, values_ndim={"friction": 0})
    set_geom_friction_torsional = dispatch_over_groups(
        RigidSolver.set_geom_friction_torsional, values_ndim={"friction_torsional": 0}
    )
    set_geom_friction_rolling = dispatch_over_groups(
        RigidSolver.set_geom_friction_rolling, values_ndim={"friction_rolling": 0}
    )
    get_geoms_friction = dispatch_over_groups(RigidSolver.get_geoms_friction)
    get_geoms_friction_torsional = dispatch_over_groups(RigidSolver.get_geoms_friction_torsional)
    get_geoms_friction_rolling = dispatch_over_groups(RigidSolver.get_geoms_friction_rolling)
    set_sol_params = dispatch_over_groups(RigidSolver.set_sol_params, {"sol_params": 1}, is_sol_params_batched)
    get_sol_params = dispatch_over_groups(RigidSolver.get_sol_params, is_info_batched=is_sol_params_batched)
    get_mass_mat = dispatch_over_groups(RigidSolver.get_mass_mat)
    get_kinetic_energy = dispatch_over_groups(RigidSolver.get_kinetic_energy)
    get_potential_energy = dispatch_over_groups(RigidSolver.get_potential_energy)
    get_AABB = dispatch_over_groups(RigidSolver.get_AABB)
    update_verts_for_geoms = dispatch_over_groups(RigidSolver.update_verts_for_geoms, values_ndim={})
    # The queries reading the state of one solver at once, which no group answers for the others
    inverse_kinematics = refuse_over_groups("inverse_kinematics")
    forward_kinematics_query = refuse_over_groups("forward_kinematics_query")
    get_terrain_height = refuse_over_groups("get_terrain_height")
    set_drone_rpm = refuse_over_groups("set_drone_rpm")

    def get_links_jacobian(self, link_idx, dof_start, n_dofs, local_point=None) -> torch.Tensor:
        """Return the Jacobian of a link in every environment (see 'KinematicSolver.get_links_jacobian')."""
        groups_jacobian = [
            solver.get_links_jacobian(link_idx, dof_start, n_dofs, local_point) for solver in self._solvers
        ]
        if self.n_envs == 0:
            return groups_jacobian[0]
        jacobian = groups_jacobian[0].new_empty((self.n_envs, *groups_jacobian[0].shape[1:]))
        for solver, group_jacobian in zip(self._solvers, groups_jacobian):
            jacobian[solver.scene_envs_idx] = group_jacobian
        return jacobian

    def detect_collision(self, env_idx: int = 0) -> np.ndarray:
        """Return the pairs of geoms in contact in one environment (see 'RigidSolver.detect_collision')."""
        ((i_group, _, envs_idx_local),) = self._partition.select([env_idx])
        return self._solvers[i_group].detect_collision(int(envs_idx_local[0]))

    @property
    def dyn_state(self):
        """The state of the solver of each group alone exists, which 'solvers' gives."""
        gs.raise_exception(
            "A scene simulating its environments by groups holds the state of each group in the rigid solver of the "
            "group ('GroupedRigidSolver.solvers'), the rigid entities of the scene differing in structure from one "
            "environment to another."
        )

    @property
    def collider(self):
        """The collider of the solver of each group alone exists, which 'solvers' gives."""
        gs.raise_exception(
            "A scene simulating its environments by groups holds the contacts of each group in the rigid solver of the "
            "group ('GroupedRigidSolver.solvers'), the rigid entities of the scene differing in structure from one "
            "environment to another."
        )

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


class GroupedRigidScopedRenderContext(ScopedRenderContext[GroupedRigidSolver]):
    """The controls of the appearance of a grouped rigid solver, which reach the scoped context of each of its groups.

    The scoped contexts of the groups draw the entities, each in the environments of its group.
    """

    def _groups(self) -> tuple[ScopedRenderContext, ...]:
        return tuple(solver.render_context for solver in self.owner.solvers)

    @property
    def link_frames(self) -> bool:
        """Whether the frames of the links are drawn (see ``KinematicVisOptions.link_frames``)."""
        return all(scoped_context.link_frames for scoped_context in self._groups())

    def set_link_frames(self, link_frames: bool) -> None:
        """Draw the frames of the links in every group, or stop drawing them."""
        for scoped_context in self._groups():
            scoped_context.set_link_frames(link_frames)

    @property
    def link_frame_size(self) -> float:
        """The length of the axes of the frames of the links (see ``KinematicVisOptions.link_frame_size``)."""
        return self._groups()[0].link_frame_size

    def set_link_frame_size(self, link_frame_size: float) -> None:
        """Set the length of the axes of the frames of the links in every group."""
        for scoped_context in self._groups():
            scoped_context.set_link_frame_size(link_frame_size)

    def get_visualize_contact(self, entity: RigidEntity) -> bool:
        """Whether the contacts of an entity of the layout solver are drawn, in every group alike."""
        return self._groups()[0].get_visualize_contact(entity)

    def set_visualize_contact(self, entity: RigidEntity, visualize_contact: bool) -> None:
        """Draw the contacts of an entity of the layout solver in every group, or stop drawing them."""
        for scoped_context in self._groups():
            scoped_context.set_visualize_contact(entity, visualize_contact)

    def get_vverts(self, vgeoms, envs_idx=None) -> torch.Tensor:
        """The vertices as drawn of the visual geoms of the scene, which the scoped context of each group holds."""
        gs.raise_exception(
            "The drawn vertices of the visual geoms are held by the scoped context of each group of a scene whose rigid "
            "entities differ in structure from one environment to another ('GroupedRigidSolver.solvers')."
        )
