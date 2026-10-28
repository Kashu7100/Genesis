from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.options.solvers import SteelOptions
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    tensor_to_array,
)

from ..base_solver import MODEL_CHANGES, StateChange, Subscriber, mutates
from ..rigid.abd.forward_kinematics import (
    func_forward_velocity,
    func_update_cartesian_space,
    kernel_forward_kinematics_links_geoms,
    kernel_forward_velocity,
)
from ..rigid.rigid_solver import RigidSolver, kernel_substep_collision
from .compliance import func_steel_compliance_rows
from .contacts import (
    func_steel_gather_warmstart,
    func_steel_match_contacts,
    func_steel_snapshot_warmstart,
    func_steel_snapshot_warmstart_rows,
    func_steel_warmstart_velocity,
    kernel_steel_reset_warmstart,
)
from .dynamics import (
    func_steel_fk_id,
    func_steel_forward_kinematics,
    func_steel_inputs,
    func_steel_integrate,
    func_steel_mass_matrix,
    func_steel_prescale_velocity,
    func_steel_publish,
    func_steel_publish_refined_velocity,
    func_steel_tau,
    func_steel_unconstrained,
    kernel_steel_pull,
)
from .friction_patches import (
    func_steel_build_patches,
    func_steel_finish_patch_impulses,
    func_steel_link_patch_rows,
    func_steel_prepare_patches,
    func_steel_seed_patch_impulses,
    func_steel_store_patches,
    kernel_steel_reset_patches,
)
from .model import STEEL_JOINT_TYPE, build_steel_model_plan
from .observables import func_steel_contact_forces, func_steel_update_status, kernel_steel_reset_status
from .pgs import func_pgs_sweeps
from .propagation import (
    func_propagation_propagate_articulation,
    func_propagation_refresh_articulation,
    func_propagation_sweep,
    func_propagation_warmstart,
    func_steel_propagation_build_rows,
    func_steel_propagation_color,
    func_steel_propagation_factor,
    func_steel_propagation_refresh,
    func_steel_propagation_row_setup,
    func_steel_propagation_velocity_pass_rhs,
)
from .rows import (
    CONTACT_PATH,
    func_steel_allocate_drive_rows,
    func_steel_allocate_rows,
    func_steel_build_rows,
    func_steel_preelim,
    func_steel_preelim_project_velocity,
    func_steel_responses,
    func_steel_row_setup,
    func_steel_velocity_pass_rhs,
    kernel_steel_gather_contacts,
)
from .sleeping import func_steel_sleep_begin, func_steel_sleep_finish, kernel_steel_sleep_wake
from .split import func_steel_split_setup, func_steel_split_solve
from .torsion import func_steel_torsion_rows, func_steel_torsion_velocity_pass

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# Bounding radius [m] of a plane, which is unbounded.
PLANE_BOUNDING_RADIUS = 1.0e6
# Default angular damping of free bodies [1/s].
DEFAULT_ANGULAR_DAMPING = 0.05
# Position-correction gain of friction patches when 'friction_anchor_beta' is None with the matrix-free solve.
DEFAULT_FRICTION_ANCHOR_BETA = 0.2

SOLVE_MODES = ("matrix_free", "split")
DRIVE_MODES = ("augmented", "rows")
FRICTION_MODES = ("fixed_normal", "bisection", "bisection_desaxce", "coulomb_newton")
ROW_SCHEDULES = ("interleaved", "contact_then_internal", "phased")
CONTACT_RESPONSES = ("immediate", "propagation", "propagation_fused", "propagation_colored")
# Geometries whose contacts admit torsion rows, convex meshes aside.
TORSION_GEOM_TYPES = (
    gs.GEOM_TYPE.PLANE,
    gs.GEOM_TYPE.SPHERE,
    gs.GEOM_TYPE.ELLIPSOID,
    gs.GEOM_TYPE.CYLINDER,
    gs.GEOM_TYPE.CAPSULE,
    gs.GEOM_TYPE.BOX,
)


def _validate_step_option(name: str, value: float, *, allow_negative: bool) -> float:
    """Return the value of a step option as a float, raising unless it is finite, and non-negative unless allowed."""
    value = float(value)
    if not np.isfinite(value) or (not allow_negative and value < 0.0):
        gs.raise_exception(f"'{name}' must be finite{'' if allow_negative else ' and non-negative'}, got {value}.")
    return value


class SteelRowFamilyWatermarks(NamedTuple):
    """High-water marks of one row family over every environment and step since the scene was built.

    high_water and raw_high_water are the largest retained and requested row counts, dropped_rows_high_water the
    largest number of rows of rejected reservations, overflow_excess_high_water the largest excess of the requested
    rows over the capacity and overflow_env_steps the number of environment-steps that exceeded the capacity.
    """

    high_water: int
    raw_high_water: int
    dropped_rows_high_water: int
    overflow_excess_high_water: int
    overflow_env_steps: int


class SteelRowWatermarks(NamedTuple):
    """High-water marks of the constraint rows of the Steel solver (see SteelSolver.constraint_row_watermarks).

    articulation, free_body and propagation are the dense, free-body and propagation row families, contact_high_water
    the largest contact count of an environment.
    """

    articulation: SteelRowFamilyWatermarks
    free_body: SteelRowFamilyWatermarks
    propagation: SteelRowFamilyWatermarks
    contact_high_water: int


class SteelSolver(RigidSolver):
    """Rigid solver simulating every rigid entity with the Steel algorithm (see 'gs.options.SteelOptions').

    It keeps the entities, the collision detection and the accessors of the rigid solver and replaces its step: the
    Steel state (joint coordinates and velocities in Steel conventions) advances by one Steel step per substep, then is
    written back to the Genesis state, whose Cartesian quantities are updated for the getters, the sensors, the
    rendering and the next collision detection. A Genesis setter of the state reloads the Steel state of the
    environments it changed and keeps their solver history (warm start, friction anchors, capacity status), which
    'scene.reset' and 'reset' discard. A setter of the model refreshes what the change enters (see _on_model_change).
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: SteelOptions) -> None:
        super().__init__(scene, sim, options)
        if self._requires_grad:
            gs.raise_exception("The Steel solver does not support differentiable simulation.")

        # Friction patches are the default of the matrix-free solve, point friction that of the split solve, of contact
        # compliance and of the friction modes other than fixed_normal.
        beta = options.friction_anchor_beta
        if beta is None:
            if options.contact_compliance:
                gs.logger.warning("'contact_compliance' uses point friction: 'friction_anchor_beta' resolves to 0.")
                beta = 0.0
            elif (
                options.friction_mode != "fixed_normal"
                and options.solve_mode == "matrix_free"
                and options.articulated_contact_response == "immediate"
            ):
                gs.logger.warning(
                    f"friction_mode='{options.friction_mode}' uses point friction: 'friction_anchor_beta' resolves "
                    "to 0."
                )
                beta = 0.0
            else:
                beta = DEFAULT_FRICTION_ANCHOR_BETA if options.solve_mode == "matrix_free" else 0.0
        elif options.contact_compliance and beta > 0.0:
            gs.raise_exception("'contact_compliance' does not support friction patches ('friction_anchor_beta' > 0).")
        elif options.friction_mode != "fixed_normal" and beta > 0.0:
            gs.raise_exception("Friction patches ('friction_anchor_beta' > 0) require friction_mode='fixed_normal'.")
        self._steel_friction_anchor_beta = beta
        self._steel_contact_w = float(np.float32(1.0 / (1.0 + options.contact_regularization)))
        # The step options, which the properties of the same name change after the build (see 'SteelOptions').
        self._steel_baumgarte = options.baumgarte
        self._steel_relaxation = options.relaxation
        self._steel_warn_constraint_overflow = options.warn_constraint_overflow
        self._steel_velocity_drive_mode = options.velocity_drive_mode
        self._steel_contact_warmstart_decay = options.contact_warmstart_decay
        self._steel_restitution_velocity_threshold = options.restitution_velocity_threshold
        self._steel_contact_speculative_scale = options.contact_speculative_scale
        self._steel_contact_gap_gate = options.contact_gap_gate
        self._steel_contact_friction_gap_max = options.contact_friction_gap_max

        if options.solve_mode == "split":
            split_features = {
                "enable_joint_velocity_limits": options.enable_joint_velocity_limits,
                "drive_mode='rows'": options.drive_mode == "rows",
                "friction patches": beta > 0.0,
                "contact_regularization": self._steel_contact_w < 1.0,
                "velocity_iterations": options.velocity_iterations > 0,
                "contact_warmstart": options.contact_warmstart,
                "contact_compliance": options.contact_compliance,
                "enable_torsional_friction": options.enable_torsional_friction,
                "enable_sleeping": options.enable_sleeping,
                "row_schedule": options.row_schedule != "interleaved",
                "friction_mode": options.friction_mode != "fixed_normal",
                "articulated_contact_response": options.articulated_contact_response != "immediate",
            }
            offenders = [name for name, is_set in split_features.items() if is_set]
            if offenders:
                gs.raise_exception(f"{offenders} require solve_mode='matrix_free'.")
        if options.articulated_contact_response != "immediate":
            propagation_features = {
                "friction_mode": options.friction_mode != "fixed_normal",
            }
            offenders = [name for name, is_set in propagation_features.items() if is_set]
            if offenders:
                gs.raise_exception(
                    f"{offenders} not supported with articulated_contact_response="
                    f"'{options.articulated_contact_response}'."
                )
            if options.articulated_contact_response == "propagation_fused" and (
                options.row_schedule == "contact_then_internal"
            ):
                gs.raise_exception(
                    "articulated_contact_response='propagation_fused' does not support "
                    "row_schedule='contact_then_internal': use 'propagation' for that schedule."
                )
        # Contact torsion and contact compliance are defined for one solve configuration each.
        for name, is_enabled, unsupported_features in (
            (
                "enable_torsional_friction",
                options.enable_torsional_friction,
                {
                    "contact_warmstart": options.contact_warmstart,
                    "friction_mode": options.friction_mode != "fixed_normal",
                    "row_schedule": options.row_schedule != "interleaved",
                    "articulated_contact_response": options.articulated_contact_response != "immediate",
                },
            ),
            (
                "contact_compliance",
                options.contact_compliance,
                {
                    "iterations=0": options.iterations == 0,
                    "velocity_iterations": options.velocity_iterations > 0,
                    "contact_warmstart": options.contact_warmstart,
                    "contact_regularization": self._steel_contact_w < 1.0,
                    "contact_shared_anchor": options.contact_shared_anchor,
                    "contact_friction_shared_anchor": options.contact_friction_shared_anchor,
                    "friction_mode": options.friction_mode != "fixed_normal",
                    "contact_friction_position_iterations": options.contact_friction_position_iterations >= 0,
                    "row_schedule": options.row_schedule != "interleaved",
                    "articulated_contact_response": options.articulated_contact_response != "immediate",
                },
            ),
        ):
            offenders = [feature for feature, is_set in unsupported_features.items() if is_set]
            if is_enabled and offenders:
                gs.raise_exception(f"'{name}' does not support {offenders}.")
        if options.enable_sleeping:
            if options.contact_warmstart or options.velocity_iterations > 0:
                gs.raise_exception(
                    "'enable_sleeping' does not support 'contact_warmstart' or 'velocity_iterations' > 0."
                )
            if options.articulated_contact_response != "immediate":
                gs.raise_exception("'enable_sleeping' requires articulated_contact_response='immediate'.")
        if options.contact_friction_position_iterations >= 0 and options.contact_warmstart:
            gs.raise_exception("'contact_friction_position_iterations' >= 0 does not support 'contact_warmstart'.")
        if options.enable_sleeping and options.contact_warmstart:
            gs.raise_exception("'enable_sleeping' does not support 'contact_warmstart'.")
        if beta > 0.0 and (options.contact_shared_anchor or options.contact_friction_shared_anchor):
            gs.logger.warning(
                "Friction patches act at their own anchors: the shared anchor options only move the normal rows."
            )

        self._steel_plan = None
        self._steel_step = 0
        # The Steel material inputs at build, one value per environment, which get_steel_info copies into SteelInfo.
        # The material setters write SteelInfo afterwards.
        self._steel_links_angular_damping = None
        self._steel_links_max_linear_velocity = None
        self._steel_links_max_angular_velocity = None
        self._steel_links_max_depenetration_velocity = None
        self._steel_geoms_restitution = None
        self._steel_dofs_velocity_limit = None
        self._steel_propagation_active = False
        # SteelOverflow bits already reported by a warning, each family warning once per solver.
        self._steel_overflow_warned = 0
        self._steel_geoms_is_torsion_supported = None
        self._steel_geoms_contact_stiffness = None
        self._steel_geoms_contact_damping = None
        # Whether a state change of selected environments vetoes their islands (see _on_state_change). It holds from the
        # first step on, until a state change of every environment, which restarts the sleep history.
        self._steel_sleep_is_history_valid = False

    def build(self):
        super().build()
        if not self.is_active:
            return

        options = self._options
        self._steel_plan = build_steel_model_plan(self)
        self._steel_n_preelim = 0
        if options.solve_mode == "split" and (self._steel_plan.n_mimics > 0 or self._steel_plan.n_connects > 0):
            gs.raise_exception("Mimic joints and loop closures require solve_mode='matrix_free'.")

        # Every free root is damped by the default rate, free bodies are unbounded in velocity and geoms do not bounce,
        # unless their material states otherwise.
        _B = self._B
        self._steel_links_angular_damping = np.full((self.n_links_, _B), DEFAULT_ANGULAR_DAMPING, dtype=gs.np_float)
        self._steel_links_max_linear_velocity = np.full((self.n_links_, _B), np.inf, dtype=gs.np_float)
        self._steel_links_max_angular_velocity = np.full((self.n_links_, _B), np.inf, dtype=gs.np_float)
        self._steel_links_max_depenetration_velocity = np.full((self.n_links_, _B), np.inf, dtype=gs.np_float)
        self._steel_geoms_restitution = np.zeros((self.n_geoms_, _B), dtype=gs.np_float)
        self._steel_dofs_velocity_limit = np.full(self.n_dofs_, np.inf, dtype=gs.np_float)
        self._steel_geoms_contact_stiffness = np.zeros((self.n_geoms_, _B), dtype=gs.np_float)
        self._steel_geoms_contact_damping = np.zeros((self.n_geoms_, _B), dtype=gs.np_float)
        for entity in self._entities:
            material = entity.material
            links_idx = slice(entity.link_start, entity.link_end)
            if material.angular_damping is not None:
                self._steel_links_angular_damping[links_idx] = material.angular_damping
            for value, array in (
                (material.max_linear_velocity, self._steel_links_max_linear_velocity),
                (material.max_angular_velocity, self._steel_links_max_angular_velocity),
                (material.max_depenetration_velocity, self._steel_links_max_depenetration_velocity),
            ):
                if value is not None:
                    array[links_idx] = value
            self._steel_geoms_restitution[entity.geom_start : entity.geom_end] = material.restitution
            if material.contact_stiffness is not None:
                self._steel_geoms_contact_stiffness[entity.geom_start : entity.geom_end] = material.contact_stiffness
                self._steel_geoms_contact_damping[entity.geom_start : entity.geom_end] = material.contact_damping
        self._steel_geoms_is_torsion_supported = np.zeros(self.n_geoms_, dtype=gs.np_bool)
        for geom in self.geoms:
            self._steel_geoms_is_torsion_supported[geom.idx] = geom.type in TORSION_GEOM_TYPES or (
                geom.type == gs.GEOM_TYPE.MESH and geom.is_convex
            )
        if options.enable_torsional_friction:
            geoms_radius = tensor_to_array(self.get_geoms_friction_torsional())
            if not np.isfinite(geoms_radius).all() or (geoms_radius < 0.0).any():
                gs.raise_exception("The torsional friction radius of every geometry must be finite and non-negative.")
            geoms_idx = np.flatnonzero(
                (geoms_radius > 0.0) & ~self._steel_geoms_is_torsion_supported[: len(geoms_radius)]
            )
            if geoms_idx.size:
                gs.raise_exception(
                    f"Contact torsion does not support the geometry of the geoms {geoms_idx.tolist()}: set their "
                    "material option 'friction_torsional' to 0."
                )
            if options.contact_compliance and (self._steel_geoms_contact_stiffness > 0.0).any():
                gs.raise_exception("'enable_torsional_friction' does not support compliant contacts.")
        if options.contact_compliance and options.enable_restitution and (self._steel_geoms_restitution > 0.0).any():
            # The compliant law replaces the position bias, which a rebound target would replace too.
            gs.raise_exception("'contact_compliance' requires zero material restitution with 'enable_restitution'.")
        # The split solve has no rebound targets.
        if options.solve_mode == "split" and options.enable_restitution and (self._steel_geoms_restitution > 0.0).any():
            gs.raise_exception(
                "Contact restitution requires solve_mode='matrix_free': set 'enable_restitution=False' to ignore the "
                "restitution of the materials."
            )
        self._steel_propagation_active = self._build_propagation_plan()
        arts_preelim_idx = self._build_preelimination_plan()
        self._warn_persistent_row_capacity()

        self.steel_config = array_class.SteelStaticConfig(
            para_level=self.rigid_config.para_level,
            batch_links_info=options.batch_links_info,
            batch_joints_info=options.batch_joints_info,
            batch_dofs_info=options.batch_dofs_info,
            solve_mode=SOLVE_MODES.index(options.solve_mode),
            drive_mode=DRIVE_MODES.index(options.drive_mode),
            friction_mode=FRICTION_MODES.index(options.friction_mode),
            row_schedule=ROW_SCHEDULES.index(options.row_schedule),
            articulated_contact_response=CONTACT_RESPONSES.index(options.articulated_contact_response),
            enable_joint_limits=options.enable_joint_limit,
            enable_joint_velocity_limits=options.enable_joint_velocity_limits,
            has_velocity_iterations=options.velocity_iterations > 0,
            has_free_roots=len(self._steel_plan.free_root_joints) > 0,
            has_free_rigid_bodies=self._steel_plan.has_free_rigid_bodies,
            has_free_and_articulated_bodies=(
                self._steel_plan.has_free_rigid_bodies and not self._steel_plan.arts_is_free_rigid.all()
            ),
            has_friction_anchors=self._steel_friction_anchor_beta > 0.0,
            has_contact_warmstart=options.contact_warmstart,
            has_sorted_contacts=options.contact_warmstart
            or self.collider.collider_config.has_non_box_plane_convex_convex,
            contact_shared_anchor=options.contact_shared_anchor,
            contact_friction_shared_anchor=options.contact_friction_shared_anchor,
            has_mimics=self._steel_plan.n_mimics > 0,
            has_preelim=self._steel_n_preelim > 0,
            has_preelim_mimics=(
                self._steel_n_preelim > 0
                and options.bilateral_preelimination_include_mimics
                and self._steel_plan.n_mimics > 0
            ),
            has_connects=self._steel_plan.n_connects > 0,
            has_fused_velocity_limits=self.velocity_limits_fused,
            # The velocity bounds of free rigid bodies are material inputs read at every step, so their rows are
            # compiled in whenever free rigid bodies exist, each body getting rows for its finite bounds alone.
            has_free_velocity_limits=self._steel_plan.has_free_rigid_bodies,
            has_propagation=self._steel_propagation_active,
            has_internal_rows=(
                options.drive_mode == "rows"
                or options.enable_joint_limit
                or self._steel_plan.n_mimics > 0
                or self._steel_plan.n_connects > 0
            ),
            has_velocity_limit_rows=(
                options.enable_joint_velocity_limits
                or self.velocity_limits_fused
                or self._steel_plan.has_free_rigid_bodies
            )
            and np.isfinite(options.velocity_limit_activation_fraction),
            propagation_same_articulation_rows=options.propagation_same_articulation_rows,
            has_torsion=options.enable_torsional_friction,
            has_compliance=options.contact_compliance,
            has_sleeping=options.enable_sleeping,
            sleep_skip_constraints=options.enable_sleeping and options.sleep_skip_constraints,
            has_row_watermark=options.row_watermark,
            preelim_max_rows=array_class.PREELIM_MAX_ROWS,
        )
        self.steel_model = array_class.get_steel_model_info(self._steel_plan, arts_preelim_idx)
        self.steel_state = array_class.get_steel_state(self)
        self.steel_dyn = array_class.get_steel_dynamics(self)
        self.steel_info = array_class.get_steel_info(self)
        max_contacts = self.collider.collider_state.contact_data.geom_a.shape[0]
        self.steel_contacts = array_class.get_steel_contacts(self, max_contacts)
        self.steel_patch_geometry = self._build_patch_geometry()
        self.steel_patch_frame = array_class.get_steel_patch_frame(self, max_contacts)
        self.steel_patch_history = array_class.get_steel_patch_history(self, max_contacts)
        self.steel_torsion_witnesses = array_class.get_steel_torsion_witnesses(self, max_contacts)
        self.steel_rows = array_class.get_steel_rows(self)
        self.steel_free_rows = array_class.get_steel_free_rows(self)
        self.steel_split = array_class.get_steel_split(self)
        self.steel_prop = array_class.get_steel_propagation(self)
        self.steel_contact_history = array_class.get_steel_contact_history(self, max_contacts)
        self.steel_sleep = self._build_sleep_state()
        self.steel_status = array_class.get_steel_status(self)
        self._refresh_steel_materials()

        self.subscribe(Subscriber(frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS}), self._on_state_change))
        self.subscribe(Subscriber(MODEL_CHANGES, self._on_model_change))
        self._on_state_change(StateChange.GEOMETRY, None)
        self.reset()
        # The step options are changed through the properties of the solver from now on (see 'SteelOptions').
        options._is_built = True

    def _build_propagation_plan(self) -> bool:
        """Return whether a propagation contact response is active, after checking that the model supports it.

        It is active when some articulation is not a free rigid body, or, for the responses routing every contact to
        propagation rows, when free rigid bodies exist. 'propagation_fused' requires the articulations that are not
        free rigid bodies to share one degree-of-freedom count, and loop closures are not supported.
        """
        options = self._options
        plan = self._steel_plan
        if options.articulated_contact_response == "immediate":
            return False
        if plan.n_connects > 0:
            gs.raise_exception(
                f"Loop closures are not supported with articulated_contact_response="
                f"'{options.articulated_contact_response}'."
            )
        arts_is_articulated = plan.arts_is_free_rigid == 0
        is_free_routed = options.articulated_contact_response in ("propagation", "propagation_colored")
        if not (arts_is_articulated.any() or (is_free_routed and plan.has_free_rigid_bodies)):
            return False
        if options.articulated_contact_response == "propagation_fused":
            sizes = np.unique(plan.arts_n_dofs[arts_is_articulated])
            if len(sizes) != 1:
                gs.raise_exception(
                    "articulated_contact_response='propagation_fused' requires every articulation that is not a free "
                    f"rigid body to have the same number of degrees of freedom, found {sizes.tolist()}: use "
                    "'propagation' instead."
                )
        return True

    def _build_patch_geometry(self) -> array_class.SteelPatchGeometry:
        """Compute the bounding sphere of every geom about its origin and of every link about its origin.

        Spheres, ellipsoids, capsules, cylinders and boxes are bounded by their dimensions, meshes by half the diagonal
        of their bounding box, terrains by the extent of their vertices about their origin and planes, which are
        unbounded, by a sphere of 1e6.
        """
        geoms_radius = np.zeros(self.n_geoms_, dtype=gs.np_float)
        for geom in self.geoms:
            if geom.type == gs.GEOM_TYPE.SPHERE:
                radius = abs(geom.data[0])
            elif geom.type == gs.GEOM_TYPE.ELLIPSOID:
                radius = np.abs(geom.data[:3]).max()
            elif geom.type in (gs.GEOM_TYPE.CAPSULE, gs.GEOM_TYPE.CYLINDER):
                radius = abs(geom.data[0]) + 0.5 * abs(geom.data[1])
            elif geom.type == gs.GEOM_TYPE.BOX:
                radius = 0.5 * np.linalg.norm(geom.data[:3])
            elif geom.type == gs.GEOM_TYPE.MESH:
                radius = 0.5 * np.linalg.norm(geom.init_verts.max(axis=0) - geom.init_verts.min(axis=0))
            elif geom.type == gs.GEOM_TYPE.TERRAIN:
                radius = np.linalg.norm(np.abs(geom.init_verts).max(axis=0))
            else:
                radius = PLANE_BOUNDING_RADIUS
            geoms_radius[geom.idx] = radius
        geoms_offset = qd_to_numpy(self.dyn_info.geoms.pos)
        links_radius = np.zeros(self.n_links_, dtype=gs.np_float)
        geoms_reach = geoms_radius + np.linalg.norm(geoms_offset, axis=-1)
        np.maximum.at(links_radius, qd_to_numpy(self.dyn_info.geoms.link_idx), geoms_reach)
        return array_class.get_steel_patch_geometry(geoms_radius, geoms_offset, links_radius)

    @property
    def compliance_contact_count(self):
        """Number of compliant contacts of every environment solved with the compliant law in the last step."""
        return qd_to_torch(self.steel_rows.compliance_n_active, copy=True)

    @property
    def compliance_skipped_contact_count(self):
        """Number of compliant contacts of every environment the row allocator gave no row in the last step.

        A contact gets no row when none of its bodies responds to impulses or its gap exceeds 'contact_gap_gate'.
        """
        return qd_to_torch(self.steel_rows.compliance_n_skipped, copy=True)

    def _build_sleep_state(self) -> array_class.SteelSleep:
        """Allocate the sleep state, marking the articulations whose fixed root supports their island and those owning
        a mimic joint or a loop closure, whose rows are built every step, so they never sleep.
        """
        plan = self._steel_plan
        if self._options.enable_sleeping and plan.n_arts == 0:
            gs.raise_exception("'enable_sleeping' requires at least one rigid entity with a degree of freedom.")
        arts_is_anchored = np.zeros(plan.n_arts_, dtype=gs.np_int)
        arts_is_anchored[: plan.n_arts] = plan.joints_type[plan.arts_joint_start[:-1]] == STEEL_JOINT_TYPE.FIXED
        arts_is_pinned = np.zeros(plan.n_arts_, dtype=gs.np_int)
        arts_is_pinned[plan.mimics_art] = 1
        arts_is_pinned[plan.connects_art] = 1
        return array_class.get_steel_sleep(self, arts_is_anchored, arts_is_pinned)

    def _build_preelimination_plan(self) -> np.ndarray:
        """Select the articulations whose bilateral rows are pre-eliminated, returning their index per articulation.

        Configurations the pre-elimination does not support warn and keep every mimic and connect row in the sweep.
        """
        options = self._options
        plan = self._steel_plan
        arts_preelim_idx = np.full(plan.n_arts_, -1, dtype=gs.np_int)
        include_mimics = options.bilateral_preelimination_include_mimics
        if not options.enable_bilateral_preelimination or (
            plan.n_connects == 0 and (plan.n_mimics == 0 or not include_mimics)
        ):
            return arts_preelim_idx
        if options.articulated_contact_response != "immediate":
            gs.logger.warning(
                "Bilateral pre-elimination does not support the propagation contact responses: it is disabled."
            )
            return arts_preelim_idx
        if options.velocity_iterations > 0:
            gs.logger.warning(
                "Bilateral pre-elimination does not support 'velocity_iterations' > 0: it is disabled for the whole "
                "solver and every articulation keeps iterative mimic and connect rows."
            )
            return arts_preelim_idx
        if plan.n_connects and (plan.connects_link_p < 0).any():
            gs.logger.warning(
                "Bilateral pre-elimination does not support loop closures to static geometry: it is disabled for the "
                "whole solver and every articulation keeps iterative mimic and connect rows."
            )
            return arts_preelim_idx
        counts = np.zeros(plan.n_arts_, dtype=gs.np_int)
        if include_mimics:
            np.add.at(counts, plan.mimics_art, 1)
        np.add.at(counts, plan.connects_art, 3)
        if counts.max() > array_class.PREELIM_MAX_ROWS:
            gs.logger.warning(
                f"An articulation owns {counts.max()} bilateral rows, more than the pre-elimination capacity of "
                f"{array_class.PREELIM_MAX_ROWS}: it is disabled for the whole solver and every articulation keeps "
                "iterative mimic and connect rows. Set 'bilateral_preelimination_include_mimics=False' to "
                "pre-eliminate the loop closures alone."
            )
            return arts_preelim_idx
        arts_idx = np.flatnonzero(counts)
        arts_preelim_idx[arts_idx] = np.arange(len(arts_idx))
        self._steel_n_preelim = len(arts_idx)
        return arts_preelim_idx

    @property
    def velocity_limits_fused(self) -> bool:
        """Whether the velocity limit of each driven degree of freedom is enforced by a clamp of its drive row."""
        options = self._options
        return (
            options.fuse_joint_velocity_limits
            and options.drive_mode == "rows"
            and self.n_dofs > 0
            and options.enable_joint_velocity_limits
            and np.isfinite(options.velocity_limit_activation_fraction)
        )

    @mutates(StateChange.ACTUATION)
    def set_dofs_velocity_limit(self, limit, dofs_idx=None) -> None:
        """Set the joint velocity limit [m/s or rad/s] of the given degrees of freedom, shared by every environment.

        The limits of prismatic, revolute and multi-axis joints are enforced when 'enable_joint_velocity_limits' is set.
        A non-positive limit is no limit.
        """
        dofs_velocity_limit = qd_to_numpy(self.steel_info.dofs_velocity_limit, copy=True)
        dofs_velocity_limit[slice(None) if dofs_idx is None else dofs_idx] = limit
        self.steel_info.dofs_velocity_limit.from_numpy(dofs_velocity_limit)

    def get_dofs_velocity_limit(self, dofs_idx=None):
        """Return the joint velocity limit of the given degrees of freedom."""
        return qd_to_numpy(self.steel_info.dofs_velocity_limit, copy=True)[
            slice(None) if dofs_idx is None else dofs_idx
        ]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- step options -----------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def baumgarte(self) -> float:
        """Fraction of the error of a row removed per step (see 'SteelOptions'), finite."""
        return self._steel_baumgarte

    @baumgarte.setter
    @gs.assert_built
    def baumgarte(self, baumgarte: float) -> None:
        self._steel_baumgarte = _validate_step_option("baumgarte", baumgarte, allow_negative=True)
        self.steel_info.baumgarte.fill(self._steel_baumgarte)

    @property
    def relaxation(self) -> float:
        """Successive over-relaxation factor of the sweeps (see 'SteelOptions'), finite."""
        return self._steel_relaxation

    @relaxation.setter
    @gs.assert_built
    def relaxation(self, relaxation: float) -> None:
        self._steel_relaxation = _validate_step_option("relaxation", relaxation, allow_negative=True)
        self.steel_info.relaxation.fill(self._steel_relaxation)

    @property
    def warn_constraint_overflow(self) -> bool:
        """Whether the first loss of rows of each row family warns (see 'SteelOptions')."""
        return self._steel_warn_constraint_overflow

    @warn_constraint_overflow.setter
    @gs.assert_built
    def warn_constraint_overflow(self, warn_constraint_overflow: bool) -> None:
        if not isinstance(warn_constraint_overflow, bool):
            gs.raise_exception(f"'warn_constraint_overflow' must be a bool, got {warn_constraint_overflow!r}.")
        self._steel_warn_constraint_overflow = warn_constraint_overflow

    @property
    def velocity_drive_mode(self) -> str:
        """Treatment of the drive rows in the velocity sweeps, 'freeze' or 'active' (see 'SteelOptions')."""
        return self._steel_velocity_drive_mode

    @velocity_drive_mode.setter
    @gs.assert_built
    def velocity_drive_mode(self, velocity_drive_mode: str) -> None:
        if velocity_drive_mode not in ("freeze", "active"):
            gs.raise_exception(f"'velocity_drive_mode' must be 'freeze' or 'active', got {velocity_drive_mode!r}.")
        self._steel_velocity_drive_mode = velocity_drive_mode

    @property
    def contact_warmstart_decay(self) -> float:
        """Scale of the carried warm-start impulses (see 'SteelOptions'), finite and non-negative."""
        return self._steel_contact_warmstart_decay

    @contact_warmstart_decay.setter
    @gs.assert_built
    def contact_warmstart_decay(self, contact_warmstart_decay: float) -> None:
        self._steel_contact_warmstart_decay = _validate_step_option(
            "contact_warmstart_decay", contact_warmstart_decay, allow_negative=False
        )
        self.steel_info.warmstart_decay.fill(self._steel_contact_warmstart_decay)

    @property
    def restitution_velocity_threshold(self) -> float:
        """Minimum incident normal speed [m/s] of a rebound (see 'SteelOptions'), finite and non-negative."""
        return self._steel_restitution_velocity_threshold

    @restitution_velocity_threshold.setter
    @gs.assert_built
    def restitution_velocity_threshold(self, restitution_velocity_threshold: float) -> None:
        self._steel_restitution_velocity_threshold = _validate_step_option(
            "restitution_velocity_threshold", restitution_velocity_threshold, allow_negative=False
        )
        self.steel_info.rebound_velocity_threshold.fill(self._rebound_velocity_threshold)

    @property
    def contact_speculative_scale(self) -> float:
        """Fraction of a positive contact gap closed within a step (see 'SteelOptions'), finite and non-negative."""
        return self._steel_contact_speculative_scale

    @contact_speculative_scale.setter
    @gs.assert_built
    def contact_speculative_scale(self, contact_speculative_scale: float) -> None:
        self._steel_contact_speculative_scale = _validate_step_option(
            "contact_speculative_scale", contact_speculative_scale, allow_negative=False
        )
        self.steel_info.contact_speculative_scale.fill(self._steel_contact_speculative_scale)

    @property
    def contact_gap_gate(self) -> float | None:
        """Gap [m] above which a contact gets no rows, None for no gate (see 'SteelOptions')."""
        return self._steel_contact_gap_gate

    @contact_gap_gate.setter
    @gs.assert_built
    def contact_gap_gate(self, contact_gap_gate: float | None) -> None:
        if contact_gap_gate is not None:
            contact_gap_gate = _validate_step_option("contact_gap_gate", contact_gap_gate, allow_negative=False)
        self._steel_contact_gap_gate = contact_gap_gate
        self.steel_info.contact_gap_limit.fill(self._contact_gap_limit)

    @property
    def contact_friction_gap_max(self) -> float | None:
        """Gap [m] above which a contact gets no friction rows, None for friction rows on every contact.

        Whether every contact gets friction rows (None or inf) is fixed at build: the setter rejects a switch between
        a finite gap and None or inf.
        """
        return self._steel_contact_friction_gap_max

    @contact_friction_gap_max.setter
    @gs.assert_built
    def contact_friction_gap_max(self, contact_friction_gap_max: float | None) -> None:
        if contact_friction_gap_max is not None:
            contact_friction_gap_max = float(contact_friction_gap_max)
            if np.isnan(contact_friction_gap_max):
                gs.raise_exception("'contact_friction_gap_max' must not be NaN.")
        is_every_contact = contact_friction_gap_max is None or np.isposinf(contact_friction_gap_max)
        if is_every_contact != np.isposinf(self._friction_gap_limit):
            gs.raise_exception("'contact_friction_gap_max' cannot switch to or from None (or inf) after the build.")
        self._steel_contact_friction_gap_max = contact_friction_gap_max
        self.steel_info.friction_gap_limit.fill(self._friction_gap_limit)

    @property
    def _rebound_velocity_threshold(self) -> float:
        """Minimum incident speed of a rebound as the kernels read it, the largest float32 without restitution."""
        if self._options.enable_restitution:
            return self._steel_restitution_velocity_threshold
        return float(np.finfo(np.float32).max)

    @property
    def _contact_gap_limit(self) -> float:
        """Gap [m] above which a contact gets no rows, as the kernels read it."""
        return float("inf") if self._steel_contact_gap_gate is None else self._steel_contact_gap_gate

    @property
    def _friction_gap_limit(self) -> float:
        """Gap [m] above which a contact gets no friction rows, as the kernels read it."""
        return float("inf") if self._steel_contact_friction_gap_max is None else self._steel_contact_friction_gap_max

    # ------------------------------------------------------------------------------------
    # ------------------------------------ materials -------------------------------------
    # ------------------------------------------------------------------------------------

    def _set_steel_material(self, values, material: qd.Tensor, idx, idx_max: int, idx_name: str, envs_idx) -> None:
        """Write one Steel material input of the given geoms or links of the given environments.

        material holds one value per geom or link and environment, idx selects the geoms or links among idx_max and
        idx_name names them in the errors.
        """
        if gs.use_zerocopy:
            if envs_idx is not None and self.n_envs == 0:
                gs.raise_exception("`envs_idx` is not supported for non-parallelized scene.")
            if isinstance(envs_idx, (int, np.integer)) and not -self.n_envs <= envs_idx < self.n_envs:
                gs.raise_exception(f"`envs_idx` out of range: {envs_idx} not in [{-self.n_envs}, {self.n_envs}).")
            mask = indices_to_mask(envs_idx, idx)
            material_t = qd_to_torch(material, transpose=True, copy=False)
            assign_indexed_tensor(material_t, mask, broadcast_tensor(values, gs.tc_float, material_t[mask].shape))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return
        values, idx, envs_idx = self._sanitize_io_variables(
            values, idx, idx_max, idx_name, envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            values = values[None]
        kernel_steel_set_material(idx, envs_idx, values, material, self.steel_config)

    @mutates(StateChange.MATERIAL)
    def set_geoms_restitution(self, restitution, geoms_idx=None, envs_idx=None) -> None:
        """Set the restitution coefficient of the given geoms in the given environments (see 'gs.materials.Rigid').

        The restitution of a contact is the mean of the coefficients of its two geoms clamped to [0, 1], a non-finite
        coefficient counting as 0. A positive coefficient requires solve_mode='matrix_free' and no 'contact_compliance'
        unless 'enable_restitution' is off.
        """
        options = self._options
        if (
            options.enable_restitution
            and (options.solve_mode == "split" or options.contact_compliance)
            and (torch.as_tensor(restitution) > 0.0).any()
        ):
            gs.raise_exception(
                "A positive restitution requires solve_mode='matrix_free' and no 'contact_compliance': set "
                "'enable_restitution=False' to ignore the restitution of the materials."
            )
        self._set_steel_material(
            restitution, self.steel_info.geoms_restitution, geoms_idx, self.n_geoms, "geoms_idx", envs_idx
        )

    def get_geoms_restitution(self, geoms_idx=None, envs_idx=None):
        """Return the restitution coefficient of the given geoms in the given environments."""
        tensor = qd_to_torch(self.steel_info.geoms_restitution, envs_idx, geoms_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    @mutates(StateChange.MATERIAL)
    def set_geoms_contact_stiffness(self, stiffness, geoms_idx=None, envs_idx=None) -> None:
        """Set the contact stiffness [N/m] of the given geoms in the given environments (see 'gs.materials.Rigid').

        A zero stiffness makes the contacts of the geom rigid. Requires 'contact_compliance', and a positive stiffness
        is rejected with 'enable_torsional_friction'.
        """
        options = self._options
        if not options.contact_compliance:
            gs.raise_exception("The contact stiffness and damping of the geoms require 'contact_compliance'.")
        if options.enable_torsional_friction and (torch.as_tensor(stiffness) > 0.0).any():
            gs.raise_exception("'enable_torsional_friction' does not support compliant contacts.")
        self._set_steel_material(
            stiffness, self.steel_info.geoms_contact_stiffness, geoms_idx, self.n_geoms, "geoms_idx", envs_idx
        )

    def get_geoms_contact_stiffness(self, geoms_idx=None, envs_idx=None):
        """Return the contact stiffness of the given geoms in the given environments, zero for a rigid geom."""
        if not self._options.contact_compliance:
            gs.raise_exception("The contact stiffness and damping of the geoms require 'contact_compliance'.")
        tensor = qd_to_torch(self.steel_info.geoms_contact_stiffness, envs_idx, geoms_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    @mutates(StateChange.MATERIAL)
    def set_geoms_contact_damping(self, damping, geoms_idx=None, envs_idx=None) -> None:
        """Set the contact damping [N s/m] of the given geoms in the given environments (see 'gs.materials.Rigid').

        It acts on the contacts of the geoms with a positive contact stiffness. Requires 'contact_compliance'.
        """
        if not self._options.contact_compliance:
            gs.raise_exception("The contact stiffness and damping of the geoms require 'contact_compliance'.")
        self._set_steel_material(
            damping, self.steel_info.geoms_contact_damping, geoms_idx, self.n_geoms, "geoms_idx", envs_idx
        )

    def get_geoms_contact_damping(self, geoms_idx=None, envs_idx=None):
        """Return the contact damping of the given geoms in the given environments."""
        if not self._options.contact_compliance:
            gs.raise_exception("The contact stiffness and damping of the geoms require 'contact_compliance'.")
        tensor = qd_to_torch(self.steel_info.geoms_contact_damping, envs_idx, geoms_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    @mutates(StateChange.MODEL)
    def set_links_angular_damping(self, angular_damping, links_idx=None, envs_idx=None) -> None:
        """Set the angular velocity damping rate [1/s] of the given links in the given environments.

        It damps the root link of a free-floating entity, as w *= 1 - angular_damping * dt every step (see
        'gs.materials.Rigid'), and is read at every step.
        """
        self._set_steel_material(
            angular_damping, self.steel_info.links_angular_damping, links_idx, self.n_links, "links_idx", envs_idx
        )

    def get_links_angular_damping(self, links_idx=None, envs_idx=None):
        """Return the angular velocity damping rate of the given links in the given environments."""
        tensor = qd_to_torch(self.steel_info.links_angular_damping, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    @mutates(StateChange.MODEL)
    def set_links_max_linear_velocity(self, max_linear_velocity, links_idx=None, envs_idx=None) -> None:
        """Set the bound [m/s] of the center of mass speed of the given links in the given environments.

        It bounds every world axis of the velocity of a free rigid body, enforced by velocity rows, and is read at every
        step. A non-finite or non-positive bound disables it.
        """
        self._set_steel_material(
            max_linear_velocity,
            self.steel_info.links_max_linear_velocity,
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
        )

    def get_links_max_linear_velocity(self, links_idx=None, envs_idx=None):
        """Return the bound of the center of mass speed of the given links in the given environments."""
        tensor = qd_to_torch(self.steel_info.links_max_linear_velocity, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    @mutates(StateChange.MODEL)
    def set_links_max_angular_velocity(self, max_angular_velocity, links_idx=None, envs_idx=None) -> None:
        """Set the bound [rad/s] of the angular speed of the given links in the given environments.

        It bounds every world axis of the angular velocity of a free rigid body, enforced by velocity rows, and is read
        at every step. A non-finite or non-positive bound disables it.
        """
        self._set_steel_material(
            max_angular_velocity,
            self.steel_info.links_max_angular_velocity,
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
        )

    def get_links_max_angular_velocity(self, links_idx=None, envs_idx=None):
        """Return the bound of the angular speed of the given links in the given environments."""
        tensor = qd_to_torch(self.steel_info.links_max_angular_velocity, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    @mutates(StateChange.MODEL)
    def set_links_max_depenetration_velocity(self, max_depenetration_velocity, links_idx=None, envs_idx=None) -> None:
        """Set the bound [m/s] of the penetration correction velocity of the given links in the given environments.

        It bounds the correction of the contacts of free rigid bodies, the smaller bound of the two bodies of a contact
        applying, and is read at every step. A non-finite or non-positive bound disables it.
        """
        self._set_steel_material(
            max_depenetration_velocity,
            self.steel_info.links_max_depenetration_velocity,
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
        )

    def get_links_max_depenetration_velocity(self, links_idx=None, envs_idx=None):
        """Return the bound of the penetration correction velocity of the given links in the given environments."""
        tensor = qd_to_torch(
            self.steel_info.links_max_depenetration_velocity, envs_idx, links_idx, transpose=True, copy=True
        )
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_parent_wrench(self, links_idx=None, envs_idx=None):
        """Return the wrench the inbound joint of each given link transmits to it, about its center of mass.

        It is the net wrench of the subtree of the link from the inverse dynamics at the start of the last step, in
        world axes: the force [N] in the first three components, the torque [N m] in the last three. It leaves out the
        constraint and contact impulses of the step. The world link of a model and the links of an entity without degree
        of freedom report zero, and a sleeping link the wrench of the last step it was awake.
        """
        force = qd_to_torch(self.steel_state.links_parent_f_lin, envs_idx, links_idx, transpose=True)
        torque = qd_to_torch(self.steel_state.links_parent_f_ang, envs_idx, links_idx, transpose=True)
        tensor = torch.cat((force, torque), dim=-1)
        return tensor[0] if self.n_envs == 0 else tensor

    def _refresh_steel_materials(self) -> None:
        """Copy the friction coefficients of the geoms, scaled by 'contact_friction_scale', into the Steel materials."""
        friction = qd_to_numpy(self.dyn_info.geoms.friction)
        friction_ratio = qd_to_numpy(self.dyn_state.geoms.friction_ratio)
        geoms_mu = (friction[:, None] if friction.ndim == 1 else friction) * friction_ratio
        self.steel_info.geoms_mu.from_numpy((geoms_mu * self._options.contact_friction_scale).astype(gs.np_float))

    # ------------------------------------------------------------------------------------
    # ------------------------------ history and notifications ---------------------------
    # ------------------------------------------------------------------------------------

    def reset(self, envs_idx=None) -> None:
        """Discard the solver history of the given environments, leaving their state unchanged.

        It clears the capacity status ('constraint_overflow') and the compliant contact counts, discards the warm-start
        impulses and the friction anchors, wakes the sleeping bodies and requests a mass matrix refresh on the next
        step, so a teleported articulation reuses no stale factorization or anchor. The other environments keep their
        history. 'scene.reset' resets the environments it restores, a setter of the state keeps their history.
        """
        is_masked = envs_idx is not None
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        if self._steel_friction_anchor_beta > 0.0:
            kernel_steel_reset_patches(envs_idx, self.steel_patch_history, self.steel_config)
        if self._options.contact_warmstart:
            kernel_steel_reset_warmstart(envs_idx, self.steel_contact_history, self.steel_config)
        kernel_steel_reset_status(envs_idx, self.steel_state, self.steel_rows, self.steel_status, self.steel_config)
        if self._options.enable_sleeping:
            # A reset of every environment restarts the sleep history, a reset of some of them only wakes them.
            kernel_steel_sleep_wake(
                envs_idx,
                self.steel_state,
                self.steel_sleep,
                self.steel_config,
                mark_changed=False,
                clear_asleep_steps=not is_masked,
            )
            if not is_masked:
                self._steel_sleep_is_history_valid = False

    def set_state(self, f, state, envs_idx=None, *, partial: bool = False) -> None:
        """Restore the state of the given environments, then discard their solver history (see 'reset')."""
        super().set_state(f, state, envs_idx, partial=partial)
        if self.is_active:
            self.reset(envs_idx)

    def add_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        """Reject the weld constraint: the Steel solver builds its equality constraints once, from the model."""
        gs.raise_exception("The Steel solver does not support adding weld constraints after the build.")

    def delete_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        """Reject the deletion: the Steel solver builds its equality constraints once, from the model."""
        gs.raise_exception("The Steel solver does not support deleting weld constraints after the build.")

    def _on_model_change(self, change: StateChange, envs_idx) -> None:
        """Refresh the Steel data that a change of the model parameters enters.

        A change of the mass distribution or of the joint-space force laws (the drive gains enter the augmented mass
        matrix) requests a mass matrix refresh of every articulation on the next step. A change of the contact material
        refreshes the friction coefficients of the geoms. Changes of the mass distribution and of the contact material
        discard the warm-start impulses of every environment, the other changes keep them. With sleeping, every island
        of the changed environments wakes and is vetoed at the next step, the gravity included: the notification names
        environments, not the entities whose parameters changed.
        """
        if change in (StateChange.INERTIA, StateChange.ACTUATION):
            self.steel_state.arts_mass_update_requested.fill(1)
        if change == StateChange.MATERIAL:
            self._refresh_steel_materials()
        if change in (StateChange.INERTIA, StateChange.MATERIAL) and self._options.contact_warmstart:
            kernel_steel_reset_warmstart(self._scene._envs_idx, self.steel_contact_history, self.steel_config)
        if self._options.enable_sleeping:
            kernel_steel_sleep_wake(
                self._scene._sanitize_envs_idx(envs_idx),
                self.steel_state,
                self.steel_sleep,
                self.steel_config,
                mark_changed=True,
                clear_asleep_steps=False,
            )

    def _on_state_change(self, change: StateChange, envs_idx) -> None:
        """Reload the Steel state of the environments a Genesis setter changed, keeping their solver history.

        With sleeping, the change wakes the islands of those environments. A change of selected environments also
        vetoes their islands at the next step, the notification naming environments, not the entities whose state
        changed. A change of every environment restarts the sleep history instead, so it wakes the islands without
        vetoing them.
        """
        if self._steel_plan.n_arts == 0:
            return
        is_masked = envs_idx is not None
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        kernel_steel_pull(
            envs_idx,
            self.dyn_state,
            self.steel_state,
            self.steel_dyn,
            self.dyn_info,
            self.rigid_info,
            self.steel_model,
            self.steel_config,
        )
        if self._options.enable_sleeping:
            kernel_steel_sleep_wake(
                envs_idx,
                self.steel_state,
                self.steel_sleep,
                self.steel_config,
                mark_changed=is_masked and self._steel_sleep_is_history_valid,
                clear_asleep_steps=not is_masked,
            )
            if not is_masked:
                self._steel_sleep_is_history_valid = False

    def _warn_persistent_row_capacity(self) -> None:
        """Warn if every step is certain to request more joint-space rows than 'articulation_constraint_max'.

        With 'enable_joint_limit' and an infinite 'joint_limit_activation_gap', every finite position limit of a
        prismatic, revolute or multi-axis degree of freedom is a row in every step.
        """
        options = self._options
        plan = self._steel_plan
        if not options.enable_joint_limit or np.isfinite(options.joint_limit_activation_gap) or self.n_dofs == 0:
            return
        dofs_is_scalar_axis = np.isin(
            plan.joints_type[plan.dofs_joint],
            (STEEL_JOINT_TYPE.PRISMATIC, STEEL_JOINT_TYPE.REVOLUTE, STEEL_JOINT_TYPE.D6),
        )
        dofs_n_finite_limits = np.isfinite(qd_to_numpy(self.dyn_info.dofs.limit)).sum(axis=-1)
        envs_n_rows = np.broadcast_to(dofs_n_finite_limits[dofs_is_scalar_axis].sum(axis=0), (self._B,))
        envs_idx = np.flatnonzero(envs_n_rows > options.articulation_constraint_max)
        if envs_idx.size:
            gs.logger.warning(
                f"Environments {envs_idx[:16].tolist()} need at least {envs_n_rows.max()} joint-space rows in every "
                f"step for their finite joint position limits alone, more than articulation_constraint_max="
                f"{options.articulation_constraint_max}. The rows beyond the capacity are dropped and flagged in "
                "'constraint_overflow'. Increase 'articulation_constraint_max', set a finite "
                "'joint_limit_activation_gap' so that only the limits near the joint position create rows, or set the "
                "unbounded limits to +/-inf."
            )

    def substep(self, f):
        if not self._is_forward_pos_updated:
            kernel_forward_kinematics_links_geoms(
                self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
            )
        if not self._is_forward_vel_updated:
            kernel_forward_velocity(
                self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, False
            )

        collider = self.collider
        collider._contact_data_cache.clear()
        kernel_substep_collision(
            self.geoms_init_AABB,
            self.dyn_state,
            collider.collider_state,
            collider.mpr.mpr_state,
            collider.gjk.gjk_state,
            collider.gjk.gjk_state.diff_contact_input,
            collider.mpr.contact0_mpr_state,
            collider.gjk.contact0_gjk_state,
            collider.mpr.multicontact_mpr_state,
            collider.gjk.multicontact_gjk_state,
            self.constraint_solver.constraint_state,
            self.dyn_info,
            self.rigid_info,
            collider.collider_info,
            self.rigid_config,
            collider.collider_config,
            collider.gjk.gjk_config,
            False,
            collider._n_possible_pairs > 0,
            collider._use_split_narrowphase,
            collider._use_coop_dedup,
            self._errno,
        )

        if self._steel_plan.n_arts > 0:
            self._steel_advance()

        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def _steel_gather_contacts(self) -> None:
        """Convert the contacts of the collision detection of the step into Steel contacts."""
        kernel_steel_gather_contacts(
            self.dyn_state,
            self.collider.collider_state,
            self.steel_state,
            self.steel_contacts,
            self.steel_model,
            self.steel_config,
        )

    @property
    def constraint_overflow(self) -> torch.Tensor:
        """Whether every environment lost constraint rows or contacts for lack of capacity since its last reset."""
        return qd_to_torch(self.steel_status.envs_overflow) != 0

    @property
    def constraint_row_watermarks(self) -> SteelRowWatermarks:
        """The high-water marks of the constraint rows since the scene was built, zero without 'row_watermark'."""
        values = [0] * (array_class.SteelRowWatermark.CONTACT_SLOT + 1)
        if self._options.row_watermark:
            values = qd_to_numpy(self.steel_status.row_watermarks).tolist()
        stride = array_class.SteelRowWatermark.FAMILY_STRIDE
        families = [SteelRowFamilyWatermarks(*values[stride * i_f : stride * (i_f + 1)]) for i_f in range(3)]
        return SteelRowWatermarks(*families, values[array_class.SteelRowWatermark.CONTACT_SLOT])

    def check_constraint_capacity(self) -> None:
        """Raise if any environment lost constraint rows or contacts for lack of capacity since its last reset.

        The lost rows leave constraints unenforced in the steps that dropped them, so their output is not trustworthy:
        increase the capacities ('articulation_constraint_max', 'free_body_constraint_max', 'max_collision_pairs') and
        reset the environments.
        """
        envs_overflow = qd_to_numpy(self.steel_status.envs_overflow)
        envs_idx = np.flatnonzero(envs_overflow)
        if envs_idx.size:
            gs.raise_exception(
                f"Steel lost constraint rows or contacts for lack of capacity in environments {envs_idx[:16].tolist()} "
                "since their last reset. Increase 'articulation_constraint_max' or 'free_body_constraint_max', or the "
                "collision buffers, then reset these environments."
            )

    def check_errno(self):
        super().check_errno()
        if not self._steel_warn_constraint_overflow:
            return
        # The warnings ride the host synchronization of the error check instead of adding one per step.
        overflow = int(np.bitwise_or.reduce(qd_to_numpy(self.steel_status.envs_overflow)))
        for bit, option in (
            (array_class.SteelOverflow.DENSE_ROWS, "articulation_constraint_max"),
            (array_class.SteelOverflow.FREE_ROWS, "free_body_constraint_max"),
            (array_class.SteelOverflow.PROPAGATION_ROWS, "articulation_constraint_max' or 'free_body_constraint_max"),
        ):
            if overflow & bit and not self._steel_overflow_warned & bit:
                self._steel_overflow_warned |= bit
                gs.logger.warning(
                    f"Steel dropped constraint rows for lack of capacity, whose constraints are not enforced. Increase "
                    f"'{option}' (see SteelSolver.check_constraint_capacity)."
                )

    def _steel_advance(self) -> None:
        """Advance the Steel state by one substep and write it back to the Genesis state.

        The contacts of the step are gathered first (see _steel_gather_contacts), then kernel_steel_step runs the step.
        With velocity iterations, it stops after the row setup and the position solve, the velocity pass and the end of
        the step follow in kernels of their own.
        """
        options = self._options
        global_refresh = self._steel_step % options.update_mass_matrix_interval == 0
        n_friction_iterations = options.contact_friction_position_iterations
        friction_start = (
            0 if n_friction_iterations < 0 else options.iterations - min(n_friction_iterations, options.iterations)
        )

        # The contacts are gathered from the poses the step starts from, which only the last stage of the step changes.
        self._steel_gather_contacts()
        regularize = self._steel_contact_w < 1.0
        kernel_steel_step(
            options.iterations,
            friction_start,
            self.dyn_state,
            self.collider.collider_state,
            self.steel_state,
            self.steel_contacts,
            self.steel_contact_history,
            self.steel_patch_frame,
            self.steel_patch_history,
            self.steel_torsion_witnesses,
            self.steel_rows,
            self.steel_free_rows,
            self.steel_prop,
            self.steel_split,
            self.steel_dyn,
            self.steel_sleep,
            self.steel_status,
            self.dyn_info,
            self.rigid_info,
            self.steel_patch_geometry,
            self.steel_info,
            self.steel_model,
            self.rigid_config,
            self.steel_config,
            global_refresh,
            regularize,
            self._errno,
        )
        if options.velocity_iterations > 0:
            # Both solves launch the one compiled solve kernel, whose instance would otherwise compile twice.
            kernel_steel_solve(
                options.iterations,
                friction_start,
                self.steel_rows.rhs,
                self.steel_free_rows.rhs,
                self.steel_prop.rhs,
                self.steel_rows,
                self.steel_free_rows,
                self.steel_prop,
                self.steel_dyn,
                self.steel_info,
                self.steel_model,
                self.steel_config,
                options.contact_warmstart,
                regularize,
                freeze_drive_rows=False,
            )
            kernel_steel_velocity_pass(
                self.steel_rows,
                self.steel_free_rows,
                self.steel_prop,
                self.steel_dyn,
                self.rigid_info,
                self.steel_info,
                self.steel_model,
                self.steel_config,
            )
            kernel_steel_solve(
                options.velocity_iterations,
                friction_start_iteration=0,
                rhs_dense=self.steel_rows.rhs_unbiased,
                rhs_free=self.steel_free_rows.rhs_unbiased,
                rhs_propagation=self.steel_prop.rhs_unbiased,
                steel_rows=self.steel_rows,
                steel_free_rows=self.steel_free_rows,
                steel_prop=self.steel_prop,
                steel_dyn=self.steel_dyn,
                steel_info=self.steel_info,
                steel_model=self.steel_model,
                steel_config=self.steel_config,
                propagate_warmstart=False,
                regularize=False,
                freeze_drive_rows=self._steel_velocity_drive_mode == "freeze",
            )
            kernel_steel_step_end(
                self.dyn_state,
                self.collider.collider_state,
                self.steel_state,
                self.steel_contacts,
                self.steel_contact_history,
                self.steel_patch_frame,
                self.steel_patch_history,
                self.steel_rows,
                self.steel_free_rows,
                self.steel_prop,
                self.steel_dyn,
                self.steel_sleep,
                self.steel_status,
                self.dyn_info,
                self.rigid_info,
                self.steel_info,
                self.steel_model,
                self.rigid_config,
                self.steel_config,
            )
        if options.enable_sleeping:
            self._steel_sleep_is_history_valid = True
        self._steel_step += 1


@qd.func
def func_steel_solve(
    n_iterations: int,
    friction_start_iteration: int,
    rhs_dense: qd.Tensor,
    rhs_free: qd.Tensor,
    rhs_propagation: qd.Tensor,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
    propagate_warmstart: bool,
    regularize: bool,
    freeze_drive_rows: bool,
):
    """Run n_iterations Gauss-Seidel iterations of the rows of every environment in the row order of 'row_schedule'.

    'interleaved' sweeps every row family in each iteration. 'contact_then_internal' runs every iteration of the contact
    rows first, then every iteration of the internal rows and the velocity limits. 'phased' solves, in each iteration,
    the internal rows, then the contact rows, then the velocity limits. A propagation response solves, in each
    iteration, the internal rows, the dense and free-body contact rows, then refreshes the link twists, sweeps the
    propagation rows, propagates their impulses through the trees, and solves the velocity limits last. With
    'contact_then_internal' its iterations leave out the internal rows and the velocity limits, which follow in
    iterations of their own. 'propagation_fused' recomputes every link twist before each sweep, the others only those
    an earlier row family of the iteration can have moved. propagate_warmstart first propagates the impulses the
    propagation rows start with through the trees.

    Every environment runs its sequence of passes in one thread, each pass being a run of sweeps of some row families
    (see func_pgs_sweeps) or a propagation step. A schedule iterated pass by pass holds 4 passes per iteration (internal
    rows, contact rows, propagation, velocity limits) between a leading propagation warm-start pass and a trailing
    pass of the rows 'contact_then_internal' defers, the passes a configuration leaves out being inert.
    """
    _B = steel_dyn.v_out.shape[1]
    n_arts = steel_model.arts_dof_start.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        n_passes = 1
        if qd.static(steel_config.has_propagation or steel_config.row_schedule == 2):
            n_passes = 4 * n_iterations + 2
        elif qd.static(steel_config.row_schedule == 1):
            n_passes = 2
        for i_pass in range(n_passes):
            pass_iterations = n_iterations
            iteration_offset = 0
            row_phase = 0
            is_sweep_pass = True
            is_propagation_pass = False
            if qd.static(steel_config.has_propagation or steel_config.row_schedule == 2):
                pass_iterations = 1
                if i_pass == 0:
                    is_sweep_pass = False
                    is_propagation_pass = propagate_warmstart
                elif i_pass == n_passes - 1:
                    pass_iterations = n_iterations
                    row_phase = 2
                    if qd.static(
                        not (
                            steel_config.has_propagation
                            and steel_config.row_schedule == 1
                            and (steel_config.has_internal_rows or steel_config.has_velocity_limit_rows)
                        )
                    ):
                        is_sweep_pass = False
                else:
                    iteration_offset = (i_pass - 1) // 4
                    i_kind = (i_pass - 1) % 4
                    if i_kind == 0:
                        row_phase = 3
                        if qd.static(
                            not steel_config.has_internal_rows
                            or (steel_config.has_propagation and steel_config.row_schedule == 1)
                        ):
                            is_sweep_pass = False
                    elif i_kind == 1:
                        row_phase = 4
                    elif i_kind == 2:
                        is_sweep_pass = False
                        is_propagation_pass = True
                    else:
                        row_phase = 5
                        if qd.static(
                            not steel_config.has_velocity_limit_rows
                            or (steel_config.has_propagation and steel_config.row_schedule == 1)
                        ):
                            is_sweep_pass = False
            elif qd.static(steel_config.row_schedule == 1):
                row_phase = 1 + i_pass
            if is_sweep_pass:
                func_pgs_sweeps(
                    i_b=i_b,
                    n_iterations=pass_iterations,
                    iteration_offset=iteration_offset,
                    friction_start_iteration=friction_start_iteration,
                    row_phase=row_phase,
                    rhs_dense=rhs_dense,
                    rhs_free=rhs_free,
                    steel_rows=steel_rows,
                    steel_free_rows=steel_free_rows,
                    steel_dyn=steel_dyn,
                    steel_info=steel_info,
                    steel_config=steel_config,
                    regularize=regularize,
                    freeze_drive_rows=freeze_drive_rows,
                )
            if qd.static(steel_config.has_propagation):
                if is_propagation_pass:
                    if i_pass == 0:
                        func_propagation_warmstart(i_b, steel_prop)
                    else:
                        for i_a in range(n_arts):
                            func_propagation_refresh_articulation(
                                i_a,
                                i_b,
                                steel_prop,
                                steel_dyn,
                                steel_model,
                                refresh_all=qd.static(
                                    steel_config.articulated_contact_response == 2
                                    or steel_config.row_schedule == 1
                                    or steel_config.has_internal_rows
                                    or steel_config.has_velocity_limit_rows
                                ),
                            )
                        func_propagation_sweep(
                            i_b,
                            iteration_offset,
                            friction_start_iteration,
                            rhs_propagation,
                            steel_prop,
                            steel_info,
                            regularize,
                        )
                    for i_a in range(n_arts):
                        func_propagation_propagate_articulation(i_a, i_b, steel_prop, steel_dyn, steel_model)


@qd.func
def func_steel_step_end(
    dt: float,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_contact_history: array_class.SteelContactHistory,
    steel_patch_frame: array_class.SteelPatchFrame,
    steel_patch_history: array_class.SteelPatchHistory,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    steel_status: array_class.SteelStatus,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    rigid_config: qd.template(),
    steel_config: qd.template(),
    sleep_linear_threshold: float,
    sleep_angular_threshold: float,
    sleep_quiet_time: float,
):
    """Integrate the solved velocity (stage 7), keep the histories of the step and publish the new state to Genesis.

    With velocity iterations, the positions follow the velocity of the position solve held in v_out_snap and the
    velocity refined by the velocity-only iterations is published. The Cartesian quantities of Genesis are updated from
    the new state. The sleep thresholds configure the sleeping islands (see func_steel_sleep_finish).
    """
    if qd.static(steel_config.has_velocity_iterations):
        func_steel_integrate(
            dt=dt,
            velocity=steel_dyn.v_out_snap,
            dyn_state=dyn_state,
            steel_state=steel_state,
            steel_dyn=steel_dyn,
            steel_sleep=steel_sleep,
            dyn_info=dyn_info,
            steel_info=steel_info,
            steel_model=steel_model,
            steel_config=steel_config,
        )
        func_steel_publish_refined_velocity(dt, steel_state, steel_dyn, steel_info, steel_model, steel_config)
    else:
        func_steel_integrate(
            dt=dt,
            velocity=steel_dyn.v_out,
            dyn_state=dyn_state,
            steel_state=steel_state,
            steel_dyn=steel_dyn,
            steel_sleep=steel_sleep,
            dyn_info=dyn_info,
            steel_info=steel_info,
            steel_model=steel_model,
            steel_config=steel_config,
        )
    func_steel_contact_forces(
        dt, dyn_state, collider_state, steel_contacts, steel_rows, steel_free_rows, steel_prop, steel_config
    )
    func_steel_forward_kinematics(dyn_state, steel_state, steel_dyn, steel_sleep, dyn_info, steel_model, steel_config)
    if qd.static(steel_config.has_sleeping):
        func_steel_sleep_finish(
            dt,
            steel_state,
            steel_dyn,
            steel_sleep,
            steel_status,
            steel_model,
            steel_config,
            sleep_linear_threshold,
            sleep_angular_threshold,
            sleep_quiet_time,
        )
    if qd.static(steel_config.has_friction_anchors):
        func_steel_finish_patch_impulses(
            CONTACT_PATH.DENSE,
            dt,
            steel_rows.row_parent,
            steel_rows.row_mu,
            steel_rows.impulses,
            steel_state,
            steel_contacts,
            steel_patch_frame,
            dyn_info,
            steel_info,
            steel_config,
        )
        if qd.static(steel_config.has_free_rigid_bodies):
            func_steel_finish_patch_impulses(
                CONTACT_PATH.FREE,
                dt,
                steel_free_rows.row_parent,
                steel_free_rows.row_mu,
                steel_free_rows.impulses,
                steel_state,
                steel_contacts,
                steel_patch_frame,
                dyn_info,
                steel_info,
                steel_config,
            )
        if qd.static(steel_config.has_propagation):
            func_steel_finish_patch_impulses(
                CONTACT_PATH.PROPAGATION,
                dt,
                steel_prop.row_parent,
                steel_prop.row_mu,
                steel_prop.impulses,
                steel_state,
                steel_contacts,
                steel_patch_frame,
                dyn_info,
                steel_info,
                steel_config,
            )
        func_steel_store_patches(steel_state, steel_contacts, steel_patch_frame, steel_patch_history, steel_config)
    if qd.static(steel_config.has_contact_warmstart):
        func_steel_snapshot_warmstart_rows(
            steel_rows.count,
            steel_rows.row_type,
            steel_rows.row_parent,
            steel_rows.impulses,
            steel_contact_history.dense_row_type,
            steel_contact_history.dense_row_parent,
            steel_contact_history.dense_impulses,
            steel_config,
        )
        if qd.static(steel_config.has_free_rigid_bodies):
            func_steel_snapshot_warmstart_rows(
                steel_free_rows.count,
                steel_free_rows.row_type,
                steel_free_rows.row_parent,
                steel_free_rows.impulses,
                steel_contact_history.free_row_type,
                steel_contact_history.free_row_parent,
                steel_contact_history.free_impulses,
                steel_config,
            )
        if qd.static(steel_config.has_propagation):
            func_steel_snapshot_warmstart_rows(
                steel_prop.count,
                steel_prop.row_type,
                steel_prop.row_parent,
                steel_prop.impulses,
                steel_contact_history.propagation_row_type,
                steel_contact_history.propagation_row_parent,
                steel_contact_history.propagation_impulses,
                steel_config,
            )
        func_steel_snapshot_warmstart(dt, steel_state, steel_contacts, steel_contact_history, steel_config)
    func_steel_publish(dt, dyn_state, steel_state, dyn_info, rigid_info, steel_model, steel_config)
    func_update_cartesian_space(
        dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms=False, is_backward=False
    )
    func_forward_velocity(dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)


@qd.kernel(graph=True, fastcache=True)
def kernel_steel_step(
    n_iterations: int,
    friction_start_iteration: int,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_contact_history: array_class.SteelContactHistory,
    steel_patch_frame: array_class.SteelPatchFrame,
    steel_patch_history: array_class.SteelPatchHistory,
    steel_torsion_witnesses: array_class.SteelTorsionWitnesses,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_split: array_class.SteelSplit,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    steel_status: array_class.SteelStatus,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_patch_geometry: array_class.SteelPatchGeometry,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    rigid_config: qd.template(),
    steel_config: qd.template(),
    global_refresh: bool,
    regularize: bool,
    errno: qd.Tensor,
):
    """Advance the Steel state by one substep from the gathered contacts and publish it, as one graph.

    Stage 1 evaluates the forward kinematics, the inverse dynamics with implicit drives and the mass matrix (refreshed
    when global_refresh), stage 2 factors it and stage 3 predicts the unconstrained velocity. Stage 4 builds the
    constraint rows: friction patches select the contacts receiving friction rows, so they are built before the row
    allocation. Stages 5 and 6 solve the rows by projected Gauss-Seidel in impulse space over n_iterations iterations,
    the friction rows from friction_start_iteration on, starting from the unconstrained prediction plus the velocity of
    the warm-start impulses, regularize applying the contact regularization weight. Stage 7 integrates and the new
    state is published (see func_steel_step_end). With velocity iterations, the step stops before stages 5 and 6.
    """
    # The constants of the step come from the info structs: a float argument makes the launch arguments uncacheable,
    # so that every launch would rebuild the arguments of every struct.
    dt = rigid_info.substep_dt[None]
    # Drive rows come first in the row order and must exist before the velocity scaling of stage 1. The sleeping
    # islands of the step are built from its contacts.
    func_steel_allocate_drive_rows(dyn_state, steel_rows, dyn_info, rigid_info, steel_model, steel_config)
    func_steel_inputs(dyn_state, steel_state, steel_dyn, dyn_info, steel_model, steel_config)
    if qd.static(steel_config.has_sleeping):
        func_steel_sleep_begin(
            dyn_state,
            steel_state,
            steel_contacts,
            steel_dyn,
            steel_sleep,
            steel_status,
            dyn_info,
            rigid_info,
            steel_info,
            steel_model,
            steel_config,
            errno,
        )

    # Stages 1 to 3.
    if qd.static(steel_config.enable_joint_velocity_limits):
        func_steel_prescale_velocity(steel_rows, steel_dyn, steel_sleep, steel_info, steel_model, steel_config)
    func_steel_fk_id(
        global_refresh, dyn_state, steel_state, steel_dyn, steel_sleep, dyn_info, rigid_info, steel_model, steel_config
    )
    func_steel_tau(dt, dyn_state, steel_state, steel_dyn, steel_sleep, dyn_info, rigid_info, steel_model, steel_config)
    func_steel_mass_matrix(steel_state, steel_dyn, dyn_info, steel_model, steel_config)
    func_steel_unconstrained(
        dt, steel_state, steel_dyn, steel_sleep, dyn_info, rigid_info, steel_info, steel_model, steel_config
    )

    # Stage 4.
    if qd.static(steel_config.has_contact_warmstart):
        func_steel_match_contacts(steel_state, steel_contacts, steel_contact_history, steel_info, steel_config)
    if qd.static(steel_config.has_friction_anchors):
        func_steel_prepare_patches(
            steel_state, steel_contacts, steel_patch_frame, steel_patch_geometry, steel_info, steel_config
        )
        func_steel_build_patches(
            steel_state,
            steel_contacts,
            steel_patch_frame,
            steel_patch_history,
            steel_sleep,
            steel_patch_geometry,
            steel_info,
            steel_config,
        )
    func_steel_allocate_rows(
        steel_state,
        steel_contacts,
        steel_rows,
        steel_free_rows,
        steel_prop,
        steel_dyn,
        steel_sleep,
        dyn_info,
        rigid_info,
        steel_info,
        steel_model,
        steel_config,
    )
    func_steel_update_status(collider_state, steel_rows, steel_free_rows, steel_prop, steel_status, steel_config, errno)
    func_steel_build_rows(
        dyn_state,
        steel_state,
        steel_contacts,
        steel_rows,
        steel_free_rows,
        steel_dyn,
        dyn_info,
        rigid_info,
        steel_info,
        steel_model,
        steel_config,
    )
    if qd.static(steel_config.has_propagation):
        if qd.static(steel_config.articulated_contact_response == 3):
            func_steel_propagation_color(steel_contacts, steel_prop, steel_config)
        func_steel_propagation_build_rows(steel_state, steel_contacts, steel_prop, steel_dyn, steel_info, steel_config)
    # Without warm start no gather reads the friction coefficients before the regions share them, so the rings are
    # linked before the torsion rows group the regions. With warm start, the free-body and propagation rows are seeded
    # before the friction coefficients of the patch anchors are shared, the dense rows after.
    if qd.static(steel_config.has_friction_anchors and not steel_config.has_contact_warmstart):
        func_steel_link_patch_rows(
            CONTACT_PATH.DENSE, steel_rows.row_parent, steel_rows.row_mu, steel_contacts, steel_config
        )
        if qd.static(steel_config.has_free_rigid_bodies):
            func_steel_link_patch_rows(
                CONTACT_PATH.FREE, steel_free_rows.row_parent, steel_free_rows.row_mu, steel_contacts, steel_config
            )
        if qd.static(steel_config.has_propagation):
            func_steel_link_patch_rows(
                CONTACT_PATH.PROPAGATION, steel_prop.row_parent, steel_prop.row_mu, steel_contacts, steel_config
            )
    if qd.static(steel_config.has_torsion):
        func_steel_torsion_rows(
            steel_state,
            steel_contacts,
            steel_rows,
            steel_free_rows,
            steel_torsion_witnesses,
            steel_dyn,
            dyn_info,
            steel_info,
            steel_model,
            steel_config,
            errno,
        )
    func_steel_responses(steel_contacts, steel_rows, steel_dyn, steel_model, steel_config)
    if qd.static(steel_config.has_preelim):
        func_steel_preelim(
            steel_info.preelim_reg_rel[None],
            steel_info.preelim_reg_floor[None],
            steel_contacts,
            steel_rows,
            steel_model,
            steel_config,
        )
    func_steel_row_setup(
        dt,
        steel_state,
        steel_contacts,
        steel_rows,
        steel_free_rows,
        steel_dyn,
        dyn_info,
        steel_info,
        steel_model,
        steel_config,
    )
    if qd.static(steel_config.has_compliance):
        func_steel_compliance_rows(dt, steel_contacts, steel_rows, steel_free_rows, steel_info, steel_config, errno)
    if qd.static(steel_config.has_contact_warmstart):
        if qd.static(steel_config.has_free_rigid_bodies):
            func_steel_gather_warmstart(
                route=CONTACT_PATH.FREE,
                dt=dt,
                count=steel_free_rows.count,
                row_type=steel_free_rows.row_type,
                row_parent=steel_free_rows.row_parent,
                row_mu=steel_free_rows.row_mu,
                impulses=steel_free_rows.impulses,
                prev_impulses=steel_contact_history.free_impulses,
                prev_row_type=steel_contact_history.free_row_type,
                prev_row_parent=steel_contact_history.free_row_parent,
                steel_contacts=steel_contacts,
                steel_contact_history=steel_contact_history,
                steel_info=steel_info,
                steel_config=steel_config,
            )
        if qd.static(steel_config.has_propagation):
            func_steel_gather_warmstart(
                route=CONTACT_PATH.PROPAGATION,
                dt=dt,
                count=steel_prop.count,
                row_type=steel_prop.row_type,
                row_parent=steel_prop.row_parent,
                row_mu=steel_prop.row_mu,
                impulses=steel_prop.impulses,
                prev_impulses=steel_contact_history.propagation_impulses,
                prev_row_type=steel_contact_history.propagation_row_type,
                prev_row_parent=steel_contact_history.propagation_row_parent,
                steel_contacts=steel_contacts,
                steel_contact_history=steel_contact_history,
                steel_info=steel_info,
                steel_config=steel_config,
            )
        if qd.static(steel_config.has_friction_anchors):
            func_steel_link_patch_rows(
                CONTACT_PATH.DENSE, steel_rows.row_parent, steel_rows.row_mu, steel_contacts, steel_config
            )
            if qd.static(steel_config.has_free_rigid_bodies):
                func_steel_link_patch_rows(
                    CONTACT_PATH.FREE, steel_free_rows.row_parent, steel_free_rows.row_mu, steel_contacts, steel_config
                )
            if qd.static(steel_config.has_propagation):
                func_steel_link_patch_rows(
                    CONTACT_PATH.PROPAGATION, steel_prop.row_parent, steel_prop.row_mu, steel_contacts, steel_config
                )
        func_steel_gather_warmstart(
            route=CONTACT_PATH.DENSE,
            dt=dt,
            count=steel_rows.count,
            row_type=steel_rows.row_type,
            row_parent=steel_rows.row_parent,
            row_mu=steel_rows.row_mu,
            impulses=steel_rows.impulses,
            prev_impulses=steel_contact_history.dense_impulses,
            prev_row_type=steel_contact_history.dense_row_type,
            prev_row_parent=steel_contact_history.dense_row_parent,
            steel_contacts=steel_contacts,
            steel_contact_history=steel_contact_history,
            steel_info=steel_info,
            steel_config=steel_config,
        )
        # The carried patch impulses replace the matched friction impulses of their anchors last.
        if qd.static(steel_config.has_friction_anchors):
            func_steel_seed_patch_impulses(
                CONTACT_PATH.DENSE,
                dt,
                steel_rows.row_parent,
                steel_rows.row_mu,
                steel_rows.impulses,
                steel_state,
                steel_contacts,
                steel_patch_frame,
                steel_patch_history,
                steel_contact_history,
                steel_info,
                steel_config,
            )
            if qd.static(steel_config.has_free_rigid_bodies):
                func_steel_seed_patch_impulses(
                    CONTACT_PATH.FREE,
                    dt,
                    steel_free_rows.row_parent,
                    steel_free_rows.row_mu,
                    steel_free_rows.impulses,
                    steel_state,
                    steel_contacts,
                    steel_patch_frame,
                    steel_patch_history,
                    steel_contact_history,
                    steel_info,
                    steel_config,
                )
            if qd.static(steel_config.has_propagation):
                func_steel_seed_patch_impulses(
                    CONTACT_PATH.PROPAGATION,
                    dt,
                    steel_prop.row_parent,
                    steel_prop.row_mu,
                    steel_prop.impulses,
                    steel_state,
                    steel_contacts,
                    steel_patch_frame,
                    steel_patch_history,
                    steel_contact_history,
                    steel_info,
                    steel_config,
                )

    # Stages 5 and 6.
    _B = steel_dyn.v_out.shape[1]
    n_dofs = steel_dyn.v_out.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        steel_dyn.v_out[i_d, i_b] = steel_dyn.v_hat[i_d, i_b]
    if qd.static(steel_config.has_contact_warmstart):
        func_steel_warmstart_velocity(steel_rows, steel_free_rows, steel_dyn, steel_model, steel_config)
    if qd.static(steel_config.has_preelim):
        func_steel_preelim_project_velocity(steel_rows, steel_dyn, steel_model, steel_config)
    if qd.static(steel_config.has_propagation):
        func_steel_propagation_factor(
            steel_state,
            steel_free_rows,
            steel_prop,
            steel_dyn,
            dyn_info,
            rigid_info,
            steel_info,
            steel_model,
            steel_config,
        )
        func_steel_propagation_refresh(steel_prop, steel_dyn, steel_model, steel_config)
        func_steel_propagation_row_setup(dt, steel_prop, steel_dyn, steel_info, steel_model, steel_config)
    # With velocity iterations, the two solves run from one compiled kernel (see SteelSolver._steel_advance).
    if qd.static(not steel_config.has_velocity_iterations):
        if qd.static(steel_config.solve_mode == 1):
            func_steel_split_setup(
                steel_contacts, steel_rows, steel_split, steel_dyn, steel_info, steel_model, steel_config
            )
            func_steel_split_solve(
                n_iterations,
                friction_start_iteration,
                steel_contacts,
                steel_rows,
                steel_free_rows,
                steel_split,
                steel_dyn,
                steel_info,
                steel_model,
                steel_config,
            )
        else:
            func_steel_solve(
                n_iterations=n_iterations,
                friction_start_iteration=friction_start_iteration,
                rhs_dense=steel_rows.rhs,
                rhs_free=steel_free_rows.rhs,
                rhs_propagation=steel_prop.rhs,
                steel_rows=steel_rows,
                steel_free_rows=steel_free_rows,
                steel_prop=steel_prop,
                steel_dyn=steel_dyn,
                steel_info=steel_info,
                steel_model=steel_model,
                steel_config=steel_config,
                propagate_warmstart=qd.static(steel_config.has_contact_warmstart),
                regularize=regularize,
                freeze_drive_rows=False,
            )
        func_steel_step_end(
            dt,
            dyn_state,
            collider_state,
            steel_state,
            steel_contacts,
            steel_contact_history,
            steel_patch_frame,
            steel_patch_history,
            steel_rows,
            steel_free_rows,
            steel_prop,
            steel_dyn,
            steel_sleep,
            steel_status,
            dyn_info,
            rigid_info,
            steel_info,
            steel_model,
            rigid_config,
            steel_config,
            steel_info.sleep_linear_threshold[None],
            steel_info.sleep_angular_threshold[None],
            steel_info.sleep_quiet_time[None],
        )


@qd.kernel(fastcache=True)
def kernel_steel_solve(
    n_iterations: int,
    friction_start_iteration: int,
    rhs_dense: qd.Tensor,
    rhs_free: qd.Tensor,
    rhs_propagation: qd.Tensor,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
    propagate_warmstart: bool,
    regularize: bool,
    freeze_drive_rows: bool,
):
    """Run the Gauss-Seidel iterations of one solve of a step with velocity iterations (see func_steel_solve)."""
    func_steel_solve(
        n_iterations,
        friction_start_iteration,
        rhs_dense,
        rhs_free,
        rhs_propagation,
        steel_rows,
        steel_free_rows,
        steel_prop,
        steel_dyn,
        steel_info,
        steel_model,
        steel_config,
        propagate_warmstart,
        regularize,
        freeze_drive_rows,
    )


@qd.kernel(graph=True, fastcache=True)
def kernel_steel_velocity_pass(
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    rigid_info: array_class.RigidInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Keep the velocity of the position solve in v_out_snap and set up the velocity-only iterations, as one graph.

    The right-hand sides of the rows drop their position bias (see func_steel_velocity_pass_rhs) and the link twists of
    a propagation response are recomputed from the solved velocity.
    """
    # See kernel_steel_step for why the substep is read from RigidInfo.
    dt = rigid_info.substep_dt[None]
    _B = steel_dyn.v_out.shape[1]
    n_dofs = steel_dyn.v_out.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        steel_dyn.v_out_snap[i_d, i_b] = steel_dyn.v_out[i_d, i_b]
    func_steel_velocity_pass_rhs(dt, steel_rows, steel_free_rows, steel_dyn, steel_info, steel_config)
    if qd.static(steel_config.has_propagation):
        func_steel_propagation_refresh(steel_prop, steel_dyn, steel_model, steel_config)
        func_steel_propagation_velocity_pass_rhs(dt, steel_prop, steel_info, steel_config)
    if qd.static(steel_config.has_torsion):
        func_steel_torsion_velocity_pass(dt, steel_rows, steel_dyn, steel_info, steel_config)


@qd.kernel(graph=True, fastcache=True)
def kernel_steel_step_end(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_contact_history: array_class.SteelContactHistory,
    steel_patch_frame: array_class.SteelPatchFrame,
    steel_patch_history: array_class.SteelPatchHistory,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    steel_status: array_class.SteelStatus,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    rigid_config: qd.template(),
    steel_config: qd.template(),
):
    """Finish a step with velocity iterations once its velocity is refined (see func_steel_step_end), as one graph."""
    # See kernel_steel_step for why the substep is read from RigidInfo.
    dt = rigid_info.substep_dt[None]
    func_steel_step_end(
        dt,
        dyn_state,
        collider_state,
        steel_state,
        steel_contacts,
        steel_contact_history,
        steel_patch_frame,
        steel_patch_history,
        steel_rows,
        steel_free_rows,
        steel_prop,
        steel_dyn,
        steel_sleep,
        steel_status,
        dyn_info,
        rigid_info,
        steel_info,
        steel_model,
        rigid_config,
        steel_config,
        steel_info.sleep_linear_threshold[None],
        steel_info.sleep_angular_threshold[None],
        steel_info.sleep_quiet_time[None],
    )


@qd.kernel(fastcache=True)
def kernel_steel_set_material(
    idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    material: qd.Tensor,
    steel_config: qd.template(),
):
    """Write the values of one Steel material input of the given geoms or links of the given environments."""
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_, i_b_ in qd.ndrange(idx.shape[0], envs_idx.shape[0]):
        i = idx[i_]
        i_b = envs_idx[i_b_]
        material[i, i_b] = values[i_b_, i_]
