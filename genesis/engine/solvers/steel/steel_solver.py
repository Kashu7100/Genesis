from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.options.solvers import SteelOptions
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

from ..base_solver import StateChange, Subscriber, mutates
from ..rigid.abd.forward_kinematics import kernel_forward_kinematics_links_geoms, kernel_forward_velocity
from ..rigid.rigid_solver import RigidSolver, kernel_substep_collision
from .contacts import (
    kernel_steel_gather_warmstart,
    kernel_steel_match_contacts,
    kernel_steel_reset_warmstart,
    kernel_steel_snapshot_warmstart,
    kernel_steel_snapshot_warmstart_rows,
    kernel_steel_warmstart_velocity,
)
from .compliance import kernel_steel_compliance_rows
from .dynamics import (
    kernel_steel_cartesian_space,
    kernel_steel_fk_id,
    kernel_steel_forward_kinematics,
    kernel_steel_inputs,
    kernel_steel_integrate,
    kernel_steel_mass_matrix,
    kernel_steel_prescale_velocity,
    kernel_steel_publish,
    kernel_steel_publish_refined_velocity,
    kernel_steel_pull,
    kernel_steel_tau,
    kernel_steel_unconstrained,
)
from .friction_patches import (
    kernel_steel_build_patches,
    kernel_steel_finish_patch_impulses,
    kernel_steel_link_patch_rows,
    kernel_steel_prepare_patches,
    kernel_steel_reset_patches,
    kernel_steel_seed_patch_impulses,
    kernel_steel_store_patches,
)
from .model import STEEL_JOINT_TYPE, build_steel_model_plan
from .observables import kernel_steel_contact_forces, kernel_steel_reset_status, kernel_steel_update_status
from .pgs import kernel_steel_pgs
from .propagation import (
    kernel_steel_propagation_build_rows,
    kernel_steel_propagation_color,
    kernel_steel_propagation_factor,
    kernel_steel_propagation_propagate,
    kernel_steel_propagation_refresh,
    kernel_steel_propagation_row_setup,
    kernel_steel_propagation_sweep,
    kernel_steel_propagation_velocity_pass_rhs,
    kernel_steel_propagation_warmstart,
)
from .rows import (
    CONTACT_PATH,
    kernel_steel_allocate_drive_rows,
    kernel_steel_allocate_rows,
    kernel_steel_build_rows,
    kernel_steel_gather_contacts,
    kernel_steel_preelim,
    kernel_steel_preelim_project_velocity,
    kernel_steel_responses,
    kernel_steel_row_setup,
    kernel_steel_velocity_pass_rhs,
)
from .sleeping import kernel_steel_sleep_begin, kernel_steel_sleep_finish, kernel_steel_sleep_wake
from .split import kernel_steel_split_setup, kernel_steel_split_solve
from .torsion import kernel_steel_torsion_rows, kernel_steel_torsion_velocity_pass

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# Bounding radius [m] of a plane, which is unbounded.
PLANE_BOUNDING_RADIUS = 1.0e6
# Default angular damping of free bodies [1/s].
DEFAULT_ANGULAR_DAMPING = 0.05
# Position-correction gain of friction patches when 'friction_anchor_beta' is None with the matrix-free solve.
DEFAULT_FRICTION_ANCHOR_BETA = 0.2
# Diagonal regularization of the pre-eliminated bilateral block, relative to each row's diagonal, and its floor.
PREELIM_REG_REL = 1.0e-3
PREELIM_REG_FLOOR_MIN = 1.0e-7

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


class SteelRowFamily(NamedTuple):
    """A family of rows that can hold contact rows, with the snapshot of its rows the warm start of the next step reads.

    route is the CONTACT_PATH of its contacts, rows the SteelRows, SteelFreeRows or SteelPropagation holding the rows.
    """

    route: CONTACT_PATH
    rows: array_class.SteelRows | array_class.SteelFreeRows | array_class.SteelPropagation
    prev_impulses: qd.Tensor
    prev_row_type: qd.Tensor
    prev_row_parent: qd.Tensor


class SteelSolver(RigidSolver):
    """Rigid solver simulating every rigid entity with the Steel algorithm (see 'gs.options.SteelOptions').

    It keeps the entities, the collision detection and the accessors of the rigid solver and replaces its step: the
    Steel state (joint coordinates and velocities in Steel conventions) advances by one Steel step per substep, then is
    written back to the Genesis state, whose Cartesian quantities are updated for the getters, the sensors, the
    rendering and the next collision detection. A Genesis setter of the state reloads the Steel state of the
    environments it changed and discards their solver history.
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
        self._steel_last_dt = None
        self._steel_links_angular_damping = None
        self._steel_links_max_linear_velocity = None
        self._steel_links_max_angular_velocity = None
        self._steel_links_max_depenetration_velocity = None
        self._steel_geoms_restitution = None
        self._steel_dofs_velocity_limit = None
        self._steel_propagation_active = False
        self._steel_row_families: list[SteelRowFamily] = []
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
        self._steel_links_angular_damping = np.full(self.n_links_, DEFAULT_ANGULAR_DAMPING, dtype=gs.np_float)
        self._steel_links_max_linear_velocity = np.full(self.n_links_, np.inf, dtype=gs.np_float)
        self._steel_links_max_angular_velocity = np.full(self.n_links_, np.inf, dtype=gs.np_float)
        self._steel_links_max_depenetration_velocity = np.full(self.n_links_, np.inf, dtype=gs.np_float)
        self._steel_geoms_restitution = np.zeros(self.n_geoms_, dtype=gs.np_float)
        self._steel_dofs_velocity_limit = np.full(self.n_dofs_, np.inf, dtype=gs.np_float)
        self._steel_geoms_contact_stiffness = np.zeros(self.n_geoms_, dtype=gs.np_float)
        self._steel_geoms_contact_damping = np.zeros(self.n_geoms_, dtype=gs.np_float)
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
        free_links_idx = self._steel_plan.joints_link[
            self._steel_plan.arts_joint_start[:-1][self._steel_plan.arts_is_free_rigid != 0]
        ]
        self._has_free_velocity_limits = bool(
            np.isfinite(self._steel_links_max_linear_velocity[free_links_idx]).any()
            or np.isfinite(self._steel_links_max_angular_velocity[free_links_idx]).any()
        )

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
            has_preelim_mimics=(
                self._steel_n_preelim > 0
                and options.bilateral_preelimination_include_mimics
                and self._steel_plan.n_mimics > 0
            ),
            has_connects=self._steel_plan.n_connects > 0,
            has_fused_velocity_limits=self.velocity_limits_fused,
            has_free_velocity_limits=self._has_free_velocity_limits,
            has_propagation=self._steel_propagation_active,
            propagation_same_articulation_rows=options.propagation_same_articulation_rows,
            has_torsion=options.enable_torsional_friction,
            has_compliance=options.contact_compliance,
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
        history = self.steel_contact_history
        self._steel_row_families.clear()
        self._steel_row_families.append(
            SteelRowFamily(
                CONTACT_PATH.DENSE,
                self.steel_rows,
                history.dense_impulses,
                history.dense_row_type,
                history.dense_row_parent,
            )
        )
        if self._steel_plan.has_free_rigid_bodies:
            self._steel_row_families.append(
                SteelRowFamily(
                    CONTACT_PATH.FREE,
                    self.steel_free_rows,
                    history.free_impulses,
                    history.free_row_type,
                    history.free_row_parent,
                )
            )
        if self._steel_propagation_active:
            self._steel_row_families.append(
                SteelRowFamily(
                    CONTACT_PATH.PROPAGATION,
                    self.steel_prop,
                    history.propagation_impulses,
                    history.propagation_row_type,
                    history.propagation_row_parent,
                )
            )
        self.steel_sleep = self._build_sleep_state()
        self.steel_status = array_class.get_steel_status(self)
        self._refresh_steel_materials()

        self.subscribe(Subscriber(frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS}), self._on_state_change))
        self.subscribe(Subscriber(frozenset({StateChange.MODEL}), self._on_model_change))
        self._on_state_change(StateChange.GEOMETRY, None)

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

    @mutates(StateChange.MODEL)
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

    def _refresh_steel_materials(self) -> None:
        """Copy the friction coefficients of the geoms, scaled by 'contact_friction_scale', into the Steel materials."""
        friction = qd_to_numpy(self.dyn_info.geoms.friction)
        friction_ratio = qd_to_numpy(self.dyn_state.geoms.friction_ratio)
        geoms_mu = (friction[:, None] if friction.ndim == 1 else friction) * friction_ratio
        self.steel_info.geoms_mu.from_numpy((geoms_mu * self._options.contact_friction_scale).astype(gs.np_float))

    def _on_model_change(self, change: StateChange, envs_idx) -> None:
        """Refresh the Steel materials, request a mass matrix refresh and discard the warm-start impulses of every
        environment after a change of the model parameters.

        With sleeping, every island of the changed environments wakes and is vetoed at the next step, the gravity
        included: the notification names environments, not the entities whose parameters changed.
        """
        self._refresh_steel_materials()
        steel_state = self.steel_state
        mass_update_requested = qd_to_numpy(steel_state.arts_mass_update_requested, copy=True)
        mass_update_requested[...] = 1
        steel_state.arts_mass_update_requested.from_numpy(mass_update_requested)
        if self._options.contact_warmstart:
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
        """Reload the Steel state of the environments a Genesis setter changed and discard their friction anchors and
        warm-start impulses.

        With sleeping, the change is a reset of the islands of those environments: they wake and drop their row
        overflow status. A change of selected environments also vetoes their islands at the next step, the
        notification naming environments, not the entities whose state changed. A change of every environment
        restarts the sleep history instead, so it wakes the islands without vetoing them.
        """
        is_masked = envs_idx is not None
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        if self._steel_friction_anchor_beta > 0.0:
            kernel_steel_reset_patches(envs_idx, self.steel_patch_history, self.steel_config)
        if self._options.contact_warmstart:
            kernel_steel_reset_warmstart(envs_idx, self.steel_contact_history, self.steel_config)
        kernel_steel_reset_status(envs_idx, self.steel_status, self.steel_config)
        if self._steel_plan.n_arts == 0:
            return
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
            self._steel_advance(self._substep_dt)

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
    def _has_internal_rows(self) -> bool:
        """Whether drive, joint-limit, mimic or loop-closure rows can exist."""
        options = self._options
        plan = self._steel_plan
        return options.drive_mode == "rows" or options.enable_joint_limit or plan.n_mimics > 0 or plan.n_connects > 0

    @property
    def _has_velocity_limit_rows(self) -> bool:
        """Whether joint or free-body velocity-limit rows, or fused drive clamps, can exist."""
        options = self._options
        return (
            options.enable_joint_velocity_limits or self.velocity_limits_fused or self._has_free_velocity_limits
        ) and np.isfinite(options.velocity_limit_activation_fraction)

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
        if not self._options.warn_constraint_overflow:
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

    def _steel_link_patch_rows(self) -> None:
        """Chain the normal rows of every friction patch region into a ring and share its friction coefficient, in every
        row family (see friction_patches.py)."""
        for family in self._steel_row_families:
            kernel_steel_link_patch_rows(
                int(family.route), family.rows.row_parent, family.rows.row_mu, self.steel_contacts, self.steel_config
            )

    def _steel_gather_warmstart(self, family: SteelRowFamily, dt: float) -> None:
        """Seed the rows of the matched contacts of one row family from the last step (see contacts.py)."""
        rows = family.rows
        kernel_steel_gather_warmstart(
            int(family.route),
            dt,
            rows.count,
            rows.row_type,
            rows.row_parent,
            rows.row_mu,
            rows.impulses,
            family.prev_impulses,
            family.prev_row_type,
            family.prev_row_parent,
            self.steel_contacts,
            self.steel_contact_history,
            self.steel_info,
            self.steel_config,
        )

    def _launch_pgs_solve(
        self, rhs_dense, rhs_free, rhs_propagation, n_iterations, *, regularize, freeze_drive_rows, friction_start
    ):
        """Run n_iterations Gauss-Seidel sweeps in the row order of 'row_schedule'.

        'interleaved' sweeps every row family in each iteration. 'contact_then_internal' runs every iteration of the
        contact rows first, then every iteration of the internal rows and the velocity limits. 'phased' solves, in each
        iteration, the internal rows, then the contact rows, then the velocity limits. A propagation response follows
        its own schedule (see _launch_propagation_solve).
        """
        if n_iterations <= 0:
            return
        if self._steel_propagation_active:
            self._launch_propagation_solve(
                rhs_dense,
                rhs_free,
                rhs_propagation,
                n_iterations,
                regularize=regularize,
                freeze_drive_rows=freeze_drive_rows,
                friction_start=friction_start,
            )
            return
        options = self._options
        args = (self.steel_rows, self.steel_free_rows, self.steel_dyn, self.steel_info, self.steel_config)
        if options.row_schedule == "contact_then_internal":
            for row_phase in (1, 2):
                kernel_steel_pgs(
                    n_iterations,
                    0,
                    friction_start,
                    row_phase,
                    regularize,
                    freeze_drive_rows,
                    rhs_dense,
                    rhs_free,
                    *args,
                )
        elif options.row_schedule == "phased":
            for i_iter in range(n_iterations):
                if self._has_internal_rows:
                    kernel_steel_pgs(
                        1, i_iter, friction_start, 3, regularize, freeze_drive_rows, rhs_dense, rhs_free, *args
                    )
                kernel_steel_pgs(
                    1, i_iter, friction_start, 4, regularize, freeze_drive_rows, rhs_dense, rhs_free, *args
                )
                if self._has_velocity_limit_rows:
                    kernel_steel_pgs(
                        1, i_iter, friction_start, 5, regularize, freeze_drive_rows, rhs_dense, rhs_free, *args
                    )
        else:
            kernel_steel_pgs(
                n_iterations, 0, friction_start, 0, regularize, freeze_drive_rows, rhs_dense, rhs_free, *args
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Propagation articulated contact responses (see steel/propagation.py)
    # ------------------------------------------------------------------------------------------------------------------

    def _propagation_setup(self, dt: float) -> None:
        """Factor the articulations, compute the link responses and twists, then the responses and biases of the rows.

        Runs once the solve velocity v_out holds the unconstrained prediction of the step.
        """
        kernel_steel_propagation_factor(
            self.steel_state,
            self.steel_free_rows,
            self.steel_prop,
            self.steel_dyn,
            self.dyn_info,
            self.rigid_info,
            self.steel_info,
            self.steel_model,
            self.steel_config,
        )
        kernel_steel_propagation_refresh(
            self.steel_prop, self.steel_dyn, self.steel_model, self.steel_config, refresh_all=True
        )
        kernel_steel_propagation_row_setup(
            dt, self.steel_prop, self.steel_dyn, self.steel_info, self.steel_model, self.steel_config
        )

    def _launch_propagation_solve(
        self, rhs_dense, rhs_free, rhs_propagation, n_iterations, *, regularize, freeze_drive_rows, friction_start
    ):
        """Run n_iterations iterations of a propagation response.

        Every iteration solves the internal rows (drive, joint-limit and mimic rows), the dense and free-body contact
        rows, then refreshes the link twists, sweeps the propagation rows, propagates their impulses through the trees,
        and solves the velocity limits last. With 'contact_then_internal' the iterations leave out the internal rows
        and the velocity limits, which follow in iterations of their own. 'propagation_fused' recomputes every link
        twist before each sweep, the others only those an earlier row family of the iteration can have moved.
        """
        options = self._options
        has_internal_rows = self._has_internal_rows
        has_velocity_limit_rows = self._has_velocity_limit_rows
        is_contact_then_internal = options.row_schedule == "contact_then_internal"
        is_refresh_forced = (
            options.articulated_contact_response == "propagation_fused"
            or is_contact_then_internal
            or has_internal_rows
            or has_velocity_limit_rows
        )
        args = (self.steel_rows, self.steel_free_rows, self.steel_dyn, self.steel_info, self.steel_config)
        for i_iter in range(n_iterations):
            if has_internal_rows and not is_contact_then_internal:
                kernel_steel_pgs(
                    1, i_iter, friction_start, 3, regularize, freeze_drive_rows, rhs_dense, rhs_free, *args
                )
            kernel_steel_pgs(1, i_iter, friction_start, 4, regularize, freeze_drive_rows, rhs_dense, rhs_free, *args)
            kernel_steel_propagation_refresh(
                self.steel_prop, self.steel_dyn, self.steel_model, self.steel_config, is_refresh_forced
            )
            kernel_steel_propagation_sweep(
                i_iter, friction_start, rhs_propagation, self.steel_prop, self.steel_info, self.steel_config, regularize
            )
            kernel_steel_propagation_propagate(self.steel_prop, self.steel_dyn, self.steel_model, self.steel_config)
            if has_velocity_limit_rows and not is_contact_then_internal:
                kernel_steel_pgs(
                    1, i_iter, friction_start, 5, regularize, freeze_drive_rows, rhs_dense, rhs_free, *args
                )
        if is_contact_then_internal and (has_internal_rows or has_velocity_limit_rows):
            kernel_steel_pgs(
                n_iterations, 0, friction_start, 2, regularize, freeze_drive_rows, rhs_dense, rhs_free, *args
            )

    def _steel_advance(self, dt: float) -> None:
        """Advance the Steel state by one step of duration dt and write it back to the Genesis state."""
        options = self._options
        # The augmented mass matrix depends on dt through the implicit drive terms.
        is_dt_changed = self._steel_last_dt is not None and abs(self._steel_last_dt - dt) > 1.0e-8
        self._steel_last_dt = dt
        global_refresh = self._steel_step % options.update_mass_matrix_interval == 0 or is_dt_changed

        # Drive rows come first in the row order and must exist before the velocity scaling of stage 1.
        kernel_steel_allocate_drive_rows(
            self.dyn_state, self.steel_rows, self.dyn_info, self.rigid_info, self.steel_model, self.steel_config
        )

        # The contacts of the step are gathered with its inputs, before the dynamics: the sleeping islands of the step
        # are built from them.
        kernel_steel_inputs(
            self.dyn_state, self.steel_state, self.steel_dyn, self.dyn_info, self.steel_model, self.steel_config
        )
        self._steel_gather_contacts()
        if options.enable_sleeping:
            kernel_steel_sleep_begin(
                self.dyn_state,
                self.steel_state,
                self.steel_contacts,
                self.steel_dyn,
                self.steel_sleep,
                self.steel_status,
                self.dyn_info,
                self.rigid_info,
                self.steel_info,
                self.steel_model,
                self.steel_config,
                self._errno,
            )

        # Stage 1: forward kinematics, inverse dynamics with implicit drives, and mass matrix. Stage 2: factorization.
        if options.enable_joint_velocity_limits:
            kernel_steel_prescale_velocity(
                self.steel_rows, self.steel_dyn, self.steel_sleep, self.steel_info, self.steel_model, self.steel_config
            )
        kernel_steel_fk_id(
            global_refresh,
            self.dyn_state,
            self.steel_state,
            self.steel_dyn,
            self.steel_sleep,
            self.dyn_info,
            self.rigid_info,
            self.steel_model,
            self.steel_config,
        )
        kernel_steel_tau(
            dt,
            self.dyn_state,
            self.steel_state,
            self.steel_dyn,
            self.steel_sleep,
            self.dyn_info,
            self.rigid_info,
            self.steel_model,
            self.steel_config,
        )
        kernel_steel_mass_matrix(self.steel_state, self.steel_dyn, self.dyn_info, self.steel_model, self.steel_config)

        # Stage 3: unconstrained acceleration and velocity prediction.
        kernel_steel_unconstrained(
            dt,
            self.steel_state,
            self.steel_dyn,
            self.steel_sleep,
            self.dyn_info,
            self.rigid_info,
            self.steel_info,
            self.steel_model,
            self.steel_config,
        )

        # Stage 4: constraint rows, responses, diagonals and right-hand sides. Friction patches select the contacts
        # receiving friction rows, so they are built before the row allocation.
        has_patches = self._steel_friction_anchor_beta > 0.0
        if options.contact_warmstart:
            kernel_steel_match_contacts(
                self.steel_state, self.steel_contacts, self.steel_contact_history, self.steel_info, self.steel_config
            )
        if has_patches:
            kernel_steel_prepare_patches(
                self.steel_state,
                self.steel_contacts,
                self.steel_patch_frame,
                self.steel_patch_geometry,
                self.steel_info,
                self.steel_config,
            )
            kernel_steel_build_patches(
                self.steel_state,
                self.steel_contacts,
                self.steel_patch_frame,
                self.steel_patch_history,
                self.steel_sleep,
                self.steel_patch_geometry,
                self.steel_info,
                self.steel_config,
            )
        kernel_steel_allocate_rows(
            self.steel_state,
            self.steel_contacts,
            self.steel_rows,
            self.steel_free_rows,
            self.steel_prop,
            self.steel_dyn,
            self.steel_sleep,
            self.dyn_info,
            self.rigid_info,
            self.steel_info,
            self.steel_model,
            self.steel_config,
        )
        kernel_steel_update_status(
            self.collider.collider_state,
            self.steel_rows,
            self.steel_free_rows,
            self.steel_prop,
            self.steel_status,
            self.steel_config,
            self._errno,
        )
        kernel_steel_build_rows(
            self.dyn_state,
            self.steel_state,
            self.steel_contacts,
            self.steel_rows,
            self.steel_free_rows,
            self.steel_dyn,
            self.dyn_info,
            self.rigid_info,
            self.steel_info,
            self.steel_model,
            self.steel_config,
        )
        if self._steel_propagation_active:
            if options.articulated_contact_response == "propagation_colored":
                kernel_steel_propagation_color(self.steel_contacts, self.steel_prop, self.steel_config)
            kernel_steel_propagation_build_rows(
                self.steel_state,
                self.steel_contacts,
                self.steel_prop,
                self.steel_dyn,
                self.steel_info,
                self.steel_config,
            )
        if has_patches and not options.contact_warmstart:
            # Without warm start no gather reads the friction coefficients before the regions share them, so the rings
            # are linked before the torsion rows group the regions.
            self._steel_link_patch_rows()
        if options.enable_torsional_friction:
            kernel_steel_torsion_rows(
                self.steel_state,
                self.steel_contacts,
                self.steel_rows,
                self.steel_free_rows,
                self.steel_torsion_witnesses,
                self.steel_dyn,
                self.dyn_info,
                self.steel_info,
                self.steel_model,
                self.steel_config,
                self._errno,
            )
        kernel_steel_responses(
            self.steel_contacts, self.steel_rows, self.steel_dyn, self.steel_model, self.steel_config
        )
        if self._steel_n_preelim > 0:
            kernel_steel_preelim(
                PREELIM_REG_REL,
                max(options.cfm, PREELIM_REG_FLOOR_MIN),
                self.steel_contacts,
                self.steel_rows,
                self.steel_model,
                self.steel_config,
            )
        kernel_steel_row_setup(
            dt,
            self.steel_state,
            self.steel_contacts,
            self.steel_rows,
            self.steel_free_rows,
            self.steel_dyn,
            self.dyn_info,
            self.steel_info,
            self.steel_model,
            self.steel_config,
        )

        if options.contact_compliance:
            kernel_steel_compliance_rows(
                dt,
                self.steel_contacts,
                self.steel_rows,
                self.steel_free_rows,
                self.steel_info,
                self.steel_config,
                self._errno,
            )

        # The free-body and propagation rows are seeded before the friction coefficients of the patch anchors are
        # shared, the dense rows after, and the carried patch impulses replace the matched friction impulses of their
        # anchors last.
        dense_family, *other_families = self._steel_row_families
        if options.contact_warmstart:
            for family in other_families:
                self._steel_gather_warmstart(family, dt)
        if has_patches and options.contact_warmstart:
            self._steel_link_patch_rows()
        if options.contact_warmstart:
            self._steel_gather_warmstart(dense_family, dt)
        if has_patches and options.contact_warmstart and options.contact_warmstart_decay > 0.0:
            for family in self._steel_row_families:
                kernel_steel_seed_patch_impulses(
                    int(family.route),
                    dt,
                    family.rows.row_parent,
                    family.rows.row_mu,
                    family.rows.impulses,
                    self.steel_state,
                    self.steel_contacts,
                    self.steel_patch_frame,
                    self.steel_patch_history,
                    self.steel_contact_history,
                    self.steel_info,
                    self.steel_config,
                )

        # Stages 5 and 6: projected Gauss-Seidel in impulse space, starting from the unconstrained prediction plus the
        # velocity of the seeded impulses.
        self.steel_dyn.v_out.copy_from(self.steel_dyn.v_hat)
        if options.contact_warmstart:
            kernel_steel_warmstart_velocity(
                self.steel_rows, self.steel_free_rows, self.steel_dyn, self.steel_model, self.steel_config
            )
        if self._steel_n_preelim > 0:
            kernel_steel_preelim_project_velocity(self.steel_rows, self.steel_dyn, self.steel_model, self.steel_config)
        if self._steel_propagation_active:
            self._propagation_setup(dt)
            # The impulses a contact warm start seeds into the propagation rows act on the velocity before the first
            # sweep.
            if options.contact_warmstart:
                kernel_steel_propagation_warmstart(self.steel_prop, self.steel_config)
                kernel_steel_propagation_propagate(self.steel_prop, self.steel_dyn, self.steel_model, self.steel_config)
        n_friction_iterations = options.contact_friction_position_iterations
        friction_start = (
            0 if n_friction_iterations < 0 else options.iterations - min(n_friction_iterations, options.iterations)
        )
        if options.solve_mode == "split":
            kernel_steel_split_setup(
                self.steel_contacts,
                self.steel_rows,
                self.steel_split,
                self.steel_dyn,
                self.steel_info,
                self.steel_model,
                self.steel_config,
            )
            kernel_steel_split_solve(
                options.iterations,
                friction_start,
                self.steel_contacts,
                self.steel_rows,
                self.steel_free_rows,
                self.steel_split,
                self.steel_dyn,
                self.steel_info,
                self.steel_model,
                self.steel_config,
            )
        else:
            self._launch_pgs_solve(
                self.steel_rows.rhs,
                self.steel_free_rows.rhs,
                self.steel_prop.rhs,
                options.iterations,
                regularize=self._steel_contact_w < 1.0,
                freeze_drive_rows=False,
                friction_start=friction_start,
            )

        # Stage 7: convert the solved velocity to accelerations and integrate. With velocity-only iterations, positions
        # follow the position solve, then those iterations refine the velocity without the position bias.
        integrate_args = (
            self.dyn_state,
            self.steel_state,
            self.steel_dyn,
            self.steel_sleep,
            self.dyn_info,
            self.steel_info,
            self.steel_model,
            self.steel_config,
        )
        if options.velocity_iterations > 0:
            self.steel_dyn.v_out_snap.copy_from(self.steel_dyn.v_out)
            kernel_steel_velocity_pass_rhs(
                dt, self.steel_rows, self.steel_free_rows, self.steel_dyn, self.steel_info, self.steel_config
            )
            if self._steel_propagation_active:
                kernel_steel_propagation_refresh(
                    self.steel_prop, self.steel_dyn, self.steel_model, self.steel_config, refresh_all=True
                )
                kernel_steel_propagation_velocity_pass_rhs(dt, self.steel_prop, self.steel_info, self.steel_config)
            if options.enable_torsional_friction:
                kernel_steel_torsion_velocity_pass(
                    dt, self.steel_rows, self.steel_dyn, self.steel_info, self.steel_config
                )
            self._launch_pgs_solve(
                self.steel_rows.rhs_unbiased,
                self.steel_free_rows.rhs_unbiased,
                self.steel_prop.rhs_unbiased,
                options.velocity_iterations,
                regularize=False,
                freeze_drive_rows=options.velocity_drive_mode == "freeze",
                friction_start=0,
            )
            kernel_steel_integrate(dt, self.steel_dyn.v_out_snap, *integrate_args)
            kernel_steel_publish_refined_velocity(
                dt, self.steel_state, self.steel_dyn, self.steel_info, self.steel_model, self.steel_config
            )
        else:
            kernel_steel_integrate(dt, self.steel_dyn.v_out, *integrate_args)
        kernel_steel_contact_forces(
            dt,
            self.dyn_state,
            self.collider.collider_state,
            self.steel_contacts,
            self.steel_rows,
            self.steel_free_rows,
            self.steel_prop,
            self.steel_config,
        )
        kernel_steel_forward_kinematics(
            self.dyn_state,
            self.steel_state,
            self.steel_dyn,
            self.steel_sleep,
            self.dyn_info,
            self.steel_model,
            self.steel_config,
        )
        if options.enable_sleeping:
            kernel_steel_sleep_finish(
                dt,
                self.steel_state,
                self.steel_dyn,
                self.steel_sleep,
                self.steel_status,
                self.steel_model,
                self.steel_config,
                options.sleep_linear_threshold,
                options.sleep_angular_threshold,
                options.sleep_quiet_time,
            )
            self._steel_sleep_is_history_valid = True
        if has_patches:
            for family in self._steel_row_families:
                kernel_steel_finish_patch_impulses(
                    int(family.route),
                    dt,
                    family.rows.row_parent,
                    family.rows.row_mu,
                    family.rows.impulses,
                    self.steel_state,
                    self.steel_contacts,
                    self.steel_patch_frame,
                    self.dyn_info,
                    self.steel_info,
                    self.steel_config,
                )
            kernel_steel_store_patches(
                self.steel_state,
                self.steel_contacts,
                self.steel_patch_frame,
                self.steel_patch_history,
                self.steel_config,
            )
        if options.contact_warmstart:
            for family in self._steel_row_families:
                rows = family.rows
                kernel_steel_snapshot_warmstart_rows(
                    rows.count,
                    rows.row_type,
                    rows.row_parent,
                    rows.impulses,
                    family.prev_row_type,
                    family.prev_row_parent,
                    family.prev_impulses,
                    self.steel_config,
                )
            kernel_steel_snapshot_warmstart(
                dt, self.steel_state, self.steel_contacts, self.steel_contact_history, self.steel_config
            )
        kernel_steel_publish(
            dt, self.dyn_state, self.steel_state, self.dyn_info, self.rigid_info, self.steel_model, self.steel_config
        )
        kernel_steel_cartesian_space(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        self._steel_step += 1
