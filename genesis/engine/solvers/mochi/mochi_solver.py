# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import sys
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.mochi_entity import MochiEntity
from genesis.engine.states.solvers import MochiSolverState
from genesis.options.solvers import MochiOptions
from genesis.utils import array_class
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array
from genesis.utils.sdf import SDF

from ..base_solver import StateChange, Subscriber, mutates
from ..kinematic_solver import KinematicSolver
from ..rigid.abd.accessor import kernel_get_kinematic_state, kernel_set_kinematic_state
from ..rigid.abd.forward_kinematics import kernel_forward_kinematics, kernel_update_geom_aabbs, kernel_update_geoms
from ..rigid.abd.misc import (
    kernel_bit_reduction,
    kernel_init_entity_fields,
    kernel_init_geom_fields,
    kernel_init_vert_fields,
    kernel_update_geoms_render_T,
)
from .colliders import query_collider
from .contact import (
    build_geom_samples,
    kernel_broadphase_pairs,
    kernel_conservative_bounds,
    kernel_contact_eval,
    kernel_init_mochi_fields,
    kernel_pairs_to_blocks,
    kernel_set_links_pair_enabled,
    kernel_zero_assembly,
)
from .data import (
    COLLIDER_TYPE,
    FRICTION_MODEL,
    INTEGRATOR,
    LINESEARCH,
    N_HISTORY,
    MochiInfo,
    MochiState,
    MochiStaticConfig,
    get_mochi_contact_state,
    get_mochi_info,
    get_mochi_state,
)
from .integration import (
    kernel_post_stage,
    kernel_reset_history,
    kernel_step_start,
    kernel_store_stage_start_poses,
)
from .linear_solver import (
    kernel_cholesky_solve_dense,
    kernel_condense_dense,
    kernel_pcg_any_active,
    kernel_pcg_init,
    kernel_pcg_iter,
)
from .newton import (
    kernel_any_active,
    kernel_apply_increment,
    kernel_convergence_check,
    kernel_linesearch_begin,
    kernel_linesearch_decide,
    kernel_reset_newton,
    kernel_residual_norms,
    kernel_store_initial_norms,
)
from .rigid_assembly import kernel_assemble_rigid, kernel_update_conv_weights

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# Conjugate gradient iterations between two host-side checks of the per-environment convergence flags.
PCG_CHECK_PERIOD = 8

_COLLIDER_TYPE_BY_NAME = {
    "none": COLLIDER_TYPE.NONE,
    "plane": COLLIDER_TYPE.PLANE,
    "sphere": COLLIDER_TYPE.SPHERE,
    "box": COLLIDER_TYPE.BOX,
    "sdf": COLLIDER_TYPE.GRID,
}
_AUTO_COLLIDER_TYPE_BY_GEOM_TYPE = {
    gs.GEOM_TYPE.PLANE: COLLIDER_TYPE.PLANE,
    gs.GEOM_TYPE.SPHERE: COLLIDER_TYPE.SPHERE,
    gs.GEOM_TYPE.BOX: COLLIDER_TYPE.BOX,
}


class MochiSolver(KinematicSolver):
    """
    Fully-implicit rigid body solver with smooth penalty contact.

    Every substep solves one nonlinear system over the generalized coordinates of all free bodies, whose residual is
    the gradient of an incremental potential (inertia, gravity, damping and the contact penalty with its friction and
    damping terms) and whose Hessian is assembled from per-link and per-contact-pair blocks. The solve is a damped
    Newton iteration with a line search; contact is re-detected at every iterate.
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: MochiOptions) -> None:
        super().__init__(scene, sim, options)
        self._options: MochiOptions = options
        self.sdf: SDF | None = None
        self._errno = None
        self._is_contacts_recorded = False
        self._skip_next_external_sync = False
        self._external_state_subscriber = Subscriber(frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS}))

    # ------------------------------------------------------------------------------------
    # ----------------------------------- add_entity -------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(self, idx, material, morph, surface, visualize_contact=False, name=None) -> MochiEntity:
        if isinstance(morph, (tuple, list)):
            gs.raise_exception("Heterogeneous morphs are not supported by the MochiSolver.")
        if isinstance(morph, (gs.morphs.Terrain, gs.morphs.USD, gs.morphs.Drone)):
            gs.raise_exception(f"Morph {type(morph).__name__} is not supported by the MochiSolver.")

        morph._enable_mujoco_compatibility = self._enable_mujoco_compatibility

        entity = MochiEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            idx_in_solver=self.n_entities,
            link_start=self.n_links,
            joint_start=self.n_joints,
            q_start=self.n_qs,
            dof_start=self.n_dofs,
            geom_start=self.n_geoms,
            cell_start=self.n_cells,
            vert_start=self.n_verts,
            free_verts_state_start=self.n_free_verts,
            fixed_verts_state_start=self.n_fixed_verts,
            face_start=self.n_faces,
            edge_start=self.n_edges,
            vgeom_start=self.n_vgeoms,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            custom_vvert_start=self.n_custom_vverts,
            custom_vface_start=self.n_custom_vfaces,
            visualize_contact=visualize_contact,
            name=name,
        )
        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------ build -----------------------------------------
    # ------------------------------------------------------------------------------------

    def build(self):
        self._n_geoms = self.n_geoms
        self._n_cells = self.n_cells
        self._n_verts = self.n_verts
        self._n_free_verts = self.n_free_verts
        self._n_fixed_verts = self.n_fixed_verts
        self._n_faces = self.n_faces
        self._n_edges = self.n_edges
        self._geoms = self.geoms

        self.n_geoms_ = max(1, self.n_geoms)
        self.n_cells_ = max(1, self.n_cells)
        self.n_verts_ = max(1, self.n_verts)
        self.n_faces_ = max(1, self.n_faces)
        self.n_edges_ = max(1, self.n_edges)
        self.n_free_verts_ = max(1, self.n_free_verts)
        self.n_fixed_verts_ = max(1, self.n_fixed_verts)

        super().build()

        if not self.is_active:
            return

        if gs.qd_float == qd.f32 and any(entity.material.penalty_coefficient >= 1e8 for entity in self._entities):
            gs.logger.warning(
                "MochiSolver runs in single precision with a contact stiffness of 1e8 Pa/m or more: the Newton system "
                "is ill-conditioned and contact accuracy degrades. Consider `gs.init(precision='64')`."
            )

        self._init_vert_fields()
        self._init_geom_fields()
        kernel_update_geoms(
            self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, True
        )
        self._init_mochi()
        self._init_sdf()
        self.subscribe(self._external_state_subscriber)
        kernel_reset_history(
            self._scene._envs_idx, self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.rigid_config
        )
        self._forward_kinematics()

    def _create_data_manager(self):
        super()._create_data_manager()
        # The kinematic data manager allocates the collision-related leaves as scalar dummies; the contact model
        # needs the real geoms and verts.
        data_manager = self.data_manager
        data_manager.dyn_info = dataclasses.replace(
            data_manager.dyn_info,
            geoms=array_class.get_geoms_info(self),
            verts=array_class.get_verts_info(self),
            faces=array_class.get_faces_info(self),
            edges=array_class.get_edges_info(self),
        )
        data_manager.dyn_state = dataclasses.replace(
            data_manager.dyn_state,
            geoms=array_class.get_geoms_state(self),
            free_verts=array_class.get_free_verts_state(self),
            fixed_verts=array_class.get_fixed_verts_state(self),
        )
        self.dyn_info = data_manager.dyn_info
        self.dyn_state = data_manager.dyn_state
        self.geoms_init_AABB = array_class.V_VEC(3, dtype=gs.qd_float, shape=(self.n_geoms_, 8))
        self._errno = data_manager.errno

    def _init_vert_fields(self):
        if self.n_verts > 0:
            geoms = self.geoms
            kernel_init_vert_fields(
                np.concatenate([np.full(geom.n_verts, geom.idx) for geom in geoms], dtype=gs.np_int),
                np.concatenate(
                    [np.arange(geom.verts_state_start, geom.verts_state_start + geom.n_verts) for geom in geoms],
                    dtype=gs.np_int,
                ),
                np.concatenate([geom.init_verts for geom in geoms], dtype=gs.np_float),
                np.concatenate([geom.init_faces + geom.vert_start for geom in geoms], dtype=gs.np_int),
                np.concatenate([geom.init_edges + geom.vert_start for geom in geoms], dtype=gs.np_int),
                np.concatenate([geom.init_normals for geom in geoms], dtype=gs.np_float),
                np.concatenate([geom.init_center_pos for geom in geoms], dtype=gs.np_float),
                np.concatenate(
                    [np.full(geom.n_verts, geom.is_fixed and not geom.entity._batch_fixed_verts) for geom in geoms],
                    dtype=gs.np_bool,
                ),
                self.dyn_info,
                self.rigid_config,
            )

    def _init_geom_fields(self):
        self._geoms_render_T = np.empty((self.n_geoms_, self._B, 4, 4), dtype=np.float32)
        if self.n_geoms == 0:
            return
        geoms = self.geoms
        geoms_center = []
        for geom in geoms:
            tmesh = geom.mesh.trimesh
            geoms_center.append(tmesh.center_mass if tmesh.is_watertight else np.mean(tmesh.vertices, axis=0))
        kernel_init_geom_fields(
            np.array([geom.link.idx for geom in geoms], dtype=gs.np_int),
            np.array([geom.vert_start for geom in geoms], dtype=gs.np_int),
            np.array([geom.face_start for geom in geoms], dtype=gs.np_int),
            np.array([geom.edge_start for geom in geoms], dtype=gs.np_int),
            np.array([geom.verts_state_start for geom in geoms], dtype=gs.np_int),
            np.array([geom.vert_end for geom in geoms], dtype=gs.np_int),
            np.array([geom.face_end for geom in geoms], dtype=gs.np_int),
            np.array([geom.edge_end for geom in geoms], dtype=gs.np_int),
            np.array([geom.verts_state_end for geom in geoms], dtype=gs.np_int),
            np.array([geom.init_pos for geom in geoms], dtype=gs.np_float),
            np.array(geoms_center, dtype=gs.np_float),
            np.array([geom.init_quat for geom in geoms], dtype=gs.np_float),
            np.array([geom.type for geom in geoms], dtype=gs.np_int),
            np.array([geom.friction for geom in geoms], dtype=gs.np_float),
            np.array([geom.friction_torsional for geom in geoms], dtype=gs.np_float),
            np.array([geom.friction_rolling for geom in geoms], dtype=gs.np_float),
            np.array([geom.sol_params for geom in geoms], dtype=gs.np_float),
            np.array([geom.data for geom in geoms], dtype=gs.np_float),
            np.array([geom.is_convex for geom in geoms], dtype=gs.np_bool),
            np.array([geom.needs_coup for geom in geoms], dtype=gs.np_int),
            np.array([geom.contype for geom in geoms], dtype=np.int32),
            np.array([geom.conaffinity for geom in geoms], dtype=np.int32),
            np.array([geom.coup_softness for geom in geoms], dtype=gs.np_float),
            np.array([geom.coup_friction for geom in geoms], dtype=gs.np_float),
            np.array([geom.coup_restitution for geom in geoms], dtype=gs.np_float),
            np.array([geom.is_fixed for geom in geoms], dtype=gs.np_bool),
            np.array([geom.metadata.get("decomposed", False) for geom in geoms], dtype=gs.np_bool),
            np.zeros(self.n_geoms, dtype=gs.np_bool),
            self.geoms_init_AABB,
            self.dyn_state,
            self.dyn_info,
            self.rigid_config,
        )

    def _init_entity_fields(self):
        if not self._entities:
            return
        entities = self._entities
        kernel_init_entity_fields(
            np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
            np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
            np.array([entity.link_start for entity in entities], dtype=gs.np_int),
            np.array([entity.link_end for entity in entities], dtype=gs.np_int),
            np.array([entity.geom_start for entity in entities], dtype=gs.np_int),
            np.array([entity.geom_end for entity in entities], dtype=gs.np_int),
            np.zeros(len(entities), dtype=gs.np_float),
            np.array([entity.is_local_collision_mask for entity in entities], dtype=gs.np_bool),
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )

    def _resolve_collider_type(self, geom):
        collider_type = geom.entity.material.collider_type
        if collider_type != "auto":
            return _COLLIDER_TYPE_BY_NAME[collider_type]
        return _AUTO_COLLIDER_TYPE_BY_GEOM_TYPE.get(geom.type, COLLIDER_TYPE.GRID)

    def _init_mochi(self):
        options = self._options
        links = self.links
        geoms = self.geoms

        # Every link is either a fixed root or a free root: articulated entities are the next stage of the port.
        for entity in self._entities:
            if entity.n_equalities > 0:
                gs.raise_exception("Equality constraints are not supported by the MochiSolver.")
        for link in links:
            is_free_root = link.parent_idx == -1 and link.n_joints == 1 and link.joints[0].type == gs.JOINT_TYPE.FREE
            if not (link.is_fixed or is_free_root):
                gs.raise_exception(
                    f"Link '{link.name}' is articulated. The MochiSolver only supports free and fixed rigid bodies."
                )

        self._layers = sorted({entity.material.contact_layer for entity in self._entities})
        n_layers = len(self._layers)
        self._layers_pair_enabled = np.ones((n_layers, n_layers), dtype=bool)
        self._entities_pair_enabled = np.ones((self.n_entities, self.n_entities), dtype=bool)

        geoms_collider_type = np.array([self._resolve_collider_type(geom) for geom in geoms], dtype=gs.np_int)
        self._has_grid_colliders = bool((geoms_collider_type == COLLIDER_TYPE.GRID).any())
        geoms_contact_params = np.array(
            [
                (
                    geom.entity.material.penalty_coefficient,
                    geom.entity.material.penalty_smoothing_half_distance,
                    geom.entity.material.penalty_threshold,
                    geom.friction,
                    geom.entity.material.friction_falloff_vel,
                    geom.entity.material.viscous_friction,
                    geom.entity.material.normal_viscous_damping,
                    geom.entity.material.max_alignment_normals,
                )
                for geom in geoms
            ],
            dtype=gs.np_float,
        ).reshape((-1, 8))

        # Contact samples of every collision geom that is not an unbounded plane, gathered per link.
        samples_pos, samples_normal, samples_weight, samples_link_idx, samples_geom_idx = [], [], [], [], []
        links_sample_start = np.zeros(self.n_links, dtype=gs.np_int)
        links_sample_end = np.zeros(self.n_links, dtype=gs.np_int)
        links_samples_aabb_min = np.zeros((self.n_links, 3), dtype=gs.np_float)
        links_samples_aabb_max = np.zeros((self.n_links, 3), dtype=gs.np_float)
        n_samples = 0
        for link in links:
            links_sample_start[link.idx] = n_samples
            for geom in link.geoms:
                if geom.type == gs.GEOM_TYPE.PLANE or not (geom.contype or geom.conaffinity):
                    continue
                pos, normal, weight = build_geom_samples(geom, options.boundary_element_type)
                samples_pos.append(pos)
                samples_normal.append(normal)
                samples_weight.append(weight)
                samples_link_idx.append(np.full(len(weight), link.idx, dtype=gs.np_int))
                samples_geom_idx.append(np.full(len(weight), geom.idx, dtype=gs.np_int))
                n_samples += len(weight)
            links_sample_end[link.idx] = n_samples
        # Bounding boxes of the sample clouds, per link.
        if n_samples > 0:
            samples_pos = np.concatenate(samples_pos).astype(gs.np_float)
            samples_normal = np.concatenate(samples_normal).astype(gs.np_float)
            samples_weight = np.concatenate(samples_weight).astype(gs.np_float)
            samples_link_idx = np.concatenate(samples_link_idx)
            samples_geom_idx = np.concatenate(samples_geom_idx)
            for link in links:
                start, end = links_sample_start[link.idx], links_sample_end[link.idx]
                if end > start:
                    links_samples_aabb_min[link.idx] = samples_pos[start:end].min(axis=0)
                    links_samples_aabb_max[link.idx] = samples_pos[start:end].max(axis=0)
        else:
            samples_pos = np.zeros((0, 3), dtype=gs.np_float)
            samples_normal = np.zeros((0, 3), dtype=gs.np_float)
            samples_weight = np.zeros((0,), dtype=gs.np_float)
            samples_link_idx = np.zeros((0,), dtype=gs.np_int)
            samples_geom_idx = np.zeros((0,), dtype=gs.np_int)
        self._n_samples = n_samples
        self.n_samples_ = max(1, n_samples)
        self._max_samples_per_link = int(max(1, (links_sample_end - links_sample_start).max(initial=0)))

        n_collider_geoms = int((geoms_collider_type != COLLIDER_TYPE.NONE).sum())
        n_links_with_samples = int((links_sample_end > links_sample_start).sum())
        self._max_pairs = options.max_contact_pairs_per_env
        if self._max_pairs is None:
            self._max_pairs = max(1, n_links_with_samples * n_collider_geoms)
        self._max_hits = max(1, 2 * n_samples) if options.record_contacts else 1

        use_dense_direct = options.linear_solver == "ldlt" or (
            options.linear_solver == "auto" and self.n_dofs <= options.dense_solver_max_dofs
        )
        self._n_pcg_iterations = options.n_pcg_iterations
        if self._n_pcg_iterations is None:
            self._n_pcg_iterations = min(max(1, self.n_dofs), 1000)

        self.mochi_config = MochiStaticConfig(
            backend=gs.backend,
            para_level=self.sim._para_level,
            integrator=INTEGRATOR.BDF2 if options.integrator == "bdf2" else INTEGRATOR.BACKWARD_EULER,
            use_newton_euler_inertia=options.use_newton_euler_inertia,
            friction_model=FRICTION_MODEL.CINF if options.friction_model == "cinf" else FRICTION_MODEL.C1,
            linesearch_type={"none": LINESEARCH.NONE, "residual_norm": LINESEARCH.RESIDUAL_NORM}.get(
                options.linesearch_type, LINESEARCH.ARMIJO
            ),
            use_fitted_friction_hessian=options.use_fitted_friction_hessian,
            friction_with_collider_normal=options.friction_with_collider_normal,
            fade_friction=options.fade_friction,
            implicit_normal_force_for_dissipation=options.implicit_normal_force_for_dissipation,
            use_dense_direct=use_dense_direct,
            has_grid_colliders=self._has_grid_colliders,
            record_contacts=options.record_contacts,
            batch_links_info=self._options.batch_links_info,
        )
        self.mochi_info = get_mochi_info(self)
        self.mochi_state = get_mochi_state(self, self._max_pairs, use_dense_direct)
        self.contact_state = get_mochi_contact_state(self, self._max_pairs, self._max_hits)

        kernel_init_mochi_fields(
            np.array([not link.is_fixed for link in links], dtype=gs.np_bool),
            np.array([link.entity.material.has_gravity for link in links], dtype=gs.np_bool),
            np.array([link.inertial_mass for link in links], dtype=gs.np_float),
            np.array([link.inertial_i for link in links], dtype=gs.np_float).reshape((-1, 3, 3)),
            np.zeros(self.n_links, dtype=gs.np_float),
            np.array([self._layers.index(link.entity.material.contact_layer) for link in links], dtype=gs.np_int),
            links_sample_start,
            links_sample_end,
            links_samples_aabb_min,
            links_samples_aabb_max,
            geoms_collider_type,
            geoms_contact_params,
            samples_pos,
            samples_normal,
            samples_weight,
            samples_link_idx,
            samples_geom_idx,
            self._compute_links_pair_enabled(),
            np.tile(np.asarray(self._init_gravity, dtype=gs.np_float), (self._B, 1)),
            self.mochi_info,
            self.rigid_config,
        )

    def _init_sdf(self):
        if not self._has_grid_colliders:
            self.sdf = SDF(self)
            return
        for geom in self.geoms:
            geom._preprocess()
        self.sdf = SDF(self)
        self.sdf.activate()

    def _compute_links_pair_enabled(self):
        """Link pair filter: layers, entity pairs, and never a link against itself or another link of its entity."""
        links = self.links
        links_layer = np.array([self._layers.index(link.entity.material.contact_layer) for link in links])
        links_entity = np.array([link._entity_idx_in_solver for link in links])
        enabled = self._layers_pair_enabled[links_layer[:, None], links_layer[None, :]]
        enabled &= self._entities_pair_enabled[links_entity[:, None], links_entity[None, :]]
        enabled &= links_entity[:, None] != links_entity[None, :]
        return enabled

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def _forward_kinematics(self):
        kernel_forward_kinematics(
            self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )
        kernel_update_geoms(
            self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, False
        )
        kernel_update_geom_aabbs(self.geoms_init_AABB, self.dyn_state, self.rigid_config)

    def _sync_external_state(self):
        # State set from outside the solver invalidates the multistep history and the rotation-derivative correction.
        if self._external_state_subscriber.pending and not self._skip_next_external_sync:
            kernel_reset_history(
                self._scene._envs_idx,
                self.dyn_state,
                self.dyn_info,
                self.mochi_info,
                self.mochi_state,
                self.rigid_config,
            )
        self._external_state_subscriber.clear()
        self._skip_next_external_sync = False

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        self._sync_external_state()

        kernel_step_start(
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.mochi_info,
            self.mochi_state,
            self.rigid_config,
            self.mochi_config,
        )
        self._forward_kinematics()
        kernel_store_stage_start_poses(self.dyn_state, self.mochi_state, self.rigid_config)
        kernel_reset_newton(self.mochi_state, self.rigid_config)
        kernel_conservative_bounds(
            self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.contact_state, self.rigid_config
        )
        kernel_broadphase_pairs(
            self.dyn_state,
            self.dyn_info,
            self.mochi_info,
            self.mochi_state,
            self.contact_state,
            self.rigid_config,
            self._errno,
        )
        self._newton_solve()
        kernel_post_stage(
            self.dyn_state, self.dyn_info, self.rigid_info, self.mochi_info, self.mochi_state, self.rigid_config
        )
        self._forward_kinematics()
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True
        self._is_contacts_recorded = False

    def _assemble(self, *, assem_res, assem_dres, skip_ls_done, record=False):
        assem_obj = self.mochi_config.linesearch_type == LINESEARCH.ARMIJO
        kernel_zero_assembly(
            self.dyn_state,
            self.mochi_state,
            self.contact_state,
            self.rigid_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
            record,
        )
        kernel_contact_eval(
            self.dyn_state,
            self.dyn_info,
            self.sdf._sdf_info,
            self.mochi_info,
            self.mochi_state,
            self.contact_state,
            self.rigid_config,
            self.mochi_config,
            self._max_samples_per_link,
            skip_ls_done,
            record,
            self._errno,
        )
        kernel_pairs_to_blocks(
            self.dyn_state,
            self.dyn_info,
            self.mochi_info,
            self.mochi_state,
            self.contact_state,
            self.rigid_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
        )
        kernel_assemble_rigid(
            self.dyn_state,
            self.dyn_info,
            self.mochi_info,
            self.mochi_state,
            self.rigid_config,
            self.mochi_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
        )
        if assem_res:
            kernel_residual_norms(self.mochi_state, self.rigid_config, skip_ls_done)

    def _linear_solve(self):
        if self.mochi_config.use_dense_direct:
            kernel_condense_dense(
                self.dyn_info, self.mochi_info, self.mochi_state, self.contact_state, self.rigid_config
            )
            kernel_cholesky_solve_dense(self.mochi_info, self.mochi_state, self.rigid_config)
        else:
            kernel_pcg_init(self.dyn_info, self.mochi_info, self.mochi_state, self.rigid_config)
            for i_iter in range(self._n_pcg_iterations):
                kernel_pcg_iter(self.dyn_info, self.mochi_info, self.mochi_state, self.contact_state, self.rigid_config)
                if (i_iter + 1) % PCG_CHECK_PERIOD == 0 and kernel_pcg_any_active(self.mochi_state) == 0:
                    break

    def _newton_solve(self):
        options = self._options
        kernel_update_conv_weights(self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.rigid_config)
        self._assemble(assem_res=True, assem_dres=True, skip_ls_done=False)
        kernel_store_initial_norms(self.rigid_info, self.mochi_state, self.rigid_config)
        kernel_convergence_check(self.mochi_info, self.mochi_state, self.rigid_config, False, self._errno)

        n_linesearch = options.n_linesearch_iterations
        if self.mochi_config.linesearch_type == LINESEARCH.NONE:
            n_linesearch = 0
        for i_iter in range(options.n_newton_iterations):
            if i_iter > 0:
                self._assemble(assem_res=True, assem_dres=True, skip_ls_done=False)
                kernel_convergence_check(self.mochi_info, self.mochi_state, self.rigid_config, False, self._errno)
            if kernel_any_active(self.mochi_state) == 0:
                break
            self._linear_solve()
            kernel_linesearch_begin(self.rigid_info, self.mochi_state, self.rigid_config)
            for i_ls in range(max(1, n_linesearch)):
                kernel_apply_increment(
                    self.dyn_info, self.rigid_info, self.mochi_info, self.mochi_state, self.rigid_config
                )
                self._forward_kinematics()
                self._assemble(assem_res=True, assem_dres=False, skip_ls_done=True)
                kernel_linesearch_decide(
                    self.rigid_info,
                    self.mochi_info,
                    self.mochi_state,
                    self.rigid_config,
                    self.mochi_config,
                    i_ls == max(1, n_linesearch) - 1,
                )
            kernel_convergence_check(self.mochi_info, self.mochi_state, self.rigid_config, True, self._errno)

    def substep_post_coupling(self, f):
        pass

    def check_errno(self):
        if gs.use_zerocopy or sys.platform == "darwin":
            errno = np.bitwise_or.reduce(qd_to_numpy(self._errno))
        else:
            errno = kernel_bit_reduction(self._errno)
        if errno & array_class.ErrorCode.OVERFLOW_MOCHI_CONTACT_PAIRS:
            gs.raise_exception(
                f"Exceeding max number of contact pairs ({self._max_pairs}). Please increase the value of "
                "MochiSolver's option 'max_contact_pairs_per_env'."
            )
        if errno & array_class.ErrorCode.OVERFLOW_MOCHI_CONTACTS:
            gs.raise_exception(f"Exceeding max number of recorded contact points ({self._max_hits}).")
        if errno & array_class.ErrorCode.MOCHI_DIVERGED:
            gs.raise_exception(
                "MochiSolver diverged: the residual of the implicit solve blew up. Reduce the time step or the "
                "contact stiffness, or increase 'n_newton_iterations'."
            )

    # ------------------------------------------------------------------------------------
    # ------------------------------------ render ----------------------------------------
    # ------------------------------------------------------------------------------------

    def update_geoms_render_T(self):
        kernel_update_geoms_render_T(self._geoms_render_T, self.dyn_state, self.rigid_info, self.rigid_config)

    # ------------------------------------------------------------------------------------
    # -------------------------------- state get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f=None):
        if not self.is_active:
            return None
        s_global = self.sim.cur_step_global
        if s_global in self._queried_states:
            return self._queried_states[s_global][0]
        state = MochiSolverState(self._scene, s_global)
        kernel_get_kinematic_state(
            state.i_pos_shift,
            state.qpos,
            state.dofs_vel,
            state.links_pos,
            state.links_quat,
            self.dyn_state,
            self.rigid_info,
            self.rigid_config,
        )
        state.qpos_prev = qd_to_torch(self.mochi_state.qpos_prev, copy=True).permute((2, 0, 1)).contiguous()
        state.dofs_vel_prev = qd_to_torch(self.mochi_state.dofs_vel_prev, copy=True).permute((2, 0, 1)).contiguous()
        state.links_vsym = qd_to_torch(self.mochi_state.links_vsym, copy=True).permute((1, 0, 2, 3)).contiguous()
        state.links_vsym_prev = (
            qd_to_torch(self.mochi_state.links_vsym_prev, copy=True).permute((2, 0, 1, 3, 4)).contiguous()
        )
        state.n_hist = qd_to_torch(self.mochi_state.n_hist, copy=True).contiguous()
        self._queried_states.append(state)
        return state

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None, *, partial: bool = False) -> None:
        if not self.is_active:
            return
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        kernel_set_kinematic_state(
            envs_idx,
            state.i_pos_shift,
            state.qpos,
            state.dofs_vel,
            state.links_pos,
            state.links_quat,
            self.dyn_state,
            self.rigid_info,
            self.rigid_config,
        )
        _kernel_set_history(
            envs_idx,
            state.qpos_prev,
            state.dofs_vel_prev,
            state.links_vsym,
            state.links_vsym_prev,
            state.n_hist,
            self.mochi_state,
            self.rigid_config,
        )
        self._forward_kinematics()
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True
        # The restored history is self-consistent; the notification this setter emits must not reset it.
        self._skip_next_external_sync = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- contact API ------------------------------------
    # ------------------------------------------------------------------------------------

    def _record_contacts(self):
        if self._is_contacts_recorded:
            return
        if not self._options.record_contacts:
            gs.raise_exception("Contact readback requires MochiSolver option 'record_contacts=True'.")
        _kernel_activate_all_envs(self.mochi_state, self.rigid_config)
        self._assemble(assem_res=False, assem_dres=False, skip_ls_done=False, record=True)
        self._is_contacts_recorded = True

    def get_contacts(self, as_tensor: bool = True, to_torch: bool = True, is_padded: bool = False):
        """
        Contact points of the current state, as a dict of arrays laid out (n_envs, n_contacts, ...) (see
        `MochiEntity.get_contacts`). Without padding, the contact axis is trimmed to the largest per-environment
        count and shorter environments carry -1 geom indices in their unused slots.
        """
        self._record_contacts()
        contact_state = self.contact_state
        n_contacts = qd_to_torch(contact_state.n_hits_total, copy=True).clamp(max=self._max_hits)
        contact_data = {
            "geom_a": qd_to_torch(contact_state.hit_geom_a, transpose=True, copy=True),
            "geom_b": qd_to_torch(contact_state.hit_geom_b, transpose=True, copy=True),
            "link_a": qd_to_torch(contact_state.hit_link_a, transpose=True, copy=True),
            "link_b": qd_to_torch(contact_state.hit_link_b, transpose=True, copy=True),
            "position": qd_to_torch(contact_state.hit_pos, transpose=True, copy=True),
            "normal": qd_to_torch(contact_state.hit_normal, transpose=True, copy=True),
            "distance": qd_to_torch(contact_state.hit_distance, transpose=True, copy=True),
            "force_a": qd_to_torch(contact_state.hit_force, transpose=True, copy=True),
            "weight": qd_to_torch(contact_state.hit_weight, transpose=True, copy=True),
        }
        contact_data["force_b"] = -contact_data["force_a"]
        slots = torch.arange(self._max_hits, device=n_contacts.device)
        is_valid = slots[None, :] < n_contacts[:, None]
        for key in ("geom_a", "geom_b", "link_a", "link_b"):
            contact_data[key] = torch.where(is_valid, contact_data[key], -1)
        if is_padded:
            contact_data["n_contacts"] = n_contacts if self.n_envs > 0 else n_contacts[0]
        else:
            n_max = int(n_contacts.max())
            contact_data = {key: value[:, :n_max] for key, value in contact_data.items()}
        if self.n_envs == 0:
            contact_data = {key: value[0] for key, value in contact_data.items()}
        if not to_torch:
            contact_data = {key: tensor_to_array(value) for key, value in contact_data.items()}
        return contact_data

    def get_collider_distances(self, geom_idx, points_world, envs_idx=None):
        """
        Signed distance and distance gradient of world points to a collider geom, in the world frame, as seen by the
        contact model (a grid collider only answers inside its grid; other points are flagged invalid).

        Returns
        -------
        (distances, gradients, is_valid) : tuple of arrays of shape (n_points,), (n_points, 3), (n_points,)
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        points = torch.as_tensor(points_world, dtype=gs.tc_float, device=gs.device).reshape((-1, 3)).contiguous()
        distances = torch.zeros((points.shape[0],), dtype=gs.tc_float, device=gs.device)
        gradients = torch.zeros((points.shape[0], 3), dtype=gs.tc_float, device=gs.device)
        is_valid = torch.zeros((points.shape[0],), dtype=torch.bool, device=gs.device)
        _kernel_get_collider_distances(
            int(geom_idx),
            int(envs_idx[0]),
            points,
            distances,
            gradients,
            is_valid,
            self.dyn_state,
            self.dyn_info,
            self.sdf._sdf_info,
            self.mochi_info,
            self.mochi_config,
        )
        return distances, gradients, is_valid

    def get_convergence_info(self):
        """Per-environment outcome of the last solve: number of Newton iterations, status (0 running, 1 converged,
        2 stopped at the iteration budget, 3 diverged), and the plain residual norm before and after the solve."""
        return {
            "n_iter": qd_to_torch(self.mochi_state.n_iter, copy=True),
            "status": qd_to_torch(self.mochi_state.status, copy=True),
            "res_norm0": qd_to_torch(self.mochi_state.res_norm0, copy=True),
            "res_norm": qd_to_torch(self.mochi_state.res_norm_sq, copy=True).sqrt(),
        }

    def set_links_contact_params(self, links_idx, **params):
        links_idx = tensor_to_array(links_idx).reshape((-1,))
        geoms_idx = np.concatenate(
            [np.arange(self.links[i_l].geom_start, self.links[i_l].geom_end) for i_l in links_idx], dtype=gs.np_int
        )
        for name, value in params.items():
            if value is None:
                continue
            field = self.mochi_info.geoms.__getattribute__(name)
            values = qd_to_numpy(field)
            values[geoms_idx] = value
            field.from_numpy(values)

    def set_links_has_gravity(self, links_idx, has_gravity):
        links_idx = tensor_to_array(links_idx).reshape((-1,))
        values = qd_to_numpy(self.mochi_info.links.has_gravity)
        values[links_idx] = has_gravity
        self.mochi_info.links.has_gravity.from_numpy(values)

    def enable_layer_contact(self, layer_a: str, layer_b: str, is_enabled: bool = True):
        """Enable or disable contact between two contact layers (symmetric)."""
        i_a, i_b = self._layers.index(layer_a), self._layers.index(layer_b)
        self._layers_pair_enabled[i_a, i_b] = self._layers_pair_enabled[i_b, i_a] = is_enabled
        kernel_set_links_pair_enabled(self._compute_links_pair_enabled(), self.mochi_info, self.rigid_config)

    def enable_entity_contact(self, entity_a, entity_b, is_enabled: bool = True):
        """Enable or disable contact between two entities (symmetric)."""
        i_a, i_b = entity_a._idx_in_solver, entity_b._idx_in_solver
        self._entities_pair_enabled[i_a, i_b] = self._entities_pair_enabled[i_b, i_a] = is_enabled
        kernel_set_links_pair_enabled(self._compute_links_pair_enabled(), self.mochi_info, self.rigid_config)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_geoms(self):
        if self.is_built:
            return self._n_geoms
        return len(self.geoms)

    @property
    def n_cells(self):
        if self.is_built:
            return self._n_cells
        return sum(entity.n_cells for entity in self._entities)

    @property
    def n_verts(self):
        if self.is_built:
            return self._n_verts
        return sum(entity.n_verts for entity in self._entities)

    @property
    def n_free_verts(self):
        if self.is_built:
            return self._n_free_verts
        return sum(link.n_verts if not link.is_fixed or link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_fixed_verts(self):
        if self.is_built:
            return self._n_fixed_verts
        return sum(link.n_verts if link.is_fixed and not link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_faces(self):
        if self.is_built:
            return self._n_faces
        return sum(entity.n_faces for entity in self._entities)

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        return sum(entity.n_edges for entity in self._entities)

    @property
    def n_equalities(self):
        return 0

    @property
    def n_samples(self):
        return self._n_samples


@qd.kernel
def _kernel_activate_all_envs(mochi_state: MochiState, rigid_config: qd.template()):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(mochi_state.is_active.shape[0]):
        mochi_state.is_active[i_b] = True
        mochi_state.ls_is_done[i_b] = False


@qd.kernel
def _kernel_set_history(
    envs_idx: qd.types.ndarray(),
    qpos_prev: qd.types.ndarray(),
    dofs_vel_prev: qd.types.ndarray(),
    links_vsym: qd.types.ndarray(),
    links_vsym_prev: qd.types.ndarray(),
    n_hist: qd.types.ndarray(),
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    n_qs = mochi_state.qpos_prev.shape[1]
    n_dofs = mochi_state.dofs_vel_prev.shape[1]
    n_links = mochi_state.links_vsym.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        mochi_state.n_hist[envs_idx[i_b_]] = n_hist[envs_idx[i_b_]]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_b_ in qd.ndrange(n_qs, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for k in qd.static(range(N_HISTORY)):
            mochi_state.qpos_prev[k, i_q, i_b] = qpos_prev[i_b, k, i_q]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b_ in qd.ndrange(n_dofs, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for k in qd.static(range(N_HISTORY)):
            mochi_state.dofs_vel_prev[k, i_d, i_b] = dofs_vel_prev[i_b, k, i_d]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b_ in qd.ndrange(n_links, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j, l in qd.static(qd.ndrange(3, 3)):
            mochi_state.links_vsym[i_l, i_b][j, l] = links_vsym[i_b, i_l, j, l]
            for k in qd.static(range(N_HISTORY)):
                mochi_state.links_vsym_prev[k, i_l, i_b][j, l] = links_vsym_prev[i_b, k, i_l, j, l]


@qd.kernel
def _kernel_get_collider_distances(
    geom_idx: qd.i32,
    env_idx: qd.i32,
    points: qd.types.ndarray(),
    distances: qd.types.ndarray(),
    gradients: qd.types.ndarray(),
    is_valid: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    mochi_info: MochiInfo,
    mochi_config: qd.template(),
):
    for i_p in range(points.shape[0]):
        pos_world = qd.Vector([points[i_p, 0], points[i_p, 1], points[i_p, 2]], dt=gs.qd_float)
        pos_g = dyn_state.geoms.pos[geom_idx, env_idx]
        quat_g = dyn_state.geoms.quat[geom_idx, env_idx]
        pos_geom = gu.qd_inv_transform_by_trans_quat(pos_world, pos_g, quat_g)
        valid, sd, grad = query_collider(geom_idx, pos_geom, dyn_info.geoms, mochi_info.geoms, sdf_info, mochi_config)
        grad_world = gu.qd_transform_by_quat(grad, quat_g)
        distances[i_p] = sd
        is_valid[i_p] = valid
        for k in qd.static(range(3)):
            gradients[i_p, k] = grad_world[k]
