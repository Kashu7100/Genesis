# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import sys
from typing import TYPE_CHECKING

import igl
import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import AABB
from genesis.engine.entities.mochi_entity import MochiEntity, MochiSoftEntity
from genesis.engine.states.solvers import MochiSolverState
from genesis.options.solvers import MochiOptions
from genesis.utils import array_class
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array
from genesis.utils.sdf import SDF

from ..base_solver import StateChange, Subscriber
from ..kinematic_solver import KinematicSolver
from ..rigid.abd.accessor import (
    kernel_control_dofs_force,
    kernel_control_dofs_position,
    kernel_control_dofs_position_velocity,
    kernel_control_dofs_velocity,
    kernel_get_dofs_control_force,
    kernel_get_kinematic_state,
    kernel_set_dofs_armature,
    kernel_set_dofs_damping,
    kernel_set_dofs_kp,
    kernel_set_dofs_kv,
    kernel_set_dofs_limit,
    kernel_set_dofs_stiffness,
)
from ..rigid.abd.forward_kinematics import kernel_forward_kinematics, kernel_update_geom_aabbs, kernel_update_geoms
from ..rigid.abd.misc import (
    kernel_bit_reduction,
    kernel_init_entity_fields,
    kernel_init_geom_fields,
    kernel_init_vert_fields,
    kernel_update_geoms_render_T,
)
from .articulated import kernel_assemble_joints, kernel_project_links_residual, kernel_update_conv_weights
from .colliders import query_collider
from .contact import (
    TRIANGLE_QUADRATURES,
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
    get_mochi_soft_info,
    get_mochi_soft_state,
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
from .rigid_assembly import kernel_assemble_links
from .soft import (
    ENTITY_PARAMS,
    SOFT_KIND_SHELL,
    SOFT_KIND_SOLID,
    SoftTetLBVH,
    build_soft_samples,
    kernel_init_shell_fields,
    kernel_init_soft_fields,
    kernel_pc_collider_aabbs,
    kernel_pc_collider_eval,
    kernel_shell_assemble,
    kernel_shell_stage_start,
    kernel_soft_apply_increment,
    kernel_soft_assemble_elements,
    kernel_soft_broadphase,
    kernel_soft_collider_aabbs,
    kernel_soft_collider_eval,
    kernel_soft_collider_query,
    kernel_soft_condense_dense,
    kernel_soft_conservative_bounds,
    kernel_soft_contact_eval,
    kernel_soft_dirichlet,
    kernel_soft_get_entity_state,
    kernel_soft_get_state,
    kernel_soft_get_state_render,
    kernel_soft_get_vertices_field,
    kernel_soft_init_render,
    kernel_soft_pairs_to_blocks,
    kernel_soft_post_stage,
    kernel_soft_set_entity_contact_params,
    kernel_soft_set_links_pair_enabled,
    kernel_soft_set_pair_enabled,
    kernel_soft_set_state,
    kernel_soft_set_vertices_fixed,
    kernel_soft_set_vertices_positions,
    kernel_soft_set_vertices_velocities,
    kernel_soft_step_start,
    kernel_soft_store_ls_ref,
    kernel_soft_update_conv_weights,
    kernel_soft_zero_assembly,
)
from .soft_materials import ELASTIC_MODEL_BY_NAME

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
# Rest-shape signed distance grid of a deformable collider: cell size relative to the mean boundary edge length,
# padding around the rest bounds and minimum resolution per axis.
SOFT_SDF_CELL_RATIO = 0.25
SOFT_SDF_PADDING = 5e-3
SOFT_SDF_MIN_RES = 6

_AUTO_COLLIDER_TYPE_BY_GEOM_TYPE = {
    gs.GEOM_TYPE.PLANE: COLLIDER_TYPE.PLANE,
    gs.GEOM_TYPE.SPHERE: COLLIDER_TYPE.SPHERE,
    gs.GEOM_TYPE.BOX: COLLIDER_TYPE.BOX,
}


class MochiSolver(KinematicSolver):
    """
    Fully-implicit multi-physics solver with smooth penalty contact.

    Every substep solves one nonlinear system over the generalized coordinates of all rigid and articulated bodies and
    the vertex positions of all deformable bodies, whose residual is the gradient of an incremental potential (inertia,
    gravity, damping, elasticity and the contact penalty with its friction and damping terms) and whose Hessian is
    assembled from per-link, per-contact-pair and per-tetrahedron blocks. The solve is a damped Newton iteration with a
    line search; contact is re-detected at every iterate.
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: MochiOptions) -> None:
        super().__init__(scene, sim, options)
        self._options: MochiOptions = options
        self.sdf: SDF | None = None
        self._errno = None
        self._is_contacts_recorded = False
        # Set when the kinematic state is changed from outside the solver; the multistep history is then rebuilt at
        # the start of the next step.
        self._is_external_state_dirty = False
        self._external_state_subscriber = Subscriber(
            frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS}), callback=self._on_external_state_change
        )
        # Deformable bodies live outside the kinematic tree.
        self._soft_entities = gs.List()
        self.soft_info = None
        self.soft_state = None
        self._soft_vverts_render = None
        self._soft_vverts_vert_idx = None

    # ------------------------------------------------------------------------------------
    # ----------------------------------- add_entity -------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(self, idx, material, morph, surface, visualize_contact=False, name=None):
        if isinstance(morph, (tuple, list)):
            gs.raise_exception("Heterogeneous morphs are not supported by the MochiSolver.")
        if isinstance(morph, (gs.morphs.Terrain, gs.morphs.USD, gs.morphs.Drone)):
            gs.raise_exception(f"Morph {type(morph).__name__} is not supported by the MochiSolver.")

        if isinstance(material, (gs.materials.Mochi.Elastic, gs.materials.Mochi.Shell)):
            if not isinstance(morph, (gs.morphs.Box, gs.morphs.Sphere, gs.morphs.Cylinder, gs.morphs.Mesh)):
                gs.raise_exception(
                    f"Morph {type(morph).__name__} is not supported for deformable Mochi bodies (Box, Sphere, "
                    "Cylinder or Mesh expected)."
                )
            entity = MochiSoftEntity(
                scene=self._scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                idx=idx,
                idx_in_solver=self.n_soft_entities,
                v_start=self.n_soft_verts,
                el_start=self.n_soft_elems,
                s_start=self.n_soft_surfaces,
                vvert_start=self.n_soft_vverts,
                vface_start=self.n_soft_vfaces,
                name=name,
            )
            self._soft_entities.append(entity)
            return entity

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

        self._n_soft_entities = self.n_soft_entities
        self._n_soft_verts = self.n_soft_verts
        self._n_soft_elems = self.n_soft_elems
        self.n_soft_entities_ = max(1, self.n_soft_entities)
        self.n_soft_verts_ = max(1, self.n_soft_verts)
        self.n_soft_elems_ = max(1, self.n_soft_elems)
        self.n_shell_elems_ = max(1, self.n_shell_elems)
        self.n_dofs_total_ = max(1, self.n_dofs_total)

        if gs.qd_float == qd.f32 and any(
            entity.material.penalty_coefficient >= 1e8 for entity in (*self._entities, *self._soft_entities)
        ):
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
        self._init_soft()
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

        for entity in self._entities:
            if entity.n_equalities > 0:
                gs.logger.warning(
                    f"Entity '{entity.uid}' defines {entity.n_equalities} equality constraint(s), which the MochiSolver "
                    "does not enforce."
                )
        dofs_entity_mass = np.zeros(self.n_dofs_total_, dtype=gs.np_float)
        for entity in self._entities:
            dofs_entity_mass[entity.dof_start : entity.dof_end] = sum(link.inertial_mass for link in entity.links)
        for entity in self._soft_entities:
            dof_start = self.n_dofs + 3 * entity.v_start
            dofs_entity_mass[dof_start : dof_start + entity.n_dofs] = entity.mass

        self._layers = sorted({entity.material.contact_layer for entity in (*self._entities, *self._soft_entities)})
        n_layers = len(self._layers)
        self._layers_pair_enabled = np.ones((n_layers, n_layers), dtype=bool)
        self._entities_pair_enabled = np.ones((self.n_entities, self.n_entities), dtype=bool)
        self._soft_entities_pair_enabled = np.ones((self.n_soft_entities, self.n_soft_entities), dtype=bool)
        self._soft_rigid_entities_pair_enabled = np.ones((self.n_soft_entities, self.n_entities), dtype=bool)

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
            options.linear_solver == "auto" and self.n_dofs_total <= options.dense_solver_max_dofs
        )
        self._n_pcg_iterations = options.n_pcg_iterations
        if self._n_pcg_iterations is None:
            self._n_pcg_iterations = min(max(1, self.n_dofs_total), 1000)

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
            has_soft=self.has_soft,
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
            dofs_entity_mass,
            np.tile(np.asarray(self._init_gravity, dtype=gs.np_float), (self._B, 1)),
            self.mochi_info,
            self.rigid_config,
        )

    def _init_soft(self):
        """Rest data, contact samples and material parameters of the deformable bodies."""
        options = self._options
        entities = self._soft_entities
        n_links_ = self.n_links_
        _B = self._B

        if self.has_soft:
            verts_rest = np.concatenate([entity.init_positions for entity in entities]).astype(gs.np_float)
            verts_entity_idx = np.concatenate(
                [np.full(entity.n_vertices, entity.idx_in_solver, dtype=gs.np_int) for entity in entities]
            )
            solids = [entity for entity in entities if not entity.is_shell]
            shells = [entity for entity in entities if entity.is_shell]
            elems_v = np.concatenate(
                [np.zeros((0, 4), dtype=gs.np_int)] + [entity.elems + entity.v_start for entity in solids]
            ).astype(gs.np_int)
            elems_entity_idx = np.concatenate(
                [np.zeros((0,), dtype=gs.np_int)]
                + [np.full(entity.n_elements, entity.idx_in_solver, dtype=gs.np_int) for entity in solids]
            )
            shell_elems_v = np.concatenate(
                [np.zeros((0, 3), dtype=gs.np_int)] + [entity.elems + entity.v_start for entity in shells]
            ).astype(gs.np_int)
            shell_elems_hinge = np.concatenate(
                [np.zeros((0, 3), dtype=gs.np_int)] + [self._shell_hinges(entity) for entity in shells]
            ).astype(gs.np_int)
            shell_elems_entity_idx = np.concatenate(
                [np.zeros((0,), dtype=gs.np_int)]
                + [np.full(entity.n_elements, entity.idx_in_solver, dtype=gs.np_int) for entity in shells]
            )
            bary, ref_weights = TRIANGLE_QUADRATURES[options.boundary_element_type]
            samples_tri, samples_bary, samples_weight, samples_entity_idx = [], [], [], []
            entities_sample_range = np.zeros((len(entities), 2), dtype=gs.np_int)
            n_samples = 0
            for entity in entities:
                tri, sample_bary, weight = build_soft_samples(
                    entity.init_positions, entity.surface_triangles, bary, ref_weights
                )
                samples_tri.append(tri + entity.v_start)
                samples_bary.append(sample_bary)
                samples_weight.append(weight)
                samples_entity_idx.append(np.full(len(weight), entity.idx_in_solver, dtype=gs.np_int))
                entities_sample_range[entity.idx_in_solver] = (n_samples, n_samples + len(weight))
                n_samples += len(weight)
            samples_tri = np.concatenate(samples_tri).astype(gs.np_int)
            samples_bary = np.concatenate(samples_bary).astype(gs.np_float)
            samples_weight = np.concatenate(samples_weight).astype(gs.np_float)
            samples_entity_idx = np.concatenate(samples_entity_idx)
            sdf_grids = [self._build_soft_sdf_grid(entity) for entity in entities]
            sdf_values = np.concatenate([grid["values"] for grid in sdf_grids]).astype(gs.np_float)
            sdf_starts = np.cumsum([0] + [len(grid["values"]) for grid in sdf_grids[:-1]])
            entities_params = np.array(
                [
                    [
                        entity.mass,
                        float(entity.material.has_gravity),
                        *self._solid_material_params(entity),
                        entity.material.mass_damping,
                        entity.material.stiffness_damping,
                        entity.material.penalty_coefficient,
                        entity.material.penalty_smoothing_half_distance,
                        entity.material.penalty_threshold,
                        entity.material.friction,
                        entity.material.friction_falloff_vel,
                        entity.material.viscous_friction,
                        entity.material.normal_viscous_damping,
                        entity.material.max_alignment_normals,
                        entity.v_start,
                        entity.v_end,
                        entities_sample_range[entity.idx_in_solver, 0],
                        entities_sample_range[entity.idx_in_solver, 1],
                        float(grid["collider_type"]),
                        float(sdf_start),
                        *grid["res"],
                        *grid["origin"],
                        *grid["cell"],
                        float(SOFT_KIND_SHELL if entity.is_shell else SOFT_KIND_SOLID),
                        *self._shell_material_params(entity),
                    ]
                    for entity, grid, sdf_start in zip(entities, sdf_grids, sdf_starts)
                ],
                dtype=gs.np_float,
            ).reshape((-1, len(ENTITY_PARAMS)))
        else:
            verts_rest = np.zeros((0, 3), dtype=gs.np_float)
            verts_entity_idx = np.zeros((0,), dtype=gs.np_int)
            elems_v = np.zeros((0, 4), dtype=gs.np_int)
            elems_entity_idx = np.zeros((0,), dtype=gs.np_int)
            shell_elems_v = np.zeros((0, 3), dtype=gs.np_int)
            shell_elems_hinge = np.zeros((0, 3), dtype=gs.np_int)
            shell_elems_entity_idx = np.zeros((0,), dtype=gs.np_int)
            samples_tri = np.zeros((0, 3), dtype=gs.np_int)
            samples_bary = np.zeros((0, 3), dtype=gs.np_float)
            samples_weight = np.zeros((0,), dtype=gs.np_float)
            samples_entity_idx = np.zeros((0,), dtype=gs.np_int)
            entities_params = np.zeros((0, len(ENTITY_PARAMS)), dtype=gs.np_float)
            entities_sample_range = np.zeros((0, 2), dtype=gs.np_int)
            sdf_values = np.zeros((0,), dtype=gs.np_float)
            n_samples = 0
        self._n_soft_samples = n_samples
        self.n_soft_samples_ = max(1, n_samples)
        self._max_samples_per_soft_entity = int(
            max(1, (entities_sample_range[:, 1] - entities_sample_range[:, 0]).max(initial=0))
        )

        n_collider_geoms = sum(1 for geom in self.geoms if self._resolve_collider_type(geom) != COLLIDER_TYPE.NONE)
        self._max_soft_pairs = options.max_contact_pairs_per_env
        if self._max_soft_pairs is None:
            self._max_soft_pairs = max(1, len(entities) * n_collider_geoms)
        self._max_soft_hits = max(1, 2 * n_samples)
        self._n_soft_sdf_voxels = len(sdf_values)
        self.n_soft_sdf_voxels_ = max(1, len(sdf_values))

        # Deformable colliders: every rigid and deformable sample point is located in the deformed tetrahedra through
        # a bounding-volume hierarchy rebuilt at every assembly.
        self._has_soft_colliders = any(
            grid["collider_type"] == COLLIDER_TYPE.GRID for grid in (sdf_grids if self.has_soft else ())
        )
        self._has_pc_colliders = any(
            grid["collider_type"] == COLLIDER_TYPE.POINT_CLOUD for grid in (sdf_grids if self.has_soft else ())
        )
        self._n_soft_queries = self.n_samples + n_samples
        self._max_sc_hits = max(1, 2 * self._n_soft_queries) if self._has_soft_colliders else 1
        self._max_pc_hits = max(1, 4 * self._n_soft_queries) if self._has_pc_colliders else 1
        if self._has_soft_colliders or self._has_pc_colliders:
            self._soft_query_aabb = AABB(_B, max(1, self._n_soft_queries))
            queries_entity_idx = np.concatenate([np.full(self.n_samples, -1, dtype=gs.np_int), samples_entity_idx])
        if self._has_pc_colliders:
            self._pc_aabb = AABB(_B, self.n_soft_verts_)
            n_results_per_point = max(1, -(-8 * self._n_soft_queries // self.n_soft_verts_))
            self._pc_bvh = SoftTetLBVH(
                self._pc_aabb, verts_entity_idx, queries_entity_idx, max_n_query_result_per_aabb=n_results_per_point
            )
        if self._has_soft_colliders or self._has_pc_colliders:
            self._soft_tet_aabb = AABB(_B, self.n_soft_elems_)
        if self._has_soft_colliders:
            # A sample point overlaps the bounds of many tetrahedra before the inclusion test; those of its own entity
            # are filtered out by the hierarchy.
            n_results_per_tet = max(1, -(-16 * self._n_soft_queries // self.n_soft_elems_))
            self._soft_tet_bvh = SoftTetLBVH(
                self._soft_tet_aabb, elems_entity_idx, queries_entity_idx, max_n_query_result_per_aabb=n_results_per_tet
            )

        self.soft_info = get_mochi_soft_info(self)
        self.soft_state = get_mochi_soft_state(
            self, self._max_soft_pairs, self._max_soft_hits, self._max_sc_hits, self._max_pc_hits
        )
        kernel_init_soft_fields(
            verts_rest,
            verts_entity_idx,
            elems_v,
            elems_entity_idx,
            samples_tri,
            samples_bary,
            samples_weight,
            samples_entity_idx,
            entities_params,
            self._compute_soft_links_pair_enabled(),
            self._compute_soft_pair_enabled(),
            sdf_values,
            self.soft_info,
            self.soft_state,
            self.rigid_config,
        )
        if self.n_shell_elems > 0:
            kernel_init_shell_fields(
                shell_elems_v, shell_elems_hinge, shell_elems_entity_idx, self.soft_info, self.rigid_config
            )

        # Render geometry of the deformable surfaces.
        n_soft_vverts_ = max(1, self.n_soft_vverts)
        self._soft_vverts_render = array_class.V_VEC(3, dtype=qd.f32, shape=(n_soft_vverts_, _B))
        self._soft_vverts_vert_idx = array_class.V(dtype=gs.qd_int, shape=(n_soft_vverts_,))
        if self.n_soft_vverts > 0:
            vert_idx = np.concatenate(
                [vgeom.sim_verts_idx + entity.v_start for entity in entities for vgeom in entity.vgeoms]
            ).astype(gs.np_int)
            kernel_soft_init_render(vert_idx, self._soft_vverts_vert_idx)
        self._envs_offset = np.asarray(self._scene.envs_offset, dtype=gs.np_float).reshape((_B, 3))

    @staticmethod
    def _shell_hinges(entity):
        """Opposite vertex (global index) of the neighboring triangle across each edge of every shell triangle, -1
        across a boundary edge. Edge e is the edge opposite the local vertex e."""
        faces = np.asarray(entity.elems, dtype=np.int64)
        edge_to_faces = {}
        for i_f, face in enumerate(faces):
            for e in range(3):
                key = tuple(sorted((int(face[(e + 1) % 3]), int(face[(e + 2) % 3]))))
                edge_to_faces.setdefault(key, []).append(i_f)
        hinges = np.full((len(faces), 3), -1, dtype=gs.np_int)
        for i_f, face in enumerate(faces):
            for e in range(3):
                key = tuple(sorted((int(face[(e + 1) % 3]), int(face[(e + 2) % 3]))))
                for j_f in edge_to_faces[key]:
                    if j_f != i_f:
                        other = [int(v) for v in faces[j_f] if int(v) not in key]
                        hinges[i_f, e] = other[0] + entity.v_start
                        break
        return hinges

    @staticmethod
    def _solid_material_params(entity):
        """(model, mu, lambda, density) columns: the elastic model of a solid, the areal density of a shell."""
        material = entity.material
        if entity.is_shell:
            return (0.0, 0.0, 0.0, material.areal_density)
        return (float(ELASTIC_MODEL_BY_NAME[material.model]), material.mu, material.lam, material.rho)

    @staticmethod
    def _shell_material_params(entity):
        material = entity.material
        if entity.is_shell:
            return (
                material.membrane_mu,
                material.membrane_lambda,
                material.bending_alpha,
                material.bending_beta,
                material.collider_radius,
            )
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    def _build_soft_sdf_grid(self, entity):
        """Rest-shape signed distance grid of a deformable solid collider (igl over the boundary triangles); shells are
        point-cloud colliders and bodies that do not act as colliders get an empty grid."""
        if entity.material.collider_type == "none" or entity.is_shell:
            collider_type = COLLIDER_TYPE.NONE
            if entity.is_shell and entity.material.collider_type != "none":
                collider_type = COLLIDER_TYPE.POINT_CLOUD
            return {
                "collider_type": collider_type,
                "values": np.zeros((0,), dtype=gs.np_float),
                "res": (1, 1, 1),
                "origin": (0.0, 0.0, 0.0),
                "cell": (1.0, 1.0, 1.0),
            }
        verts = np.asarray(entity.init_positions, dtype=np.float64)
        faces = np.asarray(entity.surface_triangles, dtype=np.int64)
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        mean_edge = float(np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1).mean())
        lower = verts.min(axis=0) - SOFT_SDF_PADDING
        upper = verts.max(axis=0) + SOFT_SDF_PADDING
        # Voxels of at most a quarter of the mean boundary edge, and at least SOFT_SDF_MIN_RES per axis so that thin
        # bodies are resolved through their thickness.
        res = np.maximum(np.ceil((upper - lower) / (SOFT_SDF_CELL_RATIO * mean_edge)).astype(int) + 1, SOFT_SDF_MIN_RES)
        cell = (upper - lower) / (res - 1)
        axes = [lower[k] + cell[k] * np.arange(res[k]) for k in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
        values, *_ = igl.signed_distance(grid, verts, faces.astype(np.int32))
        return {
            "collider_type": COLLIDER_TYPE.GRID,
            "values": np.asarray(values, dtype=gs.np_float),
            "res": tuple(int(r) for r in res),
            "origin": tuple(float(x) for x in lower),
            "cell": tuple(float(c) for c in cell),
        }

    def _compute_soft_pair_enabled(self):
        """Deformable entity pair contact filter from the contact layers (an entity never collides with itself)."""
        enabled = np.ones((self.n_soft_entities_, self.n_soft_entities_), dtype=bool)
        if self.has_soft:
            entities_layer = np.array(
                [self._layers.index(entity.material.contact_layer) for entity in self._soft_entities], dtype=int
            )
            n = self.n_soft_entities
            enabled[:n, :n] = self._layers_pair_enabled[entities_layer[:, None], entities_layer[None, :]]
            enabled[:n, :n] &= self._soft_entities_pair_enabled
            np.fill_diagonal(enabled, False)
        return enabled

    def _compute_soft_links_pair_enabled(self):
        """Deformable entity / rigid link contact filter from the contact layers."""
        links = self.links
        enabled = np.ones((self.n_soft_entities_, self.n_links_), dtype=bool)
        if self.has_soft and self.n_links > 0:
            entities_layer = np.array(
                [self._layers.index(entity.material.contact_layer) for entity in self._soft_entities], dtype=int
            )
            links_layer = np.array(
                [self._layers.index(link.entity.material.contact_layer) for link in links], dtype=int
            )
            links_entity = np.array([link._entity_idx_in_solver for link in links], dtype=int)
            enabled[: self.n_soft_entities, : self.n_links] = self._layers_pair_enabled[
                entities_layer[:, None], links_layer[None, :]
            ]
            enabled[: self.n_soft_entities, : self.n_links] &= self._soft_rigid_entities_pair_enabled[:, links_entity]
        return enabled

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
        links_layer = np.array([self._layers.index(link.entity.material.contact_layer) for link in links], dtype=int)
        links_entity = np.array([link._entity_idx_in_solver for link in links], dtype=int)
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

    def _on_external_state_change(self, change, envs_idx):
        self._is_external_state_dirty = True

    def _sync_external_state(self):
        # State set from outside the solver invalidates the multistep history and the rotation-derivative correction.
        if self._is_external_state_dirty:
            kernel_reset_history(
                self._scene._envs_idx,
                self.dyn_state,
                self.dyn_info,
                self.mochi_info,
                self.mochi_state,
                self.rigid_config,
            )
            self._is_external_state_dirty = False

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
        if self.has_soft:
            kernel_soft_step_start(
                self.mochi_state, self.soft_info, self.soft_state, self.rigid_config, self.mochi_config
            )
            if self.n_shell_elems > 0:
                kernel_shell_stage_start(self.soft_info, self.soft_state, self.rigid_config)
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
        if self.has_soft:
            kernel_soft_conservative_bounds(self.mochi_info, self.soft_info, self.soft_state, self.rigid_config)
            kernel_soft_broadphase(
                self.dyn_state,
                self.dyn_info,
                self.mochi_info,
                self.mochi_state,
                self.contact_state,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
                self._errno,
            )
        self._newton_solve()
        kernel_post_stage(
            self.dyn_state, self.dyn_info, self.rigid_info, self.mochi_info, self.mochi_state, self.rigid_config
        )
        if self.has_soft:
            kernel_soft_post_stage(self.mochi_state, self.soft_state, self.rigid_config)
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
        if self.has_soft:
            kernel_soft_zero_assembly(
                self.mochi_state, self.soft_state, self.rigid_config, assem_dres, skip_ls_done, record
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
        if self.has_soft:
            kernel_soft_contact_eval(
                self.dyn_state,
                self.dyn_info,
                self.sdf._sdf_info,
                self.mochi_info,
                self.mochi_state,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
                self.mochi_config,
                self._max_samples_per_soft_entity,
                assem_res,
                assem_dres,
                skip_ls_done,
                record,
                self._errno,
            )
            if self._has_soft_colliders or self._has_pc_colliders:
                kernel_soft_collider_aabbs(
                    self.dyn_state,
                    self.mochi_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self._soft_tet_aabb.aabbs,
                    self._soft_query_aabb.aabbs,
                    self.n_samples,
                    self.rigid_config,
                )
            if self._has_pc_colliders:
                kernel_pc_collider_aabbs(
                    self.mochi_state, self.soft_info, self.soft_state, self._pc_aabb.aabbs, self.rigid_config
                )
                self._pc_bvh.build()
                if kernel_soft_collider_query(self._pc_bvh, self._soft_query_aabb.aabbs):
                    gs.raise_exception("Exceeding the capacity of the point-cloud collider queries.")
                kernel_pc_collider_eval(
                    self._pc_bvh.query_result,
                    self._pc_bvh.query_result_count,
                    self.dyn_state,
                    self.dyn_info,
                    self.mochi_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.rigid_config,
                    self.mochi_config,
                    self.n_samples,
                    assem_obj,
                    assem_res,
                    assem_dres,
                    skip_ls_done,
                    record,
                    self._errno,
                )
            if self._has_soft_colliders:
                self._soft_tet_bvh.build()
                if kernel_soft_collider_query(self._soft_tet_bvh, self._soft_query_aabb.aabbs):
                    gs.raise_exception("Exceeding the capacity of the deformable collider queries.")
                kernel_soft_collider_eval(
                    self._soft_tet_bvh.query_result,
                    self._soft_tet_bvh.query_result_count,
                    self.dyn_state,
                    self.dyn_info,
                    self.mochi_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.rigid_config,
                    self.mochi_config,
                    self.n_samples,
                    assem_obj,
                    assem_res,
                    assem_dres,
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
        kernel_assemble_links(
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
        kernel_assemble_joints(
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.mochi_info,
            self.mochi_state,
            self.rigid_config,
            assem_obj,
            assem_res,
            assem_dres,
            skip_ls_done,
        )
        if self.has_soft:
            kernel_soft_pairs_to_blocks(
                self.dyn_state,
                self.mochi_info,
                self.mochi_state,
                self.soft_state,
                self.rigid_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
            )
            kernel_soft_assemble_elements(
                self.mochi_info,
                self.mochi_state,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
                assem_obj,
                assem_res,
                assem_dres,
                skip_ls_done,
            )
            if self.n_shell_elems > 0:
                kernel_shell_assemble(
                    self.mochi_info,
                    self.mochi_state,
                    self.soft_info,
                    self.soft_state,
                    self.rigid_config,
                    assem_obj,
                    assem_res,
                    assem_dres,
                    skip_ls_done,
                )
        if assem_res:
            if self.has_soft:
                kernel_soft_dirichlet(
                    self.mochi_state, self.soft_info, self.soft_state, self.rigid_config, skip_ls_done
                )
            kernel_project_links_residual(
                self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.rigid_config, skip_ls_done
            )
            kernel_residual_norms(self.mochi_state, self.rigid_config, skip_ls_done)

    def _linear_solve(self):
        if self.mochi_config.use_dense_direct:
            kernel_condense_dense(
                self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.contact_state, self.rigid_config
            )
            if self.has_soft:
                kernel_soft_condense_dense(
                    self.dyn_state, self.dyn_info, self.mochi_state, self.soft_info, self.soft_state, self.rigid_config
                )
            kernel_cholesky_solve_dense(self.mochi_info, self.mochi_state, self.rigid_config)
        else:
            kernel_pcg_init(
                self.dyn_state,
                self.dyn_info,
                self.mochi_info,
                self.mochi_state,
                self.soft_info,
                self.soft_state,
                self.rigid_config,
                self.mochi_config,
            )
            for i_iter in range(self._n_pcg_iterations):
                kernel_pcg_iter(
                    self.dyn_state,
                    self.dyn_info,
                    self.mochi_info,
                    self.mochi_state,
                    self.contact_state,
                    self.soft_info,
                    self.soft_state,
                    self.rigid_config,
                    self.mochi_config,
                )
                if (i_iter + 1) % PCG_CHECK_PERIOD == 0 and kernel_pcg_any_active(self.mochi_state) == 0:
                    break

    def _newton_solve(self):
        options = self._options
        kernel_update_conv_weights(self.dyn_state, self.dyn_info, self.mochi_info, self.mochi_state, self.rigid_config)
        if self.has_soft:
            kernel_soft_update_conv_weights(
                self.mochi_info, self.mochi_state, self.soft_info, self.soft_state, self.rigid_config
            )
        self._assemble(assem_res=True, assem_dres=True, skip_ls_done=False)
        kernel_store_initial_norms(self.rigid_info, self.mochi_state, self.rigid_config)
        if self.has_soft:
            kernel_soft_store_ls_ref(self.mochi_state, self.soft_state, self.rigid_config, False)
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
            if self.has_soft:
                kernel_soft_store_ls_ref(self.mochi_state, self.soft_state, self.rigid_config, False)
            for i_ls in range(max(1, n_linesearch)):
                kernel_apply_increment(
                    self.dyn_info, self.rigid_info, self.mochi_info, self.mochi_state, self.rigid_config
                )
                if self.has_soft:
                    kernel_soft_apply_increment(self.mochi_state, self.soft_info, self.soft_state, self.rigid_config)
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
                if self.has_soft:
                    kernel_soft_store_ls_ref(self.mochi_state, self.soft_state, self.rigid_config, True)
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

    def get_soft_state_render(self, f):
        """Environment-offset render vertex positions of the deformable surfaces, shape (n_vverts, B), as (positions,
        None, None) (UVs and faces are read from the visual geoms)."""
        if not self.has_soft or self.n_soft_vverts == 0:
            return None, None, None
        kernel_soft_get_state_render(
            self._soft_vverts_render, self._soft_vverts_vert_idx, self._envs_offset, self.soft_state, self.rigid_config
        )
        return self._soft_vverts_render, None, None

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
        state.links_vel = qd_to_torch(self.mochi_state.links_vel, copy=True).permute((1, 0, 2)).contiguous()
        state.links_ang = qd_to_torch(self.mochi_state.links_ang, copy=True).permute((1, 0, 2)).contiguous()
        state.links_vsym = qd_to_torch(self.mochi_state.links_vsym, copy=True).permute((1, 0, 2, 3)).contiguous()
        state.links_vel_prev = (
            qd_to_torch(self.mochi_state.links_vel_prev, copy=True).permute((2, 0, 1, 3)).contiguous()
        )
        state.links_ang_prev = (
            qd_to_torch(self.mochi_state.links_ang_prev, copy=True).permute((2, 0, 1, 3)).contiguous()
        )
        state.links_vsym_prev = (
            qd_to_torch(self.mochi_state.links_vsym_prev, copy=True).permute((2, 0, 1, 3, 4)).contiguous()
        )
        state.n_hist = qd_to_torch(self.mochi_state.n_hist, copy=True).contiguous()
        if self.has_soft:
            kernel_soft_get_state(
                state.soft_pos,
                state.soft_vel,
                state.soft_pos_prev,
                state.soft_vel_prev,
                self.soft_state,
                self.rigid_config,
            )
        self._queried_states.append(state)
        return state

    def set_state(self, f, state, envs_idx=None, *, partial: bool = False) -> None:
        if not self.is_active:
            return
        # The kinematic state is restored (and subscribers notified) by the base class; the integration history is
        # restored on top, leaving nothing for the external-state sync of the next step to rebuild.
        super().set_state(f, state, envs_idx, partial=False)
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        _kernel_set_history(
            envs_idx,
            state.qpos_prev,
            state.dofs_vel_prev,
            state.links_vel,
            state.links_ang,
            state.links_vsym,
            state.links_vel_prev,
            state.links_ang_prev,
            state.links_vsym_prev,
            state.n_hist,
            self.mochi_state,
            self.rigid_config,
        )
        kernel_update_geoms(
            self._scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, False
        )
        kernel_update_geom_aabbs(self.geoms_init_AABB, self.dyn_state, self.rigid_config)
        if self.has_soft:
            kernel_soft_set_state(
                envs_idx,
                state.soft_pos,
                state.soft_vel,
                state.soft_pos_prev,
                state.soft_vel_prev,
                self.soft_state,
                self.rigid_config,
            )
        self._is_external_state_dirty = False
        self._is_contacts_recorded = False

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

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def _set_dofs_info(self, tensor_list, dofs_idx, name, envs_idx=None):
        tensor_list = list(tensor_list)
        for j, tensor in enumerate(tensor_list):
            tensor, dofs_idx, envs_idx_ = self._sanitize_io_variables(
                tensor,
                dofs_idx,
                self.n_dofs,
                "dofs_idx",
                envs_idx,
                batched=self._options.batch_dofs_info,
                skip_allocation=True,
            )
            if self.n_envs == 0 and self._options.batch_dofs_info:
                tensor = tensor[None]
            tensor_list[j] = tensor
        if name == "kp":
            kernel_set_dofs_kp(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "kv":
            kernel_set_dofs_kv(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "stiffness":
            kernel_set_dofs_stiffness(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "armature":
            kernel_set_dofs_armature(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "damping":
            kernel_set_dofs_damping(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "limit":
            kernel_set_dofs_limit(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def set_dofs_kp(self, kp, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kp], dofs_idx, "kp", envs_idx)

    def set_dofs_kv(self, kv, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kv], dofs_idx, "kv", envs_idx)

    def set_dofs_stiffness(self, stiffness, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([stiffness], dofs_idx, "stiffness", envs_idx)

    def set_dofs_armature(self, armature, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([armature], dofs_idx, "armature", envs_idx)

    def set_dofs_damping(self, damping, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([damping], dofs_idx, "damping", envs_idx)

    def set_dofs_limit(self, lower, upper, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx)

    def _sanitize_control(self, tensor, dofs_idx, envs_idx):
        tensor, dofs_idx, envs_idx = self._sanitize_io_variables(
            tensor, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            tensor = tensor[None]
        return tensor, dofs_idx, envs_idx

    def control_dofs_force(self, force, dofs_idx=None, envs_idx=None):
        force, dofs_idx, envs_idx = self._sanitize_control(force, dofs_idx, envs_idx)
        kernel_control_dofs_force(dofs_idx, envs_idx, force, self.dyn_state, self.rigid_config)

    def control_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None):
        velocity, dofs_idx, envs_idx = self._sanitize_control(velocity, dofs_idx, envs_idx)
        kernel_control_dofs_velocity(dofs_idx, envs_idx, velocity, self.dyn_state, self.rigid_config)

    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        position, dofs_idx, envs_idx = self._sanitize_control(position, dofs_idx, envs_idx)
        kernel_control_dofs_position(dofs_idx, envs_idx, position, self.dyn_state, self.rigid_config)

    def control_dofs_position_velocity(self, position, velocity, dofs_idx=None, envs_idx=None):
        position, dofs_idx, envs_idx_ = self._sanitize_control(position, dofs_idx, envs_idx)
        velocity, _, _ = self._sanitize_control(velocity, dofs_idx, envs_idx)
        kernel_control_dofs_position_velocity(
            dofs_idx, envs_idx_, position, velocity, self.dyn_state, self.rigid_config
        )

    def get_dofs_control_force(self, dofs_idx=None, envs_idx=None):
        _tensor, dofs_idx, envs_idx = self._sanitize_io_variables(None, dofs_idx, self.n_dofs, "dofs_idx", envs_idx)
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_dofs_control_force(dofs_idx, envs_idx, tensor, self.dyn_state, self.dyn_info, self.rigid_config)
        return _tensor

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
        if self.has_soft:
            kernel_soft_set_links_pair_enabled(
                self._compute_soft_links_pair_enabled(), self.soft_info, self.rigid_config
            )
            kernel_soft_set_pair_enabled(self._compute_soft_pair_enabled(), self.soft_info, self.rigid_config)

    def enable_entity_contact(self, entity_a, entity_b, is_enabled: bool = True):
        """Enable or disable contact between two entities (symmetric)."""
        is_soft_a = isinstance(entity_a, MochiSoftEntity)
        is_soft_b = isinstance(entity_b, MochiSoftEntity)
        i_a, i_b = entity_a._idx_in_solver, entity_b._idx_in_solver
        if is_soft_a and is_soft_b:
            self._soft_entities_pair_enabled[i_a, i_b] = self._soft_entities_pair_enabled[i_b, i_a] = is_enabled
        elif is_soft_a:
            self._soft_rigid_entities_pair_enabled[i_a, i_b] = is_enabled
        elif is_soft_b:
            self._soft_rigid_entities_pair_enabled[i_b, i_a] = is_enabled
        else:
            self._entities_pair_enabled[i_a, i_b] = self._entities_pair_enabled[i_b, i_a] = is_enabled
        kernel_set_links_pair_enabled(self._compute_links_pair_enabled(), self.mochi_info, self.rigid_config)
        if self.has_soft:
            kernel_soft_set_links_pair_enabled(
                self._compute_soft_links_pair_enabled(), self.soft_info, self.rigid_config
            )
            kernel_soft_set_pair_enabled(self._compute_soft_pair_enabled(), self.soft_info, self.rigid_config)

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

    # ------------------------------------------------------------------------------------
    # --------------------------------- deformable bodies --------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_links > 0 or self.n_soft_entities > 0

    @property
    def has_soft(self):
        return self.n_soft_entities > 0

    @property
    def soft_entities(self):
        return self._soft_entities

    @property
    def n_soft_entities(self):
        return len(self._soft_entities)

    @property
    def n_soft_verts(self):
        return sum(entity.n_vertices for entity in self._soft_entities)

    @property
    def n_soft_elems(self):
        return sum(entity.n_elements for entity in self._soft_entities if not entity.is_shell)

    @property
    def n_shell_elems(self):
        return sum(entity.n_elements for entity in self._soft_entities if entity.is_shell)

    @property
    def n_soft_surfaces(self):
        return sum(entity.n_surfaces for entity in self._soft_entities)

    @property
    def n_soft_vverts(self):
        return sum(entity.n_vverts for entity in self._soft_entities)

    @property
    def n_soft_vfaces(self):
        return sum(entity.n_vfaces for entity in self._soft_entities)

    @property
    def n_soft_samples(self):
        return self._n_soft_samples

    @property
    def n_dofs_total(self):
        """Degrees of freedom of the Newton system: rigid, then 3 per deformable vertex."""
        return self.n_dofs + 3 * self.n_soft_verts

    def _soft_field_of_entity(self, entity, field, envs_idx):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        out = torch.empty((len(envs_idx), entity.n_vertices, 3), dtype=gs.tc_float, device=gs.device)
        kernel_soft_get_vertices_field(envs_idx, entity.v_start, out, field, self.rigid_config)
        return out[0] if self.n_envs == 0 else out

    def _soft_values_of_entity(self, entity, values, envs_idx):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        values = torch.as_tensor(values, dtype=gs.tc_float, device=gs.device)
        if values.ndim == 2:
            values = values[None].expand(len(envs_idx), -1, -1)
        if values.shape != (len(envs_idx), entity.n_vertices, 3):
            gs.raise_exception(
                f"Expected an array of shape ({len(envs_idx)}, {entity.n_vertices}, 3) or ({entity.n_vertices}, 3), "
                f"got {tuple(values.shape)}."
            )
        return envs_idx, values.contiguous()

    def get_soft_entity_state(self, entity, pos, vel):
        kernel_soft_get_entity_state(entity.v_start, pos, vel, self.soft_state, self.rigid_config)

    def get_soft_vertices_position(self, entity, envs_idx=None):
        return self._soft_field_of_entity(entity, self.soft_state.verts_pos, envs_idx)

    def get_soft_vertices_velocity(self, entity, envs_idx=None):
        return self._soft_field_of_entity(entity, self.soft_state.verts_vel, envs_idx)

    def get_soft_vertices_contact_force(self, entity, envs_idx=None):
        self._record_contacts()
        return self._soft_field_of_entity(entity, self.soft_state.verts_contact_force, envs_idx)

    def set_soft_vertices_position(self, entity, pos, envs_idx=None):
        envs_idx, pos = self._soft_values_of_entity(entity, pos, envs_idx)
        kernel_soft_set_vertices_positions(envs_idx, entity.v_start, pos, self.soft_state, self.rigid_config)
        self._is_external_state_dirty = True
        self._is_contacts_recorded = False

    def set_soft_vertices_velocity(self, entity, vel, envs_idx=None):
        envs_idx, vel = self._soft_values_of_entity(entity, vel, envs_idx)
        kernel_soft_set_vertices_velocities(envs_idx, entity.v_start, vel, self.soft_state, self.rigid_config)
        self._is_external_state_dirty = True

    def set_soft_vertices_fixed(self, entity, verts_idx, is_fixed, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = np.asarray(verts_idx, dtype=gs.np_int) + entity.v_start
        kernel_soft_set_vertices_fixed(envs_idx, verts_idx, int(bool(is_fixed)), self.soft_state, self.rigid_config)
        self._is_external_state_dirty = True

    def set_soft_entity_contact_params(self, entity, **params):
        material = entity.material
        keys = (
            "penalty_coefficient",
            "friction",
            "penalty_smoothing_half_distance",
            "penalty_threshold",
            "friction_falloff_vel",
            "viscous_friction",
            "normal_viscous_damping",
        )
        values = []
        for key in keys:
            value = params.get(key)
            if value is None:
                value = getattr(material, key)
            else:
                setattr(material, key, value)
            values.append(float(value))
        kernel_soft_set_entity_contact_params(entity.idx_in_solver, np.array(values, dtype=gs.np_float), self.soft_info)
        self._is_contacts_recorded = False


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
    links_vel: qd.types.ndarray(),
    links_ang: qd.types.ndarray(),
    links_vsym: qd.types.ndarray(),
    links_vel_prev: qd.types.ndarray(),
    links_ang_prev: qd.types.ndarray(),
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
        for j in qd.static(range(3)):
            mochi_state.links_vel[i_l, i_b][j] = links_vel[i_b, i_l, j]
            mochi_state.links_ang[i_l, i_b][j] = links_ang[i_b, i_l, j]
            for k in qd.static(range(N_HISTORY)):
                mochi_state.links_vel_prev[k, i_l, i_b][j] = links_vel_prev[i_b, k, i_l, j]
                mochi_state.links_ang_prev[k, i_l, i_b][j] = links_ang_prev[i_b, k, i_l, j]
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
