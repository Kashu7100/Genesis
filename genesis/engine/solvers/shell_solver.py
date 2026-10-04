from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.entities.shell_entity import ShellEntity
from genesis.engine.materials.shell import Shell
from genesis.engine.states.solvers import ShellSolverState
from genesis.utils.misc import broadcast_tensor, qd_to_torch

from .base_solver import GravityMixin, Solver, TimeBasedMixin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


# Edge of the grid the position of every vertex is anchored to, in m (see verts_pos_cell in array_class.py). A power of
# two keeps the cells exact in floating point, and its size bounds the offsets, whose precision it sets.
POS_GRID = 2.0**-10

# Floor of the norms the shell kernels divide by, guarding degenerate geometry alone. An additive epsilon (as in
# 'norm(gs.EPS)') would bias the edges, areas and normals of fine meshes, whose squared magnitudes approach it in single
# precision.
NORM_FLOOR = 1e-30


class ShellSolver(GravityMixin, TimeBasedMixin, Solver):
    """
    Solver of thin elastoplastic sheets that tear and crack.

    Each substep integrates the sheets by one linearized backward Euler step: membrane stretching (Saint Venant-Kirchhoff
    on the Green strain, plane stress) and hinge bending, both with stiffness-proportional damping. The linear system
    is solved by matrix-free preconditioned conjugate gradient (PCG), every environment iterating until its own
    residual converges. The preconditioner adds to block-Jacobi a coarse correction where each patch of vertices moves
    affinely, which resolves the stiff membranes (paper, metal, glass) whose block-Jacobi iterations would only converge
    after hundreds of iterations. Positions are anchored to a fine grid, keeping strains precise in single precision
    wherever the sheet is. The rigid coupling then corrects the vertex velocities, before the positions advance, the
    material yields plastically, and the vertices whose surrounding stress exceeds the tensile strength split along
    the mesh edges that relieve the most stress.
    """

    material_cls = Shell

    def __init__(self, scene: "Scene", sim: "Simulator", options):
        super().__init__(scene, sim, options)

        self._n_pcg_iterations = options.n_pcg_iterations
        self._pcg_threshold = options.pcg_threshold
        self._fracture_capacity = options.fracture_capacity
        self._n_coarse_patches = options.n_coarse_patches
        self._coarse_update_interval = options.coarse_update_interval
        # Substeps run since the coarse matrices were last factorized, None forcing an update at the next one
        self._coarse_age: int | None = None

        self._static_config: array_class.ShellStaticConfig | None = None
        self._shell_info: array_class.ShellInfo | None = None
        self._shell_state: array_class.ShellState | None = None
        self._shell_scratch: array_class.ShellScratch | None = None

    def add_entity(self, idx, material, morph, surface, visualize_contact=False, name=None, desc=None) -> ShellEntity:
        entity = ShellEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            vert_start=self.n_verts,
            face_start=self.n_faces,
            hinge_start=self.n_hinges,
            name=name,
        )
        self._entities.append(entity)
        return entity

    def build(self):
        super().build()
        self._n_verts = self.n_verts
        self._n_faces = self.n_faces
        self._n_hinges = self.n_hinges

        if self.is_active:
            materials = [entity.material for entity in self._entities]
            self._static_config = array_class.ShellStaticConfig(
                has_fracture=any(material.tensile_strength is not None for material in materials),
                has_plasticity=any(
                    material.yield_stress is not None or material.yield_curvature is not None for material in materials
                ),
                has_coarse_space=self._n_coarse_patches > 0,
                coarse_block_dim=1 if gs.backend == gs.cpu else 32,
            )
            entities_coarse_dim = [
                9 * entity.patches.n_patches if entity.patches is not None else 0 for entity in self._entities
            ]
            self._entities_coarse_dof_start = np.cumsum([0, *entities_coarse_dim])[:-1]
            self._entities_coarse_matrix_start = np.cumsum([0, *(dim**2 for dim in entities_coarse_dim)])[:-1]
            self._entities_coarse_dim = np.array(entities_coarse_dim)
            self._shell_info = array_class.get_shell_info(
                len(self._entities), self._n_verts, self._n_faces, self._n_hinges, int(self._entities_coarse_dim.sum())
            )
            self._shell_state = array_class.get_shell_state(
                len(self._entities), self._n_verts, self._n_faces, self._n_hinges, self._B
            )
            self._shell_scratch = array_class.get_shell_scratch(
                self._n_verts,
                self._n_faces,
                self._n_hinges,
                int(self._entities_coarse_dim.sum()),
                int(np.square(self._entities_coarse_dim).sum()),
                self._B,
                self._static_config.has_fracture,
            )
            self._init_info_and_state()

        self._build_gravity()

    def _init_info_and_state(self):
        """Fill the rest mesh of every entity and the initial state of every environment."""
        n_verts, n_faces, n_hinges, B = self._n_verts, self._n_faces, self._n_hinges, self._B
        info, state = self._shell_info, self._shell_state

        entities_material = np.array(
            [
                (
                    material.E / (1.0 - material.nu**2),
                    material.nu,
                    material.bending_scale * material.E / (12.0 * (1.0 - material.nu**2)),
                    material.damping,
                    material.tensile_strength or 0.0,
                    material.bending_fracture_scale,
                    material.yield_stress or 0.0,
                    material.plastic_flow_rate,
                    material.yield_curvature or 0.0,
                )
                for material in (entity.material for entity in self._entities)
            ],
            dtype=gs.np_float,
        )
        for i, tensor in enumerate(
            (
                info.entities_stretching_modulus,
                info.entities_nu,
                info.entities_bending_modulus,
                info.entities_damping,
                info.entities_tensile_strength,
                info.entities_bending_fracture_scale,
                info.entities_yield_stress,
                info.entities_plastic_flow_rate,
                info.entities_yield_curvature,
            )
        ):
            tensor.from_numpy(np.ascontiguousarray(entities_material[:, i]))
        info.entities_vert_start.from_numpy(np.array([e.vert_start for e in self._entities], dtype=gs.np_int))
        info.entities_vert_end.from_numpy(
            np.array([e.vert_start + e.n_verts_max for e in self._entities], dtype=gs.np_int)
        )
        info.entities_coarse_dof_start.from_numpy(self._entities_coarse_dof_start.astype(gs.np_int))
        info.entities_coarse_dim.from_numpy(self._entities_coarse_dim.astype(gs.np_int))
        info.entities_coarse_matrix_start.from_numpy(self._entities_coarse_matrix_start.astype(gs.np_int))
        coarse_dofs_entity = np.repeat(np.arange(len(self._entities), dtype=gs.np_int), self._entities_coarse_dim)
        info.coarse_dofs_entity.from_numpy(coarse_dofs_entity if len(coarse_dofs_entity) else np.zeros(1, gs.np_int))

        faces_entity = np.zeros(n_faces, dtype=gs.np_int)
        faces_mass = np.zeros(n_faces, dtype=gs.np_float)
        faces_rest_area = np.zeros(n_faces, dtype=gs.np_float)
        faces_Dm = np.zeros((n_faces, 2, 2), dtype=gs.np_float)
        faces_basis = np.zeros((n_faces, 3, 2), dtype=gs.np_float)
        faces_hinge = np.full((n_faces, 3), -1, dtype=gs.np_int)
        hinges_entity = np.zeros(max(n_hinges, 1), dtype=gs.np_int)
        # A scene without interior edge keeps one hinge whose endpoints never match across its faces, so that it never
        # reads as intact.
        hinges_corner = np.tile(np.array([0, 1, 0, 0], dtype=gs.np_int), (max(n_hinges, 1), 1))
        hinges_opposite_corner = np.zeros((max(n_hinges, 1), 2), dtype=gs.np_int)
        hinges_rest_angle = np.zeros(max(n_hinges, 1), dtype=gs.np_float)
        hinges_rest_len = np.ones(max(n_hinges, 1), dtype=gs.np_float)
        hinges_rest_area = np.ones(max(n_hinges, 1), dtype=gs.np_float)
        verts_coarse_dof = np.full(n_verts, -1, dtype=gs.np_int)
        verts_coarse_phi = np.zeros((n_verts, 3), dtype=gs.np_float)
        verts_fan_start = np.zeros(n_verts, dtype=gs.np_int)
        verts_fan_len = np.zeros(n_verts, dtype=gs.np_int)
        verts_is_fan_closed = np.zeros(n_verts, dtype=np.bool_)
        fans_corner = np.zeros(3 * n_faces, dtype=gs.np_int)
        fans_next_hinge = np.full(3 * n_faces, -1, dtype=gs.np_int)
        verts_pos = np.zeros((n_verts, 3), dtype=gs.np_float)
        verts_origin = np.full(n_verts, -1, dtype=gs.np_int)
        corners_vert = np.zeros(3 * n_faces, dtype=gs.np_int)
        entities_n_verts = np.zeros(len(self._entities), dtype=gs.np_int)

        for i_e, entity in enumerate(self._entities):
            topology = entity.topology
            v_start, f_start, h_start = entity.vert_start, entity.face_start, entity.hinge_start
            faces_slice = slice(f_start, f_start + entity.n_faces)
            hinges_slice = slice(h_start, h_start + entity.n_hinges)
            corners_slice = slice(3 * f_start, 3 * (f_start + entity.n_faces))
            verts_slice = slice(v_start, v_start + entity.n_verts)

            faces_entity[faces_slice] = i_e
            faces_mass[faces_slice] = topology.faces_mass
            faces_rest_area[faces_slice] = topology.faces_rest_area
            faces_Dm[faces_slice] = topology.faces_Dm
            faces_basis[faces_slice] = topology.faces_basis
            faces_hinge[faces_slice] = np.where(topology.faces_hinge >= 0, topology.faces_hinge + h_start, -1)
            hinges_entity[hinges_slice] = i_e
            hinges_corner[hinges_slice] = topology.hinges_corner + 3 * f_start
            hinges_opposite_corner[hinges_slice] = topology.hinges_opposite_corner + 3 * f_start
            hinges_rest_angle[hinges_slice] = topology.hinges_rest_angle
            hinges_rest_len[hinges_slice] = topology.hinges_rest_len
            hinges_rest_area[hinges_slice] = topology.hinges_rest_area
            if entity.patches is not None:
                verts_coarse_dof[verts_slice] = self._entities_coarse_dof_start[i_e] + 9 * entity.patches.verts_patch
                verts_coarse_phi[verts_slice] = entity.patches.verts_phi
            verts_fan_start[verts_slice] = topology.verts_fan_start + 3 * f_start
            verts_fan_len[verts_slice] = topology.verts_fan_len
            verts_is_fan_closed[verts_slice] = topology.verts_is_fan_closed
            fans_corner[corners_slice] = topology.fans_corner + 3 * f_start
            fans_next_hinge[corners_slice] = np.where(
                topology.fans_next_hinge >= 0, topology.fans_next_hinge + h_start, -1
            )
            verts_pos[verts_slice] = entity.init_verts
            verts_origin[verts_slice] = np.arange(v_start, v_start + entity.n_verts, dtype=gs.np_int)
            corners_vert[corners_slice] = entity.init_faces.reshape(-1) + v_start
            entities_n_verts[i_e] = entity.n_verts

        info.faces_entity.from_numpy(faces_entity)
        info.faces_mass.from_numpy(faces_mass)
        info.faces_rest_area.from_numpy(faces_rest_area)
        info.faces_Dm.from_numpy(faces_Dm)
        info.faces_Dm_inv.from_numpy(np.linalg.inv(faces_Dm))
        info.faces_basis.from_numpy(faces_basis)
        info.faces_hinge.from_numpy(faces_hinge)
        info.hinges_entity.from_numpy(hinges_entity)
        info.hinges_corner.from_numpy(hinges_corner)
        info.hinges_opposite_corner.from_numpy(hinges_opposite_corner)
        info.hinges_rest_angle.from_numpy(hinges_rest_angle)
        info.hinges_rest_len.from_numpy(hinges_rest_len)
        info.hinges_rest_area.from_numpy(hinges_rest_area)
        info.verts_coarse_dof.from_numpy(verts_coarse_dof)
        info.verts_coarse_phi.from_numpy(verts_coarse_phi)
        info.verts_fan_start.from_numpy(verts_fan_start)
        info.verts_fan_len.from_numpy(verts_fan_len)
        info.verts_is_fan_closed.from_numpy(verts_is_fan_closed)
        info.fans_corner.from_numpy(fans_corner)
        info.fans_next_hinge.from_numpy(fans_next_hinge)

        faces_thickness = np.array([entity.material.thickness for entity in self._entities], dtype=gs.np_float)
        state.verts_pos.from_numpy(np.ascontiguousarray(np.broadcast_to(verts_pos[:, None], (n_verts, B, 3))))
        verts_pos_cell = np.round(verts_pos / POS_GRID).astype(gs.np_int)
        state.verts_pos_cell.from_numpy(np.ascontiguousarray(np.broadcast_to(verts_pos_cell[:, None], (n_verts, B, 3))))
        verts_pos_offset = (verts_pos - verts_pos_cell * POS_GRID).astype(gs.np_float)
        state.verts_pos_offset.from_numpy(
            np.ascontiguousarray(np.broadcast_to(verts_pos_offset[:, None], (n_verts, B, 3)))
        )
        state.verts_vel.from_numpy(np.zeros((n_verts, B, 3), dtype=gs.np_float))
        state.verts_origin.from_numpy(np.ascontiguousarray(np.broadcast_to(verts_origin[:, None], (n_verts, B))))
        state.verts_is_fixed.from_numpy(np.zeros((n_verts, B), dtype=np.bool_))
        state.corners_vert.from_numpy(np.ascontiguousarray(np.broadcast_to(corners_vert[:, None], (3 * n_faces, B))))
        state.entities_n_verts.from_numpy(
            np.ascontiguousarray(np.broadcast_to(entities_n_verts[:, None], (len(self._entities), B)))
        )
        state.faces_plastic.from_numpy(
            np.ascontiguousarray(np.broadcast_to(np.eye(2, dtype=gs.np_float), (n_faces, B, 2, 2)))
        )
        state.faces_thickness.from_numpy(
            np.ascontiguousarray(np.broadcast_to(faces_thickness[faces_entity][:, None], (n_faces, B)))
        )
        state.hinges_plastic_angle.from_numpy(np.zeros((max(n_hinges, 1), B), dtype=gs.np_float))

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        pass

    def process_input_grad(self):
        pass

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        kernel_shell_compute_forces(
            self._substep_dt, self._gravity, self._shell_state, self._shell_scratch, self._shell_info
        )
        state, scratch, info, config = self._shell_state, self._shell_scratch, self._shell_info, self._static_config
        if config.has_coarse_space:
            if self._coarse_age is None or self._coarse_age >= self._coarse_update_interval:
                kernel_shell_coarse_assemble(state, scratch, info)
                kernel_shell_coarse_factorize(state, scratch, info, config)
                self._coarse_age = 0
            self._coarse_age += 1
        kernel_shell_system_product(scratch.verts_dv, scratch.verts_Ap, state, scratch, info)
        kernel_shell_pcg_init(self._pcg_threshold, state, scratch, info, config)
        for _ in range(self._n_pcg_iterations):
            kernel_shell_system_product(scratch.verts_p, scratch.verts_Ap, state, scratch, info)
            kernel_shell_pcg_update(state, scratch, info, config)
        kernel_shell_apply_dv(state, scratch)

    def substep_post_coupling(self, f):
        if not self.is_active:
            return
        kernel_shell_integrate(self._substep_dt, self._shell_state, self._shell_scratch)
        if self._static_config.has_plasticity:
            kernel_shell_plastic_flow(self._substep_dt, self._shell_state, self._shell_scratch, self._shell_info)
        if self._static_config.has_fracture:
            kernel_shell_fracture(self._shell_state, self._shell_scratch, self._shell_info)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling_grad(self, f):
        pass

    def reset_grad(self):
        pass

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f):
        if not self.is_active:
            return None
        state = self._shell_state
        return ShellSolverState(
            scene=self._scene,
            verts_pos=qd_to_torch(state.verts_pos, transpose=True, copy=True),
            verts_pos_cell=qd_to_torch(state.verts_pos_cell, transpose=True, copy=True),
            verts_pos_offset=qd_to_torch(state.verts_pos_offset, transpose=True, copy=True),
            verts_vel=qd_to_torch(state.verts_vel, transpose=True, copy=True),
            verts_origin=qd_to_torch(state.verts_origin, transpose=True, copy=True),
            verts_is_fixed=qd_to_torch(state.verts_is_fixed, transpose=True, copy=True),
            corners_vert=qd_to_torch(state.corners_vert, transpose=True, copy=True),
            entities_n_verts=qd_to_torch(state.entities_n_verts, transpose=True, copy=True),
            faces_plastic=qd_to_torch(state.faces_plastic, transpose=True, copy=True),
            faces_thickness=qd_to_torch(state.faces_thickness, transpose=True, copy=True),
            hinges_plastic_angle=qd_to_torch(state.hinges_plastic_angle, transpose=True, copy=True),
        )

    def set_state(self, f, state: ShellSolverState, envs_idx=None):
        if not self.is_active:
            return
        self._coarse_age = None
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        kernel_shell_set_state(
            envs_idx,
            state.verts_pos[envs_idx].contiguous(),
            state.verts_pos_cell[envs_idx].contiguous(),
            state.verts_pos_offset[envs_idx].contiguous(),
            state.verts_vel[envs_idx].contiguous(),
            state.verts_origin[envs_idx].contiguous(),
            state.verts_is_fixed[envs_idx].contiguous(),
            state.corners_vert[envs_idx].contiguous(),
            state.entities_n_verts[envs_idx].contiguous(),
            state.faces_plastic[envs_idx].contiguous(),
            state.faces_thickness[envs_idx].contiguous(),
            state.hinges_plastic_angle[envs_idx].contiguous(),
            self._shell_state,
        )

    def _sanitize_verts_idx(self, entity: ShellEntity, verts_idx_local, envs_idx):
        """Return the pool indices of the vertices of an entity, one row per environment of `envs_idx`."""
        if verts_idx_local is None:
            verts_idx_local = torch.arange(entity.n_verts, dtype=gs.tc_int, device=gs.device)
        verts_idx_local = torch.atleast_1d(torch.as_tensor(verts_idx_local, dtype=gs.tc_int, device=gs.device))
        if verts_idx_local.ndim == 1:
            verts_idx_local = verts_idx_local.expand((len(envs_idx), verts_idx_local.shape[0]))
        if ((verts_idx_local < 0) | (verts_idx_local >= entity.n_verts_max)).any():
            gs.raise_exception(f"Vertex indices must lie in [0, {entity.n_verts_max}), got {verts_idx_local}.")
        return (verts_idx_local + entity.vert_start).contiguous()

    def set_verts_pos(self, pos, entity: ShellEntity, verts_idx_local, envs_idx):
        """Set the position of some vertices of an entity."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = self._sanitize_verts_idx(entity, verts_idx_local, envs_idx)
        pos = broadcast_tensor(pos, gs.tc_float, (*verts_idx.shape, 3), ("envs_idx", "verts_idx", ""))
        kernel_shell_set_verts_pos(verts_idx, envs_idx, pos.contiguous(), self._shell_state)

    def set_verts_vel(self, vel, entity: ShellEntity, verts_idx_local, envs_idx):
        """Set the velocity of some vertices of an entity."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = self._sanitize_verts_idx(entity, verts_idx_local, envs_idx)
        vel = broadcast_tensor(vel, gs.tc_float, (*verts_idx.shape, 3), ("envs_idx", "verts_idx", ""))
        kernel_shell_set_verts_vel(verts_idx, envs_idx, vel.contiguous(), self._shell_state)

    def set_verts_fixed(self, is_fixed: bool, entity: ShellEntity, verts_idx_local, envs_idx):
        """Fix or release some vertices of an entity."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = self._sanitize_verts_idx(entity, verts_idx_local, envs_idx)
        kernel_shell_set_verts_fixed(verts_idx, envs_idx, is_fixed, self._shell_state)

    def update_render_fields(self):
        """Refresh the per-corner positions and normals the visualizer reads."""
        kernel_shell_update_render(self._shell_state, self._shell_scratch, self._shell_info)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_entities > 0

    @property
    def n_verts(self):
        if self.is_built:
            return self._n_verts
        return sum(entity.n_verts_max for entity in self._entities)

    @property
    def n_faces(self):
        if self.is_built:
            return self._n_faces
        return sum(entity.n_faces for entity in self._entities)

    @property
    def n_hinges(self):
        if self.is_built:
            return self._n_hinges
        return sum(entity.n_hinges for entity in self._entities)

    @property
    def fracture_capacity(self) -> float:
        return self._fracture_capacity

    @property
    def n_coarse_patches(self) -> int:
        return self._n_coarse_patches

    @property
    def shell_state(self) -> array_class.ShellState:
        return self._shell_state

    @property
    def shell_info(self) -> array_class.ShellInfo:
        return self._shell_info

    @property
    def shell_scratch(self) -> array_class.ShellScratch:
        return self._shell_scratch


# ------------------------------------------------------------------------------------
# ------------------------------------- helpers --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_vert_offset(i_v: int, i_u: int, i_b: int, shell_state: array_class.ShellState):
    """Position of vertex i_v relative to vertex i_u, exact up to the precision of the small anchored offsets.

    The difference of the integer grid cells is exact, and any reordering of the floating-point sum stays at the scale
    of the edge rather than of the absolute position.
    """
    cell_v = shell_state.verts_pos_cell[i_v, i_b]
    cell_u = shell_state.verts_pos_cell[i_u, i_b]
    return (cell_v - cell_u).cast(gs.qd_float) * POS_GRID + (
        shell_state.verts_pos_offset[i_v, i_b] - shell_state.verts_pos_offset[i_u, i_b]
    )


@qd.func
def func_sym2_eigen(S: qd.types.matrix(2, 2)):
    """Eigen-decompose a symmetric 2x2 matrix, returning the larger eigenvalue, the smaller one, and the unit
    eigenvector of the larger one (the other one being its counter-clockwise perpendicular)."""
    mean = 0.5 * (S[0, 0] + S[1, 1])
    half_diff = 0.5 * (S[0, 0] - S[1, 1])
    radius = qd.sqrt(half_diff * half_diff + S[0, 1] * S[0, 1])
    angle = 0.5 * qd.atan2(S[0, 1], half_diff)
    return mean + radius, mean - radius, qd.Vector([qd.cos(angle), qd.sin(angle)], dt=gs.qd_float)


@qd.func
def func_sym2_compose(lambda_0: float, lambda_1: float, eigvec_0: qd.types.vector(2)):
    """Compose the symmetric 2x2 matrix of eigenvalues lambda_0 along eigvec_0 and lambda_1 perpendicular to it."""
    eigvec_1 = qd.Vector([-eigvec_0[1], eigvec_0[0]], dt=gs.qd_float)
    return lambda_0 * eigvec_0.outer_product(eigvec_0) + lambda_1 * eigvec_1.outer_product(eigvec_1)


@qd.func
def func_membrane_stress(F: qd.types.matrix(3, 2), modulus: float, nu: float):
    """Plane-stress Saint Venant-Kirchhoff stress of a membrane, in N/m, for the deformation gradient of its face."""
    G = 0.5 * (F.transpose() @ F - qd.Matrix.identity(gs.qd_float, 2))
    return modulus * ((1.0 - nu) * G + nu * G.trace() * qd.Matrix.identity(gs.qd_float, 2))


@qd.func
def func_face_deformation(i_f: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Return the elastic deformation gradient of a face, from its rest frame to the world, and the matrix Y mapping
    the edge vectors of the face to it (F = [x1 - x0, x2 - x0] @ Y)."""
    i_v0 = shell_state.corners_vert[3 * i_f, i_b]
    i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
    i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
    Ds = qd.Matrix.cols(
        [func_vert_offset(i_v1, i_v0, i_b, shell_state), func_vert_offset(i_v2, i_v0, i_b, shell_state)]
    )
    Y = shell_info.faces_Dm_inv[i_f] @ shell_state.faces_plastic[i_f, i_b]
    return Ds @ Y, Y


@qd.func
def func_membrane_stiffness_product(
    dx0: qd.types.vector(3),
    dx1: qd.types.vector(3),
    dx2: qd.types.vector(3),
    F: qd.types.matrix(3, 2),
    stress_pos: qd.types.matrix(2, 2),
    Y: qd.types.matrix(2, 2),
    modulus: float,
    nu: float,
):
    """Product of the membrane stiffness of a face, per unit area, with a displacement of its three vertices.

    The stiffness keeps the geometric term of the positive part of the stress only, which makes it positive
    semi-definite: the material term is dG : C : dG and the geometric one stress_pos : dF^T dF, both non-negative.
    """
    dF = (dx1 - dx0).outer_product(qd.Vector([Y[0, 0], Y[0, 1]])) + (dx2 - dx0).outer_product(
        qd.Vector([Y[1, 0], Y[1, 1]])
    )
    dG = 0.5 * (dF.transpose() @ F + F.transpose() @ dF)
    dS = modulus * ((1.0 - nu) * dG + nu * dG.trace() * qd.Matrix.identity(gs.qd_float, 2))
    Q = (F @ dS + dF @ stress_pos) @ Y.transpose()
    q1 = qd.Vector([Q[0, 0], Q[1, 0], Q[2, 0]])
    q2 = qd.Vector([Q[0, 1], Q[1, 1], Q[2, 1]])
    return -(q1 + q2), q1, q2


@qd.func
def func_hinge_angle(edge_b: qd.types.vector(3), edge_c: qd.types.vector(3), edge_d: qd.types.vector(3)):
    """Signed dihedral angle of a hinge about its edge from a to b, the first face holding (a, b, c) and the second
    (b, a, d), zero when flat, from the positions of b, c and d relative to a."""
    cross_0 = edge_b.cross(edge_c)
    cross_1 = -edge_b.cross(edge_d - edge_b)
    normal_0 = cross_0 / qd.max(cross_0.norm(), NORM_FLOOR)
    normal_1 = cross_1 / qd.max(cross_1.norm(), NORM_FLOOR)
    edge = edge_b / qd.max(edge_b.norm(), NORM_FLOOR)
    return qd.atan2(edge.dot(normal_0.cross(normal_1)), normal_0.dot(normal_1))


@qd.func
def func_hinge_angle_gradient(edge_b: qd.types.vector(3), edge_c: qd.types.vector(3), edge_d: qd.types.vector(3)):
    """Gradient of the dihedral angle of a hinge (see func_hinge_angle) with respect to a, b, c and d, as columns."""
    edge_len = qd.max(edge_b.norm(), NORM_FLOOR)
    cross_0 = edge_b.cross(edge_c)
    cross_1 = -edge_b.cross(edge_d - edge_b)
    double_area_0 = qd.max(cross_0.norm(), NORM_FLOOR)
    double_area_1 = qd.max(cross_1.norm(), NORM_FLOOR)
    # The gradient at an opposite vertex is the normal of its face over its height above the edge.
    grad_c = -cross_0 / double_area_0 * (edge_len / double_area_0)
    grad_d = -cross_1 / double_area_1 * (edge_len / double_area_1)
    s_c = edge_c.dot(edge_b) / (edge_len * edge_len)
    s_d = edge_d.dot(edge_b) / (edge_len * edge_len)
    grad_a = -((1.0 - s_c) * grad_c + (1.0 - s_d) * grad_d)
    grad_b = -(s_c * grad_c + s_d * grad_d)
    return qd.Matrix.cols([grad_a, grad_b, grad_c, grad_d])


@qd.func
def func_hinge_edges(i_va: int, i_vb: int, i_vc: int, i_vd: int, i_b: int, shell_state: array_class.ShellState):
    """Positions of the vertices b, c and d of a hinge relative to its vertex a (see func_vert_offset)."""
    return (
        func_vert_offset(i_vb, i_va, i_b, shell_state),
        func_vert_offset(i_vc, i_va, i_b, shell_state),
        func_vert_offset(i_vd, i_va, i_b, shell_state),
    )


@qd.func
def func_hinge_verts(i_h: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Return the vertices a, b, c, d of a hinge (see func_hinge_angle), and whether its two faces still share their
    edge."""
    corners = shell_info.hinges_corner[i_h]
    opposite_corners = shell_info.hinges_opposite_corner[i_h]
    i_va = shell_state.corners_vert[corners[0], i_b]
    i_vb = shell_state.corners_vert[corners[1], i_b]
    i_va_1 = shell_state.corners_vert[corners[2], i_b]
    i_vb_1 = shell_state.corners_vert[corners[3], i_b]
    i_vc = shell_state.corners_vert[opposite_corners[0], i_b]
    i_vd = shell_state.corners_vert[opposite_corners[1], i_b]
    return i_va, i_vb, i_vc, i_vd, i_va == i_va_1 and i_vb == i_vb_1


@qd.func
def func_hinge_stiffness(i_h: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Bending stiffness of a hinge, in N*m per rad^2, such that its energy is 0.5 * k * (angle - rest angle)^2.

    The factor rest_len^2 / rest_area makes the hinges of a regular mesh sum up to the bending energy of an isotropic
    plate, D / 2 * curvature^2 per unit area.
    """
    i_e = shell_info.hinges_entity[i_h]
    corners = shell_info.hinges_corner[i_h]
    thickness = 0.5 * (
        shell_state.faces_thickness[corners[0] // 3, i_b] + shell_state.faces_thickness[corners[2] // 3, i_b]
    )
    rest_len = shell_info.hinges_rest_len[i_h]
    return (
        shell_info.entities_bending_modulus[i_e] * thickness**3 * rest_len * rest_len / shell_info.hinges_rest_area[i_h]
    )


@qd.func
def func_is_vert_free(i_v: int, i_b: int, shell_state: array_class.ShellState):
    """Whether a pool slot holds a vertex that the forces move."""
    return shell_state.verts_origin[i_v, i_b] >= 0 and not shell_state.verts_is_fixed[i_v, i_b]


# ------------------------------------------------------------------------------------
# -------------------------------- implicit dynamics ---------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_shell_compute_forces(
    dt: float,
    gravity: qd.Tensor,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Assemble the right-hand side dt * (f + M g) - K v of the velocity update, the diagonal blocks of M + K, and the
    per-element data the stiffness products read (see ShellScratch)."""
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_scratch.hinges_stiffness.shape[0]

    for i_v, i_b in qd.ndrange(n_verts, B):
        shell_scratch.verts_mass[i_v, i_b] = 0.0
        shell_scratch.verts_rhs[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        shell_scratch.verts_prec[i_v, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)

    for i_c, i_b in qd.ndrange(3 * n_faces, B):
        i_v = shell_state.corners_vert[i_c, i_b]
        shell_scratch.verts_mass[i_v, i_b] += shell_info.faces_mass[i_c // 3] / 3.0

    for i_f, i_b in qd.ndrange(n_faces, B):
        i_e = shell_info.faces_entity[i_f]
        modulus = shell_info.entities_stretching_modulus[i_e] * shell_state.faces_thickness[i_f, i_b]
        nu = shell_info.entities_nu[i_e]
        rest_area = shell_info.faces_rest_area[i_f]
        F, Y = func_face_deformation(i_f, i_b, shell_state, shell_info)
        stress = func_membrane_stress(F, modulus, nu)
        lambda_0, lambda_1, eigvec_0 = func_sym2_eigen(stress)
        stress_pos = func_sym2_compose(qd.max(lambda_0, 0.0), qd.max(lambda_1, 0.0), eigvec_0)
        stiffness = dt * (dt + shell_info.entities_damping[i_e]) * rest_area
        shell_scratch.faces_F[i_f, i_b] = F
        shell_scratch.faces_stress[i_f, i_b] = stress_pos
        shell_scratch.faces_stiffness[i_f, i_b] = stiffness

        i_v0 = shell_state.corners_vert[3 * i_f, i_b]
        i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
        i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
        # Elastic force: minus the gradient of rest_area / 2 * G : stress, rest_area * F @ stress being its gradient
        # with respect to F.
        Q = rest_area * (F @ stress) @ Y.transpose()
        force_1 = -qd.Vector([Q[0, 0], Q[1, 0], Q[2, 0]])
        force_2 = -qd.Vector([Q[0, 1], Q[1, 1], Q[2, 1]])
        Kv0, Kv1, Kv2 = func_membrane_stiffness_product(
            shell_state.verts_vel[i_v0, i_b],
            shell_state.verts_vel[i_v1, i_b],
            shell_state.verts_vel[i_v2, i_b],
            F,
            stress_pos,
            Y,
            modulus,
            nu,
        )
        shell_scratch.verts_rhs[i_v0, i_b] += -dt * (force_1 + force_2) - stiffness * Kv0
        shell_scratch.verts_rhs[i_v1, i_b] += dt * force_1 - stiffness * Kv1
        shell_scratch.verts_rhs[i_v2, i_b] += dt * force_2 - stiffness * Kv2

        # Diagonal blocks of the stiffness: displacing vertex k alone changes F by dx * w_k^T.
        for k in qd.static(range(3)):
            w = -qd.Vector([Y[0, 0] + Y[1, 0], Y[0, 1] + Y[1, 1]])
            if qd.static(k > 0):
                w = qd.Vector([Y[k - 1, 0], Y[k - 1, 1]])
            M = modulus * (
                0.5 * (1.0 - nu) * w.dot(w) * qd.Matrix.identity(gs.qd_float, 2)
                + (0.5 * (1.0 - nu) + nu) * w.outer_product(w)
            )
            H = F @ M @ F.transpose() + w.dot(stress_pos @ w) * qd.Matrix.identity(gs.qd_float, 3)
            i_v = shell_state.corners_vert[3 * i_f + k, i_b]
            shell_scratch.verts_prec[i_v, i_b] += stiffness * H

    for i_h, i_b in qd.ndrange(n_hinges, B):
        shell_scratch.hinges_stiffness[i_h, i_b] = 0.0
        i_va, i_vb, i_vc, i_vd, is_intact = func_hinge_verts(i_h, i_b, shell_state, shell_info)
        if is_intact:
            edge_b, edge_c, edge_d = func_hinge_edges(i_va, i_vb, i_vc, i_vd, i_b, shell_state)
            angle = func_hinge_angle(edge_b, edge_c, edge_d)
            grad = func_hinge_angle_gradient(edge_b, edge_c, edge_d)
            i_e = shell_info.hinges_entity[i_h]
            k_bend = func_hinge_stiffness(i_h, i_b, shell_state, shell_info)
            rest_angle = shell_info.hinges_rest_angle[i_h] + shell_state.hinges_plastic_angle[i_h, i_b]
            stiffness = dt * (dt + shell_info.entities_damping[i_e]) * k_bend
            shell_scratch.hinges_grad[i_h, i_b] = grad
            shell_scratch.hinges_stiffness[i_h, i_b] = stiffness

            grad_dot_vel = (
                grad[:, 0].dot(shell_state.verts_vel[i_va, i_b])
                + grad[:, 1].dot(shell_state.verts_vel[i_vb, i_b])
                + grad[:, 2].dot(shell_state.verts_vel[i_vc, i_b])
                + grad[:, 3].dot(shell_state.verts_vel[i_vd, i_b])
            )
            coeff = -dt * k_bend * (angle - rest_angle) - stiffness * grad_dot_vel
            shell_scratch.verts_rhs[i_va, i_b] += coeff * grad[:, 0]
            shell_scratch.verts_rhs[i_vb, i_b] += coeff * grad[:, 1]
            shell_scratch.verts_rhs[i_vc, i_b] += coeff * grad[:, 2]
            shell_scratch.verts_rhs[i_vd, i_b] += coeff * grad[:, 3]
            shell_scratch.verts_prec[i_va, i_b] += stiffness * grad[:, 0].outer_product(grad[:, 0])
            shell_scratch.verts_prec[i_vb, i_b] += stiffness * grad[:, 1].outer_product(grad[:, 1])
            shell_scratch.verts_prec[i_vc, i_b] += stiffness * grad[:, 2].outer_product(grad[:, 2])
            shell_scratch.verts_prec[i_vd, i_b] += stiffness * grad[:, 3].outer_product(grad[:, 3])

    for i_b in range(B):
        shell_scratch.envs_is_solving[i_b] = True

    for i_v, i_b in qd.ndrange(n_verts, B):
        mass = shell_scratch.verts_mass[i_v, i_b]
        if func_is_vert_free(i_v, i_b, shell_state) and mass > 0.0:
            shell_scratch.verts_rhs[i_v, i_b] += dt * mass * gravity[i_b]
            shell_scratch.verts_prec[i_v, i_b] = (
                shell_scratch.verts_prec[i_v, i_b] + mass * qd.Matrix.identity(gs.qd_float, 3)
            ).inverse()
        else:
            shell_scratch.verts_rhs[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            shell_scratch.verts_prec[i_v, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            shell_scratch.verts_dv[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_shell_system_product(
    src: qd.Tensor,
    dst: qd.Tensor,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Write dst = (M + K) src in every environment still solving, matrix-free, zero on fixed and empty slots."""
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_scratch.hinges_stiffness.shape[0]

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_is_solving[i_b]:
            dst[i_v, i_b] = shell_scratch.verts_mass[i_v, i_b] * src[i_v, i_b]

    for i_f, i_b in qd.ndrange(n_faces, B):
        if shell_scratch.envs_is_solving[i_b]:
            i_e = shell_info.faces_entity[i_f]
            i_v0 = shell_state.corners_vert[3 * i_f, i_b]
            i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
            i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
            Y = shell_info.faces_Dm_inv[i_f] @ shell_state.faces_plastic[i_f, i_b]
            stiffness = shell_scratch.faces_stiffness[i_f, i_b]
            Kp0, Kp1, Kp2 = func_membrane_stiffness_product(
                src[i_v0, i_b],
                src[i_v1, i_b],
                src[i_v2, i_b],
                shell_scratch.faces_F[i_f, i_b],
                shell_scratch.faces_stress[i_f, i_b],
                Y,
                shell_info.entities_stretching_modulus[i_e] * shell_state.faces_thickness[i_f, i_b],
                shell_info.entities_nu[i_e],
            )
            dst[i_v0, i_b] += stiffness * Kp0
            dst[i_v1, i_b] += stiffness * Kp1
            dst[i_v2, i_b] += stiffness * Kp2

    for i_h, i_b in qd.ndrange(n_hinges, B):
        stiffness = shell_scratch.hinges_stiffness[i_h, i_b]
        if shell_scratch.envs_is_solving[i_b] and stiffness > 0.0:
            i_va, i_vb, i_vc, i_vd, _ = func_hinge_verts(i_h, i_b, shell_state, shell_info)
            grad = shell_scratch.hinges_grad[i_h, i_b]
            coeff = stiffness * (
                grad[:, 0].dot(src[i_va, i_b])
                + grad[:, 1].dot(src[i_vb, i_b])
                + grad[:, 2].dot(src[i_vc, i_b])
                + grad[:, 3].dot(src[i_vd, i_b])
            )
            dst[i_va, i_b] += coeff * grad[:, 0]
            dst[i_vb, i_b] += coeff * grad[:, 1]
            dst[i_vc, i_b] += coeff * grad[:, 2]
            dst[i_vd, i_b] += coeff * grad[:, 3]

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_is_solving[i_b] and not func_is_vert_free(i_v, i_b, shell_state):
            dst[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_coarse_correct(
    shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch, shell_info: array_class.ShellInfo
):
    """Solve the coarse system for the residual verts_r of every environment still solving, leaving the coarse
    correction in coarse_sol (see the coarse space in kernel_shell_coarse_factorize)."""
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]

    for i_d, i_b in qd.ndrange(shell_scratch.coarse_vec.shape[0], B):
        shell_scratch.coarse_vec[i_d, i_b] = 0.0

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_is_solving[i_b] and func_is_vert_free(i_v, i_b, shell_state):
            i_o = shell_state.verts_origin[i_v, i_b]
            i_d = shell_info.verts_coarse_dof[i_o]
            if i_d >= 0:
                phi = shell_info.verts_coarse_phi[i_o]
                r = shell_scratch.verts_r[i_v, i_b]
                for a, j in qd.static(qd.ndrange(3, 3)):
                    shell_scratch.coarse_vec[i_d + 3 * a + j, i_b] += phi[a] * r[j]

    for i_d, i_b in qd.ndrange(shell_scratch.coarse_vec.shape[0], B):
        if shell_scratch.envs_is_solving[i_b]:
            i_e = shell_info.coarse_dofs_entity[i_d]
            dof_start = shell_info.entities_coarse_dof_start[i_e]
            dim = shell_info.entities_coarse_dim[i_e]
            row_start = shell_info.entities_coarse_matrix_start[i_e] + (i_d - dof_start) * dim
            value = gs.qd_float(0.0)
            for j in range(dim):
                value += shell_scratch.coarse_matrix[i_b, row_start + j] * shell_scratch.coarse_vec[dof_start + j, i_b]
            shell_scratch.coarse_sol[i_d, i_b] = value


@qd.func
def func_precondition(
    i_v: int,
    i_b: int,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    static_config: qd.template(),
):
    """Preconditioned residual of a vertex: its block-Jacobi part plus the prolongation of the coarse correction."""
    z = shell_scratch.verts_prec[i_v, i_b] @ shell_scratch.verts_r[i_v, i_b]
    if qd.static(static_config.has_coarse_space):
        if func_is_vert_free(i_v, i_b, shell_state):
            i_o = shell_state.verts_origin[i_v, i_b]
            i_d = shell_info.verts_coarse_dof[i_o]
            if i_d >= 0:
                phi = shell_info.verts_coarse_phi[i_o]
                for a, j in qd.static(qd.ndrange(3, 3)):
                    z[j] += phi[a] * shell_scratch.coarse_sol[i_d + 3 * a + j, i_b]
    return z


@qd.kernel
def kernel_shell_pcg_init(
    pcg_threshold: float,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    static_config: qd.template(),
):
    """Start the PCG solve of the velocity update.

    The solve starts from the velocity change of the previous substep, which the system product just mapped to
    verts_Ap, since the acceleration of smooth motion varies little from one substep to the next. The preconditioner
    is block-Jacobi, plus the coarse correction of the vertex patches if enabled.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]

    for i_b in range(B):
        shell_scratch.envs_rz[i_b] = 0.0
        shell_scratch.envs_rz_threshold[i_b] = 0.0

    for i_v, i_b in qd.ndrange(n_verts, B):
        shell_scratch.verts_r[i_v, i_b] = shell_scratch.verts_rhs[i_v, i_b] - shell_scratch.verts_Ap[i_v, i_b]
        # The threshold is relative to the norm of the preconditioned right-hand side, which a warm start leaves out of
        # the initial residual.
        shell_scratch.envs_rz_threshold[i_b] += shell_scratch.verts_rhs[i_v, i_b].dot(
            shell_scratch.verts_prec[i_v, i_b] @ shell_scratch.verts_rhs[i_v, i_b]
        )

    if qd.static(static_config.has_coarse_space):
        func_coarse_correct(shell_state, shell_scratch, shell_info)

    for i_v, i_b in qd.ndrange(n_verts, B):
        z = func_precondition(i_v, i_b, shell_state, shell_scratch, shell_info, static_config)
        shell_scratch.verts_z[i_v, i_b] = z
        shell_scratch.verts_p[i_v, i_b] = z
        shell_scratch.envs_rz[i_b] += shell_scratch.verts_r[i_v, i_b].dot(z)

    for i_b in range(B):
        shell_scratch.envs_rz_threshold[i_b] = pcg_threshold * pcg_threshold * shell_scratch.envs_rz_threshold[i_b]
        shell_scratch.envs_is_solving[i_b] = shell_scratch.envs_rz[i_b] > shell_scratch.envs_rz_threshold[i_b]


@qd.kernel
def kernel_shell_pcg_update(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    static_config: qd.template(),
):
    """Finish one PCG iteration in every environment still solving, from the system product verts_Ap of verts_p."""
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]

    for i_b in range(B):
        shell_scratch.envs_step[i_b] = 0.0
        shell_scratch.envs_rz_new[i_b] = 0.0

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_is_solving[i_b]:
            shell_scratch.envs_step[i_b] += shell_scratch.verts_p[i_v, i_b].dot(shell_scratch.verts_Ap[i_v, i_b])

    # The system being positive definite on the free vertices, p^T A p vanishes only once p does.
    for i_b in range(B):
        if shell_scratch.envs_is_solving[i_b]:
            if shell_scratch.envs_step[i_b] > 0.0:
                shell_scratch.envs_step[i_b] = shell_scratch.envs_rz[i_b] / shell_scratch.envs_step[i_b]
            else:
                shell_scratch.envs_is_solving[i_b] = False

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_is_solving[i_b]:
            alpha = shell_scratch.envs_step[i_b]
            shell_scratch.verts_dv[i_v, i_b] += alpha * shell_scratch.verts_p[i_v, i_b]
            shell_scratch.verts_r[i_v, i_b] -= alpha * shell_scratch.verts_Ap[i_v, i_b]

    if qd.static(static_config.has_coarse_space):
        func_coarse_correct(shell_state, shell_scratch, shell_info)

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_is_solving[i_b]:
            z = func_precondition(i_v, i_b, shell_state, shell_scratch, shell_info, static_config)
            shell_scratch.verts_z[i_v, i_b] = z
            shell_scratch.envs_rz_new[i_b] += shell_scratch.verts_r[i_v, i_b].dot(z)

    for i_b in range(B):
        if shell_scratch.envs_is_solving[i_b]:
            shell_scratch.envs_step[i_b] = shell_scratch.envs_rz_new[i_b] / shell_scratch.envs_rz[i_b]
            shell_scratch.envs_rz[i_b] = shell_scratch.envs_rz_new[i_b]

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_is_solving[i_b]:
            shell_scratch.verts_p[i_v, i_b] = (
                shell_scratch.verts_z[i_v, i_b] + shell_scratch.envs_step[i_b] * shell_scratch.verts_p[i_v, i_b]
            )

    for i_b in range(B):
        if shell_scratch.envs_rz[i_b] <= shell_scratch.envs_rz_threshold[i_b]:
            shell_scratch.envs_is_solving[i_b] = False


@qd.func
def func_select3(i: int, x0, x1, x2):
    """Return x0, x1 or x2 by index, for indexing local values at runtime."""
    x = x0
    if i == 1:
        x = x1
    elif i == 2:
        x = x2
    return x


@qd.func
def func_coarse_add_block(
    i_e: int,
    i_d_row: int,
    i_d_col: int,
    block: qd.types.matrix(3, 3),
    i_b: int,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Add a 3x3 block to the lower triangle of the coarse matrix of an entity, at global coarse rows and columns."""
    dof_start = shell_info.entities_coarse_dof_start[i_e]
    dim = shell_info.entities_coarse_dim[i_e]
    matrix_start = shell_info.entities_coarse_matrix_start[i_e]
    for j, k in qd.static(qd.ndrange(3, 3)):
        row = i_d_row - dof_start + j
        col = i_d_col - dof_start + k
        if row >= col:
            shell_scratch.coarse_assembly[matrix_start + row * dim + col, i_b] += block[j, k]


@qd.kernel
def kernel_shell_coarse_assemble(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Assemble the coarse matrix Z^T (M + K) Z of every entity, Z spanning the displacements affine in
    the rest coordinates of each patch of vertices.

    Every patch of vertices moves by a displacement affine in the in-plane rest coordinates of its vertices, which
    captures the smooth stretching and bending that block-Jacobi preconditioning resolves slowly in stiff sheets. The
    stiffness of an element being bilinear in the displacement of its vertices, its coarse block for a pair of shape
    functions is its stiffness evaluated on the sum of the vertex weights times those shape functions, per patch.
    The Cholesky factorization drops the pivots that vanish (fixed or degenerate patches), solving the coarse system
    on the remaining unknowns.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_scratch.hinges_stiffness.shape[0]
    n_entities = shell_state.entities_n_verts.shape[0]

    for i_m, i_b in qd.ndrange(shell_scratch.coarse_assembly.shape[0], B):
        shell_scratch.coarse_assembly[i_m, i_b] = 0.0

    for i_v, i_b in qd.ndrange(n_verts, B):
        if func_is_vert_free(i_v, i_b, shell_state):
            i_o = shell_state.verts_origin[i_v, i_b]
            i_d = shell_info.verts_coarse_dof[i_o]
            if i_d >= 0:
                i_e = func_vert_entity(i_v, i_b, shell_state, shell_info)
                phi = shell_info.verts_coarse_phi[i_o]
                mass = shell_scratch.verts_mass[i_v, i_b]
                for a in range(3):
                    for b in range(a + 1):
                        coeff = mass * func_select3(a, phi[0], phi[1], phi[2]) * func_select3(b, phi[0], phi[1], phi[2])
                        func_coarse_add_block(
                            i_e,
                            i_d + 3 * a,
                            i_d + 3 * b,
                            coeff * qd.Matrix.identity(gs.qd_float, 3),
                            i_b,
                            shell_scratch,
                            shell_info,
                        )

    # Faces: the coarse block of shape functions (a of patch P, b of patch Q) evaluates the membrane stiffness on the
    # weights W_Pa = sum over the free face vertices k of patch P of phi_k[a] * w_k, w_k mapping a displacement of
    # vertex k to the change of deformation gradient (see func_membrane_stiffness_product).
    for i_f, i_b in qd.ndrange(n_faces, B):
        i_e = shell_info.faces_entity[i_f]
        if shell_info.entities_coarse_dim[i_e] > 0:
            nu = shell_info.entities_nu[i_e]
            modulus = shell_info.entities_stretching_modulus[i_e] * shell_state.faces_thickness[i_f, i_b]
            stiffness = shell_scratch.faces_stiffness[i_f, i_b]
            F = shell_scratch.faces_F[i_f, i_b]
            stress_pos = shell_scratch.faces_stress[i_f, i_b]
            Y = shell_info.faces_Dm_inv[i_f] @ shell_state.faces_plastic[i_f, i_b]
            w_1 = qd.Vector([Y[0, 0], Y[0, 1]])
            w_2 = qd.Vector([Y[1, 0], Y[1, 1]])
            w_0 = -(w_1 + w_2)
            i_v0 = shell_state.corners_vert[3 * i_f, i_b]
            i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
            i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
            is_free_0 = func_is_vert_free(i_v0, i_b, shell_state)
            is_free_1 = func_is_vert_free(i_v1, i_b, shell_state)
            is_free_2 = func_is_vert_free(i_v2, i_b, shell_state)
            i_o0 = shell_state.verts_origin[i_v0, i_b]
            i_o1 = shell_state.verts_origin[i_v1, i_b]
            i_o2 = shell_state.verts_origin[i_v2, i_b]
            dof_0 = shell_info.verts_coarse_dof[i_o0]
            dof_1 = shell_info.verts_coarse_dof[i_o1]
            dof_2 = shell_info.verts_coarse_dof[i_o2]
            phi_0 = shell_info.verts_coarse_phi[i_o0]
            phi_1 = shell_info.verts_coarse_phi[i_o1]
            phi_2 = shell_info.verts_coarse_phi[i_o2]
            for k, l, a, b in qd.ndrange(3, 3, 3, 3):
                dof_k = func_select3(k, dof_0, dof_1, dof_2)
                dof_l = func_select3(l, dof_0, dof_1, dof_2)
                # Each patch of the face is handled by its first vertex, and the pair of patches once, in the lower
                # triangle.
                is_leader_k = k == 0 or (k == 1 and dof_1 != dof_0) or (k == 2 and dof_2 != dof_0 and dof_2 != dof_1)
                is_leader_l = l == 0 or (l == 1 and dof_1 != dof_0) or (l == 2 and dof_2 != dof_0 and dof_2 != dof_1)
                if is_leader_k and is_leader_l and dof_k >= dof_l:
                    W_k = qd.Vector.zero(gs.qd_float, 2)
                    W_l = qd.Vector.zero(gs.qd_float, 2)
                    if is_free_0:
                        W_k += (dof_0 == dof_k) * func_select3(a, phi_0[0], phi_0[1], phi_0[2]) * w_0
                        W_l += (dof_0 == dof_l) * func_select3(b, phi_0[0], phi_0[1], phi_0[2]) * w_0
                    if is_free_1:
                        W_k += (dof_1 == dof_k) * func_select3(a, phi_1[0], phi_1[1], phi_1[2]) * w_1
                        W_l += (dof_1 == dof_l) * func_select3(b, phi_1[0], phi_1[1], phi_1[2]) * w_1
                    if is_free_2:
                        W_k += (dof_2 == dof_k) * func_select3(a, phi_2[0], phi_2[1], phi_2[2]) * w_2
                        W_l += (dof_2 == dof_l) * func_select3(b, phi_2[0], phi_2[1], phi_2[2]) * w_2
                    M = modulus * (
                        0.5 * (1.0 - nu) * (W_k.dot(W_l) * qd.Matrix.identity(gs.qd_float, 2) + W_l.outer_product(W_k))
                        + nu * W_k.outer_product(W_l)
                    )
                    block = stiffness * (
                        F @ M @ F.transpose() + W_k.dot(stress_pos @ W_l) * qd.Matrix.identity(gs.qd_float, 3)
                    )
                    if dof_k >= 0 and dof_l >= 0:
                        func_coarse_add_block(i_e, dof_k + 3 * a, dof_l + 3 * b, block, i_b, shell_scratch, shell_info)

    # Hinges: the same with the gradient of the dihedral angle, the stiffness being k * grad grad^T
    for i_h, i_b in qd.ndrange(n_hinges, B):
        stiffness = shell_scratch.hinges_stiffness[i_h, i_b]
        if stiffness > 0.0:
            i_e = shell_info.hinges_entity[i_h]
            if shell_info.entities_coarse_dim[i_e] > 0:
                i_va, i_vb, i_vc, i_vd, _ = func_hinge_verts(i_h, i_b, shell_state, shell_info)
                grad = shell_scratch.hinges_grad[i_h, i_b]
                for k, l, a, b in qd.ndrange(4, 4, 3, 3):
                    i_vk = i_va
                    i_vl = i_va
                    G_k = qd.Vector.zero(gs.qd_float, 3)
                    G_l = qd.Vector.zero(gs.qd_float, 3)
                    is_leader_k = True
                    is_leader_l = True
                    dof_k = -1
                    dof_l = -1
                    for m in qd.static(range(4)):
                        i_vm = i_va
                        if qd.static(m == 1):
                            i_vm = i_vb
                        elif qd.static(m == 2):
                            i_vm = i_vc
                        elif qd.static(m == 3):
                            i_vm = i_vd
                        if m == k:
                            i_vk = i_vm
                        if m == l:
                            i_vl = i_vm
                    i_ok = shell_state.verts_origin[i_vk, i_b]
                    i_ol = shell_state.verts_origin[i_vl, i_b]
                    dof_k = shell_info.verts_coarse_dof[i_ok]
                    dof_l = shell_info.verts_coarse_dof[i_ol]
                    for m in qd.static(range(4)):
                        i_vm = i_va
                        if qd.static(m == 1):
                            i_vm = i_vb
                        elif qd.static(m == 2):
                            i_vm = i_vc
                        elif qd.static(m == 3):
                            i_vm = i_vd
                        i_om = shell_state.verts_origin[i_vm, i_b]
                        dof_m = shell_info.verts_coarse_dof[i_om]
                        if m < k and dof_m == dof_k:
                            is_leader_k = False
                        if m < l and dof_m == dof_l:
                            is_leader_l = False
                        if func_is_vert_free(i_vm, i_b, shell_state):
                            phi_m = shell_info.verts_coarse_phi[i_om]
                            grad_m = qd.Vector([grad[0, m], grad[1, m], grad[2, m]])
                            if dof_m == dof_k:
                                G_k += func_select3(a, phi_m[0], phi_m[1], phi_m[2]) * grad_m
                            if dof_m == dof_l:
                                G_l += func_select3(b, phi_m[0], phi_m[1], phi_m[2]) * grad_m
                    if is_leader_k and is_leader_l and dof_k >= dof_l and dof_l >= 0:
                        func_coarse_add_block(
                            i_e,
                            dof_k + 3 * a,
                            dof_l + 3 * b,
                            stiffness * G_k.outer_product(G_l),
                            i_b,
                            shell_scratch,
                            shell_info,
                        )


@qd.kernel
def kernel_shell_coarse_factorize(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    static_config: qd.template(),
):
    """Factorize and invert the coarse matrix of every entity (see kernel_shell_coarse_assemble)."""
    B = shell_state.verts_pos.shape[1]
    n_entities = shell_state.entities_n_verts.shape[0]

    # Cholesky factorization, dropping the pivots that vanish (fixed or degenerate patches) so that the coarse system is
    # solved on the remaining unknowns, then explicit inversion, which turns the coarse solve of every iteration into a
    # parallel matrix-vector product. The lanes of a block share the rows of one entity in one environment.
    _K = qd.static(static_config.coarse_block_dim)
    if qd.static(_K > 1):
        qd.loop_config(block_dim=_K)
    for i_flat in range(n_entities * B * _K):
        tid = i_flat % _K
        i_b = (i_flat // _K) % B
        i_e = i_flat // (_K * B)
        dim = shell_info.entities_coarse_dim[i_e]
        matrix_start = shell_info.entities_coarse_matrix_start[i_e]
        for i_chunk in range((dim * dim + _K - 1) // _K):
            i_entry = i_chunk * _K + tid
            if i_entry < dim * dim:
                shell_scratch.coarse_matrix[i_b, matrix_start + i_entry] = shell_scratch.coarse_assembly[
                    matrix_start + i_entry, i_b
                ]
        if qd.static(_K > 1):
            qd.simt.block.sync()
        for j in range(dim):
            if tid == 0:
                diag = shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + j]
                pivot_sq = diag
                for k in range(j):
                    pivot_sq -= shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + k] ** 2
                pivot = gs.qd_float(0.0)
                if pivot_sq > 1e-6 * diag:
                    pivot = qd.sqrt(pivot_sq)
                shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + j] = pivot
            if qd.static(_K > 1):
                qd.simt.block.sync()
            pivot = shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + j]
            for i_chunk in range((dim - j - 1 + _K - 1) // _K):
                i = j + 1 + i_chunk * _K + tid
                if i < dim:
                    value = gs.qd_float(0.0)
                    if pivot > 0.0:
                        value = shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + j]
                        for k in range(j):
                            value -= (
                                shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + k]
                                * shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + k]
                            )
                        value = value / pivot
                    shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + j] = value
            if qd.static(_K > 1):
                qd.simt.block.sync()

        # Inverse of the lower triangular factor, one column per lane by forward substitution
        for k_chunk in range((dim + _K - 1) // _K):
            k = k_chunk * _K + tid
            if k < dim:
                for i in range(k, dim):
                    value = gs.qd_float(1.0) if i == k else gs.qd_float(0.0)
                    for m in range(k, i):
                        value -= (
                            shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + m]
                            * shell_scratch.coarse_factor_inv[i_b, matrix_start + m * dim + k]
                        )
                    pivot = shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + i]
                    shell_scratch.coarse_factor_inv[i_b, matrix_start + i * dim + k] = (
                        value / pivot if pivot > 0.0 else 0.0
                    )
        if qd.static(_K > 1):
            qd.simt.block.sync()

        # Inverse of the coarse matrix, L^-T L^-1, written over the factor that is no longer needed
        for i_chunk in range((dim * dim + _K - 1) // _K):
            i_entry = i_chunk * _K + tid
            if i_entry < dim * dim:
                i = i_entry // dim
                j = i_entry % dim
                value = gs.qd_float(0.0)
                for k in range(qd.max(i, j), dim):
                    value += (
                        shell_scratch.coarse_factor_inv[i_b, matrix_start + k * dim + i]
                        * shell_scratch.coarse_factor_inv[i_b, matrix_start + k * dim + j]
                    )
                shell_scratch.coarse_matrix[i_b, matrix_start + i_entry] = value
        if qd.static(_K > 1):
            qd.simt.block.sync()


@qd.kernel
def kernel_shell_apply_dv(shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch):
    """Add the solved velocity change to the free vertices."""
    for i_v, i_b in qd.ndrange(shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]):
        if func_is_vert_free(i_v, i_b, shell_state):
            shell_state.verts_vel[i_v, i_b] += shell_scratch.verts_dv[i_v, i_b]


@qd.kernel
def kernel_shell_integrate(dt: float, shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch):
    """Advance the position of every vertex by its velocity, moving whole grid cells from its offset to its cell.

    The offset staying below a cell, its increments keep their precision, and moving whole cells out of it is exact.
    """
    for i_v, i_b in qd.ndrange(shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]):
        if shell_state.verts_origin[i_v, i_b] >= 0:
            offset = shell_state.verts_pos_offset[i_v, i_b] + dt * shell_state.verts_vel[i_v, i_b]
            cell_shift = qd.floor(offset / POS_GRID + 0.5).cast(gs.qd_int)
            cell = shell_state.verts_pos_cell[i_v, i_b] + cell_shift
            offset = offset - cell_shift.cast(gs.qd_float) * POS_GRID
            shell_state.verts_pos_cell[i_v, i_b] = cell
            shell_state.verts_pos_offset[i_v, i_b] = offset
            shell_state.verts_pos[i_v, i_b] = cell.cast(gs.qd_float) * POS_GRID + offset


# ------------------------------------------------------------------------------------
# ----------------------------------- plasticity -------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_shell_plastic_flow(
    dt: float,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Yield the faces whose von Mises stress exceeds the yield stress, and the hinges bent past the yield curvature.

    The plastic stretching flows multiplicatively, F_el <- F_el @ V Sigma^-gamma V^T up to a dilation that thins the
    face so as to conserve its volume, gamma growing with the relative overstress. The plastic bending moves the rest
    angle of a hinge until its elastic curvature, 3 * len * angle / (2 * area), drops to the yield curvature.
    """
    B = shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_scratch.hinges_stiffness.shape[0]

    for i_f, i_b in qd.ndrange(n_faces, B):
        i_e = shell_info.faces_entity[i_f]
        yield_stress = shell_info.entities_yield_stress[i_e]
        if yield_stress > 0.0:
            thickness = shell_state.faces_thickness[i_f, i_b]
            F, _ = func_face_deformation(i_f, i_b, shell_state, shell_info)
            stress = func_membrane_stress(F, shell_info.entities_stretching_modulus[i_e], shell_info.entities_nu[i_e])
            von_mises = qd.sqrt(
                qd.max(
                    stress[0, 0] ** 2 + stress[1, 1] ** 2 - stress[0, 0] * stress[1, 1] + 3.0 * stress[0, 1] ** 2,
                    0.0,
                )
            )
            gamma = qd.math.clamp(
                dt * shell_info.entities_plastic_flow_rate[i_e] * (von_mises - yield_stress) / yield_stress, 0.0, 1.0
            )
            if gamma > 0.0:
                lambda_0, lambda_1, eigvec_0 = func_sym2_eigen(F.transpose() @ F)
                sigma_0 = qd.sqrt(qd.max(lambda_0, gs.EPS))
                sigma_1 = qd.sqrt(qd.max(lambda_1, gs.EPS))
                det_sigma = sigma_0 * sigma_1
                flow = func_sym2_compose(sigma_0 ** (-gamma), sigma_1 ** (-gamma), eigvec_0) * det_sigma ** (
                    gamma / 3.0
                )
                shell_state.faces_plastic[i_f, i_b] = shell_state.faces_plastic[i_f, i_b] @ flow
                shell_state.faces_thickness[i_f, i_b] = thickness * det_sigma ** (-gamma / 3.0)

    for i_h, i_b in qd.ndrange(n_hinges, B):
        i_e = shell_info.hinges_entity[i_h]
        yield_curvature = shell_info.entities_yield_curvature[i_e]
        if yield_curvature > 0.0:
            i_va, i_vb, i_vc, i_vd, is_intact = func_hinge_verts(i_h, i_b, shell_state, shell_info)
            if is_intact:
                edge_b, edge_c, edge_d = func_hinge_edges(i_va, i_vb, i_vc, i_vd, i_b, shell_state)
                angle = func_hinge_angle(edge_b, edge_c, edge_d)
                angle_scale = 2.0 * shell_info.hinges_rest_area[i_h] / (3.0 * shell_info.hinges_rest_len[i_h])
                angle_elastic = angle - shell_info.hinges_rest_angle[i_h] - shell_state.hinges_plastic_angle[i_h, i_b]
                curvature = angle_elastic / angle_scale
                if qd.abs(curvature) > yield_curvature:
                    shell_state.hinges_plastic_angle[i_h, i_b] += (
                        qd.math.sign(curvature) * (qd.abs(curvature) - yield_curvature) * angle_scale
                    )


# ------------------------------------------------------------------------------------
# ------------------------------------ fracture --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_fan_entry_is_owned(
    i_v: int, i_j: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo
):
    """Whether the corner of fan entry i_j still belongs to vertex i_v."""
    i_c = shell_info.fans_corner[i_j]
    return shell_state.corners_vert[i_c, i_b] == i_v


@qd.func
def func_fan_link_is_intact(
    i_v: int, i_j: int, i_j_next: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo
):
    """Whether vertex i_v holds the corners of consecutive fan entries i_j and i_j_next, joined by an intact hinge."""
    is_intact = False
    i_h = shell_info.fans_next_hinge[i_j]
    if i_h >= 0:
        if func_fan_entry_is_owned(i_v, i_j, i_b, shell_state, shell_info) and func_fan_entry_is_owned(
            i_v, i_j_next, i_b, shell_state, shell_info
        ):
            _, _, _, _, is_intact = func_hinge_verts(i_h, i_b, shell_state, shell_info)
    return is_intact


@qd.func
def func_vert_arc(
    i_v: int, i_arc: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo
):
    """Locate the corners of a vertex within the fan of the original vertex it descends from.

    The corners a vertex holds form arcs of consecutive fan entries joined by intact hinges. Returns the fan start and
    length of its original vertex, the number of arcs (zero for a vertex holding its whole closed fan, which forms a
    ring), and the first entry (relative to the fan start) and length of arc i_arc, or of the ring.
    """
    i_o = shell_state.verts_origin[i_v, i_b]
    fan_start = shell_info.verts_fan_start[i_o]
    fan_len = shell_info.verts_fan_len[i_o]
    is_closed = shell_info.verts_is_fan_closed[i_o]
    n_arcs = 0
    n_owned = 0
    arc_start = -1
    for k in range(fan_len):
        is_owned = func_fan_entry_is_owned(i_v, fan_start + k, i_b, shell_state, shell_info)
        has_link_prev = False
        if k > 0:
            has_link_prev = func_fan_link_is_intact(i_v, fan_start + k - 1, fan_start + k, i_b, shell_state, shell_info)
        elif is_closed:
            has_link_prev = func_fan_link_is_intact(
                i_v, fan_start + fan_len - 1, fan_start, i_b, shell_state, shell_info
            )
        if is_owned:
            n_owned += 1
            if not has_link_prev:
                if n_arcs == i_arc:
                    arc_start = k
                n_arcs += 1
    arc_len = 0
    if n_arcs == 0 and n_owned > 0:
        arc_start = 0
        arc_len = fan_len
    elif arc_start >= 0:
        arc_len = 1
        for k in range(1, fan_len):
            i_j = fan_start + (arc_start + k - 1) % fan_len
            i_j_next = fan_start + (arc_start + k) % fan_len
            if not func_fan_link_is_intact(i_v, i_j, i_j_next, i_b, shell_state, shell_info):
                break
            arc_len += 1
    return fan_start, fan_len, n_arcs, arc_start, arc_len


@qd.func
def func_corner_traction(
    i_c: int, i_b: int, shell_scratch: array_class.ShellScratch, shell_info: array_class.ShellInfo
):
    """Return the traction a corner sector of a vertex fan transmits across a small disc around the vertex, in the
    material space, and the rest angle of the corner.

    The traction across the arc of the disc inside the face integrates to stress @ (t_end - t_start), t being the
    unit edge directions turned a quarter counter-clockwise, which is what the fracture criterion sums over a side of
    a candidate split.
    """
    i_f = i_c // 3
    k = i_c % 3
    Dm = shell_info.faces_Dm[i_f]
    p1 = qd.Vector([Dm[0, 0], Dm[1, 0]])
    p2 = qd.Vector([Dm[0, 1], Dm[1, 1]])
    edge_start = p1
    edge_end = p2
    if k == 1:
        edge_start = p2 - p1
        edge_end = -p1
    elif k == 2:
        edge_start = -p2
        edge_end = p1 - p2
    edge_start = edge_start / qd.max(edge_start.norm(), NORM_FLOOR)
    edge_end = edge_end / qd.max(edge_end.norm(), NORM_FLOOR)
    stress = shell_scratch.faces_fracture_stress[i_f, i_b]
    traction = stress @ (qd.Vector([-edge_end[1], edge_end[0]]) - qd.Vector([-edge_start[1], edge_start[0]]))
    angle = qd.acos(qd.math.clamp(edge_start.dot(edge_end), -1.0, 1.0))
    return shell_info.faces_basis[i_f] @ traction, angle


@qd.func
def func_split_score(traction_0: qd.types.vector(3), traction_1: qd.types.vector(3), is_open: bool):
    """Stress a split relieves, in N/m: the smaller of the opposing tractions its two sides pull apart with."""
    score = gs.qd_float(0.0)
    if traction_0.dot(traction_1) < 0.0:
        traction_diff = traction_0 - traction_1
        mid = traction_diff / qd.max(traction_diff.norm(), NORM_FLOOR)
        score = 0.5 * qd.min(qd.abs(traction_0.dot(mid)), qd.abs(traction_1.dot(mid)))
        if is_open:
            score = 2.0 * score
    return score


@qd.func
def func_alloc_vert(
    i_v: int, i_e: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo
):
    """Fill a free slot of the pool of an entity with a copy of vertex i_v, returning it, or -1 if none is left."""
    i_new = shell_info.entities_vert_start[i_e] + qd.atomic_add(shell_state.entities_n_verts[i_e, i_b], 1)
    if i_new >= shell_info.entities_vert_end[i_e]:
        qd.atomic_sub(shell_state.entities_n_verts[i_e, i_b], 1)
        i_new = -1
    else:
        shell_state.verts_pos[i_new, i_b] = shell_state.verts_pos[i_v, i_b]
        shell_state.verts_pos_cell[i_new, i_b] = shell_state.verts_pos_cell[i_v, i_b]
        shell_state.verts_pos_offset[i_new, i_b] = shell_state.verts_pos_offset[i_v, i_b]
        shell_state.verts_vel[i_new, i_b] = shell_state.verts_vel[i_v, i_b]
        shell_state.verts_origin[i_new, i_b] = shell_state.verts_origin[i_v, i_b]
        shell_state.verts_is_fixed[i_new, i_b] = shell_state.verts_is_fixed[i_v, i_b]
    return i_new


@qd.func
def func_vert_entity(i_v: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Entity of a filled pool slot, read from the face of the first corner of its original vertex."""
    i_o = shell_state.verts_origin[i_v, i_b]
    i_j = shell_info.verts_fan_start[i_o]
    i_c = shell_info.fans_corner[i_j]
    return shell_info.faces_entity[i_c // 3]


@qd.kernel
def kernel_shell_fracture(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Split the vertices whose surrounding stress exceeds the tensile strength of their material.

    The stress of a face adds the bending strain of its outer layer to the membrane one, F_bend = F @ (I + h / 2 * |S|)
    for the curvature S of its hinges. A vertex then evaluates every split of its fan along the mesh edges: for a
    vertex inside the sheet, two edges roughly opposite each other, and for a vertex on a boundary or a crack, one
    edge, the boundary acting as the other side. The best split scores the tractions its sides pull apart with, over
    the tensile strength times the thickness, and breaks above one. Splitting moves the corners of one side to a new
    vertex, which the hinges along the cut lose, so their far vertices become crack tips. Two vertices sharing a face
    never split in the same pass, the higher score winning, which keeps the splits independent. A vertex whose corners
    end up in several arcs (a crack reaching a boundary or another crack) is split into one vertex per arc, separating
    the pieces.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]

    for i_f, i_b in qd.ndrange(n_faces, B):
        i_e = shell_info.faces_entity[i_f]
        if shell_info.entities_tensile_strength[i_e] > 0.0:
            thickness = shell_state.faces_thickness[i_f, i_b]
            F, _ = func_face_deformation(i_f, i_b, shell_state, shell_info)
            Dm = shell_info.faces_Dm[i_f]
            curvature = qd.Matrix.zero(gs.qd_float, 2, 2)
            faces_hinge = shell_info.faces_hinge[i_f]
            for k in range(3):
                i_h = faces_hinge[0]
                if k == 1:
                    i_h = faces_hinge[1]
                elif k == 2:
                    i_h = faces_hinge[2]
                if i_h >= 0:
                    i_va, i_vb, i_vc, i_vd, is_intact = func_hinge_verts(i_h, i_b, shell_state, shell_info)
                    if is_intact:
                        edge_b, edge_c, edge_d = func_hinge_edges(i_va, i_vb, i_vc, i_vd, i_b, shell_state)
                        angle = func_hinge_angle(edge_b, edge_c, edge_d)
                        angle_elastic = (
                            angle - shell_info.hinges_rest_angle[i_h] - shell_state.hinges_plastic_angle[i_h, i_b]
                        )
                        edge = qd.Vector([Dm[0, 0], Dm[1, 0]])
                        if k == 1:
                            edge = qd.Vector([Dm[0, 1] - Dm[0, 0], Dm[1, 1] - Dm[1, 0]])
                        elif k == 2:
                            edge = -qd.Vector([Dm[0, 1], Dm[1, 1]])
                        edge_len = qd.max(edge.norm(), NORM_FLOOR)
                        normal = qd.Vector([-edge[1], edge[0]]) / edge_len
                        curvature += angle_elastic * edge_len * normal.outer_product(normal)
            curvature = curvature / (2.0 * shell_info.faces_rest_area[i_f])
            lambda_0, lambda_1, eigvec_0 = func_sym2_eigen(curvature)
            bending = func_sym2_compose(qd.abs(lambda_0), qd.abs(lambda_1), eigvec_0)
            bending_scale = 0.5 * shell_info.entities_bending_fracture_scale[i_e] * thickness
            F_bend = F @ (qd.Matrix.identity(gs.qd_float, 2) + bending_scale * bending)
            stress = func_membrane_stress(
                F_bend, shell_info.entities_stretching_modulus[i_e] * thickness, shell_info.entities_nu[i_e]
            )
            lambda_0, lambda_1, eigvec_0 = func_sym2_eigen(stress)
            shell_scratch.faces_fracture_stress[i_f, i_b] = func_sym2_compose(
                qd.max(lambda_0, 0.0), qd.max(lambda_1, 0.0), eigvec_0
            )

    for i_v, i_b in qd.ndrange(n_verts, B):
        separation = gs.qd_float(0.0)
        split = qd.Vector([-1, -1], dt=gs.qd_int)
        if shell_state.verts_origin[i_v, i_b] >= 0:
            i_e = func_vert_entity(i_v, i_b, shell_state, shell_info)
            tensile_strength = shell_info.entities_tensile_strength[i_e]
            if tensile_strength > 0.0:
                fan_start, fan_len, n_arcs, arc_start, arc_len = func_vert_arc(i_v, 0, i_b, shell_state, shell_info)
                is_open = n_arcs == 1
                if n_arcs <= 1 and arc_len >= 2:
                    # The stress a split relieves is at most twice the largest principal stress around the vertex.
                    traction_total = qd.Vector.zero(gs.qd_float, 3)
                    angle_total = gs.qd_float(0.0)
                    stress_max = gs.qd_float(0.0)
                    thickness_sum = gs.qd_float(0.0)
                    for p in range(arc_len):
                        i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                        traction, angle = func_corner_traction(i_c, i_b, shell_scratch, shell_info)
                        traction_total += traction
                        angle_total += angle
                        lambda_0, _, _ = func_sym2_eigen(shell_scratch.faces_fracture_stress[i_c // 3, i_b])
                        stress_max = qd.max(stress_max, lambda_0)
                        thickness_sum += shell_state.faces_thickness[i_c // 3, i_b]
                    toughness = tensile_strength * thickness_sum / arc_len
                    stress_bound = 2.0 * stress_max
                    if is_open:
                        stress_bound = 2.0 * stress_bound
                    if stress_bound >= toughness:
                        if is_open:
                            traction_0 = qd.Vector.zero(gs.qd_float, 3)
                            for p in range(arc_len - 1):
                                i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                                traction, _ = func_corner_traction(i_c, i_b, shell_scratch, shell_info)
                                traction_0 += traction
                                score = func_split_score(traction_0, traction_total - traction_0, True) / toughness
                                if score > separation:
                                    separation = score
                                    split = qd.Vector([p, -1], dt=gs.qd_int)
                        else:
                            for p in range(arc_len - 1):
                                traction_1 = qd.Vector.zero(gs.qd_float, 3)
                                angle_1 = gs.qd_float(0.0)
                                for q in range(p + 1, arc_len):
                                    i_c = shell_info.fans_corner[fan_start + (arc_start + q) % fan_len]
                                    traction, angle = func_corner_traction(i_c, i_b, shell_scratch, shell_info)
                                    traction_1 += traction
                                    angle_1 += angle
                                    # The two edges of the split run roughly opposite each other across the vertex.
                                    if 0.25 * angle_total <= angle_1 and angle_1 <= 0.75 * angle_total:
                                        score = func_split_score(traction_total - traction_1, traction_1, False)
                                        score = score / toughness
                                        if score > separation:
                                            separation = score
                                            split = qd.Vector([p, q], dt=gs.qd_int)
        shell_scratch.verts_separation[i_v, i_b] = separation
        shell_scratch.verts_split[i_v, i_b] = split

    # A vertex splits only if its score beats the score of every vertex it shares a face with. The verdict goes to
    # the split of the vertex, which no other vertex reads, keeping the comparison free of races.
    for i_v, i_b in qd.ndrange(n_verts, B):
        separation = shell_scratch.verts_separation[i_v, i_b]
        if separation > 1.0:
            fan_start, fan_len, _, arc_start, arc_len = func_vert_arc(i_v, 0, i_b, shell_state, shell_info)
            is_winner = True
            for p in range(arc_len):
                i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                for k in range(1, 3):
                    i_u = shell_state.corners_vert[3 * (i_c // 3) + (i_c + k) % 3, i_b]
                    separation_u = shell_scratch.verts_separation[i_u, i_b]
                    if separation_u > separation or (separation_u == separation and i_u < i_v):
                        is_winner = False
            if not is_winner:
                shell_scratch.verts_split[i_v, i_b] = qd.Vector([-1, -1], dt=gs.qd_int)

    for i_v, i_b in qd.ndrange(n_verts, B):
        split = shell_scratch.verts_split[i_v, i_b]
        if shell_scratch.verts_separation[i_v, i_b] > 1.0 and split[0] >= 0:
            fan_start, fan_len, n_arcs, arc_start, arc_len = func_vert_arc(i_v, 0, i_b, shell_state, shell_info)
            i_new = func_alloc_vert(
                i_v, func_vert_entity(i_v, i_b, shell_state, shell_info), i_b, shell_state, shell_info
            )
            if i_new >= 0:
                p_end = split[1]
                if n_arcs == 1:
                    p_end = arc_len - 1
                for p in range(split[0] + 1, p_end + 1):
                    i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                    shell_state.corners_vert[i_c, i_b] = i_new

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_state.verts_origin[i_v, i_b] >= 0:
            _, _, n_arcs, _, _ = func_vert_arc(i_v, 0, i_b, shell_state, shell_info)
            i_e = func_vert_entity(i_v, i_b, shell_state, shell_info)
            # The arcs past the first one move out one at a time, the next one becoming arc 1 in turn.
            for i_arc_ in range(n_arcs - 1):
                fan_start, fan_len, _, arc_start, arc_len = func_vert_arc(i_v, 1, i_b, shell_state, shell_info)
                i_new = func_alloc_vert(i_v, i_e, i_b, shell_state, shell_info)
                if i_new >= 0:
                    for p in range(arc_len):
                        i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                        shell_state.corners_vert[i_c, i_b] = i_new


# ------------------------------------------------------------------------------------
# ------------------------------------ accessors -------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_shell_update_render(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Write the position and the smooth normal of every face corner, from the vertex holding it."""
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]

    for i_v, i_b in qd.ndrange(n_verts, B):
        shell_scratch.verts_normal[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_f, i_b in qd.ndrange(n_faces, B):
        i_v0 = shell_state.corners_vert[3 * i_f, i_b]
        i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
        i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
        x0 = shell_state.verts_pos[i_v0, i_b]
        normal = (shell_state.verts_pos[i_v1, i_b] - x0).cross(shell_state.verts_pos[i_v2, i_b] - x0)
        shell_scratch.verts_normal[i_v0, i_b] += normal
        shell_scratch.verts_normal[i_v1, i_b] += normal
        shell_scratch.verts_normal[i_v2, i_b] += normal

    for i_c, i_b in qd.ndrange(3 * n_faces, B):
        i_v = shell_state.corners_vert[i_c, i_b]
        shell_scratch.corners_render_pos[i_c, i_b] = shell_state.verts_pos[i_v, i_b]
        normal = shell_scratch.verts_normal[i_v, i_b]
        shell_scratch.corners_render_normal[i_c, i_b] = normal / qd.max(normal.norm(), NORM_FLOOR)


@qd.kernel
def kernel_shell_set_verts_vel(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    shell_state: array_class.ShellState,
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            shell_state.verts_vel[i_v, i_b][j] = values[i_b_, i_v_, j]


@qd.kernel
def kernel_shell_set_verts_pos(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    shell_state: array_class.ShellState,
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        pos = qd.Vector([values[i_b_, i_v_, 0], values[i_b_, i_v_, 1], values[i_b_, i_v_, 2]], dt=gs.qd_float)
        cell = qd.floor(pos / POS_GRID + 0.5).cast(gs.qd_int)
        shell_state.verts_pos[i_v, i_b] = pos
        shell_state.verts_pos_cell[i_v, i_b] = cell
        shell_state.verts_pos_offset[i_v, i_b] = pos - cell.cast(gs.qd_float) * POS_GRID


@qd.kernel
def kernel_shell_set_verts_fixed(
    verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), is_fixed: bool, shell_state: array_class.ShellState
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        shell_state.verts_is_fixed[i_v, i_b] = is_fixed


@qd.kernel
def kernel_shell_set_state(
    envs_idx: qd.types.ndarray(),
    verts_pos: qd.types.ndarray(),
    verts_pos_cell: qd.types.ndarray(),
    verts_pos_offset: qd.types.ndarray(),
    verts_vel: qd.types.ndarray(),
    verts_origin: qd.types.ndarray(),
    verts_is_fixed: qd.types.ndarray(),
    corners_vert: qd.types.ndarray(),
    entities_n_verts: qd.types.ndarray(),
    faces_plastic: qd.types.ndarray(),
    faces_thickness: qd.types.ndarray(),
    hinges_plastic_angle: qd.types.ndarray(),
    shell_state: array_class.ShellState,
):
    n_verts = shell_state.verts_pos.shape[0]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_state.hinges_plastic_angle.shape[0]
    n_entities = shell_state.entities_n_verts.shape[0]
    for i_v, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            shell_state.verts_pos[i_v, i_b][j] = verts_pos[i_b_, i_v, j]
            shell_state.verts_pos_cell[i_v, i_b][j] = verts_pos_cell[i_b_, i_v, j]
            shell_state.verts_pos_offset[i_v, i_b][j] = verts_pos_offset[i_b_, i_v, j]
            shell_state.verts_vel[i_v, i_b][j] = verts_vel[i_b_, i_v, j]
        shell_state.verts_origin[i_v, i_b] = verts_origin[i_b_, i_v]
        shell_state.verts_is_fixed[i_v, i_b] = verts_is_fixed[i_b_, i_v]
    for i_c, i_b_ in qd.ndrange(3 * n_faces, envs_idx.shape[0]):
        shell_state.corners_vert[i_c, envs_idx[i_b_]] = corners_vert[i_b_, i_c]
    for i_f, i_b_ in qd.ndrange(n_faces, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j, k in qd.static(qd.ndrange(2, 2)):
            shell_state.faces_plastic[i_f, i_b][j, k] = faces_plastic[i_b_, i_f, j, k]
        shell_state.faces_thickness[i_f, i_b] = faces_thickness[i_b_, i_f]
    for i_h, i_b_ in qd.ndrange(n_hinges, envs_idx.shape[0]):
        shell_state.hinges_plastic_angle[i_h, envs_idx[i_b_]] = hinges_plastic_angle[i_b_, i_h]
    for i_e, i_b_ in qd.ndrange(n_entities, envs_idx.shape[0]):
        shell_state.entities_n_verts[i_e, envs_idx[i_b_]] = entities_n_verts[i_b_, i_e]
