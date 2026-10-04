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


class ShellSolver(GravityMixin, TimeBasedMixin, Solver):
    """
    Solver of thin elastoplastic sheets that tear and crack.

    Each substep integrates the sheets by one linearized backward Euler step: membrane stretching (Saint Venant-Kirchhoff
    on the Green strain, plane stress) and hinge bending, both with stiffness-proportional damping. The linear system
    is solved by matrix-free preconditioned conjugate gradient (PCG), every environment iterating until its own
    residual converges. The rigid coupling then corrects the vertex velocities, before the positions advance, the
    material yields plastically, and the vertices whose surrounding stress exceeds the tensile strength split along
    the mesh edges that relieve the most stress.
    """

    material_cls = Shell

    def __init__(self, scene: "Scene", sim: "Simulator", options):
        super().__init__(scene, sim, options)

        self._n_pcg_iterations = options.n_pcg_iterations
        self._pcg_threshold = options.pcg_threshold
        self._fracture_capacity = options.fracture_capacity

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
            )
            self._shell_info = array_class.get_shell_info(
                len(self._entities), self._n_verts, self._n_faces, self._n_hinges
            )
            self._shell_state = array_class.get_shell_state(
                len(self._entities), self._n_verts, self._n_faces, self._n_hinges, self._B
            )
            self._shell_scratch = array_class.get_shell_scratch(
                self._n_verts, self._n_faces, self._n_hinges, self._B, self._static_config.has_fracture
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
        info.verts_fan_start.from_numpy(verts_fan_start)
        info.verts_fan_len.from_numpy(verts_fan_len)
        info.verts_is_fan_closed.from_numpy(verts_is_fan_closed)
        info.fans_corner.from_numpy(fans_corner)
        info.fans_next_hinge.from_numpy(fans_next_hinge)

        faces_thickness = np.array([entity.material.thickness for entity in self._entities], dtype=gs.np_float)
        state.verts_pos.from_numpy(np.ascontiguousarray(np.broadcast_to(verts_pos[:, None], (n_verts, B, 3))))
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
        state, scratch, info = self._shell_state, self._shell_scratch, self._shell_info
        kernel_shell_system_product(scratch.verts_dv, scratch.verts_Ap, state, scratch, info)
        kernel_shell_pcg_init(self._pcg_threshold, state, scratch)
        for _ in range(self._n_pcg_iterations):
            kernel_shell_system_product(scratch.verts_p, scratch.verts_Ap, state, scratch, info)
            kernel_shell_pcg_update(state, scratch)
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
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        kernel_shell_set_state(
            envs_idx,
            state.verts_pos[envs_idx].contiguous(),
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

    def set_verts_vec(self, tensor, values, entity: ShellEntity, verts_idx_local, envs_idx):
        """Write one 3-vector per vertex of an entity into a per-vertex state tensor."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = self._sanitize_verts_idx(entity, verts_idx_local, envs_idx)
        values = broadcast_tensor(values, gs.tc_float, (*verts_idx.shape, 3), ("envs_idx", "verts_idx", ""))
        kernel_shell_set_verts_vec(verts_idx, envs_idx, values.contiguous(), tensor)

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
    x0 = shell_state.verts_pos[i_v0, i_b]
    Ds = qd.Matrix.cols([shell_state.verts_pos[i_v1, i_b] - x0, shell_state.verts_pos[i_v2, i_b] - x0])
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
def func_hinge_angle(xa: qd.types.vector(3), xb: qd.types.vector(3), xc: qd.types.vector(3), xd: qd.types.vector(3)):
    """Signed dihedral angle of a hinge about its edge from a to b, the first face holding (a, b, c) and the second
    (b, a, d), zero when flat."""
    normal_0 = (xb - xa).cross(xc - xa).normalized(gs.EPS)
    normal_1 = (xa - xb).cross(xd - xb).normalized(gs.EPS)
    edge = (xb - xa).normalized(gs.EPS)
    return qd.atan2(edge.dot(normal_0.cross(normal_1)), normal_0.dot(normal_1))


@qd.func
def func_hinge_angle_gradient(
    xa: qd.types.vector(3), xb: qd.types.vector(3), xc: qd.types.vector(3), xd: qd.types.vector(3)
):
    """Gradient of the dihedral angle of a hinge (see func_hinge_angle) with respect to a, b, c and d, as columns."""
    edge = xb - xa
    edge_len = edge.norm(gs.EPS)
    cross_0 = edge.cross(xc - xa)
    cross_1 = (xa - xb).cross(xd - xb)
    double_area_0 = cross_0.norm(gs.EPS)
    double_area_1 = cross_1.norm(gs.EPS)
    # The gradient at an opposite vertex is the normal of its face over its height above the edge.
    grad_c = -cross_0 / double_area_0 * (edge_len / double_area_0)
    grad_d = -cross_1 / double_area_1 * (edge_len / double_area_1)
    s_c = (xc - xa).dot(edge) / (edge_len * edge_len)
    s_d = (xd - xa).dot(edge) / (edge_len * edge_len)
    grad_a = -((1.0 - s_c) * grad_c + (1.0 - s_d) * grad_d)
    grad_b = -(s_c * grad_c + s_d * grad_d)
    return qd.Matrix.cols([grad_a, grad_b, grad_c, grad_d])


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
            xa = shell_state.verts_pos[i_va, i_b]
            xb = shell_state.verts_pos[i_vb, i_b]
            xc = shell_state.verts_pos[i_vc, i_b]
            xd = shell_state.verts_pos[i_vd, i_b]
            angle = func_hinge_angle(xa, xb, xc, xd)
            grad = func_hinge_angle_gradient(xa, xb, xc, xd)
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


@qd.kernel
def kernel_shell_pcg_init(
    pcg_threshold: float, shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch
):
    """Start the PCG solve of the velocity update with block-Jacobi preconditioning.

    The solve starts from the velocity change of the previous substep, which the system product just mapped to
    verts_Ap, since the acceleration of smooth motion varies little from one substep to the next.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]

    for i_b in range(B):
        shell_scratch.envs_rz[i_b] = 0.0
        shell_scratch.envs_rz_threshold[i_b] = 0.0

    for i_v, i_b in qd.ndrange(n_verts, B):
        r = shell_scratch.verts_rhs[i_v, i_b] - shell_scratch.verts_Ap[i_v, i_b]
        z = shell_scratch.verts_prec[i_v, i_b] @ r
        shell_scratch.verts_r[i_v, i_b] = r
        shell_scratch.verts_z[i_v, i_b] = z
        shell_scratch.verts_p[i_v, i_b] = z
        shell_scratch.envs_rz[i_b] += r.dot(z)
        # The threshold is relative to the norm of the preconditioned right-hand side, which a warm start leaves out of
        # the initial residual.
        shell_scratch.envs_rz_threshold[i_b] += shell_scratch.verts_rhs[i_v, i_b].dot(
            shell_scratch.verts_prec[i_v, i_b] @ shell_scratch.verts_rhs[i_v, i_b]
        )

    for i_b in range(B):
        shell_scratch.envs_rz_threshold[i_b] = pcg_threshold * pcg_threshold * shell_scratch.envs_rz_threshold[i_b]
        shell_scratch.envs_is_solving[i_b] = shell_scratch.envs_rz[i_b] > shell_scratch.envs_rz_threshold[i_b]


@qd.kernel
def kernel_shell_pcg_update(shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch):
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
            r = shell_scratch.verts_r[i_v, i_b] - alpha * shell_scratch.verts_Ap[i_v, i_b]
            z = shell_scratch.verts_prec[i_v, i_b] @ r
            shell_scratch.verts_r[i_v, i_b] = r
            shell_scratch.verts_z[i_v, i_b] = z
            shell_scratch.envs_rz_new[i_b] += r.dot(z)

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


@qd.kernel
def kernel_shell_apply_dv(shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch):
    """Add the solved velocity change to the free vertices."""
    for i_v, i_b in qd.ndrange(shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]):
        if func_is_vert_free(i_v, i_b, shell_state):
            shell_state.verts_vel[i_v, i_b] += shell_scratch.verts_dv[i_v, i_b]


@qd.kernel
def kernel_shell_integrate(dt: float, shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch):
    """Advance the position of every vertex by its velocity."""
    for i_v, i_b in qd.ndrange(shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]):
        if shell_state.verts_origin[i_v, i_b] >= 0:
            shell_state.verts_pos[i_v, i_b] += dt * shell_state.verts_vel[i_v, i_b]


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
                angle = func_hinge_angle(
                    shell_state.verts_pos[i_va, i_b],
                    shell_state.verts_pos[i_vb, i_b],
                    shell_state.verts_pos[i_vc, i_b],
                    shell_state.verts_pos[i_vd, i_b],
                )
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
    edge_start = edge_start.normalized(gs.EPS)
    edge_end = edge_end.normalized(gs.EPS)
    stress = shell_scratch.faces_fracture_stress[i_f, i_b]
    traction = stress @ (qd.Vector([-edge_end[1], edge_end[0]]) - qd.Vector([-edge_start[1], edge_start[0]]))
    angle = qd.acos(qd.math.clamp(edge_start.dot(edge_end), -1.0, 1.0))
    return shell_info.faces_basis[i_f] @ traction, angle


@qd.func
def func_split_score(traction_0: qd.types.vector(3), traction_1: qd.types.vector(3), is_open: bool):
    """Stress a split relieves, in N/m: the smaller of the opposing tractions its two sides pull apart with."""
    score = gs.qd_float(0.0)
    if traction_0.dot(traction_1) < 0.0:
        mid = (traction_0 - traction_1).normalized(gs.EPS)
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
                        angle = func_hinge_angle(
                            shell_state.verts_pos[i_va, i_b],
                            shell_state.verts_pos[i_vb, i_b],
                            shell_state.verts_pos[i_vc, i_b],
                            shell_state.verts_pos[i_vd, i_b],
                        )
                        angle_elastic = (
                            angle - shell_info.hinges_rest_angle[i_h] - shell_state.hinges_plastic_angle[i_h, i_b]
                        )
                        edge = qd.Vector([Dm[0, 0], Dm[1, 0]])
                        if k == 1:
                            edge = qd.Vector([Dm[0, 1] - Dm[0, 0], Dm[1, 1] - Dm[1, 0]])
                        elif k == 2:
                            edge = -qd.Vector([Dm[0, 1], Dm[1, 1]])
                        edge_len = edge.norm(gs.EPS)
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
        shell_scratch.corners_render_normal[i_c, i_b] = shell_scratch.verts_normal[i_v, i_b].normalized(gs.EPS)


@qd.kernel
def kernel_shell_set_verts_vec(
    verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), values: qd.types.ndarray(), tensor: qd.Tensor
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            tensor[i_v, i_b][j] = values[i_b_, i_v_, j]


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
