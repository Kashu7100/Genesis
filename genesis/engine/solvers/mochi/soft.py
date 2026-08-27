# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Deformable bodies: linear tetrahedra whose vertex positions are unknowns of the implicit solve.

Per tetrahedron the incremental potential collects the inertia with the consistent (4-point quadrature) mass matrix,
gravity, mass-proportional damping, the elastic energy with a positive-semidefinite projected tangent, and the
Kelvin-Voigt stiffness damping; the 12x12 tangent block is stored per element and applied on the fly by the linear
solvers. Contact acts on quadrature samples of the boundary triangles against the rigid colliders, scattering to the
three vertices with the barycentric weights; the rigid side reuses the per-pair accumulators of the rigid contact.
"""

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .articulated import func_jacobian_times_dofs, func_jacobian_transpose_add, func_link_dof_jacobian
from .colliders import query_collider
from .contact import CONSERVATIVE_MAX_ACCEL, CONSERVATIVE_SPEED_SCALE
from .contact_utils import collision_response
from .data import (
    COLLIDER_TYPE,
    INTEGRATOR,
    N_HISTORY,
    SOLVE_STATUS,
    MochiContactState,
    MochiInfo,
    MochiSoftInfo,
    MochiSoftState,
    MochiState,
)
from .integration import BDF2_ALPHA_2
from .lie import skew
from .newton import func_is_env_active
from .soft_materials import (
    func_elastic_energy,
    func_elastic_stress,
    func_elastic_tangent,
    func_stiffness_damping_block,
    func_stiffness_damping_stress,
)

# Consistent mass matrix of a linear tetrahedron, in units of rho V / 20.
CONSISTENT_MASS = ((2.0, 1.0, 1.0, 1.0), (1.0, 2.0, 1.0, 1.0), (1.0, 1.0, 2.0, 1.0), (1.0, 1.0, 1.0, 2.0))

# Columns of the per-entity parameter table passed at initialization.
ENTITY_PARAMS = (
    "mass",
    "has_gravity",
    "model",
    "mu",
    "lam",
    "rho",
    "mass_damping",
    "stiffness_damping",
    "penalty_coefficient",
    "penalty_smoothing_half_distance",
    "penalty_threshold",
    "friction",
    "friction_falloff_vel",
    "viscous_friction",
    "normal_viscous_damping",
    "max_alignment_normals",
    "vert_start",
    "vert_end",
    "sample_start",
    "sample_end",
)


def build_soft_samples(verts, surface_tri, quadrature_bary, quadrature_weights):
    """Boundary contact samples of a tetrahedral mesh: for each boundary triangle, the quadrature points as barycentric
    coordinates with their rest area weights. Returns (triangles, barycentric coordinates, weights)."""
    tri = verts[surface_tri]
    areas_2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    n_q = len(quadrature_weights)
    triangles = np.repeat(surface_tri, n_q, axis=0)
    bary = np.tile(quadrature_bary, (len(surface_tri), 1))
    weights = (np.asarray(quadrature_weights)[None, :] * areas_2[:, None]).reshape((-1,))
    return triangles.astype(gs.np_int), bary.astype(gs.np_float), weights.astype(gs.np_float)


# ------------------------------------------------------------------------------------
# ----------------------------------- initialization ---------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_init_soft_fields(
    verts_rest: qd.types.ndarray(),
    verts_entity_idx: qd.types.ndarray(),
    elems_v: qd.types.ndarray(),
    elems_entity_idx: qd.types.ndarray(),
    samples_tri: qd.types.ndarray(),
    samples_bary: qd.types.ndarray(),
    samples_weight: qd.types.ndarray(),
    samples_entity_idx: qd.types.ndarray(),
    entities_params: qd.types.ndarray(),
    entities_links_pair_enabled: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = verts_rest.shape[0]
    n_elems = elems_v.shape[0]
    n_samples = samples_tri.shape[0]
    n_entities = entities_params.shape[0]
    n_links = entities_links_pair_enabled.shape[1]
    _B = soft_state.verts_pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v in range(n_verts):
        for k in qd.static(range(3)):
            soft_info.verts_rest[i_v][k] = verts_rest[i_v, k]
        soft_info.verts_entity_idx[i_v] = verts_entity_idx[i_v]
        soft_info.verts_mass[i_v] = 0.0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e in range(n_entities):
        soft_info.entities_mass[i_e] = entities_params[i_e, 0]
        soft_info.entities_has_gravity[i_e] = entities_params[i_e, 1] > 0.5
        soft_info.entities_model[i_e] = qd.cast(entities_params[i_e, 2], gs.qd_int)
        soft_info.entities_mu[i_e] = entities_params[i_e, 3]
        soft_info.entities_lam[i_e] = entities_params[i_e, 4]
        soft_info.entities_rho[i_e] = entities_params[i_e, 5]
        soft_info.entities_mass_damping[i_e] = entities_params[i_e, 6]
        soft_info.entities_stiffness_damping[i_e] = entities_params[i_e, 7]
        soft_info.entities_penalty_coefficient[i_e] = entities_params[i_e, 8]
        soft_info.entities_penalty_smoothing_half_distance[i_e] = entities_params[i_e, 9]
        soft_info.entities_penalty_threshold[i_e] = entities_params[i_e, 10]
        soft_info.entities_friction[i_e] = entities_params[i_e, 11]
        soft_info.entities_friction_falloff_vel[i_e] = entities_params[i_e, 12]
        soft_info.entities_viscous_friction[i_e] = entities_params[i_e, 13]
        soft_info.entities_normal_viscous_damping[i_e] = entities_params[i_e, 14]
        soft_info.entities_max_alignment_normals[i_e] = entities_params[i_e, 15]
        soft_info.entities_vert_start[i_e] = qd.cast(entities_params[i_e, 16], gs.qd_int)
        soft_info.entities_vert_end[i_e] = qd.cast(entities_params[i_e, 17], gs.qd_int)
        soft_info.entities_sample_start[i_e] = qd.cast(entities_params[i_e, 18], gs.qd_int)
        soft_info.entities_sample_end[i_e] = qd.cast(entities_params[i_e, 19], gs.qd_int)
        for i_l in range(n_links):
            soft_info.entities_links_pair_enabled[i_e, i_l] = entities_links_pair_enabled[i_e, i_l]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_el in range(n_elems):
        for k in qd.static(range(4)):
            soft_info.elems_v[i_el][k] = elems_v[i_el, k]
        soft_info.elems_entity_idx[i_el] = elems_entity_idx[i_el]
        x3 = soft_info.verts_rest[elems_v[i_el, 3]]
        Dm = qd.Matrix.cols(
            [
                soft_info.verts_rest[elems_v[i_el, 0]] - x3,
                soft_info.verts_rest[elems_v[i_el, 1]] - x3,
                soft_info.verts_rest[elems_v[i_el, 2]] - x3,
            ]
        )
        vol = Dm.determinant() / 6.0
        soft_info.elems_Dm_inv[i_el] = Dm.inverse()
        soft_info.elems_vol[i_el] = vol
        rho = soft_info.entities_rho[elems_entity_idx[i_el]]
        for k in qd.static(range(4)):
            qd.atomic_add(soft_info.verts_mass[elems_v[i_el, k]], 0.25 * rho * vol)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s in range(n_samples):
        for k in qd.static(range(3)):
            soft_info.samples_tri[i_s][k] = samples_tri[i_s, k]
            soft_info.samples_bary[i_s][k] = samples_bary[i_s, k]
        soft_info.samples_weight[i_s] = samples_weight[i_s]
        soft_info.samples_entity_idx[i_s] = samples_entity_idx[i_s]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        pos = soft_info.verts_rest[i_v]
        soft_state.verts_pos[i_v, i_b] = pos
        soft_state.verts_vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        for k in qd.static(range(N_HISTORY)):
            soft_state.verts_pos_prev[k, i_v, i_b] = pos
            soft_state.verts_vel_prev[k, i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        soft_state.verts_pos_stage_start[i_v, i_b] = pos
        soft_state.verts_vel_stage_start[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        soft_state.verts_is_fixed[i_v, i_b] = False
        soft_state.verts_contact_force[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_soft_dof(i_v, k, soft_info: MochiSoftInfo):
    return soft_info.dof_start[None] + 3 * i_v + k


@qd.func
def func_read_soft_vec(src: qd.Tensor, i_v, i_b, soft_info: MochiSoftInfo):
    i_d = func_soft_dof(i_v, 0, soft_info)
    return qd.Vector([src[i_d, i_b], src[i_d + 1, i_b], src[i_d + 2, i_b]], dt=gs.qd_float)


@qd.func
def func_add_soft_vec(dst: qd.Tensor, i_v, i_b, value, soft_info: MochiSoftInfo):
    i_d = func_soft_dof(i_v, 0, soft_info)
    for k in qd.static(range(3)):
        qd.atomic_add(dst[i_d + k, i_b], value[k])


@qd.func
def func_deformation_gradient(x0, x1, x2, x3, Dm_inv):
    return qd.Matrix.cols([x0 - x3, x1 - x3, x2 - x3]) @ Dm_inv


@qd.func
def func_shape_gradients(Dm_inv):
    """Rest gradients of the four linear shape functions (rows of Dm^-1 and minus their sum)."""
    g0 = qd.Vector([Dm_inv[0, 0], Dm_inv[0, 1], Dm_inv[0, 2]], dt=gs.qd_float)
    g1 = qd.Vector([Dm_inv[1, 0], Dm_inv[1, 1], Dm_inv[1, 2]], dt=gs.qd_float)
    g2 = qd.Vector([Dm_inv[2, 0], Dm_inv[2, 1], Dm_inv[2, 2]], dt=gs.qd_float)
    return g0, g1, g2, -(g0 + g1 + g2)


# ------------------------------------------------------------------------------------
# ----------------------------------- time integration -------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_soft_step_start(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """Shift the vertex history and build the step-start (= stage-start) positions and velocities, which also warm
    start the solve; snapshot the stage-start deformation gradients for the stiffness damping. Runs after the rigid
    step start (which advances the history counter)."""
    n_verts = soft_state.verts_pos.shape[0]
    n_elems = soft_state.elems_F_stage_start.shape[0]
    _B = soft_state.verts_pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        soft_state.verts_pos_prev[1, i_v, i_b] = soft_state.verts_pos_prev[0, i_v, i_b]
        soft_state.verts_vel_prev[1, i_v, i_b] = soft_state.verts_vel_prev[0, i_v, i_b]
        soft_state.verts_pos_prev[0, i_v, i_b] = soft_state.verts_pos[i_v, i_b]
        soft_state.verts_vel_prev[0, i_v, i_b] = soft_state.verts_vel[i_v, i_b]
        x1 = soft_state.verts_pos[i_v, i_b]
        v1 = soft_state.verts_vel[i_v, i_b]
        pos = x1
        vel = v1
        if qd.static(mochi_config.integrator == INTEGRATOR.BDF2):  # noqa: SIM102
            if mochi_state.n_hist[i_b] >= 2:
                pos = x1 + BDF2_ALPHA_2 * (soft_state.verts_pos_prev[1, i_v, i_b] - x1)
                vel = v1 + BDF2_ALPHA_2 * (soft_state.verts_vel_prev[1, i_v, i_b] - v1)
        if soft_state.verts_is_fixed[i_v, i_b]:
            pos = x1
            vel = qd.Vector.zero(gs.qd_float, 3)
        soft_state.verts_pos_step_start[i_v, i_b] = pos
        soft_state.verts_pos_stage_start[i_v, i_b] = pos
        soft_state.verts_vel_stage_start[i_v, i_b] = vel
        soft_state.verts_pos[i_v, i_b] = pos

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_el, i_b in qd.ndrange(n_elems, _B):
        v = soft_info.elems_v[i_el]
        soft_state.elems_F_stage_start[i_el, i_b] = func_deformation_gradient(
            soft_state.verts_pos_stage_start[v[0], i_b],
            soft_state.verts_pos_stage_start[v[1], i_b],
            soft_state.verts_pos_stage_start[v[2], i_b],
            soft_state.verts_pos_stage_start[v[3], i_b],
            soft_info.elems_Dm_inv[i_el],
        )


@qd.kernel
def kernel_soft_post_stage(
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Finite-difference vertex velocities over the stage; a diverged environment is reset to its previous
    configuration at rest."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        if mochi_state.status[i_b] == SOLVE_STATUS.DIVERGED:
            soft_state.verts_pos[i_v, i_b] = soft_state.verts_pos_prev[0, i_v, i_b]
            soft_state.verts_vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            continue
        h = mochi_state.dt_stage[i_b]
        soft_state.verts_vel[i_v, i_b] = (
            soft_state.verts_pos[i_v, i_b] - soft_state.verts_pos_stage_start[i_v, i_b]
        ) / h


# ------------------------------------------------------------------------------------
# --------------------------------------- Newton -------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_soft_update_conv_weights(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Convergence weights of the vertex degrees of freedom: 1 / (a_ref^2 M_entity m_vertex) with the lumped vertex
    mass, so that a unit weighted residual norm means a unit acceleration error."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    EPS = mochi_info.EPS[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        a_ref = qd.max(1.0, mochi_info.gravity[i_b].norm())
        mass_entity = soft_info.entities_mass[soft_info.verts_entity_idx[i_v]]
        w = 1.0 / (a_ref * a_ref * qd.max(mass_entity * soft_info.verts_mass[i_v], EPS))
        for k in qd.static(range(3)):
            mochi_state.conv_w[func_soft_dof(i_v, k, soft_info), i_b] = w


@qd.kernel
def kernel_soft_store_ls_ref(
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    only_done: qd.template(),
):
    """Take the current vertex positions as the line search reference of the active environments (of those that just
    accepted an iterate when only_done)."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        is_ref = mochi_state.is_active[i_b]
        if qd.static(only_done):
            is_ref = is_ref and mochi_state.ls_is_done[i_b]
        if is_ref:
            soft_state.verts_pos_ls_ref[i_v, i_b] = soft_state.verts_pos[i_v, i_b]


@qd.kernel
def kernel_soft_apply_increment(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Trial vertex positions of the environments still searching: reference minus the scaled Newton step."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        if not func_is_env_active(i_b, mochi_state, True):
            continue
        if soft_state.verts_is_fixed[i_v, i_b]:
            continue
        dx = func_read_soft_vec(mochi_state.dx, i_v, i_b, soft_info)
        soft_state.verts_pos[i_v, i_b] = soft_state.verts_pos_ls_ref[i_v, i_b] - mochi_state.ls_alpha[i_b] * dx


# ------------------------------------------------------------------------------------
# -------------------------------------- assembly ------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_soft_zero_assembly(
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_dres: qd.template(),
    skip_ls_done: qd.template(),
    record: qd.template(),
):
    n_verts = soft_state.verts_pos.shape[0]
    n_elems = soft_state.elems_H.shape[0]
    max_pairs = soft_state.pair_entity_a.shape[0]
    _B = soft_state.verts_pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            soft_state.n_soft_hits[i_b] = 0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            if qd.static(assem_dres):
                soft_state.verts_H_diag[i_v, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            if qd.static(record):
                soft_state.verts_contact_force[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(assem_dres):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_el, i_b in qd.ndrange(n_elems, _B):
            if func_is_env_active(i_b, mochi_state, skip_ls_done):
                soft_state.elems_H[i_el, i_b] = qd.Matrix.zero(gs.qd_float, 12, 12)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_b in qd.ndrange(max_pairs, _B):
        if func_is_env_active(i_b, mochi_state, skip_ls_done) and i_p < soft_state.n_pairs[i_b]:
            soft_state.acc_f[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            soft_state.acc_q[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            soft_state.acc_D[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            soft_state.acc_SD[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            soft_state.acc_SDS[i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            soft_state.acc_obj[i_p, i_b] = 0.0
            soft_state.n_hits[i_p, i_b] = 0


@qd.kernel
def kernel_soft_assemble_elements(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.template(),
    skip_ls_done: qd.template(),
):
    """Inertia, gravity, mass damping, elastic stress and stiffness damping of every tetrahedron of the running
    environments: residual into the vertex degrees of freedom, 12x12 tangent block per element (positive-semidefinite
    by construction) and its diagonal 3x3 blocks into the vertex preconditioner."""
    n_elems = soft_state.elems_H.shape[0]
    _B = soft_state.verts_pos.shape[1]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_el, i_b in qd.ndrange(n_elems, _B):
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        i_e = soft_info.elems_entity_idx[i_el]
        v = soft_info.elems_v[i_el]
        h = mochi_state.dt_stage[i_b]
        vol = soft_info.elems_vol[i_el]
        rho = soft_info.entities_rho[i_e]
        mu = soft_info.entities_mu[i_e]
        lam = soft_info.entities_lam[i_e]
        model = soft_info.entities_model[i_e]
        Dm_inv = soft_info.elems_Dm_inv[i_el]

        x = qd.Matrix.zero(gs.qd_float, 4, 3)
        a = qd.Matrix.zero(gs.qd_float, 4, 3)  # x - (x_ss + h v_ss)
        b = qd.Matrix.zero(gs.qd_float, 4, 3)  # x - x_ss
        for f in qd.static(range(4)):
            pos = soft_state.verts_pos[v[f], i_b]
            pos_ss = soft_state.verts_pos_stage_start[v[f], i_b]
            vel_ss = soft_state.verts_vel_stage_start[v[f], i_b]
            for k in qd.static(range(3)):
                x[f, k] = pos[k]
                a[f, k] = pos[k] - pos_ss[k] - h * vel_ss[k]
                b[f, k] = pos[k] - pos_ss[k]

        # Inertia and mass damping through the consistent mass matrix (rho V / 20) [2 1 1 1; ...] (x) I.
        m_unit = rho * vol / 20.0
        c_inertia = m_unit / (h * h)
        c_damping = m_unit * soft_info.entities_mass_damping[i_e] / h
        energy = gs.qd_float(0.0)
        res = qd.Matrix.zero(gs.qd_float, 4, 3)
        for f in qd.static(range(4)):
            for g in qd.static(range(4)):
                m_fg = gs.qd_float(CONSISTENT_MASS[f][g])
                for k in qd.static(range(3)):
                    res[f, k] += m_fg * (c_inertia * a[g, k] + c_damping * b[g, k])
                    if qd.static(assem_obj):
                        energy += 0.5 * m_fg * (c_inertia * a[f, k] * a[g, k] + c_damping * b[f, k] * b[g, k])

        # Gravity: one quadrature point, each node carries a quarter of the weight.
        if soft_info.entities_has_gravity[i_e]:
            gravity = mochi_info.gravity[i_b]
            for f in qd.static(range(4)):
                for k in qd.static(range(3)):
                    res[f, k] -= 0.25 * rho * vol * gravity[k]
                    if qd.static(assem_obj):
                        energy -= 0.25 * rho * vol * gravity[k] * x[f, k]

        # Elastic stress: F = Ds Dm^-1, residual V P grad N_f.
        F = func_deformation_gradient(
            soft_state.verts_pos[v[0], i_b],
            soft_state.verts_pos[v[1], i_b],
            soft_state.verts_pos[v[2], i_b],
            soft_state.verts_pos[v[3], i_b],
            Dm_inv,
        )
        g0, g1, g2, g3 = func_shape_gradients(Dm_inv)
        grads = qd.Matrix.rows([g0, g1, g2, g3])
        P = func_elastic_stress(model, F, mu, lam)
        if qd.static(assem_obj):
            energy += vol * func_elastic_energy(model, F, mu, lam)
        kappa = soft_info.entities_stiffness_damping[i_e] / h
        has_stiffness_damping = kappa > 0.0
        if has_stiffness_damping:
            energy_damping, S_visc = func_stiffness_damping_stress(
                F, soft_state.elems_F_stage_start[i_el, i_b], mu, lam, kappa
            )
            P += F @ S_visc
            if qd.static(assem_obj):
                energy += vol * energy_damping
        for f in qd.static(range(4)):
            g_f = qd.Vector([grads[f, 0], grads[f, 1], grads[f, 2]], dt=gs.qd_float)
            Pg = vol * (P @ g_f)
            for k in qd.static(range(3)):
                res[f, k] += Pg[k]

        if qd.static(assem_res):
            for f in qd.static(range(4)):
                i_d = func_soft_dof(v[f], 0, soft_info)
                for k in qd.static(range(3)):
                    qd.atomic_add(mochi_state.res[i_d + k, i_b], res[f, k])
        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], energy)

        if qd.static(assem_dres):
            C = func_elastic_tangent(model, F, mu, lam, EPS, True)
            K = qd.Matrix.zero(gs.qd_float, 12, 12)
            for f in qd.static(range(4)):
                g_f = qd.Vector([grads[f, 0], grads[f, 1], grads[f, 2]], dt=gs.qd_float)
                for g in qd.static(range(4)):
                    g_g = qd.Vector([grads[g, 0], grads[g, 1], grads[g, 2]], dt=gs.qd_float)
                    block = qd.Matrix.zero(gs.qd_float, 3, 3)
                    for r in qd.static(range(3)):
                        for c in qd.static(range(3)):
                            value = gs.qd_float(0.0)
                            for m in qd.static(range(3)):
                                for n in qd.static(range(3)):
                                    value += g_f[m] * C[3 * r + m, 3 * c + n] * g_g[n]
                            block[r, c] = vol * value
                    if has_stiffness_damping:
                        block += vol * func_stiffness_damping_block(F, g_f, g_g, mu, lam, kappa)
                    m_fg = gs.qd_float(CONSISTENT_MASS[f][g]) * (c_inertia + c_damping)
                    for k in qd.static(range(3)):
                        block[k, k] += m_fg
                    for r in qd.static(range(3)):
                        for c in qd.static(range(3)):
                            K[3 * f + r, 3 * g + c] = block[r, c]
                    if qd.static(f == g):
                        qd.atomic_add(soft_state.verts_H_diag[v[f], i_b], block)
            soft_state.elems_H[i_el, i_b] = K


@qd.kernel
def kernel_soft_dirichlet(
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    skip_ls_done: qd.template(),
):
    """Zero the residual of the fixed vertices (their rows of the Newton system are identities)."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        if func_is_env_active(i_b, mochi_state, skip_ls_done) and soft_state.verts_is_fixed[i_v, i_b]:
            for k in qd.static(range(3)):
                mochi_state.res[func_soft_dof(i_v, k, soft_info), i_b] = 0.0


# ------------------------------------------------------------------------------------
# --------------------------------------- contact ------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_soft_conservative_bounds(
    mochi_info: MochiInfo,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """World bounds of every deformable entity that hold for the whole step, from the stage-start positions and
    velocities of its vertices."""
    n_verts = soft_state.verts_pos.shape[0]
    n_entities = soft_state.entities_step_aabb_min.shape[0]
    _B = soft_state.verts_pos.shape[1]
    dt = mochi_info.dt[None]
    margin = mochi_info.broadphase_margin[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_b in qd.ndrange(n_entities, _B):
        soft_state.entities_step_aabb_min[i_e, i_b] = qd.Vector([gs.qd_float(1e30)] * 3, dt=gs.qd_float)
        soft_state.entities_step_aabb_max[i_e, i_b] = qd.Vector([gs.qd_float(-1e30)] * 3, dt=gs.qd_float)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        i_e = soft_info.verts_entity_idx[i_v]
        speed = soft_state.verts_vel_stage_start[i_v, i_b].norm()
        speed = CONSERVATIVE_SPEED_SCALE * speed + CONSERVATIVE_MAX_ACCEL * dt
        if soft_info.entities_has_gravity[i_e]:
            speed += mochi_info.gravity[i_b].norm() * dt
        pad = margin + speed * dt
        pos = soft_state.verts_pos_stage_start[i_v, i_b]
        for k in qd.static(range(3)):
            qd.atomic_min(soft_state.entities_step_aabb_min[i_e, i_b][k], pos[k] - pad)
            qd.atomic_max(soft_state.entities_step_aabb_max[i_e, i_b][k], pos[k] + pad)


@qd.kernel
def kernel_soft_broadphase(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    errno: qd.Tensor,
):
    """Enumerate the (deformable entity, collider geom) pairs whose conservative bounds overlap within the step."""
    n_entities = soft_state.entities_step_aabb_min.shape[0]
    n_geoms = dyn_state.geoms.pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    max_pairs = soft_state.pair_entity_a.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        soft_state.n_pairs[i_b] = 0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_gb, i_b in qd.ndrange(n_entities, n_geoms, _B):
        if not mochi_state.is_active[i_b]:
            continue
        if soft_info.entities_sample_end[i_e] <= soft_info.entities_sample_start[i_e]:
            continue
        if mochi_info.geoms.collider_type[i_gb] == COLLIDER_TYPE.NONE:
            continue
        i_lb = dyn_info.geoms.link_idx[i_gb]
        if not soft_info.entities_links_pair_enabled[i_e, i_lb]:
            continue
        if mochi_info.geoms.collider_type[i_gb] != COLLIDER_TYPE.PLANE:
            band = (
                contact_state.links_step_pad[i_lb, i_b]
                + mochi_info.geoms.penalty_threshold[i_gb]
                + 2.0 * mochi_info.geoms.penalty_smoothing_half_distance[i_gb]
            )
            geom_min = dyn_state.geoms.aabb_min[i_gb, i_b] - band
            geom_max = dyn_state.geoms.aabb_max[i_gb, i_b] + band
            if (soft_state.entities_step_aabb_max[i_e, i_b] < geom_min).any():
                continue
            if (soft_state.entities_step_aabb_min[i_e, i_b] > geom_max).any():
                continue
        i_p = qd.atomic_add(soft_state.n_pairs[i_b], 1)
        if i_p < max_pairs:
            soft_state.pair_entity_a[i_p, i_b] = i_e
            soft_state.pair_link_b[i_p, i_b] = i_lb
            soft_state.pair_geom_b[i_p, i_b] = i_gb
        else:
            qd.atomic_or(errno[i_b], array_class.ErrorCode.OVERFLOW_MOCHI_CONTACT_PAIRS)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        soft_state.n_pairs[i_b] = qd.min(soft_state.n_pairs[i_b], max_pairs)


@qd.kernel
def kernel_soft_contact_eval(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    max_samples_per_entity: int,
    assem_res: qd.template(),
    assem_dres: qd.template(),
    skip_ls_done: qd.template(),
    record: qd.template(),
    errno: qd.Tensor,
):
    """Evaluate every boundary sample of every candidate pair against its collider at the current vertex positions:
    residual on the three vertices (barycentric weights), per-sample matrix D = -w df/dp recorded for the vertex and
    coupling blocks, and the rigid-side accumulators of the pair."""
    max_pairs = soft_state.pair_entity_a.shape[0]
    max_hits = soft_state.hit_sample.shape[0]
    _B = soft_state.verts_pos.shape[1]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_s_, i_b in qd.ndrange(max_pairs, max_samples_per_entity, _B):
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        if i_p >= soft_state.n_pairs[i_b]:
            continue
        i_e = soft_state.pair_entity_a[i_p, i_b]
        i_s = soft_info.entities_sample_start[i_e] + i_s_
        if i_s >= soft_info.entities_sample_end[i_e]:
            continue
        i_lb = soft_state.pair_link_b[i_p, i_b]
        i_gb = soft_state.pair_geom_b[i_p, i_b]

        tri = soft_info.samples_tri[i_s]
        bary = soft_info.samples_bary[i_s]
        x0 = soft_state.verts_pos[tri[0], i_b]
        x1 = soft_state.verts_pos[tri[1], i_b]
        x2 = soft_state.verts_pos[tri[2], i_b]
        pos = bary[0] * x0 + bary[1] * x1 + bary[2] * x2
        thr = mochi_info.geoms.penalty_threshold[i_gb]
        h = mochi_info.geoms.penalty_smoothing_half_distance[i_gb]
        band = thr + 2.0 * h
        if mochi_info.geoms.collider_type[i_gb] != COLLIDER_TYPE.PLANE:
            if (pos < dyn_state.geoms.aabb_min[i_gb, i_b] - band).any():
                continue
            if (pos > dyn_state.geoms.aabb_max[i_gb, i_b] + band).any():
                continue

        pos_g = dyn_state.geoms.pos[i_gb, i_b]
        quat_g = dyn_state.geoms.quat[i_gb, i_b]
        pos_geom = gu.qd_inv_transform_by_trans_quat(pos, pos_g, quat_g)
        is_valid, d, grad = query_collider(i_gb, pos_geom, dyn_info.geoms, mochi_info.geoms, sdf_info, mochi_config)
        if not is_valid or d > band:
            continue

        # Colliding normal of the deformed triangle and stage displacement of the sample, in the collider frame.
        normal_world = gu.qd_normalize((x1 - x0).cross(x2 - x0), EPS)
        normal_geom = gu.qd_inv_transform_by_quat(normal_world, quat_g)
        pos_start = (
            bary[0] * soft_state.verts_pos_stage_start[tri[0], i_b]
            + bary[1] * soft_state.verts_pos_stage_start[tri[1], i_b]
            + bary[2] * soft_state.verts_pos_stage_start[tri[2], i_b]
        )
        pos_geom_start = gu.qd_inv_transform_by_trans_quat(
            pos_start, mochi_state.geoms_pos_stage_start[i_gb, i_b], mochi_state.geoms_quat_stage_start[i_gb, i_b]
        )
        p_rel = pos_geom - pos_geom_start
        d_start = d - grad.dot(p_rel)

        is_static_b = not mochi_info.links.is_dynamic[i_lb]
        k = qd.sqrt(soft_info.entities_penalty_coefficient[i_e] * mochi_info.geoms.penalty_coefficient[i_gb])
        falloff = qd.sqrt(soft_info.entities_friction_falloff_vel[i_e] * mochi_info.geoms.friction_falloff_vel[i_gb])
        if is_static_b:
            k = soft_info.entities_penalty_coefficient[i_e]
            falloff = soft_info.entities_friction_falloff_vel[i_e]
        mu = qd.sqrt(soft_info.entities_friction[i_e] * mochi_info.geoms.friction[i_gb])
        c_visc = qd.sqrt(soft_info.entities_viscous_friction[i_e] * mochi_info.geoms.viscous_friction[i_gb])
        c_ndamp = qd.sqrt(
            soft_info.entities_normal_viscous_damping[i_e] * mochi_info.geoms.normal_viscous_damping[i_gb]
        )
        max_align = mochi_info.geoms.max_alignment_normals[i_gb]

        energy, force_geom, dforce_geom, _ = collision_response(
            d,
            grad,
            normal_geom,
            p_rel,
            d_start,
            k,
            h,
            thr,
            mu,
            falloff,
            c_visc,
            c_ndamp,
            max_align,
            mochi_state.dt_stage[i_b],
            EPS,
            mochi_config,
        )

        w = soft_info.samples_weight[i_s]
        R_g = gu.qd_quat_to_R(quat_g, EPS)
        force = R_g @ force_geom
        D = -w * (R_g @ dforce_geom @ R_g.transpose())
        r_b = pos - dyn_state.links.pos[i_lb, i_b]
        S_b = skew(r_b)

        if qd.static(assem_res):
            for i in qd.static(range(3)):
                func_add_soft_vec(mochi_state.res, tri[i], i_b, -(w * bary[i]) * force, soft_info)
        qd.atomic_add(soft_state.acc_f[i_p, i_b], w * force)
        qd.atomic_add(soft_state.acc_q[i_p, i_b], w * r_b.cross(force))
        qd.atomic_add(soft_state.acc_D[i_p, i_b], D)
        qd.atomic_add(soft_state.acc_SD[i_p, i_b], S_b @ D)
        qd.atomic_add(soft_state.acc_SDS[i_p, i_b], S_b @ D @ S_b)
        qd.atomic_add(soft_state.acc_obj[i_p, i_b], w * energy)
        qd.atomic_add(soft_state.n_hits[i_p, i_b], 1)

        if qd.static(assem_dres):
            for i in qd.static(range(3)):
                qd.atomic_add(soft_state.verts_H_diag[tri[i], i_b], (bary[i] * bary[i]) * D)
        if qd.static(assem_dres or record):
            i_h = qd.atomic_add(soft_state.n_soft_hits[i_b], 1)
            if i_h < max_hits:
                soft_state.hit_sample[i_h, i_b] = i_s
                soft_state.hit_link_b[i_h, i_b] = -1 if is_static_b else i_lb
                soft_state.hit_r_b[i_h, i_b] = r_b
                soft_state.hit_D[i_h, i_b] = D
                soft_state.hit_force[i_h, i_b] = w * force
                soft_state.hit_pos[i_h, i_b] = pos
                soft_state.hit_normal[i_h, i_b] = gu.qd_normalize(R_g @ grad, EPS)
                soft_state.hit_distance[i_h, i_b] = d
            else:
                qd.atomic_or(errno[i_b], array_class.ErrorCode.OVERFLOW_MOCHI_CONTACTS)
        if qd.static(record):
            for i in qd.static(range(3)):
                qd.atomic_add(soft_state.verts_contact_force[tri[i], i_b], (w * bary[i]) * force)
            qd.atomic_add(dyn_state.links.contact_force[i_lb, i_b], -w * force)


@qd.kernel
def kernel_soft_pairs_to_blocks(
    dyn_state: array_class.DynState,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    assem_obj: qd.template(),
    assem_res: qd.template(),
    assem_dres: qd.template(),
    skip_ls_done: qd.template(),
):
    """Rigid side of the deformable-rigid pairs: residual and 6x6 block of the collider link (its point Jacobian is
    -[I, -[r_b]x], see the rigid kernel_pairs_to_blocks)."""
    max_pairs = soft_state.pair_entity_a.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_b in qd.ndrange(max_pairs, _B):
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        if i_p >= soft_state.n_pairs[i_b] or soft_state.n_hits[i_p, i_b] == 0:
            continue
        i_lb = soft_state.pair_link_b[i_p, i_b]
        if qd.static(assem_obj):
            qd.atomic_add(mochi_state.obj[i_b], soft_state.acc_obj[i_p, i_b])
        if not mochi_info.links.is_dynamic[i_lb]:
            continue
        if qd.static(assem_res):
            F = soft_state.acc_f[i_p, i_b]
            Q = soft_state.acc_q[i_p, i_b]
            for k in qd.static(range(3)):
                qd.atomic_add(mochi_state.links_res[i_lb, i_b][k], F[k])
                qd.atomic_add(mochi_state.links_res[i_lb, i_b][3 + k], Q[k])
        if qd.static(assem_dres):
            Dbar = soft_state.acc_D[i_p, i_b]
            Sh = soft_state.acc_SD[i_p, i_b]
            Sh2 = soft_state.acc_SDS[i_p, i_b]
            ShT = Sh.transpose()
            for k in qd.static(range(3)):
                for l in qd.static(range(3)):
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][k, l], Dbar[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][k, 3 + l], ShT[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][3 + k, l], Sh[k, l])
                    qd.atomic_add(mochi_state.H_diag[i_lb, i_b][3 + k, 3 + l], -Sh2[k, l])


# ------------------------------------------------------------------------------------
# ----------------------------------- linear algebra ---------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_soft_point_displacement(
    i_lb,
    i_b,
    r_b,
    src: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """Displacement of the collider point (lever arm r_b about the link origin) per the link's ancestor increments."""
    v6 = func_jacobian_times_dofs(i_lb, i_b, src, dyn_state, dyn_info, rigid_config)
    vel = qd.Vector([v6[0], v6[1], v6[2]], dt=gs.qd_float)
    ang = qd.Vector([v6[3], v6[4], v6[5]], dt=gs.qd_float)
    return vel + ang.cross(r_b)


@qd.func
def func_soft_point_force_add(
    i_lb,
    i_b,
    r_b,
    g,
    dst: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    """dst += J_b^T (g, r_b x g): a force g at the collider point as a wrench on the link's ancestor degrees of
    freedom."""
    torque = r_b.cross(g)
    func_jacobian_transpose_add(
        i_lb,
        i_b,
        qd.Vector([g[0], g[1], g[2], torque[0], torque[1], torque[2]], dt=gs.qd_float),
        dst,
        dyn_state,
        dyn_info,
        rigid_config,
    )


@qd.func
def func_soft_matvec(
    src: qd.Tensor,
    dst: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """dst += H_soft src for the running conjugate gradient environments: element blocks, per-sample vertex blocks
    b_i b_j D and the coupling with the collider links -b_i D J_b; fixed vertices act as identity rows."""
    n_verts = soft_state.verts_pos.shape[0]
    n_elems = soft_state.elems_H.shape[0]
    max_hits = soft_state.hit_sample.shape[0]
    _B = soft_state.verts_pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_el, i_b in qd.ndrange(n_elems, _B):
        if not mochi_state.pcg_is_active[i_b]:
            continue
        v = soft_info.elems_v[i_el]
        x = qd.Vector.zero(gs.qd_float, 12)
        for f in qd.static(range(4)):
            if not soft_state.verts_is_fixed[v[f], i_b]:
                s = func_read_soft_vec(src, v[f], i_b, soft_info)
                for k in qd.static(range(3)):
                    x[3 * f + k] = s[k]
        y = soft_state.elems_H[i_el, i_b] @ x
        for f in qd.static(range(4)):
            if not soft_state.verts_is_fixed[v[f], i_b]:
                func_add_soft_vec(
                    dst, v[f], i_b, qd.Vector([y[3 * f], y[3 * f + 1], y[3 * f + 2]], dt=gs.qd_float), soft_info
                )

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_h, i_b in qd.ndrange(max_hits, _B):
        if not mochi_state.pcg_is_active[i_b] or i_h >= soft_state.n_soft_hits[i_b]:
            continue
        i_s = soft_state.hit_sample[i_h, i_b]
        tri = soft_info.samples_tri[i_s]
        bary = soft_info.samples_bary[i_s]
        D = soft_state.hit_D[i_h, i_b]
        i_lb = soft_state.hit_link_b[i_h, i_b]
        # Relative displacement of the sample against the collider point.
        dp = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(3)):
            if not soft_state.verts_is_fixed[tri[i], i_b]:
                dp += bary[i] * func_read_soft_vec(src, tri[i], i_b, soft_info)
        r_b = soft_state.hit_r_b[i_h, i_b]
        if i_lb >= 0:
            dp -= func_soft_point_displacement(i_lb, i_b, r_b, src, dyn_state, dyn_info, rigid_config)
        g = D @ dp
        for i in qd.static(range(3)):
            if not soft_state.verts_is_fixed[tri[i], i_b]:
                func_add_soft_vec(dst, tri[i], i_b, bary[i] * g, soft_info)
        if i_lb >= 0:
            # The rigid-rigid part J_b^T D J_b is already in the link block; only the coupling remains.
            g_soft = qd.Vector.zero(gs.qd_float, 3)
            for i in qd.static(range(3)):
                if not soft_state.verts_is_fixed[tri[i], i_b]:
                    g_soft += bary[i] * func_read_soft_vec(src, tri[i], i_b, soft_info)
            func_soft_point_force_add(i_lb, i_b, r_b, -(D @ g_soft), dst, dyn_state, dyn_info, rigid_config)

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        if mochi_state.pcg_is_active[i_b] and soft_state.verts_is_fixed[i_v, i_b]:
            for k in qd.static(range(3)):
                i_d = func_soft_dof(i_v, k, soft_info)
                dst[i_d, i_b] = src[i_d, i_b]


@qd.func
def func_soft_precondition(
    r: qd.Tensor,
    z: qd.Tensor,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    eps,
):
    """z = M^-1 r on the vertex degrees of freedom with the block-Jacobi preconditioner of the 3x3 vertex blocks."""
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        if not mochi_state.pcg_is_active[i_b]:
            continue
        r_v = func_read_soft_vec(r, i_v, i_b, soft_info)
        z_v = r_v
        if not soft_state.verts_is_fixed[i_v, i_b]:
            H = soft_state.verts_H_diag[i_v, i_b]
            det = H.determinant()
            if det > eps:
                z_v = H.inverse() @ r_v
            else:
                for k in qd.static(range(3)):
                    z_v[k] = r_v[k] / qd.max(H[k, k], eps)
        i_d = func_soft_dof(i_v, 0, soft_info)
        for k in qd.static(range(3)):
            z[i_d + k, i_b] = z_v[k]


@qd.kernel
def kernel_soft_condense_dense(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Add the deformable blocks to the dense Hessian of every running environment and impose the Dirichlet rows and
    columns (zero off-diagonal, unit diagonal) of the fixed vertices."""
    n_verts = soft_state.verts_pos.shape[0]
    n_elems = soft_state.elems_H.shape[0]
    max_hits = soft_state.hit_sample.shape[0]
    n_dofs = mochi_state.res.shape[0]
    _B = soft_state.verts_pos.shape[1]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_el, i_b in qd.ndrange(n_elems, _B):
        if not mochi_state.is_active[i_b]:
            continue
        v = soft_info.elems_v[i_el]
        K = soft_state.elems_H[i_el, i_b]
        for f in qd.static(range(4)):
            for g in qd.static(range(4)):
                i_d = func_soft_dof(v[f], 0, soft_info)
                j_d = func_soft_dof(v[g], 0, soft_info)
                for r in qd.static(range(3)):
                    for c in qd.static(range(3)):
                        qd.atomic_add(mochi_state.H_dense[i_b, i_d + r, j_d + c], K[3 * f + r, 3 * g + c])

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_h, i_b in qd.ndrange(max_hits, _B):
        if not mochi_state.is_active[i_b] or i_h >= soft_state.n_soft_hits[i_b]:
            continue
        i_s = soft_state.hit_sample[i_h, i_b]
        tri = soft_info.samples_tri[i_s]
        bary = soft_info.samples_bary[i_s]
        D = soft_state.hit_D[i_h, i_b]
        for i in qd.static(range(3)):
            for j in qd.static(range(3)):
                i_d = func_soft_dof(tri[i], 0, soft_info)
                j_d = func_soft_dof(tri[j], 0, soft_info)
                for r in qd.static(range(3)):
                    for c in qd.static(range(3)):
                        qd.atomic_add(mochi_state.H_dense[i_b, i_d + r, j_d + c], bary[i] * bary[j] * D[r, c])
        i_lb = soft_state.hit_link_b[i_h, i_b]
        if i_lb >= 0:
            r_b = soft_state.hit_r_b[i_h, i_b]
            i_a = i_lb
            while i_a != -1:
                I_a = [i_a, i_b] if qd.static(rigid_config.batch_links_info) else i_a
                for k_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
                    vel, ang = func_link_dof_jacobian(i_lb, k_d, i_b, dyn_state)
                    column = -(D @ (vel + ang.cross(r_b)))
                    for i in qd.static(range(3)):
                        i_d = func_soft_dof(tri[i], 0, soft_info)
                        for r in qd.static(range(3)):
                            qd.atomic_add(mochi_state.H_dense[i_b, i_d + r, k_d], bary[i] * column[r])
                            qd.atomic_add(mochi_state.H_dense[i_b, k_d, i_d + r], bary[i] * column[r])
                i_a = dyn_info.links.parent_idx[I_a]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        if not mochi_state.is_active[i_b] or not soft_state.verts_is_fixed[i_v, i_b]:
            continue
        for k in qd.static(range(3)):
            i_d = func_soft_dof(i_v, k, soft_info)
            for j_d in range(n_dofs):
                mochi_state.H_dense[i_b, i_d, j_d] = 0.0
                mochi_state.H_dense[i_b, j_d, i_d] = 0.0
            mochi_state.H_dense[i_b, i_d, i_d] = 1.0


# ------------------------------------------------------------------------------------
# --------------------------------------- io -----------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_soft_get_state(
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    pos_prev: qd.types.ndarray(),
    vel_prev: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = soft_state.verts_pos.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b in qd.ndrange(n_verts, _B):
        for k in qd.static(range(3)):
            pos[i_b, i_v, k] = soft_state.verts_pos[i_v, i_b][k]
            vel[i_b, i_v, k] = soft_state.verts_vel[i_v, i_b][k]
            for j in qd.static(range(N_HISTORY)):
                pos_prev[i_b, j, i_v, k] = soft_state.verts_pos_prev[j, i_v, i_b][k]
                vel_prev[i_b, j, i_v, k] = soft_state.verts_vel_prev[j, i_v, i_b][k]


@qd.kernel
def kernel_soft_set_state(
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    pos_prev: qd.types.ndarray(),
    vel_prev: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = soft_state.verts_pos.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for k in qd.static(range(3)):
            soft_state.verts_pos[i_v, i_b][k] = pos[i_b, i_v, k]
            soft_state.verts_vel[i_v, i_b][k] = vel[i_b, i_v, k]
            for j in qd.static(range(N_HISTORY)):
                soft_state.verts_pos_prev[j, i_v, i_b][k] = pos_prev[i_b, j, i_v, k]
                soft_state.verts_vel_prev[j, i_v, i_b][k] = vel_prev[i_b, j, i_v, k]


@qd.kernel
def kernel_soft_get_entity_state(
    v_start: qd.i32,
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = pos.shape[1]
    _B = pos.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b in qd.ndrange(n_verts, _B):
        for k in qd.static(range(3)):
            pos[i_b, i_v_, k] = soft_state.verts_pos[v_start + i_v_, i_b][k]
            vel[i_b, i_v_, k] = soft_state.verts_vel[v_start + i_v_, i_b][k]


@qd.kernel
def kernel_soft_get_vertices_field(
    envs_idx: qd.types.ndarray(),
    v_start: qd.i32,
    out: qd.types.ndarray(),
    field: qd.Tensor,
    rigid_config: qd.template(),
):
    n_verts = out.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        for k in qd.static(range(3)):
            out[i_b_, i_v_, k] = field[v_start + i_v_, envs_idx[i_b_]][k]


@qd.kernel
def kernel_soft_set_vertices_positions(
    envs_idx: qd.types.ndarray(),
    v_start: qd.i32,
    pos: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """Set vertex positions; their velocity history is reset (the next step starts from rest at the new positions
    unless velocities are set afterwards)."""
    n_verts = pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = v_start + i_v_
        value = qd.Vector([pos[i_b_, i_v_, 0], pos[i_b_, i_v_, 1], pos[i_b_, i_v_, 2]], dt=gs.qd_float)
        soft_state.verts_pos[i_v, i_b] = value
        soft_state.verts_vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        for j in qd.static(range(N_HISTORY)):
            soft_state.verts_pos_prev[j, i_v, i_b] = value
            soft_state.verts_vel_prev[j, i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_soft_set_vertices_velocities(
    envs_idx: qd.types.ndarray(),
    v_start: qd.i32,
    vel: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_verts = vel.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = v_start + i_v_
        value = qd.Vector([vel[i_b_, i_v_, 0], vel[i_b_, i_v_, 1], vel[i_b_, i_v_, 2]], dt=gs.qd_float)
        soft_state.verts_vel[i_v, i_b] = value
        for j in qd.static(range(N_HISTORY)):
            soft_state.verts_vel_prev[j, i_v, i_b] = value


@qd.kernel
def kernel_soft_set_vertices_fixed(
    envs_idx: qd.types.ndarray(),
    verts_idx: qd.types.ndarray(),
    is_fixed: qd.i32,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_v_]
        soft_state.verts_is_fixed[i_v, i_b] = is_fixed != 0
        if is_fixed != 0:
            soft_state.verts_vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            for j in qd.static(range(N_HISTORY)):
                soft_state.verts_vel_prev[j, i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_soft_set_entity_contact_params(
    i_e: qd.i32,
    params: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
):
    soft_info.entities_penalty_coefficient[i_e] = params[0]
    soft_info.entities_friction[i_e] = params[1]
    soft_info.entities_penalty_smoothing_half_distance[i_e] = params[2]
    soft_info.entities_penalty_threshold[i_e] = params[3]
    soft_info.entities_friction_falloff_vel[i_e] = params[4]
    soft_info.entities_viscous_friction[i_e] = params[5]
    soft_info.entities_normal_viscous_damping[i_e] = params[6]


@qd.kernel
def kernel_soft_get_state_render(
    vverts_render: qd.Tensor,
    vverts_vert_idx: qd.Tensor,
    envs_offset: qd.types.ndarray(),
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    n_vverts = vverts_vert_idx.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_vv, i_b in qd.ndrange(n_vverts, _B):
        pos = soft_state.verts_pos[vverts_vert_idx[i_vv], i_b]
        for k in qd.static(range(3)):
            vverts_render[i_vv, i_b][k] = qd.cast(pos[k] + envs_offset[i_b, k], qd.f32)


@qd.kernel
def kernel_soft_init_render(vert_idx: qd.types.ndarray(), vverts_vert_idx: qd.Tensor):
    for i_vv in range(vert_idx.shape[0]):
        vverts_vert_idx[i_vv] = vert_idx[i_vv]


@qd.kernel
def kernel_soft_set_links_pair_enabled(
    entities_links_pair_enabled: qd.types.ndarray(),
    soft_info: MochiSoftInfo,
    rigid_config: qd.template(),
):
    n_entities = entities_links_pair_enabled.shape[0]
    n_links = entities_links_pair_enabled.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_l in qd.ndrange(n_entities, n_links):
        soft_info.entities_links_pair_enabled[i_e, i_l] = entities_links_pair_enabled[i_e, i_l]
