# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Time integration of the MochiSolver: multistep history, stage-start extrapolation (backward Euler and BDF2) and
the finite-difference velocity recovered from the solved poses."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .data import INTEGRATOR, N_HISTORY, SOLVE_STATUS, MochiInfo, MochiState
from .lie import sym, vee, vsym_from_omega

# BDF2 extrapolation coefficient of the step before the previous one: x_start = x_-1 + BDF2_ALPHA_2 (x_-2 - x_-1).
BDF2_ALPHA_2 = -1.0 / 3.0
BDF2_BETA = 2.0 / 3.0


@qd.kernel
def kernel_step_start(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """Shift the multistep history by one step and build the stage-start reference of the new step: the previous pose
    for backward Euler, the two-step extrapolation for BDF2, which is also the warm start of the Newton solve."""
    n_qs = rigid_info.qpos.shape[0]
    n_dofs = dyn_state.dofs.vel.shape[0]
    n_links = dyn_state.links.pos.shape[0]
    _B = mochi_state.n_hist.shape[0]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        mochi_state.n_hist[i_b] = qd.min(mochi_state.n_hist[i_b] + 1, N_HISTORY)
        beta = gs.qd_float(1.0)
        if qd.static(mochi_config.integrator == INTEGRATOR.BDF2):  # noqa: SIM102
            if mochi_state.n_hist[i_b] >= 2:
                beta = BDF2_BETA
        mochi_state.dt_stage[i_b] = beta * mochi_info.dt[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_b in qd.ndrange(n_qs, _B):
        mochi_state.qpos_prev[1, i_q, i_b] = mochi_state.qpos_prev[0, i_q, i_b]
        mochi_state.qpos_prev[0, i_q, i_b] = rigid_info.qpos[i_q, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        mochi_state.dofs_vel_prev[1, i_d, i_b] = mochi_state.dofs_vel_prev[0, i_d, i_b]
        mochi_state.dofs_vel_prev[0, i_d, i_b] = dyn_state.dofs.vel[i_d, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        mochi_state.links_vsym_prev[1, i_l, i_b] = mochi_state.links_vsym_prev[0, i_l, i_b]
        mochi_state.links_vsym_prev[0, i_l, i_b] = mochi_state.links_vsym[i_l, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if not mochi_info.links.is_dynamic[i_l]:
            continue
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        q_start = dyn_info.links.q_start[I_l]
        dof_start = dyn_info.links.dof_start[I_l]

        is_bdf2 = False
        if qd.static(mochi_config.integrator == INTEGRATOR.BDF2):
            is_bdf2 = mochi_state.n_hist[i_b] >= 2

        if is_bdf2:
            for k in qd.static(range(3)):
                x1 = mochi_state.qpos_prev[0, q_start + k, i_b]
                x2 = mochi_state.qpos_prev[1, q_start + k, i_b]
                mochi_state.qpos_step_start[q_start + k, i_b] = x1 + BDF2_ALPHA_2 * (x2 - x1)
            quat1 = qd.Vector(
                [
                    mochi_state.qpos_prev[0, q_start + 3, i_b],
                    mochi_state.qpos_prev[0, q_start + 4, i_b],
                    mochi_state.qpos_prev[0, q_start + 5, i_b],
                    mochi_state.qpos_prev[0, q_start + 6, i_b],
                ],
                dt=gs.qd_float,
            )
            quat2 = qd.Vector(
                [
                    mochi_state.qpos_prev[1, q_start + 3, i_b],
                    mochi_state.qpos_prev[1, q_start + 4, i_b],
                    mochi_state.qpos_prev[1, q_start + 5, i_b],
                    mochi_state.qpos_prev[1, q_start + 6, i_b],
                ],
                dt=gs.qd_float,
            )
            # Lie extrapolation: exp(alpha_2 log(q_-2 q_-1^-1)) q_-1.
            rotvec = gu.qd_quat_to_rotvec(gu.qd_quat_mul(quat2, gu.qd_inv_quat(quat1)), EPS)
            quat0 = gu.qd_transform_quat_by_quat(quat1, gu.qd_rotvec_to_quat(BDF2_ALPHA_2 * rotvec, EPS))
            for k in qd.static(range(4)):
                mochi_state.qpos_step_start[q_start + 3 + k, i_b] = quat0[k]
            for k in qd.static(range(6)):
                v1 = mochi_state.dofs_vel_prev[0, dof_start + k, i_b]
                v2 = mochi_state.dofs_vel_prev[1, dof_start + k, i_b]
                mochi_state.dofs_vel_stage_start[dof_start + k, i_b] = v1 + BDF2_ALPHA_2 * (v2 - v1)
            S1 = mochi_state.links_vsym_prev[0, i_l, i_b]
            S2 = mochi_state.links_vsym_prev[1, i_l, i_b]
            mochi_state.links_vsym_stage_start[i_l, i_b] = S1 + BDF2_ALPHA_2 * (S2 - S1)
        else:
            for k in qd.static(range(7)):
                mochi_state.qpos_step_start[q_start + k, i_b] = mochi_state.qpos_prev[0, q_start + k, i_b]
            for k in qd.static(range(6)):
                mochi_state.dofs_vel_stage_start[dof_start + k, i_b] = mochi_state.dofs_vel_prev[0, dof_start + k, i_b]
            mochi_state.links_vsym_stage_start[i_l, i_b] = mochi_state.links_vsym_prev[0, i_l, i_b]

        # Single-stage schemes: the stage starts at the step start, which is also the warm start of the solve.
        for k in qd.static(range(7)):
            mochi_state.qpos_stage_start[q_start + k, i_b] = mochi_state.qpos_step_start[q_start + k, i_b]
            rigid_info.qpos[q_start + k, i_b] = mochi_state.qpos_step_start[q_start + k, i_b]


@qd.kernel
def kernel_set_qpos_from(
    src: qd.Tensor,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Copy a (n_qs, B) buffer into the generalized coordinates."""
    n_qs = rigid_info.qpos.shape[0]
    _B = rigid_info.qpos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_b in qd.ndrange(n_qs, _B):
        rigid_info.qpos[i_q, i_b] = src[i_q, i_b]


@qd.kernel
def kernel_store_stage_start_poses(
    dyn_state: array_class.DynState,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Snapshot the link and geom poses of the stage-start configuration (forward kinematics must have been run on
    the stage-start coordinates)."""
    n_links = dyn_state.links.pos.shape[0]
    n_geoms = dyn_state.geoms.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        mochi_state.links_pos_stage_start[i_l, i_b] = dyn_state.links.pos[i_l, i_b]
        mochi_state.links_quat_stage_start[i_l, i_b] = dyn_state.links.quat[i_l, i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_b in qd.ndrange(n_geoms, _B):
        mochi_state.geoms_pos_stage_start[i_g, i_b] = dyn_state.geoms.pos[i_g, i_b]
        mochi_state.geoms_quat_stage_start[i_g, i_b] = dyn_state.geoms.quat[i_g, i_b]


@qd.kernel
def kernel_post_stage(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Recover the velocity of every dynamic link by finite differences over the stage. The angular velocity is the
    antisymmetric part of (R - R_start)/h R^T and the symmetric part is kept so the next step extrapolates the same
    rotation increment. A diverged environment is reset to its previous pose at rest."""
    n_links = dyn_state.links.pos.shape[0]
    _B = dyn_state.links.pos.shape[1]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if not mochi_info.links.is_dynamic[i_l]:
            continue
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        q_start = dyn_info.links.q_start[I_l]
        dof_start = dyn_info.links.dof_start[I_l]

        if mochi_state.status[i_b] == SOLVE_STATUS.DIVERGED:
            for k in qd.static(range(7)):
                rigid_info.qpos[q_start + k, i_b] = mochi_state.qpos_prev[0, q_start + k, i_b]
            for k in qd.static(range(6)):
                dyn_state.dofs.vel[dof_start + k, i_b] = 0.0
            mochi_state.links_vsym[i_l, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            continue

        h = mochi_state.dt_stage[i_b]
        pos = qd.Vector(
            [rigid_info.qpos[q_start, i_b], rigid_info.qpos[q_start + 1, i_b], rigid_info.qpos[q_start + 2, i_b]],
            dt=gs.qd_float,
        )
        quat = qd.Vector(
            [
                rigid_info.qpos[q_start + 3, i_b],
                rigid_info.qpos[q_start + 4, i_b],
                rigid_info.qpos[q_start + 5, i_b],
                rigid_info.qpos[q_start + 6, i_b],
            ],
            dt=gs.qd_float,
        )
        pos_start = qd.Vector(
            [
                mochi_state.qpos_stage_start[q_start, i_b],
                mochi_state.qpos_stage_start[q_start + 1, i_b],
                mochi_state.qpos_stage_start[q_start + 2, i_b],
            ],
            dt=gs.qd_float,
        )
        quat_start = qd.Vector(
            [
                mochi_state.qpos_stage_start[q_start + 3, i_b],
                mochi_state.qpos_stage_start[q_start + 4, i_b],
                mochi_state.qpos_stage_start[q_start + 5, i_b],
                mochi_state.qpos_stage_start[q_start + 6, i_b],
            ],
            dt=gs.qd_float,
        )
        vel = (pos - pos_start) / h
        R = gu.qd_quat_to_R(quat, EPS)
        R_start = gu.qd_quat_to_R(quat_start, EPS)
        F = ((R - R_start) / h) @ R.transpose()
        omega = vee(F)
        for k in qd.static(range(3)):
            dyn_state.dofs.vel[dof_start + k, i_b] = vel[k]
            dyn_state.dofs.vel[dof_start + 3 + k, i_b] = omega[k]
        mochi_state.links_vsym[i_l, i_b] = sym(F)


@qd.kernel
def kernel_reset_history(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Invalidate the multistep history of the given environments (their next step is a backward Euler step) and
    rebuild the symmetric rotation-derivative correction from the current angular velocities, as required after the
    state has been set from outside the solver."""
    n_links = dyn_state.links.pos.shape[0]
    dt = mochi_info.dt[None]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        mochi_state.n_hist[envs_idx[i_b_]] = 0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b_ in qd.ndrange(n_links, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        S = qd.Matrix.zero(gs.qd_float, 3, 3)
        if mochi_info.links.is_dynamic[i_l]:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            dof_start = dyn_info.links.dof_start[I_l]
            omega = qd.Vector(
                [
                    dyn_state.dofs.vel[dof_start + 3, i_b],
                    dyn_state.dofs.vel[dof_start + 4, i_b],
                    dyn_state.dofs.vel[dof_start + 5, i_b],
                ],
                dt=gs.qd_float,
            )
            S = vsym_from_omega(omega, dt, EPS)
        mochi_state.links_vsym[i_l, i_b] = S
