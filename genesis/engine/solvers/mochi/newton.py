# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Per-environment control of the damped Newton iterations: residual norms, line search and convergence."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .data import LINESEARCH, SOLVE_STATUS, MochiInfo, MochiState


@qd.func
def func_is_env_active(i_b, mochi_state: MochiState, skip_ls_done: qd.template()):
    """Whether the Newton solve of an environment is still running, optionally excluding the environments whose line
    search has already accepted an iterate."""
    is_active = mochi_state.is_active[i_b]
    if qd.static(skip_ls_done):
        is_active = is_active and not mochi_state.ls_is_done[i_b]
    return is_active


@qd.kernel
def kernel_reset_newton(mochi_state: MochiState, rigid_config: qd.template()):
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        mochi_state.is_active[i_b] = True
        mochi_state.status[i_b] = SOLVE_STATUS.RUNNING
        mochi_state.n_iter[i_b] = 0
        mochi_state.ls_alpha[i_b] = 1.0
        mochi_state.ls_is_done[i_b] = False
        mochi_state.res_norm0[i_b] = 0.0
        mochi_state.res_norm0_w[i_b] = 0.0


@qd.kernel
def kernel_residual_norms(mochi_state: MochiState, rigid_config: qd.template(), skip_ls_done: qd.template()):
    """Plain and convergence-weighted squared norms of the residual of every running environment."""
    n_dofs = mochi_state.res.shape[0]
    _B = mochi_state.res.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            mochi_state.res_norm_sq[i_b] = 0.0
            mochi_state.res_w_sq[i_b] = 0.0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            r = mochi_state.res[i_d, i_b]
            qd.atomic_add(mochi_state.res_norm_sq[i_b], r * r)
            qd.atomic_add(mochi_state.res_w_sq[i_b], mochi_state.conv_w[i_d, i_b] * r * r)


@qd.kernel
def kernel_store_initial_norms(
    rigid_info: array_class.RigidInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Record the residual norms of the initial iterate, the reference of the relative convergence and divergence
    tests, and take it as the first line search reference."""
    n_qs = rigid_info.qpos.shape[0]
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        mochi_state.res_norm0[i_b] = qd.sqrt(mochi_state.res_norm_sq[i_b])
        mochi_state.res_norm0_w[i_b] = qd.sqrt(mochi_state.res_w_sq[i_b])
        mochi_state.ls_ref_norm_sq[i_b] = mochi_state.res_norm_sq[i_b]
        mochi_state.obj_ref[i_b] = mochi_state.obj[i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_b in qd.ndrange(n_qs, _B):
        mochi_state.qpos_ls_ref[i_q, i_b] = rigid_info.qpos[i_q, i_b]


@qd.kernel
def kernel_linesearch_begin(
    rigid_info: array_class.RigidInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Open a line search along the freshly solved Newton step: full step first, the current (accepted) iterate as
    reference, and the directional derivative of the objective along the step for the Armijo rule."""
    n_qs = rigid_info.qpos.shape[0]
    n_dofs = mochi_state.res.shape[0]
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        if mochi_state.is_active[i_b]:
            mochi_state.ls_alpha[i_b] = 1.0
            mochi_state.ls_is_done[i_b] = False
            mochi_state.ls_ref_norm_sq[i_b] = mochi_state.res_norm_sq[i_b]
            mochi_state.obj_ref[i_b] = mochi_state.obj[i_b]
            mochi_state.ls_slope[i_b] = 0.0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_b in qd.ndrange(n_qs, _B):
        if mochi_state.is_active[i_b]:
            mochi_state.qpos_ls_ref[i_q, i_b] = rigid_info.qpos[i_q, i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.is_active[i_b]:
            # The step taken is -dx, so the slope of the objective along it is -res . dx.
            qd.atomic_add(mochi_state.ls_slope[i_b], -mochi_state.res[i_d, i_b] * mochi_state.dx[i_d, i_b])


@qd.kernel
def kernel_apply_increment(
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """Move every dynamic link of the environments still searching to the trial iterate: the line search reference
    minus the scaled Newton step, composed on the rotation manifold."""
    n_links = mochi_info.links.is_dynamic.shape[0]
    _B = mochi_state.is_active.shape[0]
    EPS = mochi_info.EPS[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if not mochi_info.links.is_dynamic[i_l] or not func_is_env_active(i_b, mochi_state, True):
            continue
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        q_start = dyn_info.links.q_start[I_l]
        dof_start = dyn_info.links.dof_start[I_l]
        alpha = mochi_state.ls_alpha[i_b]
        for k in qd.static(range(3)):
            rigid_info.qpos[q_start + k, i_b] = (
                mochi_state.qpos_ls_ref[q_start + k, i_b] - alpha * mochi_state.dx[dof_start + k, i_b]
            )
        rotvec = -alpha * qd.Vector(
            [
                mochi_state.dx[dof_start + 3, i_b],
                mochi_state.dx[dof_start + 4, i_b],
                mochi_state.dx[dof_start + 5, i_b],
            ],
            dt=gs.qd_float,
        )
        quat_ref = qd.Vector(
            [
                mochi_state.qpos_ls_ref[q_start + 3, i_b],
                mochi_state.qpos_ls_ref[q_start + 4, i_b],
                mochi_state.qpos_ls_ref[q_start + 5, i_b],
                mochi_state.qpos_ls_ref[q_start + 6, i_b],
            ],
            dt=gs.qd_float,
        )
        quat = gu.qd_transform_quat_by_quat(quat_ref, gu.qd_rotvec_to_quat(rotvec, EPS))
        for k in qd.static(range(4)):
            rigid_info.qpos[q_start + 3 + k, i_b] = quat[k]


@qd.kernel
def kernel_linesearch_decide(
    rigid_info: array_class.RigidInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    is_last: qd.template(),
):
    """Accept the trial iterate of every searching environment when it improves on the reference (or on the last
    trial regardless, so that the solve always progresses), else halve the step."""
    n_qs = rigid_info.qpos.shape[0]
    _B = mochi_state.is_active.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        if not func_is_env_active(i_b, mochi_state, True):
            continue
        is_improved = True
        if qd.static(mochi_config.linesearch_type == LINESEARCH.RESIDUAL_NORM):
            is_improved = mochi_state.res_norm_sq[i_b] <= mochi_state.ls_ref_norm_sq[i_b]
        elif qd.static(mochi_config.linesearch_type == LINESEARCH.ARMIJO):
            is_improved = mochi_state.obj[i_b] <= (
                mochi_state.obj_ref[i_b]
                + mochi_info.linesearch_wolfe1[None] * mochi_state.ls_alpha[i_b] * mochi_state.ls_slope[i_b]
            )
        if is_improved or qd.static(is_last):
            mochi_state.ls_is_done[i_b] = True
            mochi_state.ls_ref_norm_sq[i_b] = mochi_state.res_norm_sq[i_b]
            mochi_state.obj_ref[i_b] = mochi_state.obj[i_b]
        else:
            mochi_state.ls_alpha[i_b] = mochi_state.ls_alpha[i_b] * mochi_info.linesearch_alpha[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_b in qd.ndrange(n_qs, _B):
        if mochi_state.is_active[i_b] and mochi_state.ls_is_done[i_b]:
            mochi_state.qpos_ls_ref[i_q, i_b] = rigid_info.qpos[i_q, i_b]


@qd.kernel
def kernel_convergence_check(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
    increment_iter: qd.template(),
    errno: qd.Tensor,
):
    """Classify every running environment from the residual of its accepted iterate: converged on the weighted norm
    (absolute or relative to the initial iterate), diverged on residual blow-up, stopped at the iteration budget."""
    _B = mochi_state.is_active.shape[0]
    abs_tol = mochi_info.newton_abs_tol[None]
    rel_tol = mochi_info.newton_rel_tol[None]
    div_abs_tol = mochi_info.explosion_abs_tol[None]
    div_rel_tol = mochi_info.explosion_rel_tol[None]
    max_iter = mochi_info.n_newton_iterations[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        if not mochi_state.is_active[i_b]:
            continue
        if qd.static(increment_iter):
            mochi_state.n_iter[i_b] = mochi_state.n_iter[i_b] + 1
        res_norm = qd.sqrt(mochi_state.res_norm_sq[i_b])
        res_norm_w = qd.sqrt(mochi_state.res_w_sq[i_b])
        is_diverged = qd.math.isnan(res_norm)
        if div_abs_tol > 0.0:
            is_diverged |= res_norm > div_abs_tol
            is_diverged |= (res_norm > div_rel_tol * mochi_state.res_norm0[i_b]) and (res_norm_w > abs_tol)
        if is_diverged:
            mochi_state.status[i_b] = SOLVE_STATUS.DIVERGED
            mochi_state.is_active[i_b] = False
            qd.atomic_or(errno[i_b], array_class.ErrorCode.MOCHI_DIVERGED)
        elif res_norm_w <= abs_tol or res_norm_w <= rel_tol * mochi_state.res_norm0_w[i_b]:
            mochi_state.status[i_b] = SOLVE_STATUS.CONVERGED
            mochi_state.is_active[i_b] = False
        elif mochi_state.n_iter[i_b] >= max_iter:
            mochi_state.status[i_b] = SOLVE_STATUS.STOPPED
            mochi_state.is_active[i_b] = False


@qd.kernel
def kernel_any_active(mochi_state: MochiState) -> qd.i32:
    n_active = 0
    for i_b in range(mochi_state.is_active.shape[0]):
        if mochi_state.is_active[i_b]:
            n_active += 1
    return n_active
