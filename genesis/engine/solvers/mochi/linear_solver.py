# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Linear solve of the Newton system H dx = res from its per-link diagonal and per-pair off-diagonal 6x6 blocks: a
dense Cholesky factorization of the condensed matrix, or a block-Jacobi preconditioned conjugate gradient working
directly on the blocks."""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .data import MochiContactState, MochiInfo, MochiState


@qd.kernel
def kernel_condense_dense(
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    rigid_config: qd.template(),
):
    """Scatter the Hessian blocks of every running environment into its dense matrix."""
    n_dofs = mochi_state.res.shape[0]
    n_links = mochi_state.H_diag.shape[0]
    max_pairs = contact_state.pair_link_a.shape[0]
    _B = mochi_state.is_active.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b, i_d, j_d in qd.ndrange(_B, n_dofs, n_dofs):
        if mochi_state.is_active[i_b]:
            mochi_state.H_dense[i_b, i_d, j_d] = 0.0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if not mochi_state.is_active[i_b] or not mochi_info.links.is_dynamic[i_l]:
            continue
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        dof_start = dyn_info.links.dof_start[I_l]
        for k in qd.static(range(6)):
            for l in qd.static(range(6)):
                mochi_state.H_dense[i_b, dof_start + k, dof_start + l] = mochi_state.H_diag[i_l, i_b][k, l]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_b in qd.ndrange(max_pairs, _B):
        if not mochi_state.is_active[i_b] or i_p >= contact_state.n_pairs[i_b]:
            continue
        if contact_state.n_hits[i_p, i_b] == 0:
            continue
        i_la = contact_state.pair_link_a[i_p, i_b]
        i_lb = contact_state.pair_link_b[i_p, i_b]
        if not (mochi_info.links.is_dynamic[i_la] and mochi_info.links.is_dynamic[i_lb]):
            continue
        I_la = [i_la, i_b] if qd.static(rigid_config.batch_links_info) else i_la
        I_lb = [i_lb, i_b] if qd.static(rigid_config.batch_links_info) else i_lb
        dof_a = dyn_info.links.dof_start[I_la]
        dof_b = dyn_info.links.dof_start[I_lb]
        for k in qd.static(range(6)):
            for l in qd.static(range(6)):
                value = mochi_state.H_off[i_p, i_b][k, l]
                qd.atomic_add(mochi_state.H_dense[i_b, dof_a + k, dof_b + l], value)
                qd.atomic_add(mochi_state.H_dense[i_b, dof_b + l, dof_a + k], value)


@qd.kernel
def kernel_cholesky_solve_dense(
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    """In-place Cholesky factorization of the dense matrix of every running environment followed by the two
    triangular solves. The pivot is floored relative to the original diagonal so a nearly singular row still factors."""
    n_dofs = mochi_state.res.shape[0]
    _B = mochi_state.is_active.shape[0]
    EPS = mochi_info.EPS[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        if not mochi_state.is_active[i_b]:
            continue
        for i_d in range(n_dofs):
            diag = mochi_state.H_dense[i_b, i_d, i_d]
            tmp = diag
            for k_d in range(i_d):
                tmp = tmp - mochi_state.H_dense[i_b, i_d, k_d] ** 2
            mochi_state.H_dense[i_b, i_d, i_d] = qd.sqrt(qd.max(tmp, EPS * qd.max(diag, EPS)))
            inv = 1.0 / mochi_state.H_dense[i_b, i_d, i_d]
            for j_d in range(i_d + 1, n_dofs):
                dot = gs.qd_float(0.0)
                for k_d in range(i_d):
                    dot = dot + mochi_state.H_dense[i_b, j_d, k_d] * mochi_state.H_dense[i_b, i_d, k_d]
                mochi_state.H_dense[i_b, j_d, i_d] = (mochi_state.H_dense[i_b, j_d, i_d] - dot) * inv
        # L y = res
        for i_d in range(n_dofs):
            s = mochi_state.res[i_d, i_b]
            for k_d in range(i_d):
                s = s - mochi_state.H_dense[i_b, i_d, k_d] * mochi_state.dx[k_d, i_b]
            mochi_state.dx[i_d, i_b] = s / mochi_state.H_dense[i_b, i_d, i_d]
        # L^T dx = y
        for i_d_ in range(n_dofs):
            i_d = n_dofs - 1 - i_d_
            s = mochi_state.dx[i_d, i_b]
            for k_d in range(i_d + 1, n_dofs):
                s = s - mochi_state.H_dense[i_b, k_d, i_d] * mochi_state.dx[k_d, i_b]
            mochi_state.dx[i_d, i_b] = s / mochi_state.H_dense[i_b, i_d, i_d]


@qd.func
def func_apply_block_preconditioner(i_l, i_b, dof_start, src: qd.Tensor, dst: qd.Tensor, mochi_state: MochiState):
    """Block-Jacobi preconditioner: the 3x3 translational and rotational diagonal blocks of the link are inverted
    separately."""
    H = mochi_state.H_diag[i_l, i_b]
    H_t = qd.Matrix.zero(gs.qd_float, 3, 3)
    H_r = qd.Matrix.zero(gs.qd_float, 3, 3)
    v_t = qd.Vector.zero(gs.qd_float, 3)
    v_r = qd.Vector.zero(gs.qd_float, 3)
    for k, l in qd.static(qd.ndrange(3, 3)):
        H_t[k, l] = H[k, l]
        H_r[k, l] = H[3 + k, 3 + l]
    for k in qd.static(range(3)):
        v_t[k] = src[dof_start + k, i_b]
        v_r[k] = src[dof_start + 3 + k, i_b]
    z_t = H_t.inverse() @ v_t
    z_r = H_r.inverse() @ v_r
    for k in qd.static(range(3)):
        dst[dof_start + k, i_b] = z_t[k]
        dst[dof_start + 3 + k, i_b] = z_r[k]


@qd.kernel
def kernel_pcg_init(
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    rigid_config: qd.template(),
):
    n_dofs = mochi_state.res.shape[0]
    n_links = mochi_state.H_diag.shape[0]
    _B = mochi_state.is_active.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        mochi_state.pcg_is_active[i_b] = mochi_state.is_active[i_b]
        mochi_state.pcg_rTz[i_b] = 0.0
        mochi_state.pcg_rTr[i_b] = 0.0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.pcg_is_active[i_b]:
            mochi_state.dx[i_d, i_b] = 0.0
            mochi_state.pcg_r[i_d, i_b] = mochi_state.res[i_d, i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if mochi_state.pcg_is_active[i_b] and mochi_info.links.is_dynamic[i_l]:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            dof_start = dyn_info.links.dof_start[I_l]
            func_apply_block_preconditioner(i_l, i_b, dof_start, mochi_state.pcg_r, mochi_state.pcg_z, mochi_state)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.pcg_is_active[i_b]:
            r = mochi_state.pcg_r[i_d, i_b]
            z = mochi_state.pcg_z[i_d, i_b]
            mochi_state.pcg_p[i_d, i_b] = z
            qd.atomic_add(mochi_state.pcg_rTz[i_b], r * z)
            qd.atomic_add(mochi_state.pcg_rTr[i_b], r * r)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        mochi_state.pcg_rTr0[i_b] = mochi_state.pcg_rTr[i_b]
        if mochi_state.pcg_rTr[i_b] <= 0.0:
            mochi_state.pcg_is_active[i_b] = False


@qd.kernel
def kernel_pcg_iter(
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    rigid_config: qd.template(),
):
    n_dofs = mochi_state.res.shape[0]
    n_links = mochi_state.H_diag.shape[0]
    max_pairs = contact_state.pair_link_a.shape[0]
    _B = mochi_state.is_active.shape[0]
    rel_tol = mochi_info.pcg_rel_tol[None]

    # Ap = H p from the blocks
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if mochi_state.pcg_is_active[i_b] and mochi_info.links.is_dynamic[i_l]:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            dof_start = dyn_info.links.dof_start[I_l]
            p = qd.Vector.zero(gs.qd_float, 6)
            for k in qd.static(range(6)):
                p[k] = mochi_state.pcg_p[dof_start + k, i_b]
            Ap = mochi_state.H_diag[i_l, i_b] @ p
            for k in qd.static(range(6)):
                mochi_state.pcg_Ap[dof_start + k, i_b] = Ap[k]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_p, i_b in qd.ndrange(max_pairs, _B):
        if not mochi_state.pcg_is_active[i_b] or i_p >= contact_state.n_pairs[i_b]:
            continue
        if contact_state.n_hits[i_p, i_b] == 0:
            continue
        i_la = contact_state.pair_link_a[i_p, i_b]
        i_lb = contact_state.pair_link_b[i_p, i_b]
        if not (mochi_info.links.is_dynamic[i_la] and mochi_info.links.is_dynamic[i_lb]):
            continue
        I_la = [i_la, i_b] if qd.static(rigid_config.batch_links_info) else i_la
        I_lb = [i_lb, i_b] if qd.static(rigid_config.batch_links_info) else i_lb
        dof_a = dyn_info.links.dof_start[I_la]
        dof_b = dyn_info.links.dof_start[I_lb]
        p_a = qd.Vector.zero(gs.qd_float, 6)
        p_b = qd.Vector.zero(gs.qd_float, 6)
        for k in qd.static(range(6)):
            p_a[k] = mochi_state.pcg_p[dof_a + k, i_b]
            p_b[k] = mochi_state.pcg_p[dof_b + k, i_b]
        H_off = mochi_state.H_off[i_p, i_b]
        Ap_a = H_off @ p_b
        Ap_b = H_off.transpose() @ p_a
        for k in qd.static(range(6)):
            qd.atomic_add(mochi_state.pcg_Ap[dof_a + k, i_b], Ap_a[k])
            qd.atomic_add(mochi_state.pcg_Ap[dof_b + k, i_b], Ap_b[k])

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        mochi_state.pcg_pTAp[i_b] = 0.0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.pcg_is_active[i_b]:
            qd.atomic_add(mochi_state.pcg_pTAp[i_b], mochi_state.pcg_p[i_d, i_b] * mochi_state.pcg_Ap[i_d, i_b])

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        if mochi_state.pcg_is_active[i_b]:
            if mochi_state.pcg_pTAp[i_b] <= 0.0:
                mochi_state.pcg_is_active[i_b] = False
            mochi_state.pcg_rTz_new[i_b] = 0.0
            mochi_state.pcg_rTr[i_b] = 0.0
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.pcg_is_active[i_b]:
            alpha = mochi_state.pcg_rTz[i_b] / mochi_state.pcg_pTAp[i_b]
            mochi_state.dx[i_d, i_b] += alpha * mochi_state.pcg_p[i_d, i_b]
            mochi_state.pcg_r[i_d, i_b] -= alpha * mochi_state.pcg_Ap[i_d, i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if mochi_state.pcg_is_active[i_b] and mochi_info.links.is_dynamic[i_l]:
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            dof_start = dyn_info.links.dof_start[I_l]
            func_apply_block_preconditioner(i_l, i_b, dof_start, mochi_state.pcg_r, mochi_state.pcg_z, mochi_state)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.pcg_is_active[i_b]:
            r = mochi_state.pcg_r[i_d, i_b]
            qd.atomic_add(mochi_state.pcg_rTz_new[i_b], r * mochi_state.pcg_z[i_d, i_b])
            qd.atomic_add(mochi_state.pcg_rTr[i_b], r * r)
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        if mochi_state.pcg_is_active[i_b]:
            if mochi_state.pcg_rTr[i_b] <= rel_tol * rel_tol * mochi_state.pcg_rTr0[i_b]:
                mochi_state.pcg_is_active[i_b] = False
            beta = mochi_state.pcg_rTz_new[i_b] / mochi_state.pcg_rTz[i_b]
            mochi_state.pcg_rTz[i_b] = mochi_state.pcg_rTz_new[i_b]
            mochi_state.pcg_pTAp[i_b] = beta
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.pcg_is_active[i_b]:
            mochi_state.pcg_p[i_d, i_b] = (
                mochi_state.pcg_z[i_d, i_b] + mochi_state.pcg_pTAp[i_b] * mochi_state.pcg_p[i_d, i_b]
            )


@qd.kernel
def kernel_pcg_any_active(mochi_state: MochiState) -> qd.i32:
    n_active = 0
    for i_b in range(mochi_state.pcg_is_active.shape[0]):
        if mochi_state.pcg_is_active[i_b]:
            n_active += 1
    return n_active
