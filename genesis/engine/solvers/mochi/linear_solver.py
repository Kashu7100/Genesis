# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Linear solve of the Newton system H dx = res, where H is the projection of the per-link and per-contact-pair 6x6
blocks onto the degrees of freedom through the link Jacobians plus the joint-space diagonal: a dense Cholesky
factorization of the condensed matrix, or a Jacobi-preconditioned conjugate gradient applying the blocks on the fly."""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .articulated import (
    func_jacobian_column_dot,
    func_jacobian_times_dofs,
    func_jacobian_transpose_add,
    func_link_dof_jacobian,
)
from .data import MochiContactState, MochiInfo, MochiSoftInfo, MochiSoftState, MochiState
from .soft import func_soft_matvec, func_soft_precondition


@qd.func
def func_add_projected_block(
    i_la,
    i_lb,
    i_b,
    block,
    H_dense: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    is_symmetric_pair: qd.template(),
):
    """H_dense += J_a^T block J_b over the ancestor degrees of freedom of links a and b (and the transpose block when
    the two links differ)."""
    i_a = i_la
    while i_a != -1:
        I_a = [i_a, i_b] if qd.static(rigid_config.batch_links_info) else i_a
        for i_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
            vel, ang = func_link_dof_jacobian(i_la, i_d, i_b, dyn_state)
            column = qd.Vector([vel[0], vel[1], vel[2], ang[0], ang[1], ang[2]], dt=gs.qd_float)
            row = block.transpose() @ column
            i_c = i_lb
            while i_c != -1:
                I_c = [i_c, i_b] if qd.static(rigid_config.batch_links_info) else i_c
                for j_d in range(dyn_info.links.dof_start[I_c], dyn_info.links.dof_end[I_c]):
                    value = func_jacobian_column_dot(i_lb, j_d, i_b, row, dyn_state)
                    qd.atomic_add(H_dense[i_b, i_d, j_d], value)
                    if qd.static(is_symmetric_pair):
                        qd.atomic_add(H_dense[i_b, j_d, i_d], value)
                i_c = dyn_info.links.parent_idx[I_c]
        i_a = dyn_info.links.parent_idx[I_a]


@qd.kernel
def kernel_condense_dense(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    rigid_config: qd.template(),
):
    """Assemble the dense Hessian of every running environment from the projected blocks and the joint diagonal."""
    n_dofs = mochi_state.res.shape[0]
    n_links = mochi_state.H_diag.shape[0]
    max_pairs = contact_state.pair_link_a.shape[0]
    _B = mochi_state.is_active.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b, i_d, j_d in qd.ndrange(_B, n_dofs, n_dofs):
        if mochi_state.is_active[i_b]:
            mochi_state.H_dense[i_b, i_d, j_d] = 0.0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.is_active[i_b]:
            mochi_state.H_dense[i_b, i_d, i_d] = mochi_state.dofs_H_diag[i_d, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if not mochi_state.is_active[i_b] or not mochi_info.links.is_dynamic[i_l]:
            continue
        func_add_projected_block(
            i_l, i_l, i_b, mochi_state.H_diag[i_l, i_b], mochi_state.H_dense, dyn_state, dyn_info, rigid_config, False
        )

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
        func_add_projected_block(
            i_la, i_lb, i_b, mochi_state.H_off[i_p, i_b], mochi_state.H_dense, dyn_state, dyn_info, rigid_config, True
        )


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
def func_matvec(
    src: qd.Tensor,
    dst: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """dst = H src for the running conjugate gradient environments, applying the projected blocks on the fly."""
    n_dofs = mochi_state.res.shape[0]
    n_links = mochi_state.H_diag.shape[0]
    max_pairs = contact_state.pair_link_a.shape[0]
    _B = mochi_state.is_active.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.pcg_is_active[i_b]:
            dst[i_d, i_b] = mochi_state.dofs_H_diag[i_d, i_b] * src[i_d, i_b]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if mochi_state.pcg_is_active[i_b] and mochi_info.links.is_dynamic[i_l]:
            v = func_jacobian_times_dofs(i_l, i_b, src, dyn_state, dyn_info, rigid_config)
            func_jacobian_transpose_add(
                i_l, i_b, mochi_state.H_diag[i_l, i_b] @ v, dst, dyn_state, dyn_info, rigid_config
            )

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
        H_off = mochi_state.H_off[i_p, i_b]
        v_a = func_jacobian_times_dofs(i_la, i_b, src, dyn_state, dyn_info, rigid_config)
        v_b = func_jacobian_times_dofs(i_lb, i_b, src, dyn_state, dyn_info, rigid_config)
        func_jacobian_transpose_add(i_la, i_b, H_off @ v_b, dst, dyn_state, dyn_info, rigid_config)
        func_jacobian_transpose_add(i_lb, i_b, H_off.transpose() @ v_a, dst, dyn_state, dyn_info, rigid_config)

    if qd.static(mochi_config.has_soft):
        func_soft_matvec(src, dst, dyn_state, dyn_info, mochi_info, mochi_state, soft_info, soft_state, rigid_config)


@qd.func
def func_apply_preconditioner(
    r: qd.Tensor,
    z: qd.Tensor,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """z = M^-1 r: Jacobi on the rigid degrees of freedom, block Jacobi (3x3 per vertex) on the deformable ones."""
    n_dofs = mochi_state.res.shape[0]
    _B = mochi_state.is_active.shape[0]
    EPS = mochi_info.EPS[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if mochi_state.pcg_is_active[i_b]:
            z[i_d, i_b] = r[i_d, i_b] / qd.max(mochi_state.pcg_diag[i_d, i_b], EPS)
    if qd.static(mochi_config.has_soft):
        func_soft_precondition(r, z, mochi_state, soft_info, soft_state, rigid_config, EPS)


@qd.kernel
def kernel_pcg_init(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    """Start the conjugate gradient from dx = 0 with the preconditioner built from the projected diagonal."""
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
            mochi_state.pcg_diag[i_d, i_b] = mochi_state.dofs_H_diag[i_d, i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        if not mochi_state.pcg_is_active[i_b] or not mochi_info.links.is_dynamic[i_l]:
            continue
        H = mochi_state.H_diag[i_l, i_b]
        i_a = i_l
        while i_a != -1:
            I_a = [i_a, i_b] if qd.static(rigid_config.batch_links_info) else i_a
            for i_d in range(dyn_info.links.dof_start[I_a], dyn_info.links.dof_end[I_a]):
                vel, ang = func_link_dof_jacobian(i_l, i_d, i_b, dyn_state)
                column = qd.Vector([vel[0], vel[1], vel[2], ang[0], ang[1], ang[2]], dt=gs.qd_float)
                qd.atomic_add(mochi_state.pcg_diag[i_d, i_b], column.dot(H @ column))
            i_a = dyn_info.links.parent_idx[I_a]
    func_apply_preconditioner(
        mochi_state.pcg_r, mochi_state.pcg_z, mochi_info, mochi_state, soft_info, soft_state, rigid_config, mochi_config
    )
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
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
):
    n_dofs = mochi_state.res.shape[0]
    _B = mochi_state.is_active.shape[0]
    rel_tol = mochi_info.pcg_rel_tol[None]

    func_matvec(
        mochi_state.pcg_p,
        mochi_state.pcg_Ap,
        dyn_state,
        dyn_info,
        mochi_info,
        mochi_state,
        contact_state,
        soft_info,
        soft_state,
        rigid_config,
        mochi_config,
    )

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
            mochi_state.pcg_r[i_d, i_b] = mochi_state.pcg_r[i_d, i_b] - alpha * mochi_state.pcg_Ap[i_d, i_b]
    func_apply_preconditioner(
        mochi_state.pcg_r, mochi_state.pcg_z, mochi_info, mochi_state, soft_info, soft_state, rigid_config, mochi_config
    )
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
