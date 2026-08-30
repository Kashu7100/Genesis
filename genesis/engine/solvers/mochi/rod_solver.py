# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Exact factorization of every open rod's own Hessian block, used as the conjugate gradient preconditioner of the rod
degrees of freedom (mochi preconditions rods per actor with an incomplete factorization; on a path graph the complete
banded one costs the same). The stretching stiffness of a rod couples neighboring nodes far more strongly than its
inertia, so a diagonal preconditioner leaves the conjugate gradient with a condition number that grows with the
stiffness and the node count; the banded factor removes it and only the contact couplings remain to iterate on."""

import quadrants as qd

import genesis as gs

from .data import ROD_BAND, MochiSoftInfo, MochiSoftState, MochiState


@qd.func
def func_rod_band_add(row, col, value, i_b, soft_state: MochiSoftState):
    """Accumulate an entry of the lower triangle (row >= col within the band); rows are global band rows."""
    if row >= col and row - col <= ROD_BAND:
        soft_state.rod_band[row, row - col, i_b] += value


@qd.func
def func_rod_band_factor(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    eps,
):
    """Assemble the banded Hessian block of every open rod of the running conjugate gradient environments from the
    element and stencil blocks (Dirichlet rows and columns replaced by the identity) and factor it in place."""
    n_entities = soft_info.entities_band_n.shape[0]
    _B = soft_state.verts_pos.shape[1]
    dof_start = soft_info.dof_start[None]
    twist_dof_start = soft_info.twist_dof_start[None]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_slot in qd.ndrange(n_entities, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_entities, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        n = soft_info.entities_band_n[i_e]
        if n == 0 or not mochi_state.pcg_is_active[i_b]:
            continue
        start = soft_info.entities_band_start[i_e]
        for r in range(n):
            for d in qd.static(range(ROD_BAND + 1)):
                soft_state.rod_band[start + r, d, i_b] = 0.0
            soft_state.rod_band[start + r, 0, i_b] = mochi_state.dofs_H_diag[soft_info.band_rows_dof[start + r], i_b]
        for i_r in range(soft_info.entities_rod_elem_start[i_e], soft_info.entities_rod_elem_end[i_e]):
            if soft_info.rod_elems_L[i_r] <= 0.0:
                continue
            v = soft_info.rod_elems_v[i_r]
            T = soft_state.rod_elems_H[i_r, i_b]
            c_inertia = soft_state.rod_elems_inertia[i_r, i_b]
            r0 = soft_info.dofs_band_row[dof_start + 3 * v[0]]
            r1 = soft_info.dofs_band_row[dof_start + 3 * v[1]]
            for k, l in qd.static(qd.ndrange(3, 3)):
                diag = c_inertia if qd.static(k == l) else 0.0
                func_rod_band_add(r0 + k, r0 + l, T[k, l] + diag, i_b, soft_state)
                func_rod_band_add(r1 + k, r1 + l, T[k, l] + diag, i_b, soft_state)
                func_rod_band_add(r1 + k, r0 + l, -T[k, l], i_b, soft_state)
                func_rod_band_add(r0 + k, r1 + l, -T[k, l], i_b, soft_state)
        for i_s in range(soft_info.entities_rod_stencil_start[i_e], soft_info.entities_rod_stencil_end[i_e]):
            if soft_info.rod_stencils_L[i_s] <= 0.0:
                continue
            sv = soft_info.rod_stencils_v[i_s]
            se = soft_info.rod_stencils_e[i_s]
            rows = qd.Vector.zero(gs.qd_int, 11)
            for k in qd.static(range(3)):
                rows[k] = soft_info.dofs_band_row[dof_start + 3 * sv[0] + k]
                rows[4 + k] = soft_info.dofs_band_row[dof_start + 3 * sv[1] + k]
                rows[8 + k] = soft_info.dofs_band_row[dof_start + 3 * sv[2] + k]
            rows[3] = soft_info.dofs_band_row[twist_dof_start + se[0]]
            rows[7] = soft_info.dofs_band_row[twist_dof_start + se[1]]
            K = soft_state.rod_stencils_H[i_s, i_b]
            for p in qd.static(range(11)):
                for q in qd.static(range(11)):
                    func_rod_band_add(rows[p], rows[q], K[p, q], i_b, soft_state)
        # Dirichlet rows and columns: identity.
        for r in range(n):
            i_d = soft_info.band_rows_dof[start + r]
            if i_d < twist_dof_start and soft_state.verts_is_fixed[(i_d - dof_start) // 3, i_b]:
                for d in qd.static(range(ROD_BAND + 1)):
                    soft_state.rod_band[start + r, d, i_b] = 0.0
                for d in qd.static(range(1, ROD_BAND + 1)):
                    if r + d < n:
                        soft_state.rod_band[start + r + d, d, i_b] = 0.0
                soft_state.rod_band[start + r, 0, i_b] = 1.0
        # Banded Cholesky in place.
        for j in range(n):
            diag = soft_state.rod_band[start + j, 0, i_b]
            for k in range(qd.max(j - ROD_BAND, 0), j):
                diag -= soft_state.rod_band[start + j, j - k, i_b] ** 2
            diag = qd.sqrt(qd.max(diag, eps))
            soft_state.rod_band[start + j, 0, i_b] = diag
            for i in range(j + 1, qd.min(j + ROD_BAND, n - 1) + 1):
                value = soft_state.rod_band[start + i, i - j, i_b]
                for k in range(qd.max(i - ROD_BAND, 0), j):
                    value -= soft_state.rod_band[start + i, i - k, i_b] * soft_state.rod_band[start + j, j - k, i_b]
                soft_state.rod_band[start + i, i - j, i_b] = value / diag


@qd.func
def func_rod_band_solve(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    r: qd.Tensor,
    z: qd.Tensor,
    mochi_state: MochiState,
    soft_info: MochiSoftInfo,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
):
    """z = (L L^T)^-1 r on the degrees of freedom of every open rod (forward and backward band substitutions)."""
    n_entities = soft_info.entities_band_n.shape[0]
    _B = soft_state.verts_pos.shape[1]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_e, i_slot in qd.ndrange(n_entities, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_entities, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        n = soft_info.entities_band_n[i_e]
        if n == 0 or not mochi_state.pcg_is_active[i_b]:
            continue
        start = soft_info.entities_band_start[i_e]
        for i in range(n):
            value = r[soft_info.band_rows_dof[start + i], i_b]
            for k in range(qd.max(i - ROD_BAND, 0), i):
                value -= soft_state.rod_band[start + i, i - k, i_b] * z[soft_info.band_rows_dof[start + k], i_b]
            z[soft_info.band_rows_dof[start + i], i_b] = value / soft_state.rod_band[start + i, 0, i_b]
        for i_ in range(n):
            i = n - 1 - i_
            value = z[soft_info.band_rows_dof[start + i], i_b]
            for k in range(i + 1, qd.min(i + ROD_BAND, n - 1) + 1):
                value -= soft_state.rod_band[start + k, k - i, i_b] * z[soft_info.band_rows_dof[start + k], i_b]
            z[soft_info.band_rows_dof[start + i], i_b] = value / soft_state.rod_band[start + i, 0, i_b]
