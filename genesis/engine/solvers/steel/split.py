"""Split solve of the Steel constraint rows (solve_mode='split', stages 5 and 6 of a step).

The dense rows of an environment are assembled into its Delassus matrix C = J H^-1 J^T and solved by projected
Gauss-Seidel in impulse space, the residual of row i being rhs_i + sum_j C_ij lambda_j with rhs = bias + J v_hat. The
generalized velocity follows as v = v_hat + Y lambda. The free-body rows are then solved matrix-free against that
velocity, recomputing J v of each row from the live velocity and applying every impulse change at once. When the
model holds both articulated and free bodies, every iteration runs one dense sweep and one free-body sweep: the dense
right-hand side absorbs J dv of each free-body sweep and the velocity carries the accumulated free-body change, so the
two blocks form one block Gauss-Seidel iteration. Environments are independent and solved in parallel.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .pgs import func_free_apply_impulse, func_friction_pair_candidate
from .rows import ROW_TYPE, func_row_articulations


@qd.func
def func_split_accumulate_rhs(
    i_r: int,
    i_b: int,
    velocity: qd.Tensor,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_model: array_class.SteelModelInfo,
):
    """Add the product of the Jacobian of a dense row with a generalized velocity to its right-hand side.

    Dense row i_r gets J v for the velocity in the given tensor, one articulation at a time: the articulations of the
    row contribute their partial sums in decreasing DOF count, then increasing index, the order of the reference
    implementation on CPU (the sum of three terms depends on it at the float rounding level).
    """
    i_a0, i_a1 = func_row_articulations(i_r, i_b, steel_contacts, steel_rows, steel_model)
    if i_a1 >= 0:
        n0 = steel_model.arts_n_dofs[i_a0]
        n1 = steel_model.arts_n_dofs[i_a1]
        if n1 > n0 or (n1 == n0 and i_a1 < i_a0):
            i_a0, i_a1 = i_a1, i_a0
    for i_a_ in qd.static(range(2)):
        i_a = i_a0 if qd.static(i_a_ == 0) else i_a1
        if i_a >= 0:
            dof_start = steel_model.arts_dof_start[i_a]
            jv = gs.qd_float(0.0)
            for i_d in range(dof_start, dof_start + steel_model.arts_n_dofs[i_a]):
                jv = jv + steel_rows.J[i_r, i_d, i_b] * velocity[i_d, i_b]
            steel_rows.rhs[i_r, i_b] = steel_rows.rhs[i_r, i_b] + jv


@qd.func
def func_steel_split_setup(
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_split: array_class.SteelSplit,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Assemble the Delassus matrix of the dense rows of every environment, and add J v_hat to their right-hand sides.

    C_ij sums the partial products J_i . Y_j of the articulations row i acts on, and the diagonal of row i is
    C_ii + cfm. The right-hand sides hold the bias of the rows on entry.
    """
    _B = steel_rows.count.shape[0]
    capacity = steel_rows.row_type.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_s, i_b in qd.ndrange(capacity, capacity, _B):
        n_rows = steel_rows.count[i_b]
        if i_r < n_rows and i_s < n_rows:
            i_a0, i_a1 = func_row_articulations(i_r, i_b, steel_contacts, steel_rows, steel_model)
            c = gs.qd_float(0.0)
            for i_a_ in qd.static(range(2)):
                i_a = i_a0 if qd.static(i_a_ == 0) else i_a1
                if i_a >= 0:
                    dof_start = steel_model.arts_dof_start[i_a]
                    partial = gs.qd_float(0.0)
                    for i_d in range(dof_start, dof_start + steel_model.arts_n_dofs[i_a]):
                        partial = partial + steel_rows.J[i_r, i_d, i_b] * steel_rows.Y[i_s, i_d, i_b]
                    c = c + partial
            steel_split.C[i_r, i_s, i_b] = c
            if i_r == i_s:
                steel_rows.diag[i_r, i_b] = c + steel_info.cfm[None]

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(capacity, _B):
        if i_r < steel_rows.count[i_b]:
            func_split_accumulate_rhs(i_r, i_b, steel_dyn.v_hat, steel_contacts, steel_rows, steel_model)


@qd.func
def func_split_dense_sweeps(
    i_b: int,
    n_iterations: int,
    iteration_offset: int,
    friction_start_iteration: int,
    steel_rows: array_class.SteelRows,
    steel_split: array_class.SteelSplit,
    steel_info: array_class.SteelInfo,
):
    """Run Gauss-Seidel sweeps over the Delassus system of the dense rows of one environment.

    Environment i_b runs n_iterations sweeps in row order. Contact and joint-limit rows are unilateral. The first
    friction row of a contact solves both tangent impulses on the Coulomb disk of the current normal impulse. Friction
    rows hold a zero impulse in the iterations before friction_start_iteration, counted from iteration_offset.
    """
    omega = steel_info.relaxation[None]
    n_rows = steel_rows.count[i_b]
    for it in range(n_iterations):
        for i_r in range(n_rows):
            row_type = steel_rows.row_type[i_r, i_b]
            i_p = steel_rows.row_parent[i_r, i_b]
            is_friction = row_type == ROW_TYPE.FRICTION
            is_active = True
            if is_friction and iteration_offset + it < friction_start_iteration:
                steel_rows.impulses[i_r, i_b] = 0.0
                is_active = False
            if is_friction and i_r != i_p + 1:
                is_active = False
            denom = steel_rows.diag[i_r, i_b]
            if denom <= 0.0 and not is_friction:
                is_active = False
            if is_active:
                res = steel_rows.rhs[i_r, i_b]
                for i_j in range(n_rows):
                    res = res + steel_split.C[i_r, i_j, i_b] * steel_rows.impulses[i_j, i_b]
                if is_friction:
                    radius = qd.max(steel_rows.row_mu[i_r, i_b] * steel_rows.impulses[i_p, i_b], 0.0)
                    i_s = i_p + 2
                    pair = qd.Vector.zero(gs.qd_float, 2)
                    if radius > 0.0:
                        res_s = steel_rows.rhs[i_s, i_b]
                        for i_j in range(n_rows):
                            res_s = res_s + steel_split.C[i_s, i_j, i_b] * steel_rows.impulses[i_j, i_b]
                        pair = func_friction_pair_candidate(
                            denom,
                            steel_split.C[i_r, i_s, i_b],
                            steel_rows.diag[i_s, i_b],
                            qd.Vector([res, res_s], dt=gs.qd_float),
                            qd.Vector([steel_rows.impulses[i_r, i_b], steel_rows.impulses[i_s, i_b]], dt=gs.qd_float),
                            radius,
                            omega,
                            steel_info,
                        )
                        norm = pair.norm()
                        if norm > radius:
                            pair = pair * (radius / norm)
                    steel_rows.impulses[i_r, i_b] = pair[0]
                    steel_rows.impulses[i_s, i_b] = pair[1]
                else:
                    delta = -res / denom
                    steel_rows.impulses[i_r, i_b] = qd.max(steel_rows.impulses[i_r, i_b] + omega * delta, 0.0)


@qd.func
def func_split_apply_dense_impulses(i_b: int, steel_rows: array_class.SteelRows, steel_dyn: array_class.SteelDynamics):
    """Write v_out = v_hat + Y lambda for one environment from the impulses of its dense rows."""
    n_rows = steel_rows.count[i_b]
    for i_d in range(steel_dyn.v_out.shape[0]):
        dv = gs.qd_float(0.0)
        for i_r in range(n_rows):
            dv = dv + steel_rows.Y[i_r, i_d, i_b] * steel_rows.impulses[i_r, i_b]
        steel_dyn.v_out[i_d, i_b] = steel_dyn.v_hat[i_d, i_b] + dv


@qd.func
def func_split_free_sweeps(
    i_b: int,
    n_iterations: int,
    iteration_offset: int,
    friction_start_iteration: int,
    steel_free_rows: array_class.SteelFreeRows,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
):
    """Run Gauss-Seidel sweeps over the free-body rows of one environment, updating v_out.

    Environment i_b runs n_iterations sweeps in row order. J v of every row is recomputed from v_out, each side summing
    its linear then its angular terms, and every impulse change is applied at once. Contacts are unilateral, the first
    friction row of a contact solves both tangent impulses on the Coulomb disk of the current normal impulse and
    applies the second one first, and the velocity limits, last in the row order, are stateless clamps applying the
    unrelaxed impulse that removes the current overshoot. Friction rows hold a zero impulse in the iterations before
    friction_start_iteration, counted from iteration_offset.
    """
    omega = steel_info.relaxation[None]
    n_rows = steel_free_rows.count[i_b]
    for it in range(n_iterations):
        for i_r in range(n_rows):
            row_type = steel_free_rows.row_type[i_r, i_b]
            i_p = steel_free_rows.row_parent[i_r, i_b]
            is_friction = row_type == ROW_TYPE.FRICTION
            is_active = True
            if is_friction and iteration_offset + it < friction_start_iteration:
                steel_free_rows.impulses[i_r, i_b] = 0.0
                is_active = False
            if is_friction and i_r != i_p + 1:
                is_active = False
            inv_diag = steel_free_rows.eff_mass_inv[i_r, i_b]
            if inv_diag <= 0.0 and not is_friction:
                is_active = False
            if is_active:
                dof_a = steel_free_rows.dof_a[i_r, i_b]
                dof_b = steel_free_rows.dof_b[i_r, i_b]
                old = steel_free_rows.impulses[i_r, i_b]
                new = old
                # Linear then angular part of the Jacobian of each side, summed in that order.
                J_r = (
                    (steel_free_rows.J_a_lin[i_r, i_b], steel_free_rows.J_a_ang[i_r, i_b]),
                    (steel_free_rows.J_b_lin[i_r, i_b], steel_free_rows.J_b_ang[i_r, i_b]),
                )
                if is_friction:
                    i_s = i_p + 2
                    res = qd.Vector([steel_free_rows.rhs[i_r, i_b], steel_free_rows.rhs[i_s, i_b]], dt=gs.qd_float)
                    J_s = (
                        (steel_free_rows.J_a_lin[i_s, i_b], steel_free_rows.J_a_ang[i_s, i_b]),
                        (steel_free_rows.J_b_lin[i_s, i_b], steel_free_rows.J_b_ang[i_s, i_b]),
                    )
                    M_s = (
                        (steel_free_rows.MiJt_a_lin[i_s, i_b], steel_free_rows.MiJt_a_ang[i_s, i_b]),
                        (steel_free_rows.MiJt_b_lin[i_s, i_b], steel_free_rows.MiJt_b_ang[i_s, i_b]),
                    )
                    cross = gs.qd_float(0.0)
                    for k in qd.static(range(6)):
                        for side in qd.static(range(2)):
                            dof = dof_a if qd.static(side == 0) else dof_b
                            if dof >= 0:
                                J_rk = J_r[side][k // 3][k % 3]
                                v = steel_dyn.v_out[dof + k, i_b]
                                res[0] = res[0] + J_rk * v
                                res[1] = res[1] + J_s[side][k // 3][k % 3] * v
                                cross = cross + J_rk * M_s[side][k // 3][k % 3]
                    inv_s = steel_free_rows.eff_mass_inv[i_s, i_b]
                    radius = qd.max(steel_free_rows.row_mu[i_r, i_b] * steel_free_rows.impulses[i_p, i_b], 0.0)
                    old_s = steel_free_rows.impulses[i_s, i_b]
                    pair = func_friction_pair_candidate(
                        1.0 / inv_diag if inv_diag > 0.0 else 0.0,
                        cross,
                        1.0 / inv_s if inv_s > 0.0 else 0.0,
                        res,
                        qd.Vector([old, old_s], dt=gs.qd_float),
                        radius,
                        omega,
                        steel_info,
                    )
                    norm = pair.norm()
                    if norm > radius:
                        pair = pair * (radius / norm)
                    steel_free_rows.impulses[i_s, i_b] = pair[1]
                    if pair[1] != old_s:
                        func_free_apply_impulse(i_s, i_b, pair[1] - old_s, steel_free_rows, steel_dyn)
                    new = pair[0]
                else:
                    jv = gs.qd_float(0.0)
                    for side in qd.static(range(2)):
                        dof = dof_a if qd.static(side == 0) else dof_b
                        if dof >= 0:
                            for k in qd.static(range(6)):
                                jv = jv + J_r[side][k // 3][k % 3] * steel_dyn.v_out[dof + k, i_b]
                    res = jv + steel_free_rows.rhs[i_r, i_b]
                    delta = -res * inv_diag
                    if row_type == ROW_TYPE.JOINT_VELOCITY_LIMIT:
                        # Stateless: the impulse of the visit is the change it applies.
                        new = delta if res < 0.0 else 0.0
                        old = 0.0
                    else:
                        new = qd.max(old + omega * delta, 0.0)
                steel_free_rows.impulses[i_r, i_b] = new
                if new != old:
                    func_free_apply_impulse(i_r, i_b, new - old, steel_free_rows, steel_dyn)


@qd.func
def func_steel_split_solve(
    n_iterations: int,
    friction_start_iteration: int,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_split: array_class.SteelSplit,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Solve the dense rows on their Delassus systems and the free-body rows matrix-free, writing the velocity v_out.

    Without free bodies, the dense rows run every iteration and v_out = v_hat + Y lambda. Without articulated bodies
    besides free bodies, the free-body rows then run every iteration on that velocity. With both, every iteration runs
    one dense sweep, sets v_out = v_hat + Y lambda + the accumulated free-body velocity change, runs one free-body
    sweep, and adds J dv of its velocity change dv to the dense right-hand sides.
    """
    _B = steel_dyn.v_out.shape[1]
    n_dofs = steel_dyn.v_out.shape[0]
    for i_b in range(_B):
        if qd.static(not steel_config.has_free_rigid_bodies):
            func_split_dense_sweeps(i_b, n_iterations, 0, friction_start_iteration, steel_rows, steel_split, steel_info)
            func_split_apply_dense_impulses(i_b, steel_rows, steel_dyn)
        elif qd.static(not steel_config.has_free_and_articulated_bodies):
            func_split_dense_sweeps(i_b, n_iterations, 0, friction_start_iteration, steel_rows, steel_split, steel_info)
            func_split_apply_dense_impulses(i_b, steel_rows, steel_dyn)
            func_split_free_sweeps(
                i_b, n_iterations, 0, friction_start_iteration, steel_free_rows, steel_dyn, steel_info
            )
        else:
            for i_d in range(n_dofs):
                steel_split.v_free_sum[i_d, i_b] = 0.0
            for it in range(n_iterations):
                func_split_dense_sweeps(i_b, 1, it, friction_start_iteration, steel_rows, steel_split, steel_info)
                func_split_apply_dense_impulses(i_b, steel_rows, steel_dyn)
                for i_d in range(n_dofs):
                    steel_dyn.v_out[i_d, i_b] = steel_dyn.v_out[i_d, i_b] + steel_split.v_free_sum[i_d, i_b]
                    steel_split.v_free_delta[i_d, i_b] = steel_dyn.v_out[i_d, i_b]
                func_split_free_sweeps(i_b, 1, it, friction_start_iteration, steel_free_rows, steel_dyn, steel_info)
                for i_d in range(n_dofs):
                    dv = steel_dyn.v_out[i_d, i_b] - steel_split.v_free_delta[i_d, i_b]
                    steel_split.v_free_sum[i_d, i_b] = steel_split.v_free_sum[i_d, i_b] + dv
                    steel_split.v_free_delta[i_d, i_b] = dv
                for i_r in range(steel_rows.count[i_b]):
                    func_split_accumulate_rhs(
                        i_r, i_b, steel_split.v_free_delta, steel_contacts, steel_rows, steel_model
                    )
