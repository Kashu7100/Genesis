"""Projected Gauss-Seidel solve of the Steel constraint rows in impulse space (stages 5 and 6 of a step).

The matrix-free sweep visits the rows of an environment in turn and recomputes the velocity J v of each row from the
live generalized velocity, every impulse change updating the velocity at once through the row response. Environments
are independent and solved in parallel. A sweep visits the dense rows in index order, then the free-body contact rows,
then the velocity limits. The solve stops early after a sweep that changed no impulse and no velocity, which is an exact
fixed point, so later sweeps would repeat it.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .compliance import func_compliance_refresh_rhs
from .rows import ROW_TYPE, func_free_row_dot
from .torsion import func_torsion_sweep


@qd.func
def func_friction_pair_candidate(
    a: float,
    c: float,
    d: float,
    residual: qd.types.vector(2),
    old: qd.types.vector(2),
    radius: float,
    omega: float,
    steel_info: array_class.SteelInfo,
):
    """Minimize the quadratic of the two tangent impulses of a contact on the disk of the given radius, then relax.

    The tangent block [a c; c d] is diagonalized in an orthonormal eigenbasis, which avoids cancellation in nearly
    singular blocks. Sticking takes the incremental block-inverse step, which preserves a feasible zero-residual
    impulse exactly. Sliding solves (A + alpha I) x = b with |x| = radius by safeguarded Newton with a bisection
    fallback. A feasible sliding impulse already satisfying the KKT conditions to the relative accuracy of the root
    solve is kept, re-solving it would introduce roundoff cycles. The caller projects the relaxed result on the disk.
    """
    result = qd.Vector.zero(gs.qd_float, 2)
    if radius > 0.0:
        scale = qd.max(qd.max(a, d), steel_info.friction_pair_scale_floor[None])
        a = a / scale
        c = c / scale
        d = d / scale
        largest = 0.5 * (a + d + qd.sqrt((a - d) * (a - d) + 4.0 * c * c))
        smallest = gs.qd_float(0.0)
        if largest > 0.0:
            smallest = qd.max((a * d - c * c) / largest, 0.0)
        axis = qd.Vector([c, largest - a], dt=gs.qd_float)
        if a >= d:
            axis = qd.Vector([largest - d, c], dt=gs.qd_float)
        if axis.norm() > 0.0:
            axis = axis.normalized()
        else:
            axis = qd.Vector([1.0, 0.0], dt=gs.qd_float)
        perpendicular = qd.Vector([-axis[1], axis[0]], dt=gs.qd_float)
        residual_rotated = qd.Vector([axis.dot(residual), perpendicular.dot(residual)], dt=gs.qd_float) / scale
        candidate = old
        is_sticking = False
        if largest > 0.0 and (smallest > 0.0 or residual_rotated[1] == 0.0):
            correction = (residual_rotated[0] / largest) * axis
            if smallest > 0.0:
                correction = correction + (residual_rotated[1] / smallest) * perpendicular
            candidate = old - correction
            is_sticking = candidate.norm() <= radius
        is_kept = False
        if not is_sticking:
            old_norm = old.norm()
            residual_norm = residual.norm()
            cross_residual = residual[0] * old[1] - residual[1] * old[0]
            is_kept = (
                old_norm > 0.0
                and old_norm <= radius
                and radius - old_norm <= steel_info.friction_pair_tolerance[None] * radius
                and residual.dot(old) <= 0.0
                and qd.abs(cross_residual) <= steel_info.friction_pair_tolerance[None] * old_norm * residual_norm
            )
            if not is_kept:
                old_rotated = qd.Vector([axis.dot(old), perpendicular.dot(old)], dt=gs.qd_float)
                b = qd.Vector([largest * old_rotated[0], smallest * old_rotated[1]], dt=gs.qd_float) - residual_rotated
                solution = qd.Vector.zero(gs.qd_float, 2)
                lo = gs.qd_float(0.0)
                hi = b.norm() / radius
                if hi > 0.0:
                    # Both bounds are below the multiplier root and bound the starting norm by sqrt(2) * radius. The
                    # reciprocal norm is increasing and concave, which gives Newton a useful step near singularity.
                    lo = qd.max(qd.max(hi - largest, qd.abs(b[1]) / radius - smallest), 0.0)
                    alpha = lo
                    is_converged = False
                    for _ in range(steel_info.friction_pair_newton_iterations[None]):
                        inv0 = 1.0 / (largest + alpha)
                        inv1 = gs.qd_float(0.0)
                        if smallest + alpha > 0.0:
                            inv1 = 1.0 / (smallest + alpha)
                        trial = old_rotated - qd.Vector(
                            [
                                (residual_rotated[0] + alpha * old_rotated[0]) * inv0,
                                (residual_rotated[1] + alpha * old_rotated[1]) * inv1,
                            ],
                            dt=gs.qd_float,
                        )
                        norm = trial.norm()
                        if qd.abs(norm - radius) <= steel_info.friction_pair_tolerance[None] * radius or (
                            alpha == 0.0 and norm <= radius
                        ):
                            solution = trial
                            is_converged = True
                            break
                        if norm > radius:
                            lo = alpha
                        else:
                            hi = alpha
                        slope = trial[0] * trial[0] * inv0 + trial[1] * trial[1] * inv1
                        alpha_newton = alpha
                        if slope > 0.0:
                            alpha_newton = alpha + (norm / radius - 1.0) * norm * norm / slope
                        alpha = 0.5 * (lo + hi)
                        if alpha_newton > lo and alpha_newton < hi:
                            alpha = alpha_newton
                    if not is_converged:
                        for _ in range(steel_info.friction_pair_bisection_iterations[None]):
                            alpha = 0.5 * (lo + hi)
                            trial = qd.Vector([b[0] / (largest + alpha), b[1] / (smallest + alpha)], dt=gs.qd_float)
                            if trial.norm() > radius:
                                lo = alpha
                            else:
                                hi = alpha
                        solution = qd.Vector([b[0] / (largest + hi), b[1] / (smallest + hi)], dt=gs.qd_float)
                candidate = old + (solution[0] - old_rotated[0]) * axis + (solution[1] - old_rotated[1]) * perpendicular
        result = old
        if not is_kept:
            result = old + omega * (candidate - old)
    return result


@qd.func
def func_dense_apply_impulse(
    i_r: int, i_b: int, dlam: float, steel_rows: array_class.SteelRows, steel_dyn: array_class.SteelDynamics
):
    """Apply the impulse change dlam of dense row i_r to the velocity."""
    for i_d in range(steel_rows.Y.shape[1]):
        steel_dyn.v_out[i_d, i_b] = steel_dyn.v_out[i_d, i_b] + steel_rows.Y[i_r, i_d, i_b] * dlam


@qd.func
def func_free_apply_impulse(
    i_r: int, i_b: int, dlam: float, steel_free_rows: array_class.SteelFreeRows, steel_dyn: array_class.SteelDynamics
):
    """Apply the impulse change dlam of free-body row i_r to the velocity of its bodies."""
    for side in qd.static(range(2)):
        dof = steel_free_rows.dof_a[i_r, i_b] if qd.static(side == 0) else steel_free_rows.dof_b[i_r, i_b]
        if dof >= 0:
            M_lin = (
                steel_free_rows.MiJt_a_lin[i_r, i_b] if qd.static(side == 0) else steel_free_rows.MiJt_b_lin[i_r, i_b]
            )
            M_ang = (
                steel_free_rows.MiJt_a_ang[i_r, i_b] if qd.static(side == 0) else steel_free_rows.MiJt_b_ang[i_r, i_b]
            )
            for k in qd.static(range(3)):
                steel_dyn.v_out[dof + k, i_b] = steel_dyn.v_out[dof + k, i_b] + M_lin[k] * dlam
                steel_dyn.v_out[dof + 3 + k, i_b] = steel_dyn.v_out[dof + 3 + k, i_b] + M_ang[k] * dlam


@qd.func
def func_free_row_cross(i_r: int, i_s: int, i_b: int, steel_free_rows: array_class.SteelFreeRows):
    """Return J_r . H^-1 J_s^T of two free-body rows of one contact, which share their bodies."""
    c = gs.qd_float(0.0)
    if steel_free_rows.dof_a[i_r, i_b] >= 0:
        c = c + steel_free_rows.J_a_lin[i_r, i_b].dot(steel_free_rows.MiJt_a_lin[i_s, i_b])
        c = c + steel_free_rows.J_a_ang[i_r, i_b].dot(steel_free_rows.MiJt_a_ang[i_s, i_b])
    if steel_free_rows.dof_b[i_r, i_b] >= 0:
        c = c + steel_free_rows.J_b_lin[i_r, i_b].dot(steel_free_rows.MiJt_b_lin[i_s, i_b])
        c = c + steel_free_rows.J_b_ang[i_r, i_b].dot(steel_free_rows.MiJt_b_ang[i_s, i_b])
    return c


@qd.func
def func_dense_row_cross(i_r: int, i_s: int, i_b: int, steel_rows: array_class.SteelRows):
    """Return J_r . Y_s of two dense rows."""
    c = gs.qd_float(0.0)
    for i_d in range(steel_rows.J.shape[1]):
        c = c + steel_rows.J[i_r, i_d, i_b] * steel_rows.Y[i_s, i_d, i_b]
    return c


@qd.func
def func_pooled_normal_impulse(i_p: int, i_b: int, row_parent: qd.Tensor, impulses: qd.Tensor):
    """Return the normal impulse of row i_p plus that of every other normal row of its friction patch ring."""
    load = impulses[i_p, i_b]
    i_k = row_parent[i_p, i_b]
    for _ in range(row_parent.shape[0]):
        if i_k >= 0 and i_k != i_p:
            load = load + impulses[i_k, i_b]
            i_k = row_parent[i_k, i_b]
    return load


@qd.func
def func_tangent_newton_step(u_t1: float, u_t2: float, G_t1t1: float, G_t1t2: float, G_t2t2: float, guard: float):
    """Return the tangent impulse change cancelling the tangent velocity (u_t1, u_t2), 0 for a singular block."""
    det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2
    d_t1 = gs.qd_float(0.0)
    d_t2 = gs.qd_float(0.0)
    if qd.abs(det) > guard:
        d_t1 = (-u_t1 * G_t2t2 + u_t2 * G_t1t2) / det
        d_t2 = (u_t1 * G_t1t2 - u_t2 * G_t1t1) / det
    return d_t1, d_t2


@qd.func
def func_clamp_to_cone(t1: float, t2: float, limit: float, guard: float):
    """Scale the tangent impulse (t1, t2) back to the disk of radius limit."""
    magnitude = qd.sqrt(t1 * t1 + t2 * t2)
    if magnitude > limit and magnitude > guard:
        scale = limit / magnitude
        t1 = t1 * scale
        t2 = t2 * scale
    return t1, t2


@qd.func
def func_bisection_contact(
    u: qd.types.vector(3),
    G: qd.types.matrix(3, 3),
    old: qd.types.vector(3),
    target_n: float,
    mu: float,
    steel_info: array_class.SteelInfo,
):
    """Solve the normal and tangent impulses of one contact together by bisection on the normal impulse.

    For every probe of the normal impulse, the tangent impulses cancel the tangent velocity (block solve of the tangent
    Delassus block) and are clamped to the Coulomb disk, and the normal velocity decides on which side the probe
    falls. u is the contact velocity (normal, tangent, tangent) at the old impulses, G the 3x3 Delassus block. A
    contact separating at zero load gets zero impulses. A degenerate normal response keeps the old impulses.
    """
    new = old
    if G[0, 0] >= steel_info.bisection_guard[None]:
        if u[0] + G[0, 0] * (0.0 - old[0]) >= target_n:
            new = qd.Vector.zero(gs.qd_float, 3)
        else:
            lo = gs.qd_float(0.0)
            hi = qd.max(qd.max(old[0] * 2.0, (target_n - u[0]) / G[0, 0] + old[0]), 1.0)
            for _ in range(steel_info.bisection_iterations[None]):
                mid = 0.5 * (lo + hi)
                d_n = mid - old[0]
                d_t1, d_t2 = func_tangent_newton_step(
                    u[1] + G[0, 1] * d_n,
                    u[2] + G[0, 2] * d_n,
                    G[1, 1],
                    G[1, 2],
                    G[2, 2],
                    steel_info.bisection_guard[None],
                )
                trial_t1, trial_t2 = func_clamp_to_cone(
                    old[1] + d_t1, old[2] + d_t2, mu * mid, steel_info.bisection_guard[None]
                )
                u_n_trial = u[0] + G[0, 0] * d_n + G[0, 1] * (trial_t1 - old[1]) + G[0, 2] * (trial_t2 - old[2])
                if u_n_trial < target_n:
                    lo = mid
                else:
                    hi = mid
            new[0] = 0.5 * (lo + hi)
            d_n = new[0] - old[0]
            d_t1, d_t2 = func_tangent_newton_step(
                u[1] + G[0, 1] * d_n, u[2] + G[0, 2] * d_n, G[1, 1], G[1, 2], G[2, 2], steel_info.bisection_guard[None]
            )
            new[1], new[2] = func_clamp_to_cone(
                old[1] + d_t1, old[2] + d_t2, mu * new[0], steel_info.bisection_guard[None]
            )
    return new


@qd.func
def func_coulomb_newton_contact(
    u: qd.types.vector(3),
    G: qd.types.matrix(3, 3),
    old: qd.types.vector(3),
    target_n: float,
    mu: float,
    steel_info: array_class.SteelInfo,
):
    """Solve the Coulomb cone of one contact exactly by a bracketed scalar Newton iteration.

    The normal impulse is eliminated through the normal row, leaving the tangent problem (A_T + x I) s = c_T on the
    Schur complement A_T of the tangent block. Sticking keeps x = 0, sliding searches the multiplier x where the
    tangent impulse reaches the cone, |s| = mu (w_NT . s - b_N) / W_N, first doubling a bracket, then by Newton steps
    kept inside the bracket. The normal impulse is not projected on the positive axis.
    """
    new = old
    if G[0, 0] >= steel_info.bisection_guard[None]:
        u_free = u - G @ old
        if u_free[0] < target_n:
            bN = u_free[0] - target_n
            WN = G[0, 0]
            wNT0 = G[0, 1]
            wNT1 = G[0, 2]
            AT00 = G[1, 1] - (wNT0 * wNT0) / WN
            AT01 = G[1, 2] - (wNT0 * wNT1) / WN
            AT10 = G[1, 2] - (wNT1 * wNT0) / WN
            AT11 = G[2, 2] - (wNT1 * wNT1) / WN
            cT0 = u_free[1] - (bN / WN) * wNT0
            cT1 = u_free[2] - (bN / WN) * wNT1
            inv_det0 = 1.0 / (AT00 * AT11 - AT01 * AT10)
            s_0 = (AT11 * cT0 - AT01 * cT1) * inv_det0
            s_1 = (AT00 * cT1 - AT10 * cT0) * inv_det0
            phi0 = qd.sqrt(s_0 * s_0 + s_1 * s_1) - mu * (wNT0 * s_0 + wNT1 * s_1 - bN) / WN
            if phi0 > 0.0:
                hi = gs.qd_float(1.0)
                for _ in range(steel_info.coulomb_newton_bracket_doublings[None]):
                    a_hi = AT00 + hi
                    d_hi = AT11 + hi
                    idet_hi = 1.0 / (a_hi * d_hi - AT01 * AT10)
                    sh0 = (d_hi * cT0 - AT01 * cT1) * idet_hi
                    sh1 = (a_hi * cT1 - AT10 * cT0) * idet_hi
                    if qd.sqrt(sh0 * sh0 + sh1 * sh1) - mu * (wNT0 * sh0 + wNT1 * sh1 - bN) / WN < 0.0:
                        break
                    hi = hi * 2.0
                lo = gs.qd_float(0.0)
                x = 0.5 * (lo + hi)
                tol = steel_info.coulomb_newton_tolerance[None] + steel_info.coulomb_newton_tolerance[None] * phi0
                for _ in range(steel_info.coulomb_newton_iterations[None]):
                    ax = AT00 + x
                    dx = AT11 + x
                    idet = 1.0 / (ax * dx - AT01 * AT10)
                    s0 = (dx * cT0 - AT01 * cT1) * idet
                    s1 = (ax * cT1 - AT10 * cT0) * idet
                    t0 = (dx * s0 - AT01 * s1) * idet
                    t1 = (ax * s1 - AT10 * s0) * idet
                    ns = qd.sqrt(s0 * s0 + s1 * s1)
                    fx = ns - mu * (wNT0 * s0 + wNT1 * s1 - bN) / WN
                    dfx = -(s0 * t0 + s1 * t1) / ns + mu * (wNT0 * t0 + wNT1 * t1) / WN
                    s_0 = s0
                    s_1 = s1
                    if qd.abs(fx) < tol or qd.abs(hi - lo) < steel_info.coulomb_newton_tolerance[None] * (1.0 + hi):
                        break
                    if fx > 0.0:
                        lo = x
                    else:
                        hi = x
                    x_new = 0.5 * (lo + hi)
                    if dfx != 0.0:
                        x_newton = x - fx / dfx
                        if x_newton > lo and x_newton < hi:
                            x_new = x_newton
                    x = x_new
            new[1] = -s_0
            new[2] = -s_1
            new[0] = -(wNT0 * new[1] + wNT1 * new[2] + bN) / WN
        else:
            new = qd.Vector.zero(gs.qd_float, 3)
    return new


@qd.func
def func_pgs_sweeps(
    i_b: int,
    n_iterations: int,
    iteration_offset: int,
    friction_start_iteration: int,
    row_phase: int,
    rhs_dense: qd.Tensor,
    rhs_free: qd.Tensor,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
    regularize: bool,
    freeze_drive_rows: bool,
):
    """Run n_iterations Gauss-Seidel sweeps of the rows of environment i_b, updating the velocity v_out.

    A row update is lambda += omega * (w * (-r / D) - (1 - w) * lambda) with the residual r = J v + rhs, the diagonal D
    and the regularization weight w (1 unless regularize), projected on lambda >= 0 for contacts and joint limits.
    Drive rows follow their force-drive update, and keep their impulse under freeze_drive_rows. Friction rows hold a
    zero impulse before friction_start_iteration, the sweeps counting from iteration_offset. The tangent rows of a
    contact are solved together at the first one (see the friction modes). Velocity limits are stateless clamps solved
    after the contact rows, each visit applying the impulse that removes the current overshoot. row_phase selects the
    row families of the sweep: 0 every family, 1 and 4 the contacts, 2 the internal rows and the velocity limits, 3 the
    internal rows, 5 the velocity limits. Contact torsion rows are solved after the free-body rows (see
    func_torsion_sweep). With contact compliance, every sweep starts by refreshing the right-hand sides from the current
    impulses (see func_compliance_refresh_rhs).
    """
    omega = steel_info.relaxation[None]
    is_contact_phase = row_phase == 0 or row_phase == 1 or row_phase == 4
    is_internal_phase = row_phase == 0 or row_phase == 2 or row_phase == 3
    is_velocity_limit_phase = row_phase == 0 or row_phase == 2 or row_phase == 5
    m_dense = steel_rows.count[i_b]
    dense_main_end = m_dense if row_phase != 5 else 0
    m_free = 0
    free_contact_end = 0
    if qd.static(steel_config.has_free_rigid_bodies):
        m_free = steel_free_rows.count[i_b]
        free_contact_end = qd.min(steel_free_rows.contact_rows_end[i_b], m_free)
    free_main_end = free_contact_end if is_contact_phase else 0
    for it in range(n_iterations):
        global_iter = iteration_offset + it
        is_changed = False
        if qd.static(steel_config.has_compliance):
            func_compliance_refresh_rhs(
                i_b, m_dense, m_free, rhs_dense, rhs_free, steel_rows, steel_free_rows, steel_config
            )
        for i_r in range(dense_main_end):
            row_type = steel_rows.row_type[i_r, i_b]
            is_friction = row_type == ROW_TYPE.FRICTION
            is_contact_row = row_type == ROW_TYPE.CONTACT or is_friction
            is_skipped = row_type == ROW_TYPE.JOINT_VELOCITY_LIMIT or row_type == ROW_TYPE.TORSION
            if freeze_drive_rows and row_type == ROW_TYPE.JOINT_TARGET:
                is_skipped = True
            if (is_contact_row and not is_contact_phase) or (not is_contact_row and not is_internal_phase):
                is_skipped = True
            if not is_skipped and is_friction and global_iter < friction_start_iteration:
                steel_rows.impulses[i_r, i_b] = 0.0
                is_skipped = True
            denom = steel_rows.diag[i_r, i_b]
            if denom <= 0.0 and not (qd.static(steel_config.friction_mode == 0) and is_friction):
                is_skipped = True
            if not is_skipped:
                jv = gs.qd_float(0.0)
                for i_d in range(steel_rows.J.shape[1]):
                    jv = jv + steel_rows.J[i_r, i_d, i_b] * steel_dyn.v_out[i_d, i_b]
                res = jv + rhs_dense[i_r, i_b]
                old = steel_rows.impulses[i_r, i_b]
                w = steel_rows.row_w[i_r, i_b] if regularize else 1.0
                delta = gs.qd_float(0.0)
                if denom > 0.0:
                    delta = -res / denom * w - (1.0 - w) * old
                new = old + omega * delta
                if row_type == ROW_TYPE.JOINT_TARGET:
                    if qd.static(steel_config.drive_mode == 1):
                        new = (
                            old * steel_rows.drive_impulse_multiplier[i_r, i_b]
                            + jv * steel_rows.drive_vel_multiplier[i_r, i_b]
                            + steel_rows.drive_target_vel_bias[i_r, i_b]
                        )
                        impulse_range = steel_rows.drive_impulse_range[i_r, i_b]
                        new = qd.math.clamp(new, impulse_range[0], impulse_range[1])
                elif is_friction:
                    new = old
                    i_p = steel_rows.row_parent[i_r, i_b]
                    if i_r == i_p + 1:
                        i_s = i_p + 2
                        if qd.static(steel_config.friction_mode == 1 or steel_config.friction_mode == 2):
                            # Bisection: the normal and both tangent impulses of the contact are solved together.
                            u = qd.Vector.zero(gs.qd_float, 3)
                            G = qd.Matrix.zero(gs.qd_float, 3, 3)
                            for k in qd.static(range(3)):
                                for i_d in range(steel_rows.J.shape[1]):
                                    u[k] = u[k] + steel_rows.J[i_p + k, i_d, i_b] * steel_dyn.v_out[i_d, i_b]
                            for k in qd.static(range(3)):
                                for m in qd.static(range(k, 3)):
                                    G[k, m] = func_dense_row_cross(i_p + k, i_p + m, i_b, steel_rows)
                            old3 = qd.Vector(
                                [steel_rows.impulses[i_p + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float
                            )
                            new3 = func_bisection_contact(
                                u, G, old3, -rhs_dense[i_p, i_b], steel_rows.row_mu[i_r, i_b], steel_info
                            )
                            for k in qd.static((0, 2)):
                                dlam_k = new3[k] - old3[k]
                                steel_rows.impulses[i_p + k, i_b] = new3[k]
                                if dlam_k != 0.0:
                                    is_changed = True
                                    func_dense_apply_impulse(i_p + k, i_b, dlam_k, steel_rows, steel_dyn)
                            new = new3[1]
                        else:
                            load = func_pooled_normal_impulse(i_p, i_b, steel_rows.row_parent, steel_rows.impulses)
                            radius = qd.max(steel_rows.row_mu[i_r, i_b] * load, 0.0)
                            r1 = gs.qd_float(0.0)
                            for i_d in range(steel_rows.J.shape[1]):
                                r1 = r1 + steel_rows.J[i_s, i_d, i_b] * steel_dyn.v_out[i_d, i_b]
                            r1 = r1 + rhs_dense[i_s, i_b]
                            old_s = steel_rows.impulses[i_s, i_b]
                            pair = func_friction_pair_candidate(
                                denom,
                                func_dense_row_cross(i_r, i_s, i_b, steel_rows),
                                steel_rows.diag[i_s, i_b],
                                qd.Vector([res, r1], dt=gs.qd_float),
                                qd.Vector([old, old_s], dt=gs.qd_float),
                                radius,
                                omega,
                                steel_info,
                            )
                            norm = pair.norm()
                            if norm > radius:
                                pair = pair * (radius / norm)
                            new = pair[0]
                            dlam_s = pair[1] - old_s
                            steel_rows.impulses[i_s, i_b] = pair[1]
                            if dlam_s != 0.0:
                                is_changed = True
                                func_dense_apply_impulse(i_s, i_b, dlam_s, steel_rows, steel_dyn)
                elif row_type == ROW_TYPE.CONTACT or row_type == ROW_TYPE.JOINT_LIMIT:
                    new = qd.max(new, 0.0)
                dlam = new - old
                steel_rows.impulses[i_r, i_b] = new
                if dlam != 0.0:
                    is_changed = True
                    func_dense_apply_impulse(i_r, i_b, dlam, steel_rows, steel_dyn)

        if qd.static(steel_config.has_free_rigid_bodies):
            for i_r in range(free_main_end):
                row_type = steel_free_rows.row_type[i_r, i_b]
                i_p = steel_free_rows.row_parent[i_r, i_b]
                is_friction = row_type == ROW_TYPE.FRICTION
                is_skipped = is_friction and i_r != i_p + 1
                if is_friction and global_iter < friction_start_iteration:
                    steel_free_rows.impulses[i_r, i_b] = 0.0
                    is_skipped = True
                inv_diag = steel_free_rows.eff_mass_inv[i_r, i_b]
                if inv_diag <= 0.0 and not (qd.static(steel_config.friction_mode == 0) and is_friction):
                    is_skipped = True
                radius = gs.qd_float(0.0)
                if qd.static(steel_config.friction_mode == 0):
                    if not is_skipped and is_friction:
                        load = func_pooled_normal_impulse(
                            i_p, i_b, steel_free_rows.row_parent, steel_free_rows.impulses
                        )
                        radius = qd.max(steel_free_rows.row_mu[i_r, i_b] * load, 0.0)
                        if (
                            radius == 0.0
                            and steel_free_rows.impulses[i_r, i_b] == 0.0
                            and steel_free_rows.impulses[i_r + 1, i_b] == 0.0
                        ):
                            is_skipped = True
                if not is_skipped:
                    jv = func_free_row_dot(i_r, i_b, steel_dyn.v_out, steel_free_rows)
                    res = jv + rhs_free[i_r, i_b]
                    old = steel_free_rows.impulses[i_r, i_b]
                    delta = -res * inv_diag
                    if row_type == ROW_TYPE.CONTACT:
                        w = steel_free_rows.row_w[i_r, i_b] if regularize else 1.0
                        delta = -res * inv_diag * w - (1.0 - w) * old
                    new = old + omega * delta
                    if row_type == ROW_TYPE.CONTACT:
                        new = qd.max(new, 0.0)
                    elif is_friction:
                        i_s = i_r + 1
                        if qd.static(steel_config.friction_mode == 0):
                            old_s = steel_free_rows.impulses[i_s, i_b]
                            pair = qd.Vector.zero(gs.qd_float, 2)
                            if radius > 0.0:
                                inv_s = steel_free_rows.eff_mass_inv[i_s, i_b]
                                r1 = func_free_row_dot(i_s, i_b, steel_dyn.v_out, steel_free_rows) + rhs_free[i_s, i_b]
                                pair = func_friction_pair_candidate(
                                    1.0 / inv_diag if inv_diag > 0.0 else 0.0,
                                    func_free_row_cross(i_r, i_s, i_b, steel_free_rows),
                                    1.0 / inv_s if inv_s > 0.0 else 0.0,
                                    qd.Vector([res, r1], dt=gs.qd_float),
                                    qd.Vector([old, old_s], dt=gs.qd_float),
                                    radius,
                                    omega,
                                    steel_info,
                                )
                                norm = pair.norm()
                                if norm > radius:
                                    pair = pair * (radius / norm)
                            new = pair[0]
                            dlam_s = pair[1] - old_s
                            steel_free_rows.impulses[i_s, i_b] = pair[1]
                            if dlam_s != 0.0:
                                is_changed = True
                                func_free_apply_impulse(i_s, i_b, dlam_s, steel_free_rows, steel_dyn)
                        else:
                            u = qd.Vector.zero(gs.qd_float, 3)
                            G = qd.Matrix.zero(gs.qd_float, 3, 3)
                            for k in qd.static(range(3)):
                                u[k] = func_free_row_dot(i_p + k, i_b, steel_dyn.v_out, steel_free_rows)
                                for m in qd.static(range(k, 3)):
                                    G[k, m] = func_free_row_cross(i_p + k, i_p + m, i_b, steel_free_rows)
                                    G[m, k] = G[k, m]
                            old3 = qd.Vector(
                                [steel_free_rows.impulses[i_p + k, i_b] for k in qd.static(range(3))],
                                dt=gs.qd_float,
                            )
                            mu = steel_free_rows.row_mu[i_r, i_b]
                            target_n = -rhs_free[i_p, i_b]
                            if qd.static(steel_config.friction_mode == 2):
                                # de Saxce maximum-dissipation bias.
                                target_n = target_n + mu * qd.sqrt(u[1] * u[1] + u[2] * u[2])
                            new3 = old3
                            if qd.static(steel_config.friction_mode == 3):
                                new3 = func_coulomb_newton_contact(u, G, old3, target_n, mu, steel_info)
                            else:
                                new3 = func_bisection_contact(u, G, old3, target_n, mu, steel_info)
                            for k in qd.static((0, 2)):
                                dlam_k = new3[k] - old3[k]
                                steel_free_rows.impulses[i_p + k, i_b] = new3[k]
                                if dlam_k != 0.0:
                                    is_changed = True
                                    func_free_apply_impulse(i_p + k, i_b, dlam_k, steel_free_rows, steel_dyn)
                            new = new3[1]
                    dlam = new - old
                    steel_free_rows.impulses[i_r, i_b] = new
                    if dlam != 0.0:
                        is_changed = True
                        func_free_apply_impulse(i_r, i_b, dlam, steel_free_rows, steel_dyn)

        if qd.static(steel_config.has_torsion):
            if is_contact_phase:
                if func_torsion_sweep(i_b, m_dense, omega, rhs_dense, steel_rows, steel_dyn):
                    is_changed = True

        if is_velocity_limit_phase:
            if qd.static(steel_config.enable_joint_velocity_limits):
                for i_r in range(m_dense):
                    denom = steel_rows.diag[i_r, i_b]
                    if steel_rows.row_type[i_r, i_b] == ROW_TYPE.JOINT_VELOCITY_LIMIT and denom > 0.0:
                        res = rhs_dense[i_r, i_b]
                        for i_d in range(steel_rows.J.shape[1]):
                            res = res + steel_rows.J[i_r, i_d, i_b] * steel_dyn.v_out[i_d, i_b]
                        dlam = -res / denom if res < 0.0 else 0.0
                        steel_rows.impulses[i_r, i_b] = dlam
                        if dlam != 0.0:
                            is_changed = True
                            func_dense_apply_impulse(i_r, i_b, dlam, steel_rows, steel_dyn)
            if qd.static(steel_config.has_fused_velocity_limits):
                for i_r in range(m_dense):
                    denom = steel_rows.diag[i_r, i_b]
                    if steel_rows.row_type[i_r, i_b] == ROW_TYPE.JOINT_TARGET and denom > 0.0:
                        qdot_max = steel_rows.drive_vel_limit[i_r, i_b]
                        jv = gs.qd_float(0.0)
                        for i_d in range(steel_rows.J.shape[1]):
                            jv = jv + steel_rows.J[i_r, i_d, i_b] * steel_dyn.v_out[i_d, i_b]
                        if qd.abs(jv) > qdot_max:
                            dlam = (qd.math.clamp(jv, -qdot_max, qdot_max) - jv) / denom
                            if dlam != 0.0:
                                is_changed = True
                                func_dense_apply_impulse(i_r, i_b, dlam, steel_rows, steel_dyn)
            if qd.static(steel_config.has_free_velocity_limits):
                for i_r in range(free_contact_end, m_free):
                    inv_diag = steel_free_rows.eff_mass_inv[i_r, i_b]
                    if steel_free_rows.row_type[i_r, i_b] == ROW_TYPE.JOINT_VELOCITY_LIMIT and inv_diag > 0.0:
                        res = func_free_row_dot(i_r, i_b, steel_dyn.v_out, steel_free_rows) + rhs_free[i_r, i_b]
                        dlam = -res * inv_diag if res < 0.0 else 0.0
                        steel_free_rows.impulses[i_r, i_b] = dlam
                        if dlam != 0.0:
                            is_changed = True
                            func_free_apply_impulse(i_r, i_b, dlam, steel_free_rows, steel_dyn)

        if global_iter >= friction_start_iteration and not is_changed:
            break
