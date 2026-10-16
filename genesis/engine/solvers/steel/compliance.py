"""Contact compliance of the Steel solver: implicit unilateral spring-damper contacts.

A contact between geoms with a contact stiffness k [N/m] and damping c [N s/m] (see 'gs.materials.Rigid') pushes with
F = max(0, -k phi_next - c u_next) over a step of duration h, phi_next = phi + h u_next being the gap at the end of the
step and u_next the separating velocity. With d = h k + c (h k across an open gap, which carries no dashpot), its normal
row solves u_next + k phi / d + lambda / (h d) = 0 under lambda >= 0: the bias k phi / d replaces the position
correction and the impulse-space compliance gamma = 1 / (h d) the constraint force mixing. The compliant residual
changes with the row's own impulse, so the solve refreshes it before every sweep, which is exact Gauss-Seidel since an
impulse only changes at its own row. Friction keeps the pair coefficient, bounded by the compliant normal impulse.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .rows import CONTACT_PATH


@qd.func
def func_contact_compliance(
    i_c: int, i_b: int, steel_contacts: array_class.SteelContacts, steel_info: array_class.SteelInfo
):
    """Return the stiffness and damping of a contact, zero stiffness for a rigid contact.

    The contact stiffnesses of the two geoms act as springs in series, k = k_a k_b / (k_a + k_b), a rigid geom (zero
    stiffness) leaving the other's. Their dampings combine as c = (c_a k_b^2 + c_b k_a^2) / (k_a + k_b)^2, the series
    combination of two spring-dampers to first order in frequency, which keeps the relaxation time c / k of two equal
    materials.
    """
    i_g0 = steel_contacts.shape0[i_c, i_b]
    i_g1 = steel_contacts.shape1[i_c, i_b]
    k_a = steel_info.geoms_contact_stiffness[i_g0]
    k_b = steel_info.geoms_contact_stiffness[i_g1]
    c_a = steel_info.geoms_contact_damping[i_g0]
    c_b = steel_info.geoms_contact_damping[i_g1]
    k = k_b
    c = c_b
    if k_b <= 0.0:
        k = k_a
        c = c_a
    elif k_a > 0.0:
        k_sum = k_a + k_b
        k = k_a * k_b / k_sum
        c = (c_a * k_b * k_b + c_b * k_a * k_a) / (k_sum * k_sum)
    return k, c


@qd.kernel(fastcache=True)
def kernel_steel_compliance_rows(
    dt: float,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
    errno: qd.Tensor,
):
    """Turn the normal row of every compliant contact into its implicit spring-damper row.

    Every row's current right-hand side becomes its base, its compliance zero. A compliant normal row then takes the
    base bias - target and the compliance gamma, which replaces the constraint force mixing in its diagonal (dense) or
    its inverse effective mass (free body). A compliant contact the allocator gave no row on purpose (no responding
    body, gap beyond the gates) is counted as skipped. Dropped rows, or a compliance or bias beyond the float32 range,
    set an error and solve no row of the environment in this step.
    """
    _B = steel_rows.count.shape[0]
    cfm = steel_info.cfm[None]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        m_dense = steel_rows.count[i_b]
        m_free = 0
        is_dropped = steel_rows.n_requested[i_b] > m_dense
        if qd.static(steel_config.has_free_rigid_bodies):
            m_free = steel_free_rows.count[i_b]
            is_dropped = is_dropped or steel_free_rows.n_requested[i_b] > m_free
        if is_dropped:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.STEEL_ROWS_DROPPED
        is_invalid = False

        for i_r in range(m_dense):
            steel_rows.compliance_base[i_r, i_b] = steel_rows.rhs[i_r, i_b]
            steel_rows.compliance_gamma[i_r, i_b] = 0.0
        if qd.static(steel_config.has_free_rigid_bodies):
            for i_r in range(m_free):
                steel_free_rows.compliance_base[i_r, i_b] = steel_free_rows.rhs[i_r, i_b]
                steel_free_rows.compliance_gamma[i_r, i_b] = 0.0

        n_active = 0
        n_skipped = 0
        for i_c in range(steel_contacts.n[i_b]):
            k, c = func_contact_compliance(i_c, i_b, steel_contacts, steel_info)
            if k > 0.0:
                path = steel_contacts.path[i_c, i_b]
                slot = steel_contacts.slot[i_c, i_b]
                if path == CONTACT_PATH.NONE:
                    if steel_contacts.slots_needed[i_c, i_b] == 0:
                        n_skipped = n_skipped + 1
                else:
                    phi = gs.qd_float(0.0)
                    if path == CONTACT_PATH.DENSE:
                        phi = steel_rows.phi[slot, i_b]
                    else:
                        if qd.static(steel_config.has_free_rigid_bodies):
                            phi = steel_free_rows.phi[slot, i_b]
                    # A dashpot acts across penetration alone.
                    d = dt * k
                    if phi <= 0.0:
                        d = d + c
                    gamma = 1.0 / (dt * d)
                    bias = k * phi / d
                    if not (
                        qd.abs(gamma) <= steel_info.float32_max[None] and qd.abs(bias) <= steel_info.float32_max[None]
                    ):
                        is_invalid = True
                    elif path == CONTACT_PATH.DENSE:
                        steel_rows.compliance_gamma[slot, i_b] = gamma
                        steel_rows.compliance_base[slot, i_b] = bias - steel_rows.target_velocity[slot, i_b]
                        steel_rows.diag[slot, i_b] = steel_rows.diag[slot, i_b] + gamma - cfm
                    else:
                        if qd.static(steel_config.has_free_rigid_bodies):
                            eff_mass_inv = steel_free_rows.eff_mass_inv[slot, i_b]
                            if eff_mass_inv <= 0.0:
                                is_invalid = True
                            else:
                                steel_free_rows.compliance_gamma[slot, i_b] = gamma
                                steel_free_rows.compliance_base[slot, i_b] = (
                                    bias - steel_free_rows.target_velocity[slot, i_b]
                                )
                                steel_free_rows.eff_mass_inv[slot, i_b] = 1.0 / (1.0 / eff_mass_inv - cfm + gamma)
                    n_active = n_active + 1
        steel_rows.compliance_n_active[i_b] = n_active
        steel_rows.compliance_n_skipped[i_b] = n_skipped

        if is_invalid:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.STEEL_COMPLIANCE_INVALID
        if is_dropped or is_invalid:
            # An incomplete contact system is not solved: the environment keeps no row in this step.
            steel_rows.count[i_b] = 0
            if qd.static(steel_config.has_free_rigid_bodies):
                steel_free_rows.count[i_b] = 0


@qd.func
def func_compliance_refresh_rhs(
    i_b: int,
    m_dense: int,
    m_free: int,
    rhs_dense: qd.Tensor,
    rhs_free: qd.Tensor,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_config: qd.template(),
):
    """Refresh the right-hand side of every row of an environment from its current impulse: base + gamma * lambda."""
    for i_r in range(m_dense):
        rhs_dense[i_r, i_b] = (
            steel_rows.compliance_base[i_r, i_b] + steel_rows.compliance_gamma[i_r, i_b] * steel_rows.impulses[i_r, i_b]
        )
    if qd.static(steel_config.has_free_rigid_bodies):
        for i_r in range(m_free):
            rhs_free[i_r, i_b] = (
                steel_free_rows.compliance_base[i_r, i_b]
                + steel_free_rows.compliance_gamma[i_r, i_b] * steel_free_rows.impulses[i_r, i_b]
            )
