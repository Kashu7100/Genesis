"""Contact torsion of the Steel solver: one bounded angular row per coherent contact group of articulated bodies.

A torsion row acts on the relative angular velocity of the two bodies of its group about the group's lead normal. Its
impulse is bounded by r * max(mu * N - S, 0), N being the final normal load of the group's normal rows in the sweep, S
the norm of the tangent impulses its anchors use and mu the undivided friction coefficient of the group, so sliding and
spin share one Coulomb budget. r is the effective radius of the group, the largest torsional friction radius of its
geoms. With point friction a group is a coplanar cluster of touching contacts of one geom pair, every member carrying
its own tangent pair. With friction patches a group is one patch region, the normal rows linked by the row_parent ring,
whose anchors alone carry tangent rows. Only dense rows get torsion: free-body contacts never do. Rows are rebuilt every
step, without spin history.
"""

from enum import IntEnum

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .rows import CONTACT_PATH, ROW_TYPE, func_contact_points_world, func_dense_row_dot


class TORSION_GROUP(IntEnum):
    """Membership marks of a torsion row besides the index of its group."""

    # Mark of a torsion row the velocity pass retires: its group no longer touches.
    RETIRED = -2


@qd.func
def func_is_anchor_normal(i_n: int, m_dense: int, i_b: int, steel_rows: array_class.SteelRows):
    """Return whether a contact normal row is followed by its own tangent pair within the active dense rows."""
    is_anchor = False
    if i_n + 2 < m_dense:
        is_anchor = True
        for k in qd.static(range(1, 3)):
            if steel_rows.row_type[i_n + k, i_b] != ROW_TYPE.FRICTION or steel_rows.row_parent[i_n + k, i_b] != i_n:
                is_anchor = False
    return is_anchor


@qd.func
def func_contact_torsion_radius(
    i_c: int, i_b: int, steel_contacts: array_class.SteelContacts, dyn_info: array_class.DynInfo
):
    """Return the torsion radius of a contact: the larger torsional friction radius of its two geoms."""
    i_g0 = steel_contacts.shape0[i_c, i_b]
    i_g1 = steel_contacts.shape1[i_c, i_b]
    return qd.max(dyn_info.geoms.friction_torsional[i_g0], dyn_info.geoms.friction_torsional[i_g1])


@qd.func
def func_append_torsion_row(
    i_c: int,
    i_r: int,
    i_b: int,
    steel_witnesses: array_class.SteelTorsionWitnesses,
    capacity: int,
    errno: qd.Tensor,
):
    """Assign the next torsion row to a group, returning the following row and whether the row overflows.

    The group is the one of representative contact i_c, the row i_r. A row beyond the capacity is not assigned and sets
    the overflow error of the environment.
    """
    is_overflow = i_r >= capacity
    if is_overflow:
        errno[i_b] = errno[i_b] | array_class.ErrorCode.STEEL_TORSION_ROWS_OVERFLOW
    else:
        steel_witnesses.spin_row[i_c, i_b] = i_r
        i_r = i_r + 1
    return i_r, is_overflow


@qd.func
def func_accumulate_angular_jacobian_row(
    i_r: int,
    i_b: int,
    i_l: int,
    sign: float,
    direction: qd.types.vector(3),
    steel_rows: array_class.SteelRows,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
):
    """Add the angular velocity of a link along a direction to the Jacobian of a dense row, over its ancestor DOFs.

    Row i_r gets sign * direction . w, w being the angular velocity of link i_l.
    """
    i_a = steel_model.links_art[i_l]
    i_j = steel_model.links_joint[i_l]
    for _ in range(steel_model.arts_joint_start[i_a + 1] - steel_model.arts_joint_start[i_a]):
        if i_j >= 0:
            i_d_start = steel_model.joints_dof_start[i_j]
            for i_d in range(i_d_start, i_d_start + steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]):
                steel_rows.J[i_r, i_d, i_b] = steel_rows.J[i_r, i_d, i_b] + sign * direction.dot(
                    steel_dyn.dofs_S_ang[i_d, i_b]
                )
            i_j = steel_model.joints_ancestor[i_j]


@qd.kernel(fastcache=True)
def kernel_steel_torsion_rows(
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_witnesses: array_class.SteelTorsionWitnesses,
    steel_dyn: array_class.SteelDynamics,
    dyn_info: array_class.DynInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
    errno: qd.Tensor,
):
    """Group the dense contacts of every environment and append one torsion row per group after the other rows.

    A dense contact is admitted when one of its geoms has a positive torsional friction radius, both geoms have a
    supported geometry and its rows are the normal row followed, with point friction, by its own tangent pair. With
    point friction, a contact separated by more than touch_tolerance (see SteelInfo) is left out, and the admitted
    contacts of one geom pair form clusters greedily in contact order: a contact joins the first cluster all of whose
    members share its normal within torsion_normal_cosine and its contact plane within touch_tolerance. With friction
    patches, a group is the region of the row_parent ring of the normal rows, touching or not, and a region holding a
    rejected contact gets no row. A group gets a row when it has an anchor, a touching member and a positive friction
    coefficient (the first anchor's, multiplied back by the anchor count of a region). Rows follow in contact order,
    all clusters of a geom pair at its first contact.

    The row acts along the normal of the first touching member (the lead): J v = n . (w_a - w_b), bounded by the
    effective radius times the remaining Coulomb budget of the group (see func_torsion_sweep). Dropped rows of the
    environment and torsion rows beyond the capacity set an error and solve no row of the environment in this step.
    """
    _B = steel_rows.count.shape[0]
    capacity = steel_rows.row_type.shape[0]
    n_dofs = steel_rows.J.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        for i_r in range(capacity):
            steel_rows.torsion_group[i_r, i_b] = -1
        count = steel_rows.count[i_b]
        n_contacts = steel_contacts.n[i_b]
        # Rows dropped for capacity would leave contacts out of their groups.
        is_failed = steel_rows.n_requested[i_b] > count
        if is_failed:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.STEEL_ROWS_DROPPED

        # Admission of the dense contact normal rows.
        for i_c in range(n_contacts):
            steel_witnesses.eligible[i_c, i_b] = 0
            steel_witnesses.group[i_c, i_b] = -1
            steel_witnesses.next_member[i_c, i_b] = -1
            steel_witnesses.tail[i_c, i_b] = i_c
            steel_witnesses.first_pair[i_c, i_b] = i_c
            steel_witnesses.lead[i_c, i_b] = -1
            steel_witnesses.spin_row[i_c, i_b] = -1
            steel_witnesses.coefficient[i_c, i_b] = 0.0
            if qd.static(steel_config.has_friction_anchors):
                steel_witnesses.is_rejected[i_c, i_b] = False
                steel_witnesses.is_ring_broken[i_c, i_b] = False
            slot = steel_contacts.slot[i_c, i_b]
            if steel_contacts.path[i_c, i_b] == CONTACT_PATH.DENSE and slot >= 0:
                steel_witnesses.eligible[i_c, i_b] = -1
                i_g0 = steel_contacts.shape0[i_c, i_b]
                i_g1 = steel_contacts.shape1[i_c, i_b]
                is_anchor = steel_contacts.slots_needed[i_c, i_b] == 3
                # Point friction admits the normal rows followed by their own tangent pair alone.
                is_slot_admitted = is_anchor
                if qd.static(steel_config.has_friction_anchors):
                    is_slot_admitted = True
                is_admitted = (
                    func_contact_torsion_radius(i_c, i_b, steel_contacts, dyn_info) > 0.0
                    and steel_info.geoms_is_torsion_supported[i_g0]
                    and steel_info.geoms_is_torsion_supported[i_g1]
                    and slot < count
                    and is_slot_admitted
                )
                if is_admitted:
                    is_admitted = steel_rows.row_type[slot, i_b] == ROW_TYPE.CONTACT
                if is_admitted and is_anchor:
                    is_admitted = func_is_anchor_normal(slot, count, i_b, steel_rows)
                if is_admitted:
                    p_a, p_b, n_c, gap = func_contact_points_world(i_c, i_b, steel_state, steel_contacts)
                    # Point friction leaves separated contacts out, patch regions keep every member.
                    is_open = gap > steel_info.touch_tolerance[None]
                    if qd.static(steel_config.has_friction_anchors):
                        is_open = False
                    if is_open:
                        steel_witnesses.eligible[i_c, i_b] = 0
                    else:
                        steel_witnesses.eligible[i_c, i_b] = 1
                        steel_witnesses.is_anchor[i_c, i_b] = is_anchor
                        steel_witnesses.normal[i_c, i_b] = n_c
                        steel_witnesses.point[i_c, i_b] = 0.5 * (p_a + p_b)
                        steel_witnesses.gap[i_c, i_b] = gap

        # Grouping, in contact order.
        for i_c in range(n_contacts):
            if steel_witnesses.eligible[i_c, i_b] == 1:
                if qd.static(steel_config.has_friction_anchors):
                    # The first admitted contact of a region walks its load ring and claims every admitted member.
                    if steel_witnesses.group[i_c, i_b] < 0:
                        slot_c = steel_contacts.slot[i_c, i_b]
                        i_s = slot_c
                        is_walking = True
                        for _ in range(capacity):
                            if is_walking:
                                i_m = -1
                                if i_s >= 0 and i_s < count:
                                    i_m = steel_rows.row_contact[i_s, i_b]
                                    # The ring walks the normal rows of the region alone.
                                    if i_m >= 0 and (
                                        steel_contacts.path[i_m, i_b] != CONTACT_PATH.DENSE
                                        or steel_contacts.slot[i_m, i_b] != i_s
                                    ):
                                        i_m = -1
                                if i_m < 0:
                                    steel_witnesses.is_ring_broken[i_c, i_b] = True
                                    is_walking = False
                                else:
                                    if steel_witnesses.eligible[i_m, i_b] == 1:
                                        steel_witnesses.group[i_m, i_b] = i_c
                                    elif steel_witnesses.eligible[i_m, i_b] == -1:
                                        steel_witnesses.is_rejected[i_c, i_b] = True
                                    i_s = steel_rows.row_parent[i_s, i_b]
                                    if i_s == slot_c:
                                        is_walking = False
                        if is_walking:
                            steel_witnesses.is_ring_broken[i_c, i_b] = True
                else:
                    # Greedy first-fit clustering of the admitted contacts of one geom pair.
                    n_c = steel_witnesses.normal[i_c, i_b]
                    point_c = steel_witnesses.point[i_c, i_b]
                    i_g0 = steel_contacts.shape0[i_c, i_b]
                    i_g1 = steel_contacts.shape1[i_c, i_b]
                    i_pair_first = i_c
                    i_cluster = -1
                    for i_o in range(i_c):
                        if (
                            steel_witnesses.eligible[i_o, i_b] == 1
                            and steel_contacts.shape0[i_o, i_b] == i_g0
                            and steel_contacts.shape1[i_o, i_b] == i_g1
                        ):
                            i_pair_first = qd.min(i_pair_first, i_o)
                            if i_cluster < 0 and steel_witnesses.group[i_o, i_b] == i_o:
                                is_compatible = True
                                i_m = i_o
                                for _ in range(n_contacts):
                                    if i_m >= 0:
                                        if (
                                            n_c.dot(steel_witnesses.normal[i_m, i_b])
                                            < steel_info.torsion_normal_cosine[None]
                                        ):
                                            is_compatible = False
                                        if (
                                            qd.abs(n_c.dot(point_c - steel_witnesses.point[i_m, i_b]))
                                            > steel_info.touch_tolerance[None]
                                        ):
                                            is_compatible = False
                                        i_m = steel_witnesses.next_member[i_m, i_b]
                                if is_compatible:
                                    i_cluster = i_o
                    if i_cluster < 0:
                        i_cluster = i_c
                    else:
                        i_tail = steel_witnesses.tail[i_cluster, i_b]
                        steel_witnesses.next_member[i_tail, i_b] = i_c
                        steel_witnesses.tail[i_cluster, i_b] = i_c
                    steel_witnesses.group[i_c, i_b] = i_cluster
                    steel_witnesses.first_pair[i_c, i_b] = i_pair_first

        # Budget of every group, accumulated in contact order.
        for i_c in range(n_contacts):
            if steel_witnesses.eligible[i_c, i_b] == 1 and steel_witnesses.group[i_c, i_b] == i_c:
                radius = gs.qd_float(0.0)
                n_anchors = 0
                i_first_anchor = -1
                i_touching = -1
                for i_m in range(n_contacts):
                    if steel_witnesses.eligible[i_m, i_b] == 1 and steel_witnesses.group[i_m, i_b] == i_c:
                        radius = qd.max(radius, func_contact_torsion_radius(i_m, i_b, steel_contacts, dyn_info))
                        if steel_witnesses.is_anchor[i_m, i_b]:
                            n_anchors = n_anchors + 1
                            if i_first_anchor < 0:
                                i_first_anchor = i_m
                        if i_touching < 0 and steel_witnesses.gap[i_m, i_b] <= steel_info.touch_tolerance[None]:
                            i_touching = i_m
                is_budgeted = True
                if qd.static(steel_config.has_friction_anchors):
                    if steel_witnesses.is_rejected[i_c, i_b]:
                        is_budgeted = False
                    elif steel_witnesses.is_ring_broken[i_c, i_b]:
                        errno[i_b] = errno[i_b] | array_class.ErrorCode.STEEL_TORSION_PATCH_RING
                        is_failed = True
                if is_budgeted and i_first_anchor >= 0 and i_touching >= 0:
                    slot_anchor = steel_contacts.slot[i_first_anchor, i_b]
                    coefficient = steel_rows.row_mu[slot_anchor + 1, i_b]
                    if qd.static(steel_config.has_friction_anchors):
                        # Patch anchors carry the coefficient divided by their count.
                        coefficient = coefficient * n_anchors
                    if coefficient > 0.0:
                        steel_witnesses.coefficient[i_c, i_b] = coefficient
                        steel_witnesses.radius[i_c, i_b] = radius
                        steel_witnesses.lead[i_c, i_b] = i_touching

        # Row assignment: regions in contact order, or every cluster of a geom pair at its first contact.
        i_r = count
        for i_c in range(n_contacts):
            if steel_witnesses.eligible[i_c, i_b] == 1 and steel_witnesses.group[i_c, i_b] == i_c:
                if qd.static(steel_config.has_friction_anchors):
                    if steel_witnesses.lead[i_c, i_b] >= 0:
                        i_r, is_overflow = func_append_torsion_row(i_c, i_r, i_b, steel_witnesses, capacity, errno)
                        if is_overflow:
                            is_failed = True
                elif steel_witnesses.first_pair[i_c, i_b] == i_c:
                    for i_k in range(i_c, n_contacts):
                        if (
                            steel_witnesses.lead[i_k, i_b] >= 0
                            and steel_witnesses.first_pair[i_k, i_b] == i_c
                            and steel_witnesses.group[i_k, i_b] == i_k
                        ):
                            i_r, is_overflow = func_append_torsion_row(i_k, i_r, i_b, steel_witnesses, capacity, errno)
                            if is_overflow:
                                is_failed = True

        if is_failed:
            # An incomplete contact system is not solved: the environment keeps no row in this step.
            steel_rows.count[i_b] = 0
            steel_rows.n_requested[i_b] = 0
            if qd.static(steel_config.has_free_rigid_bodies):
                steel_free_rows.count[i_b] = 0
                steel_free_rows.n_requested[i_b] = 0
        else:
            for i_c in range(n_contacts):
                if steel_witnesses.eligible[i_c, i_b] == 1:
                    i_rep = steel_witnesses.group[i_c, i_b]
                    i_s = steel_witnesses.spin_row[i_rep, i_b]
                    if i_s >= 0:
                        slot = steel_contacts.slot[i_c, i_b]
                        steel_rows.torsion_group[slot, i_b] = i_s
            for i_c in range(n_contacts):
                i_s = steel_witnesses.spin_row[i_c, i_b]
                if i_s >= 0:
                    i_lead = steel_witnesses.lead[i_c, i_b]
                    steel_rows.row_type[i_s, i_b] = ROW_TYPE.TORSION
                    # The lead contact gives the articulations the row acts on.
                    steel_rows.row_contact[i_s, i_b] = i_lead
                    steel_rows.row_ref[i_s, i_b] = -1
                    steel_rows.row_axis[i_s, i_b] = 0
                    steel_rows.row_parent[i_s, i_b] = steel_contacts.slot[i_lead, i_b]
                    steel_rows.row_mu[i_s, i_b] = steel_witnesses.coefficient[i_c, i_b]
                    steel_rows.torsion_radius[i_s, i_b] = steel_witnesses.radius[i_c, i_b]
                    steel_rows.phi[i_s, i_b] = 0.0
                    steel_rows.target_velocity[i_s, i_b] = 0.0
                    steel_rows.restitution[i_s, i_b] = 0.0
                    for i_d in range(n_dofs):
                        steel_rows.J[i_s, i_d, i_b] = 0.0
                        steel_rows.Y[i_s, i_d, i_b] = 0.0
                    n_lead = steel_witnesses.normal[i_lead, i_b]
                    i_l0 = steel_contacts.body0[i_lead, i_b]
                    i_l1 = steel_contacts.body1[i_lead, i_b]
                    if i_l0 >= 0:
                        func_accumulate_angular_jacobian_row(
                            i_s, i_b, i_l0, 1.0, n_lead, steel_rows, steel_dyn, steel_model
                        )
                    if i_l1 >= 0:
                        func_accumulate_angular_jacobian_row(
                            i_s, i_b, i_l1, -1.0, n_lead, steel_rows, steel_dyn, steel_model
                        )
            steel_rows.count[i_b] = i_r
            steel_rows.n_requested[i_b] = i_r


@qd.kernel(fastcache=True)
def kernel_steel_torsion_velocity_pass(
    dt: float,
    steel_rows: array_class.SteelRows,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Retire, for the velocity-only sweeps, every torsion row whose group no longer touches.

    A group keeps its row when one of its normal rows touches at the start of the step or reaches contact by the end
    of the step with the velocity of the position solve, within contact_end_gap_slop (see SteelInfo). A retired row
    keeps its impulse, which the sweep clamps to zero, refunding the spin of the position solve through its response.
    """
    _B = steel_rows.count.shape[0]
    capacity = steel_rows.row_type.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s, i_b in qd.ndrange(capacity, _B):
        m_dense = steel_rows.count[i_b]
        if i_s < m_dense and steel_rows.row_type[i_s, i_b] == ROW_TYPE.TORSION:
            is_touching = False
            for i_n in range(m_dense):
                if steel_rows.row_type[i_n, i_b] == ROW_TYPE.CONTACT and steel_rows.torsion_group[i_n, i_b] == i_s:
                    phi = steel_rows.phi[i_n, i_b]
                    speed = func_dense_row_dot(i_n, i_b, steel_dyn.v_out_snap, steel_rows)
                    end_gap = phi + dt * (speed - steel_rows.target_velocity[i_n, i_b])
                    if phi <= steel_info.contact_end_gap_slop[None] or end_gap <= steel_info.contact_end_gap_slop[None]:
                        is_touching = True
            steel_rows.torsion_group[i_s, i_b] = -1 if is_touching else TORSION_GROUP.RETIRED


@qd.func
def func_torsion_sweep(
    i_b: int,
    m_dense: int,
    omega: float,
    rhs_dense: qd.Tensor,
    steel_rows: array_class.SteelRows,
    steel_dyn: array_class.SteelDynamics,
):
    """Solve every torsion row of an environment once, in row order, returning whether an impulse changed.

    The budget of a group is mu * N, N being the sum of the positive normal impulses of its normal rows. When the
    tangent impulses of its anchors exceed it (a later normal row lowered the pooled load), they are scaled back onto
    it, their velocity response applied. The torsion impulse then takes a relaxed Gauss-Seidel step clamped to the
    radius times the remaining budget, zero for a retired row.
    """
    is_changed = False
    for i_s in range(m_dense):
        if steel_rows.row_type[i_s, i_b] == ROW_TYPE.TORSION:
            radius = steel_rows.torsion_radius[i_s, i_b]
            mu = qd.max(steel_rows.row_mu[i_s, i_b], 0.0)
            load = gs.qd_float(0.0)
            sliding_impulse = gs.qd_float(0.0)
            for i_n in range(m_dense):
                if steel_rows.row_type[i_n, i_b] == ROW_TYPE.CONTACT and steel_rows.torsion_group[i_n, i_b] == i_s:
                    load = load + qd.max(steel_rows.impulses[i_n, i_b], 0.0)
                    if func_is_anchor_normal(i_n, m_dense, i_b, steel_rows):
                        t1 = steel_rows.impulses[i_n + 1, i_b]
                        t2 = steel_rows.impulses[i_n + 2, i_b]
                        sliding_impulse = sliding_impulse + qd.sqrt(t1 * t1 + t2 * t2)
            budget = mu * load
            if sliding_impulse > budget:
                scale = budget / sliding_impulse
                sliding_impulse = 0.0
                for i_n in range(m_dense):
                    if (
                        steel_rows.row_type[i_n, i_b] == ROW_TYPE.CONTACT
                        and steel_rows.torsion_group[i_n, i_b] == i_s
                        and func_is_anchor_normal(i_n, m_dense, i_b, steel_rows)
                    ):
                        old1 = steel_rows.impulses[i_n + 1, i_b]
                        old2 = steel_rows.impulses[i_n + 2, i_b]
                        t1 = old1 * scale
                        t2 = old2 * scale
                        d1 = t1 - old1
                        d2 = t2 - old2
                        if d1 != 0.0 or d2 != 0.0:
                            is_changed = True
                            steel_rows.impulses[i_n + 1, i_b] = t1
                            steel_rows.impulses[i_n + 2, i_b] = t2
                            for i_d in range(steel_rows.Y.shape[1]):
                                steel_dyn.v_out[i_d, i_b] = steel_dyn.v_out[i_d, i_b] + (
                                    steel_rows.Y[i_n + 1, i_d, i_b] * d1 + steel_rows.Y[i_n + 2, i_d, i_b] * d2
                                )
                        sliding_impulse = sliding_impulse + qd.sqrt(t1 * t1 + t2 * t2)
            bound = radius * qd.max(budget - sliding_impulse, 0.0)
            if steel_rows.torsion_group[i_s, i_b] == TORSION_GROUP.RETIRED:
                bound = 0.0
            res = func_dense_row_dot(i_s, i_b, steel_dyn.v_out, steel_rows) + rhs_dense[i_s, i_b]
            old = steel_rows.impulses[i_s, i_b]
            diag = steel_rows.diag[i_s, i_b]
            trial = gs.qd_float(0.0)
            if diag > 0.0:
                trial = old - omega * res / diag
            new = qd.math.clamp(trial, -bound, bound)
            delta = new - old
            if delta != 0.0:
                is_changed = True
                steel_rows.impulses[i_s, i_b] = new
                for i_d in range(steel_rows.Y.shape[1]):
                    steel_dyn.v_out[i_d, i_b] = steel_dyn.v_out[i_d, i_b] + steel_rows.Y[i_s, i_d, i_b] * delta
    return is_changed
