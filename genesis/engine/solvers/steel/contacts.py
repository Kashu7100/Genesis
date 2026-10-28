"""Contact matching across steps and contact warm start of the Steel solver.

The contacts of a step are sorted by geom pair (see kernel_steel_gather_contacts), so the contacts of one pair are
contiguous. Every contact is matched to the closest contact of the same geom pair of the last step whose witness
midpoint lies within match_position_threshold and whose normal is aligned within match_normal_cosine (see SteelInfo),
a contact of the last step going to the closest of the contacts claiming it (the earliest in sorted order on a tie),
the others being broken. The midpoint is symmetric in the two geoms.

With warm start, the rows of a matched contact start from the impulses its rows had at the end of the last step,
scaled by contact_warmstart_decay and by the ratio of the step to the last one. A friction row is seeded only when
both steps gave the contact friction rows. The carried tangent impulse is turned into the current tangent basis and
clamped to the friction disk of the carried normal impulse. The velocity of the seeded impulses is installed before
the first sweep, so velocity and impulses start consistent.
"""

from enum import IntEnum

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .friction_patches import func_warmstart_dt_scale, func_world_point
from .rows import CONTACT_PATH, ROW_TYPE, func_contact_tangent_basis

# Match index of a contact whose geom pair had no contact in the last step.


class MATCH(IntEnum):
    """Match index of a contact without a match in the contacts of the last step."""

    # Its geom pair had no contact in the last step.
    NOT_FOUND = -1
    # Its geom pair had contacts in the last step, none of them matching it.
    BROKEN = -2


@qd.func
def func_contact_midpoint(
    i_c: int, i_b: int, links_pos: qd.Tensor, links_quat: qd.Tensor, steel_contacts: array_class.SteelContacts
):
    """Return the world midpoint of the two witness points of a contact for the given link poses."""
    return 0.5 * (
        func_world_point(steel_contacts.body0[i_c, i_b], i_b, steel_contacts.point0[i_c, i_b], links_pos, links_quat)
        + func_world_point(steel_contacts.body1[i_c, i_b], i_b, steel_contacts.point1[i_c, i_b], links_pos, links_quat)
    )


@qd.func
def func_pair_lower_bound(
    i_b: int,
    lo: int,
    hi: int,
    i_ga: int,
    i_gb: int,
    shape0: qd.Tensor,
    shape1: qd.Tensor,
    steel_info: array_class.SteelInfo,
):
    """Return the first index of a sorted range whose geom pair does not precede a given geom pair.

    The range is [lo, hi), and hi is returned when every pair of it precedes (i_ga, i_gb).
    """
    for _ in range(steel_info.binary_search_iterations[None]):
        if lo < hi:
            mid = (lo + hi) // 2
            j_ga = shape0[mid, i_b]
            if j_ga < i_ga or (j_ga == i_ga and shape1[mid, i_b] < i_gb):
                lo = mid + 1
            else:
                hi = mid
    return lo


@qd.func
def func_steel_match_contacts(
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_contact_history: array_class.SteelContactHistory,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Match every contact of the step to a contact of the last step, writing match_index (see the module docstring).

    The first pass picks the closest candidate of every contact, a later candidate winning a distance tie. The second
    pass keeps a candidate only for the contact claiming it at the smallest distance, the earliest in sorted order on
    a tie, without falling back to a farther candidate.
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        match_index = MATCH.NOT_FOUND
        i_candidate = -1
        best_dist_sq = gs.qd_float(
            steel_info.match_position_threshold[None] * steel_info.match_position_threshold[None]
        )
        if i_c < steel_contacts.n[i_b]:
            i_ga = steel_contacts.shape0[i_c, i_b]
            i_gb = steel_contacts.shape1[i_c, i_b]
            n_prev = steel_contact_history.n[i_b]
            lo = func_pair_lower_bound(
                i_b, 0, n_prev, i_ga, i_gb, steel_contact_history.shape0, steel_contact_history.shape1, steel_info
            )
            hi = func_pair_lower_bound(
                i_b, lo, n_prev, i_ga, i_gb + 1, steel_contact_history.shape0, steel_contact_history.shape1, steel_info
            )
            if lo < hi:
                pos = func_contact_midpoint(i_c, i_b, steel_state.links_pos, steel_state.links_quat, steel_contacts)
                normal = steel_contacts.normal[i_c, i_b]
                for i_p in range(lo, hi):
                    diff = pos - steel_contact_history.pos[i_p, i_b]
                    dist_sq = diff.dot(diff)
                    if (
                        dist_sq <= best_dist_sq
                        and normal.dot(steel_contact_history.normal[i_p, i_b]) >= (steel_info.match_normal_cosine[None])
                    ):
                        best_dist_sq = dist_sq
                        i_candidate = i_p
                match_index = i_candidate if i_candidate >= 0 else MATCH.BROKEN
        steel_contacts.match_index[i_c, i_b] = match_index
        steel_contacts.match_candidate[i_c, i_b] = i_candidate
        steel_contacts.match_dist_sq[i_c, i_b] = best_dist_sq

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        i_candidate = steel_contacts.match_candidate[i_c, i_b]
        if i_candidate >= 0:
            # Every contact claiming the same candidate belongs to the same geom pair. The distances compare exactly:
            # the claim of the closest contact must win against its exact copies too, by sorted order.
            n = steel_contacts.n[i_b]
            i_ga = steel_contacts.shape0[i_c, i_b]
            i_gb = steel_contacts.shape1[i_c, i_b]
            lo = func_pair_lower_bound(i_b, 0, n, i_ga, i_gb, steel_contacts.shape0, steel_contacts.shape1, steel_info)
            hi = func_pair_lower_bound(
                i_b, lo, n, i_ga, i_gb + 1, steel_contacts.shape0, steel_contacts.shape1, steel_info
            )
            dist_sq = steel_contacts.match_dist_sq[i_c, i_b]
            for j_c in range(lo, hi):
                if j_c != i_c and steel_contacts.match_candidate[j_c, i_b] == i_candidate:
                    other_dist_sq = steel_contacts.match_dist_sq[j_c, i_b]
                    if other_dist_sq < dist_sq or (other_dist_sq == dist_sq and j_c < i_c):
                        steel_contacts.match_index[i_c, i_b] = MATCH.BROKEN


@qd.func
def func_transport_warmstart_contact(
    i_b: int,
    slot: int,
    count: int,
    normal: qd.types.vector(3),
    previous_normal: qd.types.vector(3),
    row_type: qd.Tensor,
    row_parent: qd.Tensor,
    row_mu: qd.Tensor,
    impulses: qd.Tensor,
):
    """Turn the carried tangent impulse of a contact into the current tangent basis and clamp it to its friction disk.

    The rows of the contact start at slot, and the disk is the one of its carried normal impulse.
    A carried normal impulse that is not finite or whose normal reversed is dropped.
    """
    normal_impulse = impulses[slot, i_b]
    if qd.math.isnan(normal_impulse) or qd.math.isinf(normal_impulse) or normal.dot(previous_normal) <= 0.0:
        normal_impulse = 0.0
    normal_impulse = qd.max(normal_impulse, 0.0)
    impulses[slot, i_b] = normal_impulse
    if slot + 2 < count:
        if (
            row_type[slot + 1, i_b] == ROW_TYPE.FRICTION
            and row_type[slot + 2, i_b] == ROW_TYPE.FRICTION
            and row_parent[slot + 1, i_b] == slot
            and row_parent[slot + 2, i_b] == slot
        ):
            # Contact normals point from shape0 toward shape1, the rows along the opposite direction.
            old_t0, old_t1 = func_contact_tangent_basis(-previous_normal)
            new_t0, new_t1 = func_contact_tangent_basis(-normal)
            tangent_world = impulses[slot + 1, i_b] * old_t0 + impulses[slot + 2, i_b] * old_t1
            tangent = qd.Vector([tangent_world.dot(new_t0), tangent_world.dot(new_t1)], dt=gs.qd_float)
            magnitude = tangent.norm()
            radius = qd.max(row_mu[slot + 1, i_b] * normal_impulse, 0.0)
            if qd.math.isnan(magnitude) or qd.math.isinf(magnitude) or radius <= 0.0:
                tangent = qd.Vector.zero(gs.qd_float, 2)
            elif magnitude > radius:
                tangent = tangent * (radius / magnitude)
            impulses[slot + 1, i_b] = tangent[0]
            impulses[slot + 2, i_b] = tangent[1]


@qd.func
def func_steel_gather_warmstart(
    route: int,
    dt: float,
    count: qd.Tensor,
    row_type: qd.Tensor,
    row_parent: qd.Tensor,
    row_mu: qd.Tensor,
    impulses: qd.Tensor,
    prev_impulses: qd.Tensor,
    prev_row_type: qd.Tensor,
    prev_row_parent: qd.Tensor,
    steel_contacts: array_class.SteelContacts,
    steel_contact_history: array_class.SteelContactHistory,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Seed the rows of the matched contacts of one row family from the impulses of their rows in the last step.

    The normal row is seeded when it was a contact row in both steps, a friction row when the contact had that friction
    row in both steps, then func_transport_warmstart_contact turns the tangent impulse into the current basis. Unmatched
    contacts start cold.
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_contacts.n.shape[0]
    max_rows = prev_impulses.shape[0]
    decay = steel_info.warmstart_decay[None]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        new_slot = -1
        if i_c < steel_contacts.n[i_b] and steel_contacts.path[i_c, i_b] == route:
            new_slot = steel_contacts.slot[i_c, i_b]
        n_rows = count[i_b]
        i_m = -1
        if new_slot >= 0 and new_slot < n_rows:
            i_m = steel_contacts.match_index[i_c, i_b]
        old_slot = -1
        if i_m >= 0:
            if steel_contact_history.path[i_m, i_b] == route:
                old_slot = steel_contact_history.slot[i_m, i_b]
        if old_slot >= 0 and old_slot < max_rows:
            scale = decay * func_warmstart_dt_scale(dt, steel_contact_history.dt[None])
            if row_type[new_slot, i_b] == ROW_TYPE.CONTACT and prev_row_type[old_slot, i_b] == ROW_TYPE.CONTACT:
                impulses[new_slot, i_b] = scale * prev_impulses[old_slot, i_b]
            for k in qd.static(range(1, 3)):
                new_r = new_slot + k
                old_r = old_slot + k
                if new_r < n_rows and old_r < max_rows:
                    if (
                        row_type[new_r, i_b] == ROW_TYPE.FRICTION
                        and row_parent[new_r, i_b] == new_slot
                        and prev_row_type[old_r, i_b] == ROW_TYPE.FRICTION
                        and prev_row_parent[old_r, i_b] == old_slot
                    ):
                        impulses[new_r, i_b] = scale * prev_impulses[old_r, i_b]
            func_transport_warmstart_contact(
                i_b,
                new_slot,
                n_rows,
                steel_contacts.normal[i_c, i_b],
                steel_contact_history.normal[i_m, i_b],
                row_type,
                row_parent,
                row_mu,
                impulses,
            )


@qd.func
def func_steel_warmstart_velocity(
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Add the velocity of the seeded impulses to the solve velocity: Y lambda of the dense rows, then H^-1 J^T lambda
    of the free-body rows on the degrees of freedom of their bodies."""
    _B = steel_dyn.v_out.shape[1]
    n_dofs = steel_rows.Y.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        delta = gs.qd_float(0.0)
        for i_r in range(steel_rows.count[i_b]):
            delta = delta + steel_rows.Y[i_r, i_d, i_b] * steel_rows.impulses[i_r, i_b]
        steel_dyn.v_out[i_d, i_b] = steel_dyn.v_out[i_d, i_b] + delta

    if qd.static(steel_config.has_free_rigid_bodies):
        n_arts = steel_model.arts_dof_start.shape[0]
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_a, i_b in qd.ndrange(n_arts, _B):
            if steel_model.arts_is_free_rigid[i_a] != 0:
                dof_start = steel_model.arts_dof_start[i_a]
                # Every component accumulates independently, in row order, side a before side b.
                delta_lin = qd.Vector.zero(gs.qd_float, 3)
                delta_ang = qd.Vector.zero(gs.qd_float, 3)
                for i_r in range(steel_free_rows.count[i_b]):
                    impulse = steel_free_rows.impulses[i_r, i_b]
                    # Rows without impulse contribute nothing, whatever their response.
                    if impulse != 0.0:
                        if steel_free_rows.dof_a[i_r, i_b] == dof_start:
                            delta_lin = delta_lin + steel_free_rows.MiJt_a_lin[i_r, i_b] * impulse
                            delta_ang = delta_ang + steel_free_rows.MiJt_a_ang[i_r, i_b] * impulse
                        if steel_free_rows.dof_b[i_r, i_b] == dof_start:
                            delta_lin = delta_lin + steel_free_rows.MiJt_b_lin[i_r, i_b] * impulse
                            delta_ang = delta_ang + steel_free_rows.MiJt_b_ang[i_r, i_b] * impulse
                for k in qd.static(range(3)):
                    steel_dyn.v_out[dof_start + k, i_b] = steel_dyn.v_out[dof_start + k, i_b] + delta_lin[k]
                    steel_dyn.v_out[dof_start + 3 + k, i_b] = steel_dyn.v_out[dof_start + 3 + k, i_b] + delta_ang[k]


@qd.func
def func_steel_snapshot_warmstart_rows(
    count: qd.Tensor,
    row_type: qd.Tensor,
    row_parent: qd.Tensor,
    impulses: qd.Tensor,
    prev_row_type: qd.Tensor,
    prev_row_parent: qd.Tensor,
    prev_impulses: qd.Tensor,
    steel_config: qd.template(),
):
    """Keep the row layout and the impulses of one row family for the warm start of the next step.

    The impulses past the row count of an environment are kept as zero.
    """
    capacity = row_type.shape[0]
    _B = row_type.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(capacity, _B):
        impulse = gs.qd_float(0.0)
        if i_r < count[i_b]:
            impulse = impulses[i_r, i_b]
        prev_impulses[i_r, i_b] = impulse
        prev_row_type[i_r, i_b] = row_type[i_r, i_b]
        prev_row_parent[i_r, i_b] = row_parent[i_r, i_b]


@qd.func
def func_steel_snapshot_warmstart(
    dt: float,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_contact_history: array_class.SteelContactHistory,
    steel_config: qd.template(),
):
    """Keep the contacts of the step and their rows for the matching and the warm start of the next step.

    It runs after the integration, so the poses of the detection of the step are in links_pos_prev / links_quat_prev.
    """
    _B = steel_contacts.n.shape[0]
    max_contacts = steel_contacts.shape0.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        path = CONTACT_PATH.NONE
        slot = -1
        if i_c < steel_contacts.n[i_b]:
            path = steel_contacts.path[i_c, i_b]
            slot = steel_contacts.slot[i_c, i_b]
            steel_contact_history.shape0[i_c, i_b] = steel_contacts.shape0[i_c, i_b]
            steel_contact_history.shape1[i_c, i_b] = steel_contacts.shape1[i_c, i_b]
            steel_contact_history.pos[i_c, i_b] = func_contact_midpoint(
                i_c, i_b, steel_state.links_pos_prev, steel_state.links_quat_prev, steel_contacts
            )
            steel_contact_history.normal[i_c, i_b] = steel_contacts.normal[i_c, i_b]
        steel_contact_history.path[i_c, i_b] = path
        steel_contact_history.slot[i_c, i_b] = slot

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        steel_contact_history.n[i_b] = steel_contacts.n[i_b]
    steel_contact_history.dt[None] = dt


@qd.kernel(fastcache=True)
def kernel_steel_reset_warmstart(
    envs_idx: qd.types.ndarray(),
    steel_contact_history: array_class.SteelContactHistory,
    steel_config: qd.template(),
):
    """Discard the contacts the selected environments carry to their next step, so that they all start cold."""
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        steel_contact_history.n[i_b] = 0
