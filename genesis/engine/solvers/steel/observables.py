"""Observables of the Steel solver: the contact forces of the step published to the Genesis state, and the capacity
status and row watermarks of the solver."""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .rows import CONTACT_PATH, func_contact_tangent_basis


@qd.func
def func_steel_contact_forces(
    dt: float,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_config: qd.template(),
):
    """Write the force of every contact of the step and the net contact force of every link.

    The force of a contact is its solved normal and friction impulses over dt. Genesis reports it as the force on the
    body of geom b, the body of geom a receiving its opposite. A contact without rows, or whose rows were dropped for
    capacity, reports zero. A friction patch region carries its friction on its anchors, its other contacts report
    their normal force alone.
    """
    n_links = dyn_state.links.contact_force.shape[0]
    _B = dyn_state.links.contact_force.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        dyn_state.links.contact_force[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        for i_c in range(collider_state.n_contacts[i_b]):
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            force = qd.Vector.zero(gs.qd_float, 3)
            if i_c < steel_contacts.n[i_b]:
                slot = steel_contacts.slot[i_c, i_b]
                path = steel_contacts.path[i_c, i_b]
                impulse = qd.Vector.zero(gs.qd_float, 3)
                if slot >= 0 and path == CONTACT_PATH.DENSE:
                    count = steel_rows.count[i_b]
                    if slot < count:
                        impulse[0] = steel_rows.impulses[slot, i_b]
                        if steel_contacts.slots_needed[i_c, i_b] == 3 and slot + 2 < count:
                            impulse[1] = steel_rows.impulses[slot + 1, i_b]
                            impulse[2] = steel_rows.impulses[slot + 2, i_b]
                if qd.static(steel_config.has_free_rigid_bodies):
                    if slot >= 0 and path == CONTACT_PATH.FREE:
                        count = steel_free_rows.count[i_b]
                        if slot < count:
                            impulse[0] = steel_free_rows.impulses[slot, i_b]
                            if steel_contacts.slots_needed[i_c, i_b] == 3 and slot + 2 < count:
                                impulse[1] = steel_free_rows.impulses[slot + 1, i_b]
                                impulse[2] = steel_free_rows.impulses[slot + 2, i_b]
                if qd.static(steel_config.has_propagation):
                    if slot >= 0 and path == CONTACT_PATH.PROPAGATION:
                        count = steel_prop.count[i_b]
                        if slot < count:
                            impulse[0] = steel_prop.impulses[slot, i_b]
                            if steel_contacts.slots_needed[i_c, i_b] == 3 and slot + 2 < count:
                                impulse[1] = steel_prop.impulses[slot + 1, i_b]
                                impulse[2] = steel_prop.impulses[slot + 2, i_b]
                n_c = -steel_contacts.normal[i_c, i_b]
                t0, t1 = func_contact_tangent_basis(n_c)
                # The impulses act on the body of shape 0 (geom a), Genesis reports the force on geom b.
                force = -(impulse[0] * n_c + impulse[1] * t0 + impulse[2] * t1) / dt
            collider_state.contact_data.force[i_col, i_b] = force
            i_la = collider_state.contact_data.link_a[i_col, i_b]
            i_lb = collider_state.contact_data.link_b[i_col, i_b]
            dyn_state.links.contact_force[i_la, i_b] = dyn_state.links.contact_force[i_la, i_b] - force
            dyn_state.links.contact_force[i_lb, i_b] = dyn_state.links.contact_force[i_lb, i_b] + force


@qd.func
def func_accumulate_row_watermarks(
    i_b: int,
    base: int,
    count: qd.Tensor,
    n_requested: qd.Tensor,
    n_dropped: qd.Tensor,
    row_watermarks: qd.Tensor,
    capacity: int,
):
    """Fold the row counts of one family of an environment into the watermarks starting at slot base."""
    requested = n_requested[i_b]
    qd.atomic_max(row_watermarks[base], count[i_b])
    qd.atomic_max(row_watermarks[base + 1], requested)
    qd.atomic_max(row_watermarks[base + 2], n_dropped[i_b])
    if requested > capacity:
        qd.atomic_max(row_watermarks[base + 3], requested - capacity)
        qd.atomic_add(row_watermarks[base + 4], 1)


@qd.func
def func_steel_update_status(
    collider_state: array_class.ColliderState,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_status: array_class.SteelStatus,
    steel_config: qd.template(),
    errno: qd.Tensor,
):
    """Latch the capacity status of every environment after the row allocation, and accumulate the row watermarks.

    A row family overflows when it requested more rows than its capacity or dropped a reservation, and the contacts of
    an environment are incomplete when its collision buffers overflowed. The status stays set until the environment is
    reset. With 'row_watermark', the high-water marks of the row counts of every family and of the contact count grow
    over the whole run.
    """
    _B = steel_rows.count.shape[0]
    contact_overflow_errno = (
        array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS
        | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS
        | array_class.ErrorCode.OVERFLOW_CONTACTS
    )
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        overflow = steel_status.envs_overflow[i_b]
        if steel_rows.n_requested[i_b] > steel_rows.row_type.shape[0] or steel_rows.n_dropped[i_b] > 0:
            overflow = overflow | array_class.SteelOverflow.DENSE_ROWS
        if qd.static(steel_config.has_free_rigid_bodies):
            if (
                steel_free_rows.n_requested[i_b] > steel_free_rows.row_type.shape[0]
                or steel_free_rows.n_dropped[i_b] > 0
            ):
                overflow = overflow | array_class.SteelOverflow.FREE_ROWS
        if qd.static(steel_config.has_propagation):
            if steel_prop.n_requested[i_b] > steel_prop.row_type.shape[0] or steel_prop.n_dropped[i_b] > 0:
                overflow = overflow | array_class.SteelOverflow.PROPAGATION_ROWS
        if (errno[i_b] & contact_overflow_errno) != 0:
            overflow = overflow | array_class.SteelOverflow.CONTACTS_INCOMPLETE
        steel_status.envs_overflow[i_b] = overflow

        if qd.static(steel_config.has_row_watermark):
            stride = array_class.SteelRowWatermark.FAMILY_STRIDE
            func_accumulate_row_watermarks(
                i_b,
                0,
                steel_rows.count,
                steel_rows.n_requested,
                steel_rows.n_dropped,
                steel_status.row_watermarks,
                steel_rows.row_type.shape[0],
            )
            if qd.static(steel_config.has_free_rigid_bodies):
                func_accumulate_row_watermarks(
                    i_b,
                    stride,
                    steel_free_rows.count,
                    steel_free_rows.n_requested,
                    steel_free_rows.n_dropped,
                    steel_status.row_watermarks,
                    steel_free_rows.row_type.shape[0],
                )
            if qd.static(steel_config.has_propagation):
                func_accumulate_row_watermarks(
                    i_b,
                    2 * stride,
                    steel_prop.count,
                    steel_prop.n_requested,
                    steel_prop.n_dropped,
                    steel_status.row_watermarks,
                    steel_prop.row_type.shape[0],
                )
            qd.atomic_max(
                steel_status.row_watermarks[array_class.SteelRowWatermark.CONTACT_SLOT], collider_state.n_contacts[i_b]
            )


@qd.kernel(fastcache=True)
def kernel_steel_reset_status(
    envs_idx: qd.types.ndarray(),
    steel_state: array_class.SteelState,
    steel_rows: array_class.SteelRows,
    steel_status: array_class.SteelStatus,
    steel_config: qd.template(),
):
    """Clear the capacity status and the compliant contact counts of the selected environments, and request a mass
    matrix refresh of their articulations on the next step."""
    n_arts = steel_state.arts_mass_update_requested.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        steel_status.envs_overflow[i_b] = 0
        steel_rows.compliance_n_active[i_b] = 0
        steel_rows.compliance_n_skipped[i_b] = 0
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b_ in qd.ndrange(n_arts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        steel_state.arts_mass_update_requested[i_a, i_b] = 1
