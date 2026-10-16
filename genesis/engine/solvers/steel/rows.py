"""Constraint rows of the Steel solver (stage 4 of a step, immediate articulated contact response).

Joints, limits and contacts become rows of a projected Gauss-Seidel solve in impulse space. A row touching an
articulated body is dense: its Jacobian J spans the degrees of freedom of the environment and its response is
Y = H^-1 J^T. A row whose responding sides are only free rigid bodies is a free-body row, whose response uses the
inverse spatial inertia of each body. The solve enforces J v + rhs >= 0 for unilateral rows, rhs holding the bias.

Rows are allocated serially per environment in a fixed family order (drives, mimic joints, loop closures, joint
limits, joint velocity limits, contacts) and, inside a family, in articulation, degree-of-freedom, equality and
contact order: one of the orders the reference implementation admits.
A contact reserves its normal row followed by its two tangent rows. The reservation counter keeps counting past the
capacity, so a reservation that overflows drops every later one of the step.
"""

from enum import IntEnum

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.utils.array_class import PREELIM_MAX_ROWS

from ..rigid.constraint.island import func_sort_contacts
from .dynamics import func_diag, func_drive_params, func_skew
from .model import STEEL_JOINT_TYPE


class ROW_SLOT(IntEnum):
    """Sentinel slots of the row reservations."""

    # Reservation slot that no rejected reservation has lowered.
    UNBOUNDED = 2**31 - 1


class ROW_TYPE(IntEnum):
    """Steel constraint row types, numbered as in the reference implementation."""

    CONTACT = 0
    JOINT_TARGET = 1
    FRICTION = 2
    JOINT_LIMIT = 3
    JOINT_VELOCITY_LIMIT = 4
    MIMIC = 5
    CONNECT = 6
    TORSION = 7


class CONTACT_PATH(IntEnum):
    """Row family of a contact."""

    NONE = -1
    DENSE = 0
    FREE = 1
    PROPAGATION = 2


@qd.func
def func_contact_tangent_basis(n: qd.types.vector(3)):
    """Return the deterministic tangent pair every friction row of normal n uses."""
    t0 = n.cross(qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float))
    if t0.norm_sqr() < 1.0e-12:
        t0 = n.cross(qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float))
    t0 = t0.normalized()
    return t0, n.cross(t0).normalized()


@qd.func
def func_contact_points_world(
    i_c: int, i_b: int, steel_state: array_class.SteelState, steel_contacts: array_class.SteelContacts
):
    """Return the world witness points of contact i_c, the normal from shape 1 toward shape 0 and the signed gap."""
    p_a = steel_contacts.point0[i_c, i_b]
    i_l0 = steel_contacts.body0[i_c, i_b]
    if i_l0 >= 0:
        p_a = steel_state.links_pos[i_l0, i_b] + gu.qd_transform_by_quat(p_a, steel_state.links_quat[i_l0, i_b])
    p_b = steel_contacts.point1[i_c, i_b]
    i_l1 = steel_contacts.body1[i_c, i_b]
    if i_l1 >= 0:
        p_b = steel_state.links_pos[i_l1, i_b] + gu.qd_transform_by_quat(p_b, steel_state.links_quat[i_l1, i_b])
    n_c = -steel_contacts.normal[i_c, i_b]
    return p_a, p_b, n_c, n_c.dot(p_a - p_b)


@qd.func
def func_contact_row_points(
    i_c: int,
    i_b: int,
    is_friction: bool,
    p_a: qd.types.vector(3),
    p_b: qd.types.vector(3),
    steel_contacts: array_class.SteelContacts,
    steel_config: qd.template(),
):
    """Return the points a contact row acts at: the witness points, or their midpoint with a shared anchor.

    With friction patches, the friction rows of an anchor act at its patch point on both bodies.
    """
    point_a, point_b = p_a, p_b
    is_shared = qd.static(steel_config.contact_shared_anchor)
    if qd.static(steel_config.contact_friction_shared_anchor):
        is_shared = is_shared or is_friction
    if is_shared:
        point_a = 0.5 * (p_a + p_b)
        point_b = point_a
    if qd.static(steel_config.has_friction_anchors):
        if is_friction:
            point_a = steel_contacts.patch_point[i_c, i_b]
            point_b = point_a
    return point_a, point_b


@qd.func
def func_contact_mu(i_c: int, i_b: int, steel_contacts: array_class.SteelContacts, steel_info: array_class.SteelInfo):
    """Return the friction coefficient of contact i_c: the mean of the coefficients of its two geoms."""
    i_g0 = steel_contacts.shape0[i_c, i_b]
    i_g1 = steel_contacts.shape1[i_c, i_b]
    return 0.5 * (steel_info.geoms_mu[i_g0, i_b] + steel_info.geoms_mu[i_g1, i_b])


@qd.func
def func_contact_restitution(
    i_c: int, i_b: int, steel_contacts: array_class.SteelContacts, steel_info: array_class.SteelInfo
):
    """Return the restitution of contact i_c: the mean of the coefficients of its two geoms clamped to [0, 1]."""
    e = gs.qd_float(0.0)
    for side in qd.static(range(2)):
        i_g = steel_contacts.shape0[i_c, i_b] if qd.static(side == 0) else steel_contacts.shape1[i_c, i_b]
        e_g = steel_info.geoms_restitution[i_g, i_b]
        if not (qd.math.isnan(e_g) or qd.math.isinf(e_g)):
            e = e + qd.math.clamp(e_g, 0.0, 1.0)
    return 0.5 * e


@qd.func
def func_restitution_fires(phi: float, u: float, dt: float, steel_info: array_class.SteelInfo):
    """Return whether a contact closing at normal speed -u rebounds: faster than the threshold, and touching or
    predicted to reach the surface within the step."""
    is_firing = False
    if u < -steel_info.rebound_velocity_threshold[None]:
        is_firing = (
            phi <= steel_info.contact_end_gap_slop[None] or phi + dt * u <= steel_info.contact_end_gap_slop[None]
        )
    return is_firing


@qd.func
def func_accumulate_jacobian_row(
    i_r: int,
    i_b: int,
    i_l: int,
    sign: float,
    point: qd.types.vector(3),
    direction: qd.types.vector(3),
    steel_rows: array_class.SteelRows,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
):
    """Add sign * direction . v(point) of link i_l to the Jacobian of dense row i_r, over its ancestor DOFs."""
    i_a = steel_model.links_art[i_l]
    origin = steel_dyn.arts_origin[i_a, i_b]
    lever = point - origin
    i_j = steel_model.links_joint[i_l]
    for _ in range(steel_model.arts_joint_start[i_a + 1] - steel_model.arts_joint_start[i_a]):
        if i_j >= 0:
            i_d_start = steel_model.joints_dof_start[i_j]
            for i_d in range(i_d_start, i_d_start + steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]):
                v = steel_dyn.dofs_S_lin[i_d, i_b] + steel_dyn.dofs_S_ang[i_d, i_b].cross(lever)
                steel_rows.J[i_r, i_d, i_b] = steel_rows.J[i_r, i_d, i_b] + sign * direction.dot(v)
            i_j = steel_model.joints_ancestor[i_j]


@qd.func
def func_reserve_rows(n_slots: int, counter: int, first_rejected: int, n_dropped: int, capacity: int):
    """Reserve n_slots consecutive rows, returning the first slot (-1 if rejected) and the updated counters.

    A rejected reservation records its first slot, so the row count is truncated before it, and counts its rows as
    dropped.
    """
    slot = counter
    counter = counter + n_slots
    if slot + n_slots > capacity:
        first_rejected = qd.min(first_rejected, slot)
        n_dropped = n_dropped + n_slots
        slot = -1
    return slot, counter, first_rejected, n_dropped


@qd.func
def func_is_propagation_contact(
    i_l0: int, i_l1: int, is_free: bool, steel_model: array_class.SteelModelInfo, steel_config: qd.template()
):
    """Return whether a contact between links i_l0 and i_l1 (-1 for static geometry) takes propagation rows.

    'propagation' and 'propagation_colored' route every contact there, those whose responding bodies are only free
    rigid bodies (is_free) included. Otherwise a contact touching an articulation that is not a free rigid body goes
    there, unless it pairs two links of one articulation without propagation_same_articulation_rows.
    """
    is_propagation = False
    if is_free:
        if qd.static(steel_config.articulated_contact_response == 1 or steel_config.articulated_contact_response == 3):
            is_propagation = True
    else:
        i_a0 = -1
        i_a1 = -1
        if i_l0 >= 0:
            i_a0 = steel_model.links_art[i_l0]
        if i_l1 >= 0:
            i_a1 = steel_model.links_art[i_l1]
        is_non_free0 = False
        is_non_free1 = False
        if i_a0 >= 0:
            is_non_free0 = steel_model.arts_is_free_rigid[i_a0] == 0
        if i_a1 >= 0:
            is_non_free1 = steel_model.arts_is_free_rigid[i_a1] == 0
        if is_non_free0 or is_non_free1:
            is_propagation = True
            if qd.static(not steel_config.propagation_same_articulation_rows):
                if is_non_free0 and is_non_free1 and i_a0 == i_a1:
                    is_propagation = False
    return is_propagation


@qd.kernel(fastcache=True)
def kernel_steel_gather_contacts(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Convert the Genesis contacts of the step into Steel contacts, in the deterministic Genesis contact order.

    A Genesis contact holds the midpoint of its two witness points, the penetration depth and the normal from geom b
    toward geom a. Steel takes geom a as shape 0 with the normal from shape 0 toward shape 1, the witness points
    pos -/+ depth / 2 * normal_ab expressed in the frame of their link, so the signed gap is -depth.

    When the narrow phase order is not deterministic, or for contact matching, which needs the contacts of a geom pair
    contiguous, the contacts are sorted by geom pair, then by position (see func_sort_contacts).
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        steel_contacts.n[i_b] = qd.min(collider_state.n_contacts[i_b], max_contacts)
        if qd.static(steel_config.has_sorted_contacts):
            func_sort_contacts(
                i_b,
                0,
                collider_state.n_contacts[i_b],
                collider_state.contact_sort_idx,
                collider_state.contact_data.pos,
                collider_state.contact_data.geom_a,
                collider_state.contact_data.geom_b,
                dyn_state.geoms.pos,
                dyn_state.geoms.quat,
            )

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        if i_c < steel_contacts.n[i_b]:
            i_col = collider_state.contact_sort_idx[i_c, i_b]
            normal_ab = -collider_state.contact_data.normal[i_col, i_b]
            pos = collider_state.contact_data.pos[i_col, i_b]
            half_depth = 0.5 * collider_state.contact_data.penetration[i_col, i_b]
            for side in qd.static(range(2)):
                i_l = collider_state.contact_data.link_a[i_col, i_b]
                i_g = collider_state.contact_data.geom_a[i_col, i_b]
                point = pos + half_depth * normal_ab
                if qd.static(side == 1):
                    i_l = collider_state.contact_data.link_b[i_col, i_b]
                    i_g = collider_state.contact_data.geom_b[i_col, i_b]
                    point = pos - half_depth * normal_ab
                if steel_model.links_art[i_l] < 0:
                    i_l = -1
                else:
                    point = gu.qd_inv_transform_by_quat(
                        point - steel_state.links_pos[i_l, i_b], steel_state.links_quat[i_l, i_b]
                    )
                if qd.static(side == 0):
                    steel_contacts.shape0[i_c, i_b] = i_g
                    steel_contacts.body0[i_c, i_b] = i_l
                    steel_contacts.point0[i_c, i_b] = point
                else:
                    steel_contacts.shape1[i_c, i_b] = i_g
                    steel_contacts.body1[i_c, i_b] = i_l
                    steel_contacts.point1[i_c, i_b] = point
            steel_contacts.normal[i_c, i_b] = normal_ab


@qd.func
def func_is_scalar_axis_joint(joint_type: int):
    """Return whether every degree of freedom of a joint of this type has one scalar coordinate."""
    return (
        joint_type == STEEL_JOINT_TYPE.PRISMATIC
        or joint_type == STEEL_JOINT_TYPE.REVOLUTE
        or joint_type == STEEL_JOINT_TYPE.D6
    )


@qd.func
def func_init_dense_row(
    i_r: int, i_b: int, row_type: int, row_ref: int, row_axis: int, steel_rows: array_class.SteelRows
):
    """Record the family and the source of a reserved dense row that is not a contact row."""
    steel_rows.row_type[i_r, i_b] = row_type
    steel_rows.row_contact[i_r, i_b] = -1
    steel_rows.row_ref[i_r, i_b] = row_ref
    steel_rows.row_axis[i_r, i_b] = row_axis
    steel_rows.row_parent[i_r, i_b] = -1
    steel_rows.row_mu[i_r, i_b] = 0.0
    steel_rows.phi[i_r, i_b] = 0.0
    steel_rows.target_velocity[i_r, i_b] = 0.0
    steel_rows.restitution[i_r, i_b] = 0.0


@qd.func
def func_steel_allocate_drive_rows(
    dyn_state: array_class.DynState,
    steel_rows: array_class.SteelRows,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Start the dense rows of the step with one drive row per driven degree of freedom (drive_mode='rows').

    A degree of freedom of a prismatic, revolute or multi-axis joint is driven when its drive stiffness or damping is
    positive. The drive rows are reserved before the dynamics of the step, whose velocity scaling skips the degrees of
    freedom a drive row limits.
    """
    _B = steel_rows.count.shape[0]
    n_arts = steel_model.arts_dof_start.shape[0]
    capacity = steel_rows.row_type.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        counter = 0
        if qd.static(steel_config.drive_mode == 1):
            for i_a in range(n_arts):
                for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
                    i_d_start = steel_model.joints_dof_start[i_j]
                    n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
                    for i_d in range(i_d_start, i_d_start + n_dofs):
                        steel_rows.dofs_drive_slot[i_d, i_b] = -1
                        if func_is_scalar_axis_joint(steel_model.joints_type[i_j]):
                            ke, kd, _, _ = func_drive_params(i_d, i_b, dyn_state, dyn_info, rigid_info, steel_config)
                            if ke > 0.0 or kd > 0.0:
                                slot = counter
                                counter = counter + 1
                                if slot < capacity:
                                    steel_rows.dofs_drive_slot[i_d, i_b] = slot
                                    func_init_dense_row(slot, i_b, ROW_TYPE.JOINT_TARGET, i_d, 0, steel_rows)
        steel_rows.n_requested[i_b] = counter
        steel_rows.first_rejected[i_b] = ROW_SLOT.UNBOUNDED


@qd.func
def func_steel_allocate_rows(
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Reserve the rows of every environment after its drive rows and record the row family of every contact.

    Dense rows follow the order drives, mimic joints, loop closures (three rows each, all or none), joint limits,
    joint velocity limits, contacts. A finite joint limit gets a row while the joint lies within
    joint_limit_activation_gap of it, its gap being positive inside the range. A joint velocity limit gets a lower
    and an upper row while the predicted speed reaches velocity_limit_activation_fraction of it. A contact gets rows
    when one of its bodies responds to impulses and its gap does not exceed contact_gap_limit: three rows (normal and
    two tangents) when its gap does not exceed friction_gap_limit, one otherwise. Its rows are free-body rows when
    every body it touches is a free rigid body or static. The velocity limits of free rigid bodies follow the
    free-body contact rows. With a propagation response, a contact touching an articulation that is not a free rigid
    body takes propagation rows instead of dense rows, unless it pairs two links of one articulation without
    propagation_same_articulation_rows, and so do the free-body contacts except with 'propagation_fused'. With
    sleep_skip_constraints, an articulation asleep for the step gets no limit or velocity limit row and its links do not
    respond to impulses.
    """
    _B = steel_state.q.shape[1]
    n_arts = steel_model.arts_dof_start.shape[0]
    capacity = steel_rows.row_type.shape[0]
    free_capacity = 0
    if qd.static(steel_config.has_free_rigid_bodies):
        free_capacity = steel_free_rows.row_type.shape[0]
    propagation_capacity = 0
    if qd.static(steel_config.has_propagation):
        propagation_capacity = steel_prop.row_type.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        counter = steel_rows.n_requested[i_b]
        first_rejected = steel_rows.first_rejected[i_b]
        n_dropped = 0

        if qd.static(steel_config.has_mimics):
            for i_m in range(steel_model.mimics_eq.shape[0]):
                slot, counter, first_rejected, n_dropped = func_reserve_rows(
                    1, counter, first_rejected, n_dropped, capacity
                )
                steel_rows.mimics_slot[i_m, i_b] = slot
                if slot >= 0:
                    func_init_dense_row(slot, i_b, ROW_TYPE.MIMIC, i_m, 0, steel_rows)

        if qd.static(steel_config.has_connects):
            for i_k in range(steel_model.connects_eq.shape[0]):
                slot, counter, first_rejected, n_dropped = func_reserve_rows(
                    3, counter, first_rejected, n_dropped, capacity
                )
                steel_rows.connects_slot[i_k, i_b] = slot
                if slot >= 0:
                    for axis in qd.static(range(3)):
                        func_init_dense_row(slot + axis, i_b, ROW_TYPE.CONNECT, i_k, axis, steel_rows)

        if qd.static(steel_config.enable_joint_limits):
            gap = steel_info.joint_limit_activation_gap[None]
            for i_a in range(n_arts):
                is_active = True
                if qd.static(steel_config.sleep_skip_constraints):
                    is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
                for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
                    if is_active and func_is_scalar_axis_joint(steel_model.joints_type[i_j]):
                        i_q_start = steel_model.joints_q_start[i_j]
                        i_d_start = steel_model.joints_dof_start[i_j]
                        for k in range(steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]):
                            i_d = i_d_start + k
                            I_d = [i_d, i_b] if qd.static(steel_config.batch_dofs_info) else i_d
                            q = steel_state.q[i_q_start + k, i_b]
                            q0 = rigid_info.qpos0[i_q_start + k, i_b]
                            for side in qd.static(range(2)):
                                bound = dyn_info.dofs.limit[I_d][side] - q0
                                phi = q - bound if qd.static(side == 0) else bound - q
                                if not (qd.math.isinf(bound) or qd.math.isnan(bound)) and phi <= gap:
                                    slot = counter
                                    counter = counter + 1
                                    if slot < capacity:
                                        func_init_dense_row(slot, i_b, ROW_TYPE.JOINT_LIMIT, i_d, side, steel_rows)
                                        steel_rows.phi[slot, i_b] = phi

        if qd.static(steel_config.enable_joint_velocity_limits):
            fraction = steel_info.velocity_limit_activation_fraction[None]
            for i_a in range(n_arts):
                is_active = True
                if qd.static(steel_config.sleep_skip_constraints):
                    is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
                for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
                    if is_active and func_is_scalar_axis_joint(steel_model.joints_type[i_j]):
                        i_d_start = steel_model.joints_dof_start[i_j]
                        for i_d in range(
                            i_d_start, i_d_start + steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
                        ):
                            qdot_max = steel_info.dofs_velocity_limit[i_d]
                            is_active = qdot_max > 0.0
                            if qd.static(steel_config.has_fused_velocity_limits):
                                if steel_rows.dofs_drive_slot[i_d, i_b] >= 0:
                                    is_active = False
                            if fraction > 0.0:
                                if qd.abs(steel_dyn.v_hat[i_d, i_b]) < fraction * qdot_max:
                                    is_active = False
                            if is_active:
                                for side in qd.static(range(2)):
                                    slot = counter
                                    counter = counter + 1
                                    if slot < capacity:
                                        func_init_dense_row(
                                            slot, i_b, ROW_TYPE.JOINT_VELOCITY_LIMIT, i_d, side, steel_rows
                                        )

        free_counter = 0
        free_first_rejected = ROW_SLOT.UNBOUNDED
        free_n_dropped = 0
        propagation_counter = 0
        propagation_first_rejected = ROW_SLOT.UNBOUNDED
        propagation_n_dropped = 0
        has_dense_contacts = 0
        for i_c in range(steel_contacts.n[i_b]):
            steel_contacts.path[i_c, i_b] = CONTACT_PATH.NONE
            steel_contacts.slot[i_c, i_b] = -1
            steel_contacts.slots_needed[i_c, i_b] = 0
            i_l0 = steel_contacts.body0[i_c, i_b]
            i_l1 = steel_contacts.body1[i_c, i_b]
            # A contact touching a sleeping link only touches sleeping links or static geometry, an island being all
            # awake or all asleep, so it gets no rows.
            is_responding = False
            for side in qd.static(range(2)):
                i_l = i_l0 if qd.static(side == 0) else i_l1
                if i_l >= 0:
                    is_side_responding = steel_model.links_has_response_dofs[i_l] != 0
                    if qd.static(steel_config.sleep_skip_constraints):
                        i_a = steel_model.links_art[i_l]
                        is_side_responding = is_side_responding and steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
                    is_responding = is_responding or is_side_responding
            if is_responding:
                _, _, _, phi = func_contact_points_world(i_c, i_b, steel_state, steel_contacts)
                if phi <= steel_info.contact_gap_limit[None]:
                    is_free = False
                    if qd.static(steel_config.has_free_rigid_bodies):
                        is_free = True
                        for side in qd.static(range(2)):
                            i_l = i_l0 if qd.static(side == 0) else i_l1
                            if i_l >= 0:
                                i_a = steel_model.links_art[i_l]
                                is_free = is_free and steel_model.arts_is_free_rigid[i_a] != 0
                    is_propagation = False
                    if qd.static(steel_config.has_propagation):
                        is_propagation = func_is_propagation_contact(i_l0, i_l1, is_free, steel_model, steel_config)
                    # With friction patches, every contact keeps its normal row and only anchors get friction rows.
                    is_friction = phi <= steel_info.friction_gap_limit[None]
                    if qd.static(steel_config.has_friction_anchors):
                        is_friction = is_friction and steel_contacts.patch_weight[i_c, i_b] > 0.0
                    n_slots = 3 if is_friction else 1
                    steel_contacts.slots_needed[i_c, i_b] = n_slots
                    slot = -1
                    if is_propagation:
                        if qd.static(steel_config.has_propagation):
                            slot, propagation_counter, propagation_first_rejected, propagation_n_dropped = (
                                func_reserve_rows(
                                    n_slots,
                                    propagation_counter,
                                    propagation_first_rejected,
                                    propagation_n_dropped,
                                    propagation_capacity,
                                )
                            )
                            if slot >= 0:
                                steel_contacts.path[i_c, i_b] = CONTACT_PATH.PROPAGATION
                    elif is_free:
                        if qd.static(steel_config.has_free_rigid_bodies):
                            slot, free_counter, free_first_rejected, free_n_dropped = func_reserve_rows(
                                n_slots, free_counter, free_first_rejected, free_n_dropped, free_capacity
                            )
                            if slot >= 0:
                                steel_contacts.path[i_c, i_b] = CONTACT_PATH.FREE
                                for k in range(n_slots):
                                    steel_free_rows.row_contact[slot + k, i_b] = i_c
                    else:
                        slot, counter, first_rejected, n_dropped = func_reserve_rows(
                            n_slots, counter, first_rejected, n_dropped, capacity
                        )
                        if slot >= 0:
                            steel_contacts.path[i_c, i_b] = CONTACT_PATH.DENSE
                            has_dense_contacts = 1
                            for k in range(n_slots):
                                steel_rows.row_contact[slot + k, i_b] = i_c
                    steel_contacts.slot[i_c, i_b] = slot

        steel_rows.n_requested[i_b] = counter
        steel_rows.first_rejected[i_b] = first_rejected
        steel_rows.n_dropped[i_b] = n_dropped
        steel_rows.count[i_b] = qd.min(qd.min(counter, first_rejected), capacity)
        if qd.static(steel_config.has_propagation):
            steel_prop.n_requested[i_b] = propagation_counter
            steel_prop.first_rejected[i_b] = propagation_first_rejected
            steel_prop.n_dropped[i_b] = propagation_n_dropped
            steel_prop.count[i_b] = qd.min(
                qd.min(propagation_counter, propagation_first_rejected), propagation_capacity
            )
            steel_prop.has_dense_contacts[i_b] = has_dense_contacts

        if qd.static(steel_config.has_free_rigid_bodies):
            steel_free_rows.contact_rows_end[i_b] = free_counter
            if qd.static(steel_config.has_free_velocity_limits):
                fraction = steel_info.velocity_limit_activation_fraction[None]
                for i_a in range(n_arts):
                    is_active = True
                    if qd.static(steel_config.sleep_skip_constraints):
                        is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
                    if is_active and steel_model.arts_is_free_rigid[i_a] != 0:
                        i_j = steel_model.arts_joint_start[i_a]
                        i_l = steel_model.joints_link[i_j]
                        dof_start = steel_model.arts_dof_start[i_a]
                        for axis in range(6):
                            limit = steel_info.links_max_linear_velocity[i_l, i_b]
                            if axis >= 3:
                                limit = steel_info.links_max_angular_velocity[i_l, i_b]
                            is_active = limit > 0.0 and not qd.math.isinf(limit)
                            if fraction > 0.0:
                                if qd.abs(steel_dyn.v_hat[dof_start + axis, i_b]) < fraction * limit:
                                    is_active = False
                            if is_active:
                                for side in qd.static(range(2)):
                                    slot = free_counter
                                    free_counter = free_counter + 1
                                    if slot < free_capacity:
                                        steel_free_rows.row_contact[slot, i_b] = -1
                                        steel_free_rows.row_ref[slot, i_b] = i_l
                                        steel_free_rows.row_axis[slot, i_b] = 2 * axis + side
            steel_free_rows.n_requested[i_b] = free_counter
            steel_free_rows.n_dropped[i_b] = free_n_dropped
            steel_free_rows.count[i_b] = qd.min(qd.min(free_counter, free_first_rejected), free_capacity)


@qd.func
def func_connect_anchors(
    i_k: int,
    i_b: int,
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    dyn_info: array_class.DynInfo,
    steel_model: array_class.SteelModelInfo,
):
    """Return the world anchors of loop closure i_k on its parent (A) and child (B) links."""
    i_e = steel_model.connects_eq[i_k]
    eq_data = dyn_info.equalities.eq_data[i_e, i_b]
    offset_p = steel_model.connects_anchor_p_offset[i_k]
    anchor_p = qd.Vector([eq_data[offset_p + k] for k in qd.static(range(3))], dt=gs.qd_float)
    anchor_c = qd.Vector([eq_data[3 - offset_p + k] for k in qd.static(range(3))], dt=gs.qd_float)
    i_l_p = steel_model.connects_link_p[i_k]
    i_ls_p = steel_model.connects_static_link_p[i_k]
    p_a = anchor_p
    if i_l_p >= 0:
        p_a = steel_state.links_pos[i_l_p, i_b] + gu.qd_transform_by_quat(anchor_p, steel_state.links_quat[i_l_p, i_b])
    elif i_ls_p >= 0:
        p_a = dyn_state.links.pos[i_ls_p, i_b] + gu.qd_transform_by_quat(anchor_p, dyn_state.links.quat[i_ls_p, i_b])
    i_l_c = steel_model.connects_link_c[i_k]
    p_b = steel_state.links_pos[i_l_c, i_b] + gu.qd_transform_by_quat(anchor_c, steel_state.links_quat[i_l_c, i_b])
    return p_a, p_b


@qd.func
def func_steel_build_rows(
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_dyn: array_class.SteelDynamics,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Write the Jacobian and the metadata of every reserved row.

    A contact row acts along the normal from shape 1 toward shape 0 (normal row) or along a tangent (friction rows),
    side A positive and side B negative, so J v > 0 separates. A friction row points at its normal row. A drive row
    selects its degree of freedom and carries its drive parameters, its force range turned into an impulse range by
    func_steel_responses. A limit row selects its degree of freedom, positive
    for the lower bound and negative for the upper one. A velocity-limit row asks J v >= -qdot_max. A mimic row
    enforces q - c1 q_leader - c0 = 0 on the coordinates relative to the neutral configuration. A loop-closure row is
    one world axis of the coincidence of the two anchors, normalized, inert when its Jacobian vanishes.
    """
    _B = steel_state.q.shape[1]
    capacity = steel_rows.row_type.shape[0]
    n_dofs = steel_rows.J.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(capacity, _B):
        if i_r < steel_rows.count[i_b]:
            for i_d in range(n_dofs):
                steel_rows.J[i_r, i_d, i_b] = 0.0
                steel_rows.Y[i_r, i_d, i_b] = 0.0
            i_c = steel_rows.row_contact[i_r, i_b]
            row_type = steel_rows.row_type[i_r, i_b]
            if i_c >= 0:
                p_a, p_b, n_c, phi = func_contact_points_world(i_c, i_b, steel_state, steel_contacts)
                offset = i_r - steel_contacts.slot[i_c, i_b]
                t0, t1 = func_contact_tangent_basis(n_c)
                direction = n_c
                if offset == 1:
                    direction = t0
                elif offset == 2:
                    direction = t1
                point_a, point_b = func_contact_row_points(i_c, i_b, offset > 0, p_a, p_b, steel_contacts, steel_config)
                i_l0 = steel_contacts.body0[i_c, i_b]
                i_l1 = steel_contacts.body1[i_c, i_b]
                if i_l0 >= 0:
                    func_accumulate_jacobian_row(
                        i_r, i_b, i_l0, 1.0, point_a, direction, steel_rows, steel_dyn, steel_model
                    )
                if i_l1 >= 0:
                    func_accumulate_jacobian_row(
                        i_r, i_b, i_l1, -1.0, point_b, direction, steel_rows, steel_dyn, steel_model
                    )
                steel_rows.row_ref[i_r, i_b] = -1
                steel_rows.row_mu[i_r, i_b] = func_contact_mu(i_c, i_b, steel_contacts, steel_info)
                steel_rows.target_velocity[i_r, i_b] = 0.0
                if offset == 0:
                    steel_rows.row_type[i_r, i_b] = ROW_TYPE.CONTACT
                    steel_rows.row_parent[i_r, i_b] = -1
                    steel_rows.phi[i_r, i_b] = phi
                    steel_rows.restitution[i_r, i_b] = func_contact_restitution(i_c, i_b, steel_contacts, steel_info)
                else:
                    steel_rows.row_type[i_r, i_b] = ROW_TYPE.FRICTION
                    steel_rows.row_parent[i_r, i_b] = i_r - offset
                    steel_rows.phi[i_r, i_b] = 0.0
                    if qd.static(steel_config.has_friction_anchors):
                        steel_rows.phi[i_r, i_b] = steel_contacts.patch_phi[i_c, i_b][offset - 1]
                    steel_rows.restitution[i_r, i_b] = 0.0
            elif row_type == ROW_TYPE.JOINT_TARGET:
                i_d = steel_rows.row_ref[i_r, i_b]
                I_d = [i_d, i_b] if qd.static(steel_config.batch_dofs_info) else i_d
                ke, kd, q_target, v_target = func_drive_params(i_d, i_b, dyn_state, dyn_info, rigid_info, steel_config)
                i_j = steel_model.dofs_joint[i_d]
                i_q = steel_model.joints_q_start[i_j] + (i_d - steel_model.joints_dof_start[i_j])
                steel_rows.J[i_r, i_d, i_b] = 1.0
                steel_rows.target_velocity[i_r, i_b] = v_target
                if qd.static(steel_config.drive_mode == 1):
                    steel_rows.drive_stiffness[i_r, i_b] = ke
                    steel_rows.drive_damping[i_r, i_b] = kd
                    steel_rows.drive_geom_error[i_r, i_b] = (
                        q_target - rigid_info.qpos0[i_q, i_b] - steel_state.q[i_q, i_b]
                    )
                    steel_rows.drive_impulse_range[i_r, i_b] = dyn_info.dofs.force_range[I_d]
                    vel_limit = steel_info.dofs_velocity_limit[i_d]
                    if vel_limit <= 0.0 or qd.math.isinf(vel_limit) or qd.math.isnan(vel_limit):
                        vel_limit = float("inf")
                    steel_rows.drive_vel_limit[i_r, i_b] = vel_limit
            elif row_type == ROW_TYPE.JOINT_LIMIT or row_type == ROW_TYPE.JOINT_VELOCITY_LIMIT:
                i_d = steel_rows.row_ref[i_r, i_b]
                steel_rows.J[i_r, i_d, i_b] = 1.0 if steel_rows.row_axis[i_r, i_b] == 0 else -1.0
                if row_type == ROW_TYPE.JOINT_VELOCITY_LIMIT:
                    steel_rows.target_velocity[i_r, i_b] = -steel_info.dofs_velocity_limit[i_d]
            elif row_type == ROW_TYPE.MIMIC:
                i_m = steel_rows.row_ref[i_r, i_b]
                i_e = steel_model.mimics_eq[i_m]
                i_d = steel_model.mimics_dof[i_m]
                i_d_leader = steel_model.mimics_leader_dof[i_m]
                i_q = steel_model.mimics_q[i_m]
                i_q_leader = steel_model.mimics_leader_q[i_m]
                eq_data = dyn_info.equalities.eq_data[i_e, i_b]
                c0, c1 = eq_data[0], eq_data[1]
                steel_rows.J[i_r, i_d, i_b] = 1.0
                steel_rows.J[i_r, i_d_leader, i_b] = -c1
                steel_rows.phi[i_r, i_b] = steel_state.q[i_q, i_b] - c1 * steel_state.q[i_q_leader, i_b] - c0
            elif row_type == ROW_TYPE.CONNECT:
                i_k = steel_rows.row_ref[i_r, i_b]
                axis = steel_rows.row_axis[i_r, i_b]
                p_a, p_b = func_connect_anchors(i_k, i_b, dyn_state, steel_state, dyn_info, steel_model)
                e = qd.Vector.zero(gs.qd_float, 3)
                e[axis] = 1.0
                # A static parent contributes no degree of freedom and does not move. Shared ancestors cancel.
                i_l_p = steel_model.connects_link_p[i_k]
                if i_l_p >= 0:
                    func_accumulate_jacobian_row(i_r, i_b, i_l_p, 1.0, p_a, e, steel_rows, steel_dyn, steel_model)
                func_accumulate_jacobian_row(
                    i_r, i_b, steel_model.connects_link_c[i_k], -1.0, p_b, e, steel_rows, steel_dyn, steel_model
                )
                i_a = steel_model.connects_art[i_k]
                dof_start = steel_model.arts_dof_start[i_a]
                norm_sq = gs.qd_float(0.0)
                for i_d in range(dof_start, dof_start + steel_model.arts_n_dofs[i_a]):
                    norm_sq = norm_sq + steel_rows.J[i_r, i_d, i_b] * steel_rows.J[i_r, i_d, i_b]
                phi = (p_a - p_b)[axis]
                # A nearly redundant axis of a planar linkage would relax a real velocity error against a near-zero
                # diagonal: rows are normalized, and degenerate rows left inert.
                scale = gs.qd_float(0.0)
                if norm_sq > 1.0e-8:
                    scale = 1.0 / qd.sqrt(norm_sq)
                for i_d in range(dof_start, dof_start + steel_model.arts_n_dofs[i_a]):
                    steel_rows.J[i_r, i_d, i_b] = steel_rows.J[i_r, i_d, i_b] * scale
                steel_rows.phi[i_r, i_b] = phi * scale

    if qd.static(steel_config.has_free_rigid_bodies):
        free_capacity = steel_free_rows.row_type.shape[0]
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_r, i_b in qd.ndrange(free_capacity, _B):
            if i_r < steel_free_rows.count[i_b]:
                i_c = steel_free_rows.row_contact[i_r, i_b]
                if i_r < steel_free_rows.contact_rows_end[i_b]:
                    p_a, p_b, n_c, phi = func_contact_points_world(i_c, i_b, steel_state, steel_contacts)
                    offset = i_r - steel_contacts.slot[i_c, i_b]
                    t0, t1 = func_contact_tangent_basis(n_c)
                    direction = n_c
                    if offset == 1:
                        direction = t0
                    elif offset == 2:
                        direction = t1
                    point_a, point_b = func_contact_row_points(
                        i_c, i_b, offset > 0, p_a, p_b, steel_contacts, steel_config
                    )
                    for side in qd.static(range(2)):
                        i_l = steel_contacts.body0[i_c, i_b] if qd.static(side == 0) else steel_contacts.body1[i_c, i_b]
                        point = point_a if qd.static(side == 0) else point_b
                        sign = 1.0 if qd.static(side == 0) else -1.0
                        body = -1
                        dof = -1
                        J_lin = qd.Vector.zero(gs.qd_float, 3)
                        J_ang = qd.Vector.zero(gs.qd_float, 3)
                        if i_l >= 0:
                            i_a = steel_model.links_art[i_l]
                            body = i_l
                            dof = steel_model.arts_dof_start[i_a]
                            J_lin = sign * direction
                            J_ang = sign * (point - steel_dyn.arts_origin[i_a, i_b]).cross(direction)
                        if qd.static(side == 0):
                            steel_free_rows.body_a[i_r, i_b] = body
                            steel_free_rows.dof_a[i_r, i_b] = dof
                            steel_free_rows.J_a_lin[i_r, i_b] = J_lin
                            steel_free_rows.J_a_ang[i_r, i_b] = J_ang
                        else:
                            steel_free_rows.body_b[i_r, i_b] = body
                            steel_free_rows.dof_b[i_r, i_b] = dof
                            steel_free_rows.J_b_lin[i_r, i_b] = J_lin
                            steel_free_rows.J_b_ang[i_r, i_b] = J_ang
                    steel_free_rows.row_mu[i_r, i_b] = func_contact_mu(i_c, i_b, steel_contacts, steel_info)
                    steel_free_rows.target_velocity[i_r, i_b] = 0.0
                    if offset == 0:
                        steel_free_rows.row_type[i_r, i_b] = ROW_TYPE.CONTACT
                        steel_free_rows.row_parent[i_r, i_b] = -1
                        steel_free_rows.phi[i_r, i_b] = phi
                        steel_free_rows.restitution[i_r, i_b] = func_contact_restitution(
                            i_c, i_b, steel_contacts, steel_info
                        )
                    else:
                        steel_free_rows.row_type[i_r, i_b] = ROW_TYPE.FRICTION
                        steel_free_rows.row_parent[i_r, i_b] = i_r - offset
                        # The phi of a free-body friction row holds the anchor displacement scaled by its gain.
                        steel_free_rows.phi[i_r, i_b] = 0.0
                        if qd.static(steel_config.has_friction_anchors):
                            steel_free_rows.phi[i_r, i_b] = (
                                steel_info.friction_anchor_beta[None] * steel_contacts.patch_phi[i_c, i_b][offset - 1]
                            )
                        steel_free_rows.restitution[i_r, i_b] = 0.0
                else:
                    i_l = steel_free_rows.row_ref[i_r, i_b]
                    axis = steel_free_rows.row_axis[i_r, i_b] // 2
                    sign = 1.0 if steel_free_rows.row_axis[i_r, i_b] % 2 == 0 else -1.0
                    J_lin = qd.Vector.zero(gs.qd_float, 3)
                    J_ang = qd.Vector.zero(gs.qd_float, 3)
                    limit = steel_info.links_max_linear_velocity[i_l, i_b]
                    if axis < 3:
                        J_lin[axis] = sign
                    else:
                        J_ang[axis - 3] = sign
                        limit = steel_info.links_max_angular_velocity[i_l, i_b]
                    steel_free_rows.body_a[i_r, i_b] = i_l
                    steel_free_rows.body_b[i_r, i_b] = -1
                    i_a = steel_model.links_art[i_l]
                    steel_free_rows.dof_a[i_r, i_b] = steel_model.arts_dof_start[i_a]
                    steel_free_rows.dof_b[i_r, i_b] = -1
                    steel_free_rows.J_a_lin[i_r, i_b] = J_lin
                    steel_free_rows.J_a_ang[i_r, i_b] = J_ang
                    steel_free_rows.J_b_lin[i_r, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    steel_free_rows.J_b_ang[i_r, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    steel_free_rows.row_type[i_r, i_b] = ROW_TYPE.JOINT_VELOCITY_LIMIT
                    steel_free_rows.row_parent[i_r, i_b] = -1
                    steel_free_rows.row_mu[i_r, i_b] = 0.0
                    steel_free_rows.target_velocity[i_r, i_b] = 0.0
                    steel_free_rows.restitution[i_r, i_b] = 0.0
                    # The phi of a velocity-limit row holds its bound.
                    steel_free_rows.phi[i_r, i_b] = limit


@qd.func
def func_solve_articulation_block(
    i_r: int,
    i_b: int,
    i_a: int,
    steel_rows: array_class.SteelRows,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
):
    """Write Y = H^-1 J^T of dense row i_r over the DOFs of articulation i_a, by forward and backward substitution."""
    n = steel_model.arts_n_dofs[i_a]
    i_H = steel_model.arts_H_offset[i_a]
    dof_start = steel_model.arts_dof_start[i_a]
    for i in range(n):
        v = steel_rows.J[i_r, dof_start + i, i_b]
        for k in range(i):
            v = v - steel_dyn.L[i_H + i * n + k, i_b] * steel_rows.Y[i_r, dof_start + k, i_b]
        L_ii = steel_dyn.L[i_H + i * n + i, i_b]
        steel_rows.Y[i_r, dof_start + i, i_b] = v / L_ii if L_ii != 0.0 else 0.0
    for i_ in range(n):
        i = n - 1 - i_
        v = steel_rows.Y[i_r, dof_start + i, i_b]
        for k in range(i + 1, n):
            v = v - steel_dyn.L[i_H + k * n + i, i_b] * steel_rows.Y[i_r, dof_start + k, i_b]
        L_ii = steel_dyn.L[i_H + i * n + i, i_b]
        steel_rows.Y[i_r, dof_start + i, i_b] = v / L_ii if L_ii != 0.0 else 0.0


@qd.func
def func_row_articulations(
    i_r: int,
    i_b: int,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_model: array_class.SteelModelInfo,
):
    """Return the (at most two, distinct) articulations dense row i_r acts on, -1 for none."""
    i_a0 = -1
    i_a1 = -1
    i_c = steel_rows.row_contact[i_r, i_b]
    row_type = steel_rows.row_type[i_r, i_b]
    if i_c >= 0:
        i_l0 = steel_contacts.body0[i_c, i_b]
        i_l1 = steel_contacts.body1[i_c, i_b]
        if i_l0 >= 0:
            i_a0 = steel_model.links_art[i_l0]
        if i_l1 >= 0:
            i_a1 = steel_model.links_art[i_l1]
        if i_a1 == i_a0:
            i_a1 = -1
    else:
        i_ref = steel_rows.row_ref[i_r, i_b]
        if row_type == ROW_TYPE.MIMIC:
            i_a0 = steel_model.mimics_art[i_ref]
        elif row_type == ROW_TYPE.CONNECT:
            i_a0 = steel_model.connects_art[i_ref]
        else:
            i_j = steel_model.dofs_joint[i_ref]
            i_a0 = steel_model.joints_art[i_j]
    return i_a0, i_a1


@qd.func
def func_dense_row_dot(i_r: int, i_b: int, velocity: qd.Tensor, steel_rows: array_class.SteelRows):
    """Return J v of dense row i_r for the generalized velocity in the given tensor."""
    jv = gs.qd_float(0.0)
    for i_d in range(steel_rows.J.shape[1]):
        jv = jv + steel_rows.J[i_r, i_d, i_b] * velocity[i_d, i_b]
    return jv


@qd.func
def func_free_row_dot(i_r: int, i_b: int, velocity: qd.Tensor, steel_free_rows: array_class.SteelFreeRows):
    """Return J v of free-body row i_r for the generalized velocity in the given tensor."""
    jv = gs.qd_float(0.0)
    for side in qd.static(range(2)):
        dof = steel_free_rows.dof_a[i_r, i_b] if qd.static(side == 0) else steel_free_rows.dof_b[i_r, i_b]
        if dof >= 0:
            J_lin = steel_free_rows.J_a_lin[i_r, i_b] if qd.static(side == 0) else steel_free_rows.J_b_lin[i_r, i_b]
            J_ang = steel_free_rows.J_a_ang[i_r, i_b] if qd.static(side == 0) else steel_free_rows.J_b_ang[i_r, i_b]
            for k in qd.static(range(3)):
                jv = jv + J_lin[k] * velocity[dof + k, i_b] + J_ang[k] * velocity[dof + 3 + k, i_b]
    return jv


@qd.func
def func_steel_responses(
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Compute the response Y = H^-1 J^T of every dense row over the articulations it acts on (stage 4)."""
    _B = steel_rows.count.shape[0]
    capacity = steel_rows.row_type.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(capacity, _B):
        if i_r < steel_rows.count[i_b]:
            i_a0, i_a1 = func_row_articulations(i_r, i_b, steel_contacts, steel_rows, steel_model)
            for i_a_ in qd.static(range(2)):
                i_a = i_a0 if qd.static(i_a_ == 0) else i_a1
                if i_a >= 0:
                    func_solve_articulation_block(i_r, i_b, i_a, steel_rows, steel_dyn, steel_model)


@qd.func
def func_steel_row_setup(
    dt: float,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_dyn: array_class.SteelDynamics,
    dyn_info: array_class.DynInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Compute the diagonals and right-hand sides of every row (stage 4), and zero their impulses.

    Dense rows: D = J . Y + cfm, a drive row keeping its exact unit response for its force-drive update
    lambda = clamp((1 - x) lambda - x a J v + x dt kd v* + x dt ke (q* - q)) with a = dt (dt ke + kd) and
    x = 1 / (1 + a D). The bias of a penetrating contact, a violated limit, a mimic or a loop closure removes baumgarte
    of the error per step, a separated contact may close contact_speculative_scale of its gap within the step, an
    unreached limit its full gap, a friction anchor friction_anchor_beta of its displacement. An impacting contact
    replaces its bias by the rebound target e * u, u being its incident normal velocity in the unconstrained
    prediction. Penetrating contacts that do not rebound take the regularization weight.

    Free-body rows: the response uses the inverse spatial inertia of each body (with the armature of its free joint),
    and the bias of a penetrating contact is bounded by the depenetration velocity of its bodies.
    """
    _B = steel_state.q.shape[1]
    capacity = steel_rows.row_type.shape[0]
    inv_dt = 1.0 / dt
    baumgarte = steel_info.baumgarte[None]
    contact_w = steel_info.contact_w[None]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(capacity, _B):
        if i_r < steel_rows.count[i_b]:
            i_a0, i_a1 = func_row_articulations(i_r, i_b, steel_contacts, steel_rows, steel_model)
            diag = gs.qd_float(0.0)
            for i_a_ in qd.static(range(2)):
                i_a = i_a0 if qd.static(i_a_ == 0) else i_a1
                if i_a >= 0:
                    dof_start = steel_model.arts_dof_start[i_a]
                    for i_d in range(dof_start, dof_start + steel_model.arts_n_dofs[i_a]):
                        diag = diag + steel_rows.J[i_r, i_d, i_b] * steel_rows.Y[i_r, i_d, i_b]
            row_type = steel_rows.row_type[i_r, i_b]
            if row_type != ROW_TYPE.JOINT_TARGET:
                diag = diag + steel_info.cfm[None]
            steel_rows.diag[i_r, i_b] = diag

            if qd.static(steel_config.drive_mode == 1):
                if row_type == ROW_TYPE.JOINT_TARGET:
                    ke = steel_rows.drive_stiffness[i_r, i_b]
                    kd = steel_rows.drive_damping[i_r, i_b]
                    a = dt * (dt * ke + kd)
                    x = 1.0 / (1.0 + a * diag) if diag > 0.0 else 0.0
                    steel_rows.drive_target_vel_bias[i_r, i_b] = (
                        x * dt * (kd * steel_rows.target_velocity[i_r, i_b])
                        + ke * x * dt * steel_rows.drive_geom_error[i_r, i_b]
                    )
                    steel_rows.drive_vel_multiplier[i_r, i_b] = -x * a
                    steel_rows.drive_impulse_multiplier[i_r, i_b] = 1.0 - x
                    force_range = steel_rows.drive_impulse_range[i_r, i_b]
                    steel_rows.drive_impulse_range[i_r, i_b] = qd.Vector(
                        [
                            qd.max(force_range[0] * dt, -steel_info.drive_unbounded_impulse[None]),
                            qd.min(force_range[1] * dt, steel_info.drive_unbounded_impulse[None]),
                        ],
                        dt=gs.qd_float,
                    )

            phi = steel_rows.phi[i_r, i_b]
            target = steel_rows.target_velocity[i_r, i_b]
            rhs = -target
            w = gs.qd_float(1.0)
            if row_type == ROW_TYPE.CONTACT:
                if phi <= 0.0:
                    rhs = rhs + baumgarte * phi * inv_dt
                    w = contact_w
                else:
                    rhs = rhs + steel_info.contact_speculative_scale[None] * phi * inv_dt
                e = steel_rows.restitution[i_r, i_b]
                if e > 0.0:
                    u = func_dense_row_dot(i_r, i_b, steel_dyn.v_hat, steel_rows) - target
                    if func_restitution_fires(phi, u, dt, steel_info):
                        rhs = -target + e * u
                        w = 1.0
            elif row_type == ROW_TYPE.FRICTION:
                rhs = rhs + steel_info.friction_anchor_beta[None] * phi * inv_dt
            elif row_type == ROW_TYPE.JOINT_LIMIT:
                if phi < 0.0:
                    rhs = rhs + baumgarte * phi * inv_dt
                else:
                    rhs = rhs + phi * inv_dt
            elif row_type == ROW_TYPE.MIMIC or row_type == ROW_TYPE.CONNECT:
                rhs = rhs + baumgarte * phi * inv_dt
            steel_rows.rhs[i_r, i_b] = rhs
            steel_rows.row_w[i_r, i_b] = w
            steel_rows.impulses[i_r, i_b] = 0.0

    if qd.static(steel_config.has_free_rigid_bodies):
        n_arts = steel_model.arts_dof_start.shape[0]
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_a, i_b in qd.ndrange(n_arts, _B):
            if steel_model.arts_is_free_rigid[i_a] != 0:
                i_j = steel_model.arts_joint_start[i_a]
                i_l = steel_model.joints_link[i_j]
                dof_start = steel_model.arts_dof_start[i_a]
                arm_lin = qd.Vector.zero(gs.qd_float, 3)
                arm_ang = qd.Vector.zero(gs.qd_float, 3)
                for k in qd.static(range(3)):
                    I_d = [dof_start + k, i_b] if qd.static(steel_config.batch_dofs_info) else dof_start + k
                    I_da = [dof_start + 3 + k, i_b] if qd.static(steel_config.batch_dofs_info) else dof_start + 3 + k
                    arm_lin[k] = dyn_info.dofs.armature[I_d]
                    arm_ang[k] = dyn_info.dofs.armature[I_da]
                # Inverse of the spatial inertia [[m I, -[h]x], [[h]x, I_o]] plus armature by 3x3 block inversion.
                h_cross = func_skew(steel_dyn.links_I_h[i_l, i_b])
                A = steel_dyn.links_I_mass[i_l, i_b] * qd.Matrix.identity(gs.qd_float, 3) + func_diag(arm_lin)
                B = -h_cross
                C = h_cross
                D = steel_dyn.links_I_o[i_l, i_b] + func_diag(arm_ang)
                A_inv = A.inverse()
                S_inv = (D - C @ A_inv @ B).inverse()
                steel_free_rows.links_Hinv_tl[i_l, i_b] = A_inv + A_inv @ B @ S_inv @ C @ A_inv
                steel_free_rows.links_Hinv_tr[i_l, i_b] = -A_inv @ B @ S_inv
                steel_free_rows.links_Hinv_bl[i_l, i_b] = -S_inv @ C @ A_inv
                steel_free_rows.links_Hinv_br[i_l, i_b] = S_inv

        free_capacity = steel_free_rows.row_type.shape[0]
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_r, i_b in qd.ndrange(free_capacity, _B):
            if i_r < steel_free_rows.count[i_b]:
                diag = steel_info.cfm[None]
                for side in qd.static(range(2)):
                    i_l = steel_free_rows.body_a[i_r, i_b] if qd.static(side == 0) else steel_free_rows.body_b[i_r, i_b]
                    if i_l >= 0:
                        J_lin = (
                            steel_free_rows.J_a_lin[i_r, i_b]
                            if qd.static(side == 0)
                            else steel_free_rows.J_b_lin[i_r, i_b]
                        )
                        J_ang = (
                            steel_free_rows.J_a_ang[i_r, i_b]
                            if qd.static(side == 0)
                            else steel_free_rows.J_b_ang[i_r, i_b]
                        )
                        M_lin = (
                            steel_free_rows.links_Hinv_tl[i_l, i_b] @ J_lin
                            + steel_free_rows.links_Hinv_tr[i_l, i_b] @ J_ang
                        )
                        M_ang = (
                            steel_free_rows.links_Hinv_bl[i_l, i_b] @ J_lin
                            + steel_free_rows.links_Hinv_br[i_l, i_b] @ J_ang
                        )
                        if qd.static(side == 0):
                            steel_free_rows.MiJt_a_lin[i_r, i_b] = M_lin
                            steel_free_rows.MiJt_a_ang[i_r, i_b] = M_ang
                        else:
                            steel_free_rows.MiJt_b_lin[i_r, i_b] = M_lin
                            steel_free_rows.MiJt_b_ang[i_r, i_b] = M_ang
                        diag = diag + J_lin.dot(M_lin) + J_ang.dot(M_ang)
                steel_free_rows.eff_mass_inv[i_r, i_b] = 1.0 / diag if diag > 0.0 else 0.0

                row_type = steel_free_rows.row_type[i_r, i_b]
                phi = steel_free_rows.phi[i_r, i_b]
                bias = gs.qd_float(0.0)
                w = gs.qd_float(1.0)
                if row_type == ROW_TYPE.CONTACT:
                    if phi <= 0.0:
                        w = contact_w
                    if phi < 0.0:
                        bias = baumgarte * phi * inv_dt
                        max_depenetration = steel_info.unbounded_depenetration[None]
                        i_la = steel_free_rows.body_a[i_r, i_b]
                        i_lb = steel_free_rows.body_b[i_r, i_b]
                        if i_la >= 0:
                            max_depenetration = steel_info.links_max_depenetration_velocity[i_la, i_b]
                        if i_lb >= 0:
                            v_b = steel_info.links_max_depenetration_velocity[i_lb, i_b]
                            if v_b > 0.0 and not qd.math.isinf(v_b) and v_b < max_depenetration:
                                max_depenetration = v_b
                        if max_depenetration > 0.0 and not qd.math.isinf(max_depenetration):
                            bias = qd.max(bias, -max_depenetration)
                    else:
                        bias = steel_info.contact_speculative_scale[None] * phi * inv_dt
                    e = steel_free_rows.restitution[i_r, i_b]
                    if e > 0.0:
                        u = func_free_row_dot(i_r, i_b, steel_dyn.v_hat, steel_free_rows)
                        if func_restitution_fires(phi, u, dt, steel_info):
                            bias = e * u
                            w = 1.0
                elif row_type == ROW_TYPE.FRICTION:
                    bias = phi / dt
                elif row_type == ROW_TYPE.JOINT_VELOCITY_LIMIT:
                    bias = phi
                steel_free_rows.rhs[i_r, i_b] = bias
                steel_free_rows.row_w[i_r, i_b] = w
                steel_free_rows.impulses[i_r, i_b] = 0.0


@qd.func
def func_steel_velocity_pass_rhs(
    dt: float,
    steel_rows: array_class.SteelRows,
    steel_free_rows: array_class.SteelFreeRows,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Compute the right-hand sides of the velocity-only sweeps, which solve the rows without position correction.

    An impacting contact keeps its rebound target. A separated contact whose linearized gap at the end of the step,
    with the velocity of the position solve, stays open keeps its full gap allowance phi / dt, and so does an unreached
    joint limit. Velocity limits keep their bound. Every other row asks for its target velocity alone.
    """
    _B = steel_rows.count.shape[0]
    capacity = steel_rows.row_type.shape[0]
    inv_dt = 1.0 / dt
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(capacity, _B):
        if i_r < steel_rows.count[i_b]:
            row_type = steel_rows.row_type[i_r, i_b]
            target = steel_rows.target_velocity[i_r, i_b]
            phi = steel_rows.phi[i_r, i_b]
            rhs = -target
            if row_type == ROW_TYPE.CONTACT:
                e = steel_rows.restitution[i_r, i_b]
                is_rebound = False
                if e > 0.0:
                    u = func_dense_row_dot(i_r, i_b, steel_dyn.v_hat, steel_rows) - target
                    if func_restitution_fires(phi, u, dt, steel_info):
                        rhs = rhs + e * u
                        is_rebound = True
                if not is_rebound and phi > 0.0:
                    end_gap = phi + dt * (func_dense_row_dot(i_r, i_b, steel_dyn.v_out_snap, steel_rows) - target)
                    if end_gap > steel_info.contact_end_gap_slop[None]:
                        rhs = rhs + phi * inv_dt
            elif row_type == ROW_TYPE.JOINT_LIMIT:
                if phi >= 0.0:
                    rhs = rhs + phi * inv_dt
            steel_rows.rhs_unbiased[i_r, i_b] = rhs

    if qd.static(steel_config.has_free_rigid_bodies):
        free_capacity = steel_free_rows.row_type.shape[0]
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_r, i_b in qd.ndrange(free_capacity, _B):
            if i_r < steel_free_rows.count[i_b]:
                row_type = steel_free_rows.row_type[i_r, i_b]
                target = steel_free_rows.target_velocity[i_r, i_b]
                phi = steel_free_rows.phi[i_r, i_b]
                bias = gs.qd_float(0.0)
                if row_type == ROW_TYPE.CONTACT:
                    e = steel_free_rows.restitution[i_r, i_b]
                    is_rebound = False
                    if e > 0.0:
                        u = func_free_row_dot(i_r, i_b, steel_dyn.v_hat, steel_free_rows) - target
                        if func_restitution_fires(phi, u, dt, steel_info):
                            bias = e * u
                            is_rebound = True
                    if not is_rebound and phi > 0.0:
                        end_gap = phi + dt * (
                            func_free_row_dot(i_r, i_b, steel_dyn.v_out_snap, steel_free_rows) - target
                        )
                        if end_gap > steel_info.contact_end_gap_slop[None]:
                            bias = phi * inv_dt
                elif row_type == ROW_TYPE.JOINT_VELOCITY_LIMIT:
                    bias = phi
                steel_free_rows.rhs_unbiased[i_r, i_b] = bias - target


@qd.func
def func_preelim_solve(
    i_pe: int,
    i_b: int,
    n: int,
    b: qd.types.vector(PREELIM_MAX_ROWS),
    steel_rows: array_class.SteelRows,
    steel_config: qd.template(),
):
    """Solve (S + diag(reg)) x = b for the bilateral block of pre-elimination index i_pe, by its Cholesky factor."""
    i_S = i_pe * steel_config.preelim_max_rows * steel_config.preelim_max_rows
    x = qd.Vector.zero(gs.qd_float, steel_config.preelim_max_rows)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s = s - steel_rows.preelim_L[i_S + i * n + j, i_b] * x[j]
        x[i] = s / steel_rows.preelim_L[i_S + i * n + i, i_b]
    for i_ in range(n):
        i = n - 1 - i_
        s = x[i]
        for j in range(i + 1, n):
            s = s - steel_rows.preelim_L[i_S + j * n + i, i_b] * x[j]
        x[i] = s / steel_rows.preelim_L[i_S + i * n + i, i_b]
    return x


@qd.func
def func_steel_preelim(
    reg_rel: float,
    reg_floor: float,
    steel_contacts: array_class.SteelContacts,
    steel_rows: array_class.SteelRows,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Pre-eliminate the bilateral rows (mimic joints and loop closures) of every articulation.

    The block of the bilateral rows B of an articulation is S = J_B Y_B, regularized by reg = reg_rel S_pp + reg_floor
    on its diagonal, which keeps the nearly dependent axis of a planar closure positive definite, and factored. Every
    other row of the articulation then takes the response Y_i - Y_B (S + R)^-1 (J_B Y_i), so the sweep impulses nearly
    preserve the bilateral rows. Zero rows (inert closure axes) are left out of the block.
    """
    _B = steel_rows.count.shape[0]
    n_arts = steel_model.arts_dof_start.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        i_pe = steel_model.arts_preelim_idx[i_a]
        if i_pe >= 0:
            base = i_pe * steel_config.preelim_max_rows
            n = 0
            if qd.static(steel_config.has_preelim_mimics):
                for i_m in range(steel_model.mimics_eq.shape[0]):
                    if (
                        steel_model.mimics_art[i_m] == i_a
                        and steel_rows.mimics_slot[i_m, i_b] >= 0
                        and n < steel_config.preelim_max_rows
                    ):
                        steel_rows.preelim_slots[base + n, i_b] = steel_rows.mimics_slot[i_m, i_b]
                        n = n + 1
            if qd.static(steel_config.has_connects):
                for i_k in range(steel_model.connects_eq.shape[0]):
                    if steel_model.connects_art[i_k] == i_a and steel_rows.connects_slot[i_k, i_b] >= 0:
                        for axis in range(3):
                            if n < steel_config.preelim_max_rows:
                                steel_rows.preelim_slots[base + n, i_b] = steel_rows.connects_slot[i_k, i_b] + axis
                                n = n + 1
            dof_start = steel_model.arts_dof_start[i_a]
            dof_end = dof_start + steel_model.arts_n_dofs[i_a]
            m = 0
            for p in range(n):
                i_s = steel_rows.preelim_slots[base + p, i_b]
                norm_sq = gs.qd_float(0.0)
                for i_d in range(dof_start, dof_end):
                    norm_sq = norm_sq + steel_rows.J[i_s, i_d, i_b] * steel_rows.J[i_s, i_d, i_b]
                if norm_sq > 1.0e-10:
                    steel_rows.preelim_slots[base + m, i_b] = i_s
                    m = m + 1
            for p in range(m, steel_config.preelim_max_rows):
                steel_rows.preelim_slots[base + p, i_b] = -1
            steel_rows.preelim_n[i_pe, i_b] = m

            i_S = i_pe * steel_config.preelim_max_rows * steel_config.preelim_max_rows
            for p in range(m):
                i_sp = steel_rows.preelim_slots[base + p, i_b]
                for q in range(m):
                    i_sq = steel_rows.preelim_slots[base + q, i_b]
                    acc = gs.qd_float(0.0)
                    for i_d in range(dof_start, dof_end):
                        acc = acc + steel_rows.J[i_sp, i_d, i_b] * steel_rows.Y[i_sq, i_d, i_b]
                    steel_rows.preelim_S[i_S + p * m + q, i_b] = acc
            for p in range(m):
                steel_rows.preelim_reg[base + p, i_b] = reg_rel * steel_rows.preelim_S[i_S + p * m + p, i_b] + reg_floor
            for j in range(m):
                s = steel_rows.preelim_S[i_S + j * m + j, i_b] + steel_rows.preelim_reg[base + j, i_b]
                for k in range(j):
                    r = steel_rows.preelim_L[i_S + j * m + k, i_b]
                    s = s - r * r
                s = qd.sqrt(s)
                inv_s = 1.0 / s
                steel_rows.preelim_L[i_S + j * m + j, i_b] = s
                for i in range(j + 1, m):
                    s_ij = steel_rows.preelim_S[i_S + i * m + j, i_b]
                    for k in range(j):
                        s_ij = (
                            s_ij
                            - steel_rows.preelim_L[i_S + i * m + k, i_b] * steel_rows.preelim_L[i_S + j * m + k, i_b]
                        )
                    steel_rows.preelim_L[i_S + i * m + j, i_b] = s_ij * inv_s

    capacity = steel_rows.row_type.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(capacity, _B):
        if i_r < steel_rows.count[i_b]:
            i_a0, i_a1 = func_row_articulations(i_r, i_b, steel_contacts, steel_rows, steel_model)
            for i_a_ in qd.static(range(2)):
                i_a = i_a0 if qd.static(i_a_ == 0) else i_a1
                i_pe = -1
                if i_a >= 0:
                    i_pe = steel_model.arts_preelim_idx[i_a]
                if i_pe >= 0:
                    base = i_pe * steel_config.preelim_max_rows
                    m = steel_rows.preelim_n[i_pe, i_b]
                    is_block = False
                    for p in range(m):
                        if steel_rows.preelim_slots[base + p, i_b] == i_r:
                            is_block = True
                    if m > 0 and not is_block:
                        dof_start = steel_model.arts_dof_start[i_a]
                        dof_end = dof_start + steel_model.arts_n_dofs[i_a]
                        w = qd.Vector.zero(gs.qd_float, steel_config.preelim_max_rows)
                        is_coupled = False
                        for p in range(m):
                            i_sp = steel_rows.preelim_slots[base + p, i_b]
                            acc = gs.qd_float(0.0)
                            for i_d in range(dof_start, dof_end):
                                acc = acc + steel_rows.J[i_sp, i_d, i_b] * steel_rows.Y[i_r, i_d, i_b]
                            w[p] = acc
                            if acc != 0.0:
                                is_coupled = True
                        if is_coupled:
                            z = func_preelim_solve(i_pe, i_b, m, w, steel_rows, steel_config)
                            for i_d in range(dof_start, dof_end):
                                acc = gs.qd_float(0.0)
                                for p in range(m):
                                    i_sp = steel_rows.preelim_slots[base + p, i_b]
                                    acc = acc + steel_rows.Y[i_sp, i_d, i_b] * z[p]
                                steel_rows.Y[i_r, i_d, i_b] = steel_rows.Y[i_r, i_d, i_b] - acc


@qd.func
def func_steel_preelim_project_velocity(
    steel_rows: array_class.SteelRows,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Project the solve velocity once on the bilateral rows: v -= Y_B (S + R)^-1 (J_B v + b_B).

    The regularization R makes the projection inexact: a residual R (S + R)^-1 (J_B v + b_B) remains, which the
    bilateral rows left in the sweep reduce further.
    """
    _B = steel_rows.count.shape[0]
    n_arts = steel_model.arts_dof_start.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        i_pe = steel_model.arts_preelim_idx[i_a]
        if i_pe >= 0:
            m = steel_rows.preelim_n[i_pe, i_b]
            if m > 0:
                base = i_pe * steel_config.preelim_max_rows
                dof_start = steel_model.arts_dof_start[i_a]
                dof_end = dof_start + steel_model.arts_n_dofs[i_a]
                r = qd.Vector.zero(gs.qd_float, steel_config.preelim_max_rows)
                for p in range(m):
                    i_sp = steel_rows.preelim_slots[base + p, i_b]
                    acc = steel_rows.rhs[i_sp, i_b]
                    for i_d in range(dof_start, dof_end):
                        acc = acc + steel_rows.J[i_sp, i_d, i_b] * steel_dyn.v_out[i_d, i_b]
                    r[p] = acc
                z = func_preelim_solve(i_pe, i_b, m, r, steel_rows, steel_config)
                for i_d in range(dof_start, dof_end):
                    acc = gs.qd_float(0.0)
                    for p in range(m):
                        i_sp = steel_rows.preelim_slots[base + p, i_b]
                        acc = acc + steel_rows.Y[i_sp, i_d, i_b] * z[p]
                    if acc != 0.0:
                        steel_dyn.v_out[i_d, i_b] = steel_dyn.v_out[i_d, i_b] - acc
