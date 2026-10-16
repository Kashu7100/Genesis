"""Propagation articulated contact responses of the Steel solver ('articulated_contact_response' other than
'immediate').

A propagation row acts on at most two links through 6-vector Jacobians at their centers of mass (COM), instead of the
dense response Y = H^-1 J^T over the degrees of freedom. Every link of an articulation carries its own 6x6 COM response
Phi = J_l H^-1 J_l^T, from an articulated-body factorization of its tree, and a free rigid body its inverse spatial
inertia. A Gauss-Seidel sweep solves the rows with these responses alone, updating a twist estimate of every touched
link and accumulating the impulse wrenches it receives. The cross response between two links of one tree is ignored
and compensated by mass splitting: the response of a link is scaled by the number of row-bearing links of its coupling
group. After every sweep, the accumulated wrenches are pushed exactly through every tree (articulated-body backward and
forward passes) into the joint velocities, and the link twists are recomputed exactly from them.

Spatial vectors are world-aligned (linear, angular) 6-vectors: a twist is the velocity of the COM and the angular
velocity, a wrench the force and the torque about the COM. The edge e of a link is the offset of its COM from the COM
of its parent link. A parent twist moves to the child COM as (v + w x e, w), a child wrench to the parent COM as
(f, tau + e x f).
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu

from .dynamics import func_body_inertia
from .pgs import func_friction_pair_candidate, func_pooled_normal_impulse
from .rows import (
    CONTACT_PATH,
    ROW_TYPE,
    func_contact_mu,
    func_contact_points_world,
    func_contact_restitution,
    func_contact_row_points,
    func_contact_tangent_basis,
    func_restitution_fires,
)


@qd.func
def func_lin(v: qd.types.vector(6)):
    return qd.Vector([v[0], v[1], v[2]], dt=gs.qd_float)


@qd.func
def func_ang(v: qd.types.vector(6)):
    return qd.Vector([v[3], v[4], v[5]], dt=gs.qd_float)


@qd.func
def func_vec6(lin: qd.types.vector(3), ang: qd.types.vector(3)):
    return qd.Vector([lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]], dt=gs.qd_float)


@qd.func
def func_translate_twist(v: qd.types.vector(6), e: qd.types.vector(3)):
    """Move a twist from a point to the point displaced by e."""
    ang = func_ang(v)
    return func_vec6(func_lin(v) + ang.cross(e), ang)


@qd.func
def func_translate_wrench(f: qd.types.vector(6), e: qd.types.vector(3)):
    """Move a wrench from the point displaced by e back to the point."""
    force = func_lin(f)
    return func_vec6(force, func_ang(f) + e.cross(force))


@qd.func
def func_link_edge(i_l: int, i_p: int, i_b: int, steel_prop: array_class.SteelPropagation):
    """Return the offset of the COM of link i_l from the COM of its parent link i_p."""
    return steel_prop.links_com_rel[i_l, i_b] - steel_prop.links_com_rel[i_p, i_b]


@qd.func
def func_tree_backward(
    i_a: int, i_b: int, steel_prop: array_class.SteelPropagation, steel_model: array_class.SteelModelInfo
):
    """Run the leaf-to-root pass of the impulses -links_pA of articulation i_a, writing the joint terms dofs_u.

    Every joint takes u = -S^T pA of its link and transmits pA + U D^-1 u to its parent link.
    """
    i_j_start = steel_model.arts_joint_start[i_a]
    i_j_end = steel_model.arts_joint_start[i_a + 1]
    for i_j_ in range(i_j_end - i_j_start):
        i_j = i_j_end - 1 - i_j_
        i_l = steel_model.joints_link[i_j]
        i_p = steel_model.joints_parent_link[i_j]
        i_d_start = steel_model.joints_dof_start[i_j]
        n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
        pA = steel_prop.links_pA[i_l, i_b]
        for i_d in range(i_d_start, i_d_start + n_dofs):
            steel_prop.dofs_u[i_d, i_b] = -steel_prop.dofs_S[i_d, i_b].dot(pA)
        if i_p >= 0:
            D_inv = steel_prop.links_D_inv[i_l, i_b]
            for a in range(n_dofs):
                coeff = gs.qd_float(0.0)
                for b in range(n_dofs):
                    coeff = coeff + D_inv[a, b] * steel_prop.dofs_u[i_d_start + b, i_b]
                pA = pA + steel_prop.dofs_U[i_d_start + a, i_b] * coeff
            pA_parent = func_translate_wrench(pA, func_link_edge(i_l, i_p, i_b, steel_prop))
            steel_prop.links_pA[i_p, i_b] = steel_prop.links_pA[i_p, i_b] + pA_parent


@qd.func
def func_tree_forward(
    i_a: int,
    i_b: int,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
    update_velocity: qd.template(),
):
    """Run the root-to-leaf pass of articulation i_a after func_tree_backward, writing the link twist changes.

    Every joint takes qdd = D^-1 (u - U^T pd), pd being the twist change of its parent moved to its link, and its link
    the twist change pd + S qdd. With update_velocity, qdd is added to the joint velocity v_out.
    """
    for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
        i_l = steel_model.joints_link[i_j]
        i_p = steel_model.joints_parent_link[i_j]
        i_d_start = steel_model.joints_dof_start[i_j]
        n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
        pd = qd.Vector.zero(gs.qd_float, 6)
        if i_p >= 0:
            pd = func_translate_twist(steel_prop.links_delta[i_p, i_b], func_link_edge(i_l, i_p, i_b, steel_prop))
        delta = pd
        D_inv = steel_prop.links_D_inv[i_l, i_b]
        for a in range(n_dofs):
            qdd = gs.qd_float(0.0)
            for b in range(n_dofs):
                i_d = i_d_start + b
                qdd = qdd + D_inv[a, b] * (steel_prop.dofs_u[i_d, i_b] - steel_prop.dofs_U[i_d, i_b].dot(pd))
            delta = delta + steel_prop.dofs_S[i_d_start + a, i_b] * qdd
            if qd.static(update_velocity):
                steel_dyn.v_out[i_d_start + a, i_b] = steel_dyn.v_out[i_d_start + a, i_b] + qdd
        steel_prop.links_delta[i_l, i_b] = delta


@qd.func
def func_tree_twists(
    i_a: int,
    i_b: int,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
    clear_impulse: qd.template(),
):
    """Recompute the COM twists of the links of articulation i_a from the joint velocity v_out, root to leaves."""
    for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
        i_l = steel_model.joints_link[i_j]
        i_p = steel_model.joints_parent_link[i_j]
        twist = qd.Vector.zero(gs.qd_float, 6)
        if i_p >= 0:
            twist = func_translate_twist(steel_prop.links_qd[i_p, i_b], func_link_edge(i_l, i_p, i_b, steel_prop))
        i_d_start = steel_model.joints_dof_start[i_j]
        for i_d in range(i_d_start, i_d_start + steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]):
            twist = twist + steel_prop.dofs_S[i_d, i_b] * steel_dyn.v_out[i_d, i_b]
        steel_prop.links_qd[i_l, i_b] = twist
        if qd.static(clear_impulse):
            steel_prop.links_impulse[i_l, i_b] = qd.Vector.zero(gs.qd_float, 6)


@qd.func
def func_free_body_link(i_a: int, steel_model: array_class.SteelModelInfo):
    """Return the link of free rigid body articulation i_a."""
    i_j = steel_model.arts_joint_start[i_a]
    return steel_model.joints_link[i_j]


@qd.func
def func_propagation_row_dot(i_r: int, i_b: int, steel_prop: array_class.SteelPropagation):
    """Return J v of propagation row i_r from the twists of its links."""
    jv = gs.qd_float(0.0)
    i_la = steel_prop.body_a[i_r, i_b]
    i_lb = steel_prop.body_b[i_r, i_b]
    if i_la >= 0:
        jv = jv + steel_prop.J_a[i_r, i_b].dot(steel_prop.links_qd[i_la, i_b])
    if i_lb >= 0:
        jv = jv + steel_prop.J_b[i_r, i_b].dot(steel_prop.links_qd[i_lb, i_b])
    return jv


@qd.func
def func_propagation_apply_impulse(i_r: int, i_b: int, dlam: float, steel_prop: array_class.SteelPropagation):
    """Apply the impulse change dlam of propagation row i_r to the twist estimates and the impulses of its links."""
    i_la = steel_prop.body_a[i_r, i_b]
    i_lb = steel_prop.body_b[i_r, i_b]
    if i_la >= 0:
        steel_prop.links_qd[i_la, i_b] = steel_prop.links_qd[i_la, i_b] + steel_prop.MiJt_a[i_r, i_b] * dlam
        steel_prop.links_impulse[i_la, i_b] = steel_prop.links_impulse[i_la, i_b] + steel_prop.J_a[i_r, i_b] * dlam
    if i_lb >= 0:
        steel_prop.links_qd[i_lb, i_b] = steel_prop.links_qd[i_lb, i_b] + steel_prop.MiJt_b[i_r, i_b] * dlam
        steel_prop.links_impulse[i_lb, i_b] = steel_prop.links_impulse[i_lb, i_b] + steel_prop.J_b[i_r, i_b] * dlam


@qd.kernel(fastcache=True)
def kernel_steel_propagation_color(
    steel_contacts: array_class.SteelContacts,
    steel_prop: array_class.SteelPropagation,
    steel_config: qd.template(),
):
    """Color the propagation contacts of every environment and move their rows into color order
    ('propagation_colored').

    A contact takes the lowest color that neither of its links uses yet (first-fit greedy in contact order), so the
    contacts of one color share no link. Contacts finding no color below PROPAGATION_COLOR_TAIL (see array_class.py)
    go to the tail bucket. The rows of the contacts are then reserved again color after color, in contact order within
    a color and the tail last, which is the order the sweep visits them. The contacts of one color are independent:
    solving them in any order, or in parallel, gives the same impulses.
    """
    _B = steel_prop.count.shape[0]
    # The color counters hold every color, the tail bucket and the total (see PROPAGATION_COLOR_TAIL).
    n_color_entries = steel_prop.colors_count.shape[0]
    color_tail = n_color_entries - 2
    n_mask_words = steel_prop.links_color_mask.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        n_units = 0
        for i_c in range(steel_contacts.n[i_b]):
            if steel_contacts.path[i_c, i_b] == CONTACT_PATH.PROPAGATION:
                steel_prop.units_contact[n_units, i_b] = i_c
                n_units = n_units + 1

        for i_u in range(n_units):
            i_c = steel_prop.units_contact[i_u, i_b]
            for side in qd.static(range(2)):
                i_l = steel_contacts.body0[i_c, i_b] if qd.static(side == 0) else steel_contacts.body1[i_c, i_b]
                if i_l >= 0:
                    for i_w in range(n_mask_words):
                        steel_prop.links_color_mask[i_l, i_w, i_b] = 0
        for i_k in range(n_color_entries):
            steel_prop.colors_count[i_k, i_b] = 0

        for i_u in range(n_units):
            i_c = steel_prop.units_contact[i_u, i_b]
            i_l0 = steel_contacts.body0[i_c, i_b]
            i_l1 = steel_contacts.body1[i_c, i_b]
            color = color_tail
            for i_w in range(n_mask_words):
                if color == color_tail:
                    used = 0
                    if i_l0 >= 0:
                        used = used | steel_prop.links_color_mask[i_l0, i_w, i_b]
                    if i_l1 >= 0:
                        used = used | steel_prop.links_color_mask[i_l1, i_w, i_b]
                    free_bits = ~used
                    if free_bits != 0:
                        bit = 31
                        for i_bit_ in range(32):
                            i_bit = 31 - i_bit_
                            if (free_bits >> i_bit) & 1 != 0:
                                bit = i_bit
                        color = i_w * 32 + bit
                        if i_l0 >= 0:
                            steel_prop.links_color_mask[i_l0, i_w, i_b] = steel_prop.links_color_mask[
                                i_l0, i_w, i_b
                            ] | (1 << bit)
                        if i_l1 >= 0:
                            steel_prop.links_color_mask[i_l1, i_w, i_b] = steel_prop.links_color_mask[
                                i_l1, i_w, i_b
                            ] | (1 << bit)
            steel_prop.units_color[i_u, i_b] = color
            steel_prop.colors_count[color, i_b] = steel_prop.colors_count[color, i_b] + 1

        # Exclusive prefix of the counts, then a stable counting sort of the contacts by color.
        acc = 0
        for i_k in range(n_color_entries):
            count = steel_prop.colors_count[i_k, i_b]
            steel_prop.colors_count[i_k, i_b] = acc
            acc = acc + count
        for i_u in range(n_units):
            color = steel_prop.units_color[i_u, i_b]
            pos = steel_prop.colors_count[color, i_b]
            steel_prop.colors_count[color, i_b] = pos + 1
            steel_prop.units_order[pos, i_b] = i_u

        slot = 0
        for pos in range(n_units):
            i_u = steel_prop.units_order[pos, i_b]
            i_c = steel_prop.units_contact[i_u, i_b]
            steel_contacts.slot[i_c, i_b] = slot
            slot = slot + steel_contacts.slots_needed[i_c, i_b]


@qd.kernel(fastcache=True)
def kernel_steel_propagation_build_rows(
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Write the normal row and, when reserved, the two tangent rows of every propagation contact.

    Side A of a row acts on link body0 of the contact with J_a = (d, (p_a - c_a) x d), c_a being its COM, side B on
    body1 with J_b = -(d, (p_b - c_b) x d), so J v > 0 separates. Static geometry gets no side (-1). A tangent row
    points at its normal row. Every row starts the step with a zero impulse, which a warm start may then seed.
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_prop.count.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        if i_c < steel_contacts.n[i_b] and steel_contacts.path[i_c, i_b] == CONTACT_PATH.PROPAGATION:
            slot = steel_contacts.slot[i_c, i_b]
            p_a, p_b, n_c, phi = func_contact_points_world(i_c, i_b, steel_state, steel_contacts)
            t0, t1 = func_contact_tangent_basis(n_c)
            mu = func_contact_mu(i_c, i_b, steel_contacts, steel_info)
            restitution = func_contact_restitution(i_c, i_b, steel_contacts, steel_info)
            i_l0 = steel_contacts.body0[i_c, i_b]
            i_l1 = steel_contacts.body1[i_c, i_b]
            com_a = qd.Vector.zero(gs.qd_float, 3)
            com_b = qd.Vector.zero(gs.qd_float, 3)
            if i_l0 >= 0:
                com_a = steel_dyn.links_com_pos[i_l0, i_b]
            if i_l1 >= 0:
                com_b = steel_dyn.links_com_pos[i_l1, i_b]
            for offset in range(steel_contacts.slots_needed[i_c, i_b]):
                i_r = slot + offset
                direction = n_c
                if offset == 1:
                    direction = t0
                elif offset == 2:
                    direction = t1
                point_a, point_b = func_contact_row_points(i_c, i_b, offset > 0, p_a, p_b, steel_contacts, steel_config)
                J_a = qd.Vector.zero(gs.qd_float, 6)
                J_b = qd.Vector.zero(gs.qd_float, 6)
                if i_l0 >= 0:
                    J_a = func_vec6(direction, (point_a - com_a).cross(direction))
                if i_l1 >= 0:
                    J_b = -func_vec6(direction, (point_b - com_b).cross(direction))
                steel_prop.J_a[i_r, i_b] = J_a
                steel_prop.J_b[i_r, i_b] = J_b
                steel_prop.body_a[i_r, i_b] = i_l0
                steel_prop.body_b[i_r, i_b] = i_l1
                steel_prop.row_contact[i_r, i_b] = i_c
                steel_prop.row_mu[i_r, i_b] = mu
                steel_prop.target_velocity[i_r, i_b] = 0.0
                steel_prop.impulses[i_r, i_b] = 0.0
                if offset == 0:
                    steel_prop.row_type[i_r, i_b] = ROW_TYPE.CONTACT
                    steel_prop.row_parent[i_r, i_b] = -1
                    steel_prop.phi[i_r, i_b] = phi
                    steel_prop.restitution[i_r, i_b] = restitution
                else:
                    steel_prop.row_type[i_r, i_b] = ROW_TYPE.FRICTION
                    steel_prop.row_parent[i_r, i_b] = slot
                    # The phi of a propagation friction row holds the anchor displacement scaled by its gain.
                    steel_prop.phi[i_r, i_b] = 0.0
                    if qd.static(steel_config.has_friction_anchors):
                        steel_prop.phi[i_r, i_b] = (
                            steel_info.friction_anchor_beta[None] * steel_contacts.patch_phi[i_c, i_b][offset - 1]
                        )
                    steel_prop.restitution[i_r, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_steel_propagation_factor(
    steel_state: array_class.SteelState,
    steel_free_rows: array_class.SteelFreeRows,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Factor every articulation into articulated-body terms and compute the COM response of each of its links.

    The motion subspace S of every degree of freedom moves from the articulation origin to the COM of its link. Leaves
    first, every link reduces its articulated inertia Ia (spatial inertia of the link plus the reduced inertias of its
    children) across its inbound joint: U = Ia S, D = S^T U + armature + K (K being the implicit drive term of the mass
    matrix), Ia - U D^-1 U^T joins the parent. Roots first, the response of a link to a unit wrench e follows that of
    its parent: the joint absorbs u = S^T e, the wrench p = e - U D^-1 u reaches the parent, whose twist change moves
    to the link as pd, and the link changes its twist by pd + S D^-1 (u - U^T pd). A free rigid body responds with its
    inverse spatial inertia.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    _B = steel_prop.count.shape[0]
    max_art_dofs = steel_dyn.aug_dof.shape[0] // n_arts
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        if steel_model.arts_is_free_rigid[i_a] != 0:
            if qd.static(steel_config.has_free_rigid_bodies):
                i_l = func_free_body_link(i_a, steel_model)
                tl = steel_free_rows.links_Hinv_tl[i_l, i_b]
                tr = steel_free_rows.links_Hinv_tr[i_l, i_b]
                bl = steel_free_rows.links_Hinv_bl[i_l, i_b]
                br = steel_free_rows.links_Hinv_br[i_l, i_b]
                response = qd.Matrix.zero(gs.qd_float, 6, 6)
                for r in qd.static(range(3)):
                    for c in qd.static(range(3)):
                        response[r, c] = tl[r, c]
                        response[r, 3 + c] = tr[r, c]
                        response[3 + r, c] = bl[r, c]
                        response[3 + r, 3 + c] = br[r, c]
                steel_prop.links_response[i_l, i_b] = response
        else:
            origin = steel_dyn.arts_origin[i_a, i_b]
            i_j_start = steel_model.arts_joint_start[i_a]
            i_j_end = steel_model.arts_joint_start[i_a + 1]
            for i_j in range(i_j_start, i_j_end):
                i_l = steel_model.joints_link[i_j]
                com_rel = steel_dyn.links_com_pos[i_l, i_b] - origin
                steel_prop.links_com_rel[i_l, i_b] = com_rel
                i_d_start = steel_model.joints_dof_start[i_j]
                for i_d in range(i_d_start, i_d_start + steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]):
                    S_ang = steel_dyn.dofs_S_ang[i_d, i_b]
                    steel_prop.dofs_S[i_d, i_b] = func_vec6(
                        steel_dyn.dofs_S_lin[i_d, i_b] + S_ang.cross(com_rel), S_ang
                    )
                mass, _, inertia = func_body_inertia(i_l, i_b, dyn_info, rigid_info, steel_config)
                R = gu.qd_quat_to_R(steel_state.links_quat[i_l, i_b], rigid_info.EPS[None])
                inertia = R @ inertia @ R.transpose()
                Ia = qd.Matrix.zero(gs.qd_float, 6, 6)
                for r in qd.static(range(3)):
                    Ia[r, r] = mass
                    for c in qd.static(range(3)):
                        Ia[3 + r, 3 + c] = inertia[r, c]
                steel_prop.links_Ia[i_l, i_b] = Ia

            for i_j_ in range(i_j_end - i_j_start):
                i_j = i_j_end - 1 - i_j_
                i_l = steel_model.joints_link[i_j]
                i_p = steel_model.joints_parent_link[i_j]
                i_d_start = steel_model.joints_dof_start[i_j]
                n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
                Ia = steel_prop.links_Ia[i_l, i_b]
                D = qd.Matrix.zero(gs.qd_float, 6, 6)
                for a in range(n_dofs):
                    steel_prop.dofs_U[i_d_start + a, i_b] = Ia @ steel_prop.dofs_S[i_d_start + a, i_b]
                for a in range(n_dofs):
                    i_d_a = i_d_start + a
                    for b in range(n_dofs):
                        value = steel_prop.dofs_S[i_d_a, i_b].dot(steel_prop.dofs_U[i_d_start + b, i_b])
                        if a == b:
                            I_d = [i_d_a, i_b] if qd.static(steel_config.batch_dofs_info) else i_d_a
                            value = value + dyn_info.dofs.armature[I_d]
                            for i_aug_ in range(steel_dyn.arts_aug_count[i_a, i_b]):
                                i_aug = i_a * max_art_dofs + i_aug_
                                K = steel_dyn.aug_K[i_aug, i_b]
                                if steel_dyn.aug_dof[i_aug, i_b] == i_d_a and K > 0.0:
                                    value = value + K
                        D[a, b] = value

                # Cholesky factor of D, its pivots floored, then D^-1 by forward and backward substitution.
                L = qd.Matrix.zero(gs.qd_float, 6, 6)
                for j in range(n_dofs):
                    s = D[j, j]
                    for k in range(j):
                        s = s - L[j, k] * L[j, k]
                    if s <= steel_info.pivot_floor[None]:
                        s = steel_info.pivot_floor[None]
                    s = qd.sqrt(s)
                    L[j, j] = s
                    inv_s = 1.0 / s
                    for i in range(j + 1, n_dofs):
                        v = D[i, j]
                        for k in range(j):
                            v = v - L[i, k] * L[j, k]
                        L[i, j] = v * inv_s
                D_inv = qd.Matrix.zero(gs.qd_float, 6, 6)
                for col in range(n_dofs):
                    for i in range(n_dofs):
                        v = 1.0 if i == col else 0.0
                        for k in range(i):
                            v = v - L[i, k] * D_inv[k, col]
                        D_inv[i, col] = v / L[i, i]
                    for i_ in range(n_dofs):
                        i = n_dofs - 1 - i_
                        v = D_inv[i, col]
                        for k in range(i + 1, n_dofs):
                            v = v - L[k, i] * D_inv[k, col]
                        D_inv[i, col] = v / L[i, i]
                steel_prop.links_D_inv[i_l, i_b] = D_inv

                for a in range(n_dofs):
                    W = qd.Vector.zero(gs.qd_float, 6)
                    for b in range(n_dofs):
                        W = W + D_inv[a, b] * steel_prop.dofs_U[i_d_start + b, i_b]
                    Ia = Ia - steel_prop.dofs_U[i_d_start + a, i_b].outer_product(W)
                steel_prop.links_Ia[i_l, i_b] = Ia
                if i_p >= 0:
                    # Twist translation by the edge: (v, w) -> (v + w x e, w) = X (v, w), so the child inertia joins
                    # the parent as X^T Ia X.
                    e = func_link_edge(i_l, i_p, i_b, steel_prop)
                    X = qd.Matrix.identity(gs.qd_float, 6)
                    X[0, 4] = e[2]
                    X[0, 5] = -e[1]
                    X[1, 3] = -e[2]
                    X[1, 5] = e[0]
                    X[2, 3] = e[1]
                    X[2, 4] = -e[0]
                    steel_prop.links_Ia[i_p, i_b] = steel_prop.links_Ia[i_p, i_b] + X.transpose() @ Ia @ X

            for i_j in range(i_j_start, i_j_end):
                i_l = steel_model.joints_link[i_j]
                i_p = steel_model.joints_parent_link[i_j]
                i_d_start = steel_model.joints_dof_start[i_j]
                n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
                D_inv = steel_prop.links_D_inv[i_l, i_b]
                e = qd.Vector.zero(gs.qd_float, 3)
                response_parent = qd.Matrix.zero(gs.qd_float, 6, 6)
                if i_p >= 0:
                    e = func_link_edge(i_l, i_p, i_b, steel_prop)
                    response_parent = steel_prop.links_response[i_p, i_b]
                response = qd.Matrix.zero(gs.qd_float, 6, 6)
                for col in range(6):
                    wrench = qd.Vector.zero(gs.qd_float, 6)
                    wrench[col] = 1.0
                    for a in range(n_dofs):
                        y = gs.qd_float(0.0)
                        for b in range(n_dofs):
                            y = y + D_inv[a, b] * steel_prop.dofs_S[i_d_start + b, i_b][col]
                        wrench = wrench - steel_prop.dofs_U[i_d_start + a, i_b] * y
                    pd = qd.Vector.zero(gs.qd_float, 6)
                    if i_p >= 0:
                        pd = func_translate_twist(response_parent @ func_translate_wrench(wrench, e), e)
                    delta = pd
                    for a in range(n_dofs):
                        qdd = gs.qd_float(0.0)
                        for b in range(n_dofs):
                            i_d = i_d_start + b
                            qdd = qdd + D_inv[a, b] * (
                                steel_prop.dofs_S[i_d, i_b][col] - steel_prop.dofs_U[i_d, i_b].dot(pd)
                            )
                        delta = delta + steel_prop.dofs_S[i_d_start + a, i_b] * qdd
                    for r in qd.static(range(6)):
                        response[r, col] = delta[r]
                steel_prop.links_response[i_l, i_b] = response


@qd.kernel(fastcache=True)
def kernel_steel_propagation_refresh(
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
    refresh_all: bool,
):
    """Recompute the twists of every link from the joint velocity v_out.

    Unless refresh_all, the articulations of environments without dense contact rows keep their twists: their joint
    velocity only changes through the tree propagation, which leaves the twists consistent. Free rigid bodies always
    copy their velocity, which is their COM twist.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    _B = steel_prop.count.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        if steel_model.arts_is_free_rigid[i_a] != 0:
            i_l = func_free_body_link(i_a, steel_model)
            dof_start = steel_model.arts_dof_start[i_a]
            steel_prop.links_qd[i_l, i_b] = qd.Vector(
                [steel_dyn.v_out[dof_start + k, i_b] for k in qd.static(range(6))], dt=gs.qd_float
            )
        elif refresh_all or steel_prop.has_dense_contacts[i_b] != 0:
            func_tree_twists(i_a, i_b, steel_prop, steel_dyn, steel_model, clear_impulse=False)


@qd.kernel(fastcache=True)
def kernel_steel_propagation_row_setup(
    dt: float,
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Compute the response, the diagonal and the right-hand side of every propagation row.

    The response of side x is MiJt_x = s_x Phi_x J_x^T, s_x being the number of distinct row-bearing links in the
    coupling group of its link (mass splitting), and the diagonal d = cfm + sum_x J_x . MiJt_x. Contact biases follow
    the free-body rows (see kernel_steel_row_setup), the incident velocity of a rebound being read from the link
    twists. Penetrating contacts take the regularization weight w * d / (w * d + (1 - w) * d_u), d_u being the
    unsplit diagonal, so the compliance of the fixed point follows the unsplit response. With
    propagation_same_articulation_rows, a row between two links of one articulation takes instead the exact response
    of a tree solve of its two Jacobians, unsplit.
    """
    _B = steel_prop.count.shape[0]
    n_joints = steel_model.joints_link.shape[0]
    capacity = steel_prop.row_type.shape[0]
    inv_dt = 1.0 / dt
    cfm = steel_info.cfm[None]
    baumgarte = steel_info.baumgarte[None]
    contact_w = steel_info.contact_w[None]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        for i_j in range(n_joints):
            i_l = steel_model.joints_link[i_j]
            steel_prop.links_split_seen[i_l, i_b] = 0
            steel_prop.links_group_count[i_l, i_b] = 0
        for i_r in range(steel_prop.count[i_b]):
            for side in qd.static(range(2)):
                i_l = steel_prop.body_a[i_r, i_b] if qd.static(side == 0) else steel_prop.body_b[i_r, i_b]
                if i_l >= 0:
                    if steel_prop.links_split_seen[i_l, i_b] == 0:
                        steel_prop.links_split_seen[i_l, i_b] = 1
                        i_g = steel_model.links_coupling_group[i_l]
                        steel_prop.links_group_count[i_g, i_b] = steel_prop.links_group_count[i_g, i_b] + 1

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(capacity, _B):
        if i_r < steel_prop.count[i_b]:
            i_la = steel_prop.body_a[i_r, i_b]
            i_lb = steel_prop.body_b[i_r, i_b]
            d = cfm
            d_unsplit = cfm
            for side in qd.static(range(2)):
                i_l = i_la if qd.static(side == 0) else i_lb
                J = steel_prop.J_a[i_r, i_b] if qd.static(side == 0) else steel_prop.J_b[i_r, i_b]
                MiJt = qd.Vector.zero(gs.qd_float, 6)
                if i_l >= 0:
                    i_g = steel_model.links_coupling_group[i_l]
                    split = qd.max(steel_prop.links_group_count[i_g, i_b], 1)
                    MiJt = steel_prop.links_response[i_l, i_b] @ J
                    d_unsplit = d_unsplit + J.dot(MiJt)
                    MiJt = MiJt * split
                    d = d + J.dot(MiJt)
                if qd.static(side == 0):
                    steel_prop.MiJt_a[i_r, i_b] = MiJt
                else:
                    steel_prop.MiJt_b[i_r, i_b] = MiJt
            steel_prop.eff_mass_inv[i_r, i_b] = 1.0 / d if d > 0.0 else 0.0

            bias = gs.qd_float(0.0)
            restitution_bias = gs.qd_float(0.0)
            w = gs.qd_float(1.0)
            row_type = steel_prop.row_type[i_r, i_b]
            target = steel_prop.target_velocity[i_r, i_b]
            phi = steel_prop.phi[i_r, i_b]
            if row_type == ROW_TYPE.CONTACT:
                if phi <= 0.0:
                    w = contact_w
                if phi < 0.0:
                    bias = baumgarte * phi * inv_dt
                    max_depenetration = steel_info.unbounded_depenetration[None]
                    if i_la >= 0:
                        max_depenetration = steel_info.links_max_depenetration_velocity[i_la]
                    if i_lb >= 0:
                        v_b = steel_info.links_max_depenetration_velocity[i_lb]
                        if v_b > 0.0 and not qd.math.isinf(v_b) and v_b < max_depenetration:
                            max_depenetration = v_b
                    if max_depenetration > 0.0 and not qd.math.isinf(max_depenetration):
                        bias = qd.max(bias, -max_depenetration)
                else:
                    bias = steel_info.contact_speculative_scale[None] * phi * inv_dt
                e = steel_prop.restitution[i_r, i_b]
                if e > 0.0:
                    u = func_propagation_row_dot(i_r, i_b, steel_prop) - target
                    if func_restitution_fires(phi, u, dt, steel_info):
                        bias = e * u
                        restitution_bias = bias
                        w = 1.0
            elif row_type == ROW_TYPE.FRICTION:
                bias = phi * inv_dt
            steel_prop.rhs[i_r, i_b] = bias - target
            steel_prop.restitution_bias[i_r, i_b] = restitution_bias
            # The split diagonal is used by the sweep, the unsplit one keeps the compliance of the fixed point.
            if contact_w < 1.0 and w < 1.0 and d != d_unsplit:
                w = contact_w * d / (contact_w * d + (1.0 - contact_w) * d_unsplit)
            steel_prop.row_w[i_r, i_b] = w

    if qd.static(steel_config.propagation_same_articulation_rows):
        n_arts = steel_model.arts_dof_start.shape[0]
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_a, i_b in qd.ndrange(n_arts, _B):
            if steel_model.arts_is_free_rigid[i_a] == 0:
                for i_r in range(steel_prop.count[i_b]):
                    i_la = steel_prop.body_a[i_r, i_b]
                    i_lb = steel_prop.body_b[i_r, i_b]
                    is_same_articulation = False
                    if i_la >= 0 and i_lb >= 0:
                        i_aa = steel_model.links_art[i_la]
                        i_ab = steel_model.links_art[i_lb]
                        is_same_articulation = i_aa == i_a and i_ab == i_a
                    if is_same_articulation:
                        for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
                            i_l = steel_model.joints_link[i_j]
                            steel_prop.links_pA[i_l, i_b] = qd.Vector.zero(gs.qd_float, 6)
                        J_a = steel_prop.J_a[i_r, i_b]
                        J_b = steel_prop.J_b[i_r, i_b]
                        steel_prop.links_pA[i_la, i_b] = steel_prop.links_pA[i_la, i_b] - J_a
                        steel_prop.links_pA[i_lb, i_b] = steel_prop.links_pA[i_lb, i_b] - J_b
                        func_tree_backward(i_a, i_b, steel_prop, steel_model)
                        func_tree_forward(i_a, i_b, steel_prop, steel_dyn, steel_model, update_velocity=False)
                        MiJt_a = steel_prop.links_delta[i_la, i_b]
                        MiJt_b = steel_prop.links_delta[i_lb, i_b]
                        steel_prop.MiJt_a[i_r, i_b] = MiJt_a
                        steel_prop.MiJt_b[i_r, i_b] = MiJt_b
                        d = cfm + J_a.dot(MiJt_a) + J_b.dot(MiJt_b)
                        steel_prop.eff_mass_inv[i_r, i_b] = 1.0 / d if d > 0.0 else 0.0
                        # The exact response is unsplit, so the row keeps the uniform regularization weight.
                        if contact_w < 1.0 and steel_prop.row_w[i_r, i_b] < 1.0:
                            steel_prop.row_w[i_r, i_b] = contact_w


@qd.kernel(fastcache=True)
def kernel_steel_propagation_sweep(
    global_iter: int,
    friction_start_iteration: int,
    rhs: qd.Tensor,
    steel_prop: array_class.SteelPropagation,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
    regularize: bool,
):
    """Run one Gauss-Seidel sweep of the propagation rows of every environment, in row order.

    A row update is lambda += omega * (w * (-r / d) - (1 - w) * lambda) with the residual r = J v + rhs from the link
    twists, the regularization weight w applying to contact rows under regularize, projected on lambda >= 0 for
    contacts. The first tangent row of a contact solves both tangent impulses on the friction disk of its pooled normal
    impulse, the second one is left as is, and tangent rows hold a zero impulse before friction_start_iteration. Every
    impulse change moves the twist estimates of the row's links by MiJt dlambda and accumulates J^T dlambda on them,
    which kernel_steel_propagation_propagate applies to the joint velocities. The rows of 'propagation_colored' lie in
    color order, the contacts of one color sharing no link.
    """
    _B = steel_prop.count.shape[0]
    omega = steel_info.relaxation[None]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        for i_r in range(steel_prop.count[i_b]):
            row_type = steel_prop.row_type[i_r, i_b]
            is_friction = row_type == ROW_TYPE.FRICTION
            eff_inv = steel_prop.eff_mass_inv[i_r, i_b]
            is_skipped = eff_inv <= 0.0 and not is_friction
            if is_friction and global_iter < friction_start_iteration:
                steel_prop.impulses[i_r, i_b] = 0.0
                is_skipped = True
            if not is_skipped:
                res = func_propagation_row_dot(i_r, i_b, steel_prop) + rhs[i_r, i_b]
                old = steel_prop.impulses[i_r, i_b]
                delta = -res * eff_inv
                if row_type == ROW_TYPE.CONTACT and regularize:
                    w = steel_prop.row_w[i_r, i_b]
                    delta = delta * w - (1.0 - w) * old
                new = old + omega * delta
                if row_type == ROW_TYPE.CONTACT:
                    new = qd.max(new, 0.0)
                elif is_friction:
                    new = old
                    i_p = steel_prop.row_parent[i_r, i_b]
                    if i_r == i_p + 1:
                        i_s = i_p + 2
                        load = func_pooled_normal_impulse(i_p, i_b, steel_prop.row_parent, steel_prop.impulses)
                        radius = qd.max(steel_prop.row_mu[i_r, i_b] * load, 0.0)
                        old_s = steel_prop.impulses[i_s, i_b]
                        res_s = func_propagation_row_dot(i_s, i_b, steel_prop) + rhs[i_s, i_b]
                        cross = gs.qd_float(0.0)
                        if steel_prop.body_a[i_r, i_b] >= 0:
                            cross = cross + steel_prop.J_a[i_r, i_b].dot(steel_prop.MiJt_a[i_s, i_b])
                        if steel_prop.body_b[i_r, i_b] >= 0:
                            cross = cross + steel_prop.J_b[i_r, i_b].dot(steel_prop.MiJt_b[i_s, i_b])
                        eff_inv_s = steel_prop.eff_mass_inv[i_s, i_b]
                        pair = func_friction_pair_candidate(
                            1.0 / eff_inv if eff_inv > 0.0 else 0.0,
                            cross,
                            1.0 / eff_inv_s if eff_inv_s > 0.0 else 0.0,
                            qd.Vector([res, res_s], dt=gs.qd_float),
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
                        steel_prop.impulses[i_s, i_b] = pair[1]
                        if dlam_s != 0.0:
                            func_propagation_apply_impulse(i_s, i_b, dlam_s, steel_prop)
                dlam = new - old
                steel_prop.impulses[i_r, i_b] = new
                if dlam != 0.0:
                    func_propagation_apply_impulse(i_r, i_b, dlam, steel_prop)


@qd.kernel(fastcache=True)
def kernel_steel_propagation_propagate(
    steel_prop: array_class.SteelPropagation,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Apply the impulses accumulated on the links by the sweep to the joint velocity v_out, exactly.

    Every articulation holding an impulse runs the backward and forward passes of its factorization on them, which
    adds H^-1 J^T I to its joint velocity, then recomputes the twists of its links from it, replacing the estimates of
    the sweep, and clears the impulses. A free rigid body writes its twist, exact, back to its velocity.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    _B = steel_prop.count.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        if steel_model.arts_is_free_rigid[i_a] != 0:
            i_l = func_free_body_link(i_a, steel_model)
            dof_start = steel_model.arts_dof_start[i_a]
            twist = steel_prop.links_qd[i_l, i_b]
            for k in qd.static(range(6)):
                steel_dyn.v_out[dof_start + k, i_b] = twist[k]
            steel_prop.links_impulse[i_l, i_b] = qd.Vector.zero(gs.qd_float, 6)
        else:
            has_impulse = False
            for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
                i_l = steel_model.joints_link[i_j]
                impulse = steel_prop.links_impulse[i_l, i_b]
                for k in qd.static(range(6)):
                    if impulse[k] != 0.0:
                        has_impulse = True
                steel_prop.links_pA[i_l, i_b] = -impulse
            if has_impulse:
                func_tree_backward(i_a, i_b, steel_prop, steel_model)
                func_tree_forward(i_a, i_b, steel_prop, steel_dyn, steel_model, update_velocity=True)
                func_tree_twists(i_a, i_b, steel_prop, steel_dyn, steel_model, clear_impulse=True)


@qd.kernel(fastcache=True)
def kernel_steel_propagation_warmstart(
    steel_prop: array_class.SteelPropagation,
    steel_config: qd.template(),
):
    """Move the twist estimates and accumulate the link impulses of the impulses the rows start the step with.

    kernel_steel_propagation_propagate then installs their velocity in v_out before the first sweep.
    """
    _B = steel_prop.count.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        for i_r in range(steel_prop.count[i_b]):
            impulse = steel_prop.impulses[i_r, i_b]
            if impulse != 0.0:
                func_propagation_apply_impulse(i_r, i_b, impulse, steel_prop)


@qd.kernel(fastcache=True)
def kernel_steel_propagation_velocity_pass_rhs(
    dt: float,
    steel_prop: array_class.SteelPropagation,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Compute the right-hand sides of the velocity-only sweeps of the propagation rows, from the link twists.

    An impacting contact keeps its rebound target. A separated contact whose linearized gap at the end of the step,
    with the twists of the position solve, stays open keeps its full gap allowance phi / dt. Every other row asks for
    its target velocity alone.
    """
    _B = steel_prop.count.shape[0]
    capacity = steel_prop.row_type.shape[0]
    inv_dt = 1.0 / dt
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b in qd.ndrange(capacity, _B):
        if i_r < steel_prop.count[i_b]:
            target = steel_prop.target_velocity[i_r, i_b]
            bias = gs.qd_float(0.0)
            if steel_prop.row_type[i_r, i_b] == ROW_TYPE.CONTACT:
                restitution_bias = steel_prop.restitution_bias[i_r, i_b]
                phi = steel_prop.phi[i_r, i_b]
                if restitution_bias != 0.0:
                    bias = restitution_bias
                elif phi > 0.0:
                    end_gap = phi + dt * (func_propagation_row_dot(i_r, i_b, steel_prop) - target)
                    if end_gap > steel_info.contact_end_gap_slop[None]:
                        bias = phi * inv_dt
            steel_prop.rhs_unbiased[i_r, i_b] = bias - target
