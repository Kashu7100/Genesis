"""Articulated dynamics of the Steel solver: stages 1 to 3 and 7 of a step.

Stage 1 evaluates the forward kinematics of every articulation and the bias wrenches of the inverse dynamics, the
generalized forces (inverse-dynamics backward pass, passive damping, implicit drives) and the composite rigid body mass
matrix. Stage 2 factors it, stage 3 solves the unconstrained acceleration and predicts the velocity the integrator would
realize. Stage 7 converts the constrained velocity back to an acceleration, integrates the joint coordinates with
symplectic Euler and evaluates the forward kinematics of the new configuration.

Every articulation is solved in its own frame: world axes, origin at the center of mass of the child link of its first
joint, recomputed every step. Spatial vectors are (linear, angular) pairs, the linear part being the velocity of the
material point at the origin (a force about the origin for wrenches).

With 'sleep_skip_constraints', an articulation asleep for the step (see sleeping.py) skips every stage, its mass matrix
refresh included, so its state stays frozen.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu

from .model import STEEL_JOINT_TYPE


@qd.func
def func_skew(v: qd.types.vector(3)):
    return qd.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dt=gs.qd_float)


@qd.func
def func_diag(v: qd.types.vector(3)):
    return qd.Matrix([[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]], dt=gs.qd_float)


@qd.func
def func_transform_twist(
    t_pos: qd.types.vector(3), t_quat: qd.types.vector(4), v_lin: qd.types.vector(3), v_ang: qd.types.vector(3)
):
    """Express the twist (v_lin, v_ang) given in the frame t in the frame t is relative to."""
    ang = gu.qd_transform_by_quat(v_ang, t_quat)
    lin = gu.qd_transform_by_quat(v_lin, t_quat) + t_pos.cross(ang)
    return lin, ang


@qd.func
def func_motion_cross(
    a_lin: qd.types.vector(3), a_ang: qd.types.vector(3), b_lin: qd.types.vector(3), b_ang: qd.types.vector(3)
):
    """Spatial motion cross product a x b."""
    return a_ang.cross(b_lin) + a_lin.cross(b_ang), a_ang.cross(b_ang)


@qd.func
def func_force_cross(
    a_lin: qd.types.vector(3), a_ang: qd.types.vector(3), b_lin: qd.types.vector(3), b_ang: qd.types.vector(3)
):
    """Spatial force cross product a x* b of a twist a and a wrench b."""
    return a_ang.cross(b_lin), a_ang.cross(b_ang) + a_lin.cross(b_lin)


@qd.func
def func_mul_com_inertia(
    mass: float,
    com: qd.types.vector(3),
    inertia_o: qd.types.matrix(3, 3),
    v_lin: qd.types.vector(3),
    v_ang: qd.types.vector(3),
):
    """Product of the spatial inertia of a body (mass, center of mass, inertia about the origin) with a twist."""
    return mass * (v_lin - com.cross(v_ang)), mass * com.cross(v_lin) + inertia_o @ v_ang


@qd.func
def func_mul_spatial_inertia(
    mass: float,
    h: qd.types.vector(3),
    inertia_o: qd.types.matrix(3, 3),
    v_lin: qd.types.vector(3),
    v_ang: qd.types.vector(3),
):
    """Product of a spatial inertia (mass, first moment h, inertia about the origin) with a twist."""
    return mass * v_lin - h.cross(v_ang), h.cross(v_lin) + inertia_o @ v_ang


@qd.func
def func_body_inertia(
    i_l: int, i_b: int, dyn_info: array_class.DynInfo, rigid_info: array_class.RigidInfo, steel_config: qd.template()
):
    """Return the mass, the center of mass in the link frame and the rotational inertia about it in the link axes."""
    I_l = [i_l, i_b] if qd.static(steel_config.batch_links_info) else i_l
    R = gu.qd_quat_to_R(dyn_info.links.inertial_quat[I_l], rigid_info.EPS[None])
    return (
        dyn_info.links.inertial_mass[I_l],
        dyn_info.links.inertial_pos[I_l],
        R @ dyn_info.links.inertial_i[I_l] @ R.transpose(),
    )


@qd.func
def func_joint_transform(
    i_j: int, i_b: int, steel_state: array_class.SteelState, steel_model: array_class.SteelModelInfo
):
    """Return the transform (position, quaternion) across joint i_j at its current coordinates."""
    joint_type = steel_model.joints_type[i_j]
    i_q = steel_model.joints_q_start[i_j]
    i_d = steel_model.joints_dof_start[i_j]
    pos = qd.Vector.zero(gs.qd_float, 3)
    quat = gu.qd_identity_quat()
    if joint_type == STEEL_JOINT_TYPE.PRISMATIC:
        pos = steel_model.dofs_axis[i_d] * steel_state.q[i_q, i_b]
    elif joint_type == STEEL_JOINT_TYPE.REVOLUTE:
        quat = gu.qd_axis_angle_to_quat(steel_model.dofs_axis[i_d], steel_state.q[i_q, i_b])
    elif joint_type == STEEL_JOINT_TYPE.BALL:
        quat = qd.Vector([steel_state.q[i_q + k, i_b] for k in qd.static(range(4))], dt=gs.qd_float)
    elif joint_type == STEEL_JOINT_TYPE.FREE:
        pos = qd.Vector([steel_state.q[i_q + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
        quat = qd.Vector([steel_state.q[i_q + 3 + k, i_b] for k in qd.static(range(4))], dt=gs.qd_float)
    elif joint_type == STEEL_JOINT_TYPE.D6:
        for k in range(steel_model.joints_n_lin[i_j]):
            pos = pos + steel_model.dofs_axis[i_d + k] * steel_state.q[i_q + k, i_b]
        _, quat = func_d6_angular_axes(i_j, i_b, steel_state, steel_model)
    return pos, quat


@qd.func
def func_d6_angular_axes(
    i_j: int, i_b: int, steel_state: array_class.SteelState, steel_model: array_class.SteelModelInfo
):
    """Return the angular axes of a D6 joint transported through the rotations preceding them, and its rotation.

    The rotation of the joint is the intrinsic chain of its angular coordinates: axis k is carried by the rotations of
    the axes before it, so the joint rotation is qa(A_n, q_n) ... qa(A_0, q_0) with transported axes A_k.
    """
    i_q = steel_model.joints_q_start[i_j] + steel_model.joints_n_lin[i_j]
    i_d = steel_model.joints_dof_start[i_j] + steel_model.joints_n_lin[i_j]
    axes = qd.Matrix.zero(gs.qd_float, 3, 3)
    rot = gu.qd_identity_quat()
    for k in range(steel_model.joints_n_ang[i_j]):
        axis = gu.qd_transform_by_quat(steel_model.dofs_axis[i_d + k], rot)
        axes[k, :] = axis
        rot = gu.qd_quat_mul(gu.qd_axis_angle_to_quat(axis, steel_state.q[i_q + k, i_b]), rot)
    return axes, rot


@qd.func
def func_joint_motion(
    i_j: int,
    i_b: int,
    x_pos: qd.types.vector(3),
    x_quat: qd.types.vector(4),
    qd_src: qd.Tensor,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    steel_model: array_class.SteelModelInfo,
):
    """Write the motion subspace of joint i_j in the frame x and return its joint twist and apparent bias acceleration.

    qd_src holds the joint velocities. The angular axes of a multi-axis D6 joint depend on its coordinates, which adds
    the apparent derivative sum_{j<k} (A_j x A_k) qd_j qd_k of the transported axes.
    """
    joint_type = steel_model.joints_type[i_j]
    i_d = steel_model.joints_dof_start[i_j]
    n_lin = steel_model.joints_n_lin[i_j]
    n_ang = steel_model.joints_n_ang[i_j]
    zero = qd.Vector.zero(gs.qd_float, 3)
    v_lin, v_ang = zero, zero
    c_lin, c_ang = zero, zero
    if joint_type == STEEL_JOINT_TYPE.PRISMATIC or joint_type == STEEL_JOINT_TYPE.REVOLUTE:
        axis = steel_model.dofs_axis[i_d]
        axis_lin, axis_ang = axis, zero
        if joint_type == STEEL_JOINT_TYPE.REVOLUTE:
            axis_lin, axis_ang = zero, axis
        S_lin, S_ang = func_transform_twist(x_pos, x_quat, axis_lin, axis_ang)
        steel_dyn.dofs_S_lin[i_d, i_b] = S_lin
        steel_dyn.dofs_S_ang[i_d, i_b] = S_ang
        v_lin, v_ang = S_lin * qd_src[i_d, i_b], S_ang * qd_src[i_d, i_b]
    elif joint_type == STEEL_JOINT_TYPE.BALL or joint_type == STEEL_JOINT_TYPE.FREE:
        i_d_ang = i_d + 3 if joint_type == STEEL_JOINT_TYPE.FREE else i_d
        for k in qd.static(range(3)):
            unit = qd.Vector.unit(3, k, gs.qd_float)
            if joint_type == STEEL_JOINT_TYPE.FREE:
                S_lin, S_ang = func_transform_twist(x_pos, x_quat, unit, zero)
                steel_dyn.dofs_S_lin[i_d + k, i_b] = S_lin
                steel_dyn.dofs_S_ang[i_d + k, i_b] = S_ang
                v_lin = v_lin + S_lin * qd_src[i_d + k, i_b]
                v_ang = v_ang + S_ang * qd_src[i_d + k, i_b]
            S_lin, S_ang = func_transform_twist(x_pos, x_quat, zero, unit)
            steel_dyn.dofs_S_lin[i_d_ang + k, i_b] = S_lin
            steel_dyn.dofs_S_ang[i_d_ang + k, i_b] = S_ang
            v_lin = v_lin + S_lin * qd_src[i_d_ang + k, i_b]
            v_ang = v_ang + S_ang * qd_src[i_d_ang + k, i_b]
    elif joint_type == STEEL_JOINT_TYPE.D6:
        for k in range(n_lin):
            S_lin, S_ang = func_transform_twist(x_pos, x_quat, steel_model.dofs_axis[i_d + k], zero)
            steel_dyn.dofs_S_lin[i_d + k, i_b] = S_lin
            steel_dyn.dofs_S_ang[i_d + k, i_b] = S_ang
            v_lin = v_lin + S_lin * qd_src[i_d + k, i_b]
            v_ang = v_ang + S_ang * qd_src[i_d + k, i_b]
        axes, _ = func_d6_angular_axes(i_j, i_b, steel_state, steel_model)
        c_app = zero
        for k in range(n_ang):
            i_dk = i_d + n_lin + k
            S_lin, S_ang = func_transform_twist(x_pos, x_quat, zero, axes[k, :])
            steel_dyn.dofs_S_lin[i_dk, i_b] = S_lin
            steel_dyn.dofs_S_ang[i_dk, i_b] = S_ang
            v_lin = v_lin + S_lin * qd_src[i_dk, i_b]
            v_ang = v_ang + S_ang * qd_src[i_dk, i_b]
            for j in range(k):
                c_app = c_app + axes[j, :].cross(axes[k, :]) * (qd_src[i_d + n_lin + j, i_b] * qd_src[i_dk, i_b])
        c_lin, c_ang = func_transform_twist(x_pos, x_quat, zero, c_app)
    return v_lin, v_ang, c_lin, c_ang


@qd.func
def func_joint_anchor(
    i_j: int,
    i_b: int,
    dyn_info: array_class.DynInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Return the position of the anchor of joint i_j in the frame of its link."""
    anchor = qd.Vector.zero(gs.qd_float, 3)
    i_g = steel_model.joints_anchor[i_j]
    if i_g >= 0:
        I_g = [i_g, i_b] if qd.static(steel_config.batch_joints_info) else i_g
        anchor = dyn_info.joints.pos[I_g]
    return anchor


@qd.func
def func_parent_anchor_frame(
    i_j: int,
    i_b: int,
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    dyn_info: array_class.DynInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Return the world pose of the parent anchor frame of joint i_j.

    It is the frame of the link at zero joint coordinates, moved to the joint anchor: the link offset composed with the
    pose of the parent link (Steel's, or Genesis' for static geometry), the pose Genesis holds for a fixed root link, or
    the identity for a free joint, whose coordinates are the pose of its link.
    """
    pos = qd.Vector.zero(gs.qd_float, 3)
    quat = gu.qd_identity_quat()
    if steel_model.joints_type[i_j] != STEEL_JOINT_TYPE.FREE:
        i_l = steel_model.joints_link[i_j]
        I_l = [i_l, i_b] if qd.static(steel_config.batch_links_info) else i_l
        i_p = steel_model.joints_parent_link[i_j]
        i_ps = steel_model.joints_static_parent_link[i_j]
        if i_p >= 0:
            parent_quat = steel_state.links_quat[i_p, i_b]
            pos = steel_state.links_pos[i_p, i_b] + gu.qd_transform_by_quat(dyn_info.links.pos[I_l], parent_quat)
            quat = gu.qd_quat_mul(parent_quat, dyn_info.links.quat[I_l])
        elif i_ps >= 0:
            parent_quat = dyn_state.links.quat[i_ps, i_b]
            pos = dyn_state.links.pos[i_ps, i_b] + gu.qd_transform_by_quat(dyn_info.links.pos[I_l], parent_quat)
            quat = gu.qd_quat_mul(parent_quat, dyn_info.links.quat[I_l])
        elif dyn_info.links.is_fixed[I_l]:
            pos = dyn_state.links.pos[i_l, i_b]
            quat = dyn_state.links.quat[i_l, i_b]
        else:
            pos = dyn_info.links.pos[I_l]
            quat = dyn_info.links.quat[I_l]
        pos = pos + gu.qd_transform_by_quat(func_joint_anchor(i_j, i_b, dyn_info, steel_model, steel_config), quat)
    return pos, quat


@qd.func
def func_link_transform(
    i_j: int,
    i_b: int,
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    dyn_info: array_class.DynInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Write the world pose of the link of joint i_j and of its center of mass from the pose of its parent."""
    i_l = steel_model.joints_link[i_j]
    wpj_pos, wpj_quat = func_parent_anchor_frame(i_j, i_b, dyn_state, steel_state, dyn_info, steel_model, steel_config)
    j_pos, j_quat = func_joint_transform(i_j, i_b, steel_state, steel_model)
    wcj_pos = wpj_pos + gu.qd_transform_by_quat(j_pos, wpj_quat)
    wcj_quat = gu.qd_quat_mul(wpj_quat, j_quat)
    anchor = func_joint_anchor(i_j, i_b, dyn_info, steel_model, steel_config)
    pos = wcj_pos - gu.qd_transform_by_quat(anchor, wcj_quat)
    steel_state.links_pos[i_l, i_b] = pos
    steel_state.links_quat[i_l, i_b] = wcj_quat
    I_l = [i_l, i_b] if qd.static(steel_config.batch_links_info) else i_l
    steel_dyn.links_com_pos[i_l, i_b] = pos + gu.qd_transform_by_quat(dyn_info.links.inertial_pos[I_l], wcj_quat)


@qd.func
def func_steel_prescale_velocity(
    steel_rows: array_class.SteelRows,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Scale the velocity of every articulation exceeding a joint velocity limit, before the prediction (stage 1).

    One ratio per articulation, the smallest limit-to-speed ratio of its limited degrees of freedom, scales all of
    them, the degrees of freedom whose drive row clamps the velocity excepted. Only the kinematics and the predictor
    see the scaled velocity: damping, drives and the integrator keep the velocity of the start of the step.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    _B = steel_dyn.dofs_qd_work.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        is_active = True
        if qd.static(steel_config.sleep_skip_constraints):
            is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
        if is_active:
            ratio = gs.qd_float(1.0)
            for i_pass in qd.static(range(2)):
                for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
                    joint_type = steel_model.joints_type[i_j]
                    if (
                        joint_type == STEEL_JOINT_TYPE.PRISMATIC
                        or joint_type == STEEL_JOINT_TYPE.REVOLUTE
                        or joint_type == STEEL_JOINT_TYPE.D6
                    ):
                        i_d_start = steel_model.joints_dof_start[i_j]
                        for i_d in range(
                            i_d_start, i_d_start + steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
                        ):
                            is_scaled = True
                            if qd.static(steel_config.has_fused_velocity_limits):
                                is_scaled = steel_rows.dofs_drive_slot[i_d, i_b] < 0
                            if is_scaled:
                                if qd.static(i_pass == 0):
                                    limit = steel_info.dofs_velocity_limit[i_d]
                                    qd_abs = qd.abs(steel_dyn.dofs_qd_work[i_d, i_b])
                                    if limit > 0.0 and not qd.math.isinf(limit) and qd_abs > 0.0:
                                        ratio = qd.min(ratio, limit / qd_abs)
                                elif ratio < 1.0:
                                    steel_dyn.dofs_qd_work[i_d, i_b] = steel_dyn.dofs_qd_work[i_d, i_b] * ratio


@qd.func
def func_steel_fk_id(
    global_refresh: bool,
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Evaluate the link poses, the joint motion subspaces and the bias wrenches of every articulation (stage 1).

    An articulation refreshes its mass matrix on a global refresh step (global_refresh) or when a refresh was requested
    for it alone, which consumes the request. The spatial inertia of its links is written then, and always for free
    rigid bodies, whose free-body rows read it.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        is_active = True
        if qd.static(steel_config.sleep_skip_constraints):
            is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
        # A request of an articulation asleep for the step is dropped: waking it requests a refresh again.
        is_refresh = (global_refresh or steel_state.arts_mass_update_requested[i_a, i_b] != 0) and is_active
        steel_dyn.arts_mass_update_mask[i_a, i_b] = 1 if is_refresh else 0
        steel_state.arts_mass_update_requested[i_a, i_b] = 0
        if is_active:
            i_j_start = steel_model.arts_joint_start[i_a]
            i_j_end = steel_model.arts_joint_start[i_a + 1]
            for i_j in range(i_j_start, i_j_end):
                func_link_transform(i_j, i_b, dyn_state, steel_state, steel_dyn, dyn_info, steel_model, steel_config)
                i_l = steel_model.joints_link[i_j]
                steel_dyn.links_ft_lin[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                steel_dyn.links_ft_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)

            i_l_root = steel_model.joints_link[i_j_start]
            origin = steel_dyn.links_com_pos[i_l_root, i_b]
            steel_dyn.arts_origin[i_a, i_b] = origin
            is_write_inertia = is_refresh or steel_model.arts_is_free_rigid[i_a] != 0
            gravity = rigid_info.gravity[i_b]
            for i_j in range(i_j_start, i_j_end):
                i_l = steel_model.joints_link[i_j]
                i_p = steel_model.joints_parent_link[i_j]
                parent_v_lin = qd.Vector.zero(gs.qd_float, 3)
                parent_v_ang = qd.Vector.zero(gs.qd_float, 3)
                parent_a_lin = qd.Vector.zero(gs.qd_float, 3)
                parent_a_ang = qd.Vector.zero(gs.qd_float, 3)
                if i_p >= 0:
                    parent_v_lin = steel_dyn.links_v_lin[i_p, i_b]
                    parent_v_ang = steel_dyn.links_v_ang[i_p, i_b]
                    parent_a_lin = steel_dyn.links_a_lin[i_p, i_b]
                    parent_a_ang = steel_dyn.links_a_ang[i_p, i_b]

                # The twist of a free joint is referenced at the articulation origin, so its subspace takes the
                # rotation.
                wpj_pos, wpj_quat = func_parent_anchor_frame(
                    i_j, i_b, dyn_state, steel_state, dyn_info, steel_model, steel_config
                )
                x_pos = wpj_pos - origin
                if steel_model.joints_type[i_j] == STEEL_JOINT_TYPE.FREE:
                    x_pos = qd.Vector.zero(gs.qd_float, 3)
                vj_lin, vj_ang, c_lin, c_ang = func_joint_motion(
                    i_j, i_b, x_pos, wpj_quat, steel_dyn.dofs_qd_work, steel_state, steel_dyn, steel_model
                )
                v_lin = parent_v_lin + vj_lin
                v_ang = parent_v_ang + vj_ang
                x_lin, x_ang = func_motion_cross(v_lin, v_ang, vj_lin, vj_ang)
                a_lin = parent_a_lin + x_lin + c_lin
                a_ang = parent_a_ang + x_ang + c_ang
                steel_dyn.links_v_lin[i_l, i_b] = v_lin
                steel_dyn.links_v_ang[i_l, i_b] = v_ang
                steel_dyn.links_a_lin[i_l, i_b] = a_lin
                steel_dyn.links_a_ang[i_l, i_b] = a_ang

                mass, _, inertia = func_body_inertia(i_l, i_b, dyn_info, rigid_info, steel_config)
                com = steel_dyn.links_com_pos[i_l, i_b] - origin
                R = gu.qd_quat_to_R(steel_state.links_quat[i_l, i_b], rigid_info.EPS[None])
                com_cross = func_skew(com)
                inertia_o = R @ inertia @ R.transpose() - mass * com_cross @ com_cross
                if is_write_inertia:
                    steel_dyn.links_I_mass[i_l, i_b] = mass
                    steel_dyn.links_I_h[i_l, i_b] = mass * com
                    steel_dyn.links_I_o[i_l, i_b] = inertia_o
                f_g = mass * gravity
                # The root's linear inertial wrench is kept: the solve origin is a material point of the root body, so
                # it accelerates as the body rotates, and this term carries that acceleration.
                Iv_lin, Iv_ang = func_mul_com_inertia(mass, com, inertia_o, v_lin, v_ang)
                cor_lin, cor_ang = func_force_cross(v_lin, v_ang, Iv_lin, Iv_ang)
                Ia_lin, Ia_ang = func_mul_com_inertia(mass, com, inertia_o, a_lin, a_ang)
                steel_dyn.links_f_lin[i_l, i_b] = Ia_lin + cor_lin - f_g
                steel_dyn.links_f_ang[i_l, i_b] = Ia_ang + cor_ang - com.cross(f_g)


@qd.func
def func_body_net_wrench(i_l: int, i_b: int, origin: qd.types.vector(3), steel_dyn: array_class.SteelDynamics):
    """Return the net wrench about the origin of the subtree of link i_l, external wrench subtracted."""
    f_ext = steel_dyn.links_f_ext_lin[i_l, i_b]
    com_rel = steel_dyn.links_com_pos[i_l, i_b] - origin
    tau_origin = steel_dyn.links_f_ext_ang[i_l, i_b] + com_rel.cross(f_ext)
    return (
        steel_dyn.links_f_lin[i_l, i_b] + steel_dyn.links_ft_lin[i_l, i_b] - f_ext,
        steel_dyn.links_f_ang[i_l, i_b] + steel_dyn.links_ft_ang[i_l, i_b] - tau_origin,
    )


@qd.func
def func_drive_params(
    i_d: int,
    i_b: int,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_config: qd.template(),
):
    """Return the drive stiffness, damping, position target and velocity target of DOF i_d from the Genesis control.

    Position control drives with kp and kv towards the position and velocity targets, velocity control with kv towards
    the velocity target, force control has no drive. Position targets are relative to the neutral configuration.
    """
    I_d = [i_d, i_b] if qd.static(steel_config.batch_dofs_info) else i_d
    ctrl_mode = dyn_state.dofs.ctrl_mode[i_d, i_b]
    ke = gs.qd_float(0.0)
    kd = gs.qd_float(0.0)
    if ctrl_mode == gs.CTRL_MODE.POSITION:
        ke = dyn_info.dofs.act_gain[I_d]
    if ctrl_mode == gs.CTRL_MODE.POSITION or ctrl_mode == gs.CTRL_MODE.VELOCITY:
        kd = -dyn_info.dofs.act_bias[I_d][2]
    return ke, kd, dyn_state.dofs.ctrl_pos[i_d, i_b], dyn_state.dofs.ctrl_vel[i_d, i_b]


@qd.func
def func_steel_tau(
    dt: float,
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Accumulate the generalized forces of every articulation and prepare its implicit drives (stage 1).

    The backward pass of the inverse dynamics yields tau = applied + passive - bias, the right-hand side of
    H qdd = tau. With augmented drives, each driven DOF adds the explicit force u0 = -(ke (q - q* + dt qd) + kd (qd -
    v*)), clamped to the force range, and records K = ke dt^2 + kd dt, which joins the mass matrix diagonal: this is the
    backward-Euler drive f = u0 - K qdd. The subtree wrench transmitted through the inbound joint of every link is
    published as the parent force observable, about the link center of mass.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    _B = steel_state.q.shape[1]
    max_art_dofs = steel_dyn.aug_dof.shape[0] // n_arts
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        is_active = True
        if qd.static(steel_config.sleep_skip_constraints):
            is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
        if is_active:
            origin = steel_dyn.arts_origin[i_a, i_b]
            i_j_start = steel_model.arts_joint_start[i_a]
            i_j_end = steel_model.arts_joint_start[i_a + 1]
            for i_j_ in range(i_j_end - i_j_start):
                i_j = i_j_end - 1 - i_j_
                i_l = steel_model.joints_link[i_j]
                f_lin, f_ang = func_body_net_wrench(i_l, i_b, origin, steel_dyn)
                joint_type = steel_model.joints_type[i_j]
                i_d_start = steel_model.joints_dof_start[i_j]
                n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
                for i_d in range(i_d_start, i_d_start + n_dofs):
                    I_d = [i_d, i_b] if qd.static(steel_config.batch_dofs_info) else i_d
                    tau = -(steel_dyn.dofs_S_lin[i_d, i_b].dot(f_lin) + steel_dyn.dofs_S_ang[i_d, i_b].dot(f_ang))
                    tau = tau + steel_dyn.dofs_tau[i_d, i_b]
                    if joint_type != STEEL_JOINT_TYPE.FREE:
                        tau = tau - dyn_info.dofs.damping[I_d] * steel_state.qd[i_d, i_b]
                    steel_dyn.dofs_tau[i_d, i_b] = tau
                i_p = steel_model.joints_parent_link[i_j]
                if i_p >= 0:
                    steel_dyn.links_ft_lin[i_p, i_b] = steel_dyn.links_ft_lin[i_p, i_b] + f_lin
                    steel_dyn.links_ft_ang[i_p, i_b] = steel_dyn.links_ft_ang[i_p, i_b] + f_ang

            for i_j in range(i_j_start, i_j_end):
                i_l = steel_model.joints_link[i_j]
                f_lin, f_ang = func_body_net_wrench(i_l, i_b, origin, steel_dyn)
                com_rel = steel_dyn.links_com_pos[i_l, i_b] - origin
                steel_state.links_parent_f_lin[i_l, i_b] = f_lin
                steel_state.links_parent_f_ang[i_l, i_b] = f_ang - com_rel.cross(f_lin)

            n_aug = 0
            if qd.static(steel_config.drive_mode == 0):
                for i_j in range(i_j_start, i_j_end):
                    joint_type = steel_model.joints_type[i_j]
                    if (
                        joint_type == STEEL_JOINT_TYPE.PRISMATIC
                        or joint_type == STEEL_JOINT_TYPE.REVOLUTE
                        or joint_type == STEEL_JOINT_TYPE.D6
                    ):
                        i_q_start = steel_model.joints_q_start[i_j]
                        i_d_start = steel_model.joints_dof_start[i_j]
                        for k in range(steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]):
                            i_d = i_d_start + k
                            ke, kd, q_target, v_target = func_drive_params(
                                i_d, i_b, dyn_state, dyn_info, rigid_info, steel_config
                            )
                            K = ke * dt * dt + kd * dt
                            if (ke > 0.0 or kd > 0.0) and K > 0.0 and n_aug < max_art_dofs:
                                I_d = [i_d, i_b] if qd.static(steel_config.batch_dofs_info) else i_d
                                q = steel_state.q[i_q_start + k, i_b]
                                qd_ = steel_state.qd[i_d, i_b]
                                q_target = q_target - rigid_info.qpos0[i_q_start + k, i_b]
                                u0 = -(ke * (q - q_target + dt * qd_) + kd * (qd_ - v_target))
                                force_range = dyn_info.dofs.force_range[I_d]
                                u0 = qd.math.clamp(u0, force_range[0], force_range[1])
                                i_r = i_a * max_art_dofs + n_aug
                                steel_dyn.aug_dof[i_r, i_b] = i_d
                                steel_dyn.aug_K[i_r, i_b] = K
                                steel_dyn.dofs_tau[i_d, i_b] = steel_dyn.dofs_tau[i_d, i_b] + u0
                                n_aug = n_aug + 1
            steel_dyn.arts_aug_count[i_a, i_b] = n_aug


@qd.func
def func_steel_mass_matrix(
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    dyn_info: array_class.DynInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Assemble and factor the mass matrix of every articulation that refreshes it (stages 1 and 2).

    The composite rigid body algorithm gives H[i, j] = S_i . (I_c S_j) for the DOFs i of the ancestors of the joint of
    DOF j, I_c being the composite inertia of the subtree of that joint, and 0 between unrelated DOFs. The implicit
    drive terms K join the diagonal and the factorization adds the armature: L L^T = H + diag(K) + diag(armature).
    Articulations whose refresh is not due keep their previous factor.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    _B = steel_state.q.shape[1]
    n_dofs = steel_dyn.dofs_tau.shape[0]
    max_art_dofs = steel_dyn.aug_dof.shape[0] // n_arts

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        if steel_dyn.arts_mass_update_mask[i_a, i_b] != 0:
            i_j_start = steel_model.arts_joint_start[i_a]
            i_j_end = steel_model.arts_joint_start[i_a + 1]
            for i_j in range(i_j_start, i_j_end):
                i_l = steel_model.joints_link[i_j]
                steel_dyn.links_Ic_mass[i_l, i_b] = steel_dyn.links_I_mass[i_l, i_b]
                steel_dyn.links_Ic_h[i_l, i_b] = steel_dyn.links_I_h[i_l, i_b]
                steel_dyn.links_Ic_o[i_l, i_b] = steel_dyn.links_I_o[i_l, i_b]
            for i_j_ in range(i_j_end - i_j_start):
                i_j = i_j_end - 1 - i_j_
                i_k = steel_model.joints_ancestor[i_j]
                if i_k >= i_j_start:
                    i_l = steel_model.joints_link[i_j]
                    i_lk = steel_model.joints_link[i_k]
                    steel_dyn.links_Ic_mass[i_lk, i_b] = (
                        steel_dyn.links_Ic_mass[i_lk, i_b] + steel_dyn.links_Ic_mass[i_l, i_b]
                    )
                    steel_dyn.links_Ic_h[i_lk, i_b] = steel_dyn.links_Ic_h[i_lk, i_b] + steel_dyn.links_Ic_h[i_l, i_b]
                    steel_dyn.links_Ic_o[i_lk, i_b] = steel_dyn.links_Ic_o[i_lk, i_b] + steel_dyn.links_Ic_o[i_l, i_b]
            n = steel_model.arts_n_dofs[i_a]
            i_H = steel_model.arts_H_offset[i_a]
            for k in range(n * n):
                steel_dyn.H[i_H + k, i_b] = 0.0

    # One thread per column of every refreshed mass matrix. The thread of column c writes the lower-triangle entries
    # of the rows of the DOFs of the ancestors of its joint, and those of its own joint at or below c.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        i_j_pivot = steel_model.dofs_joint[i_d]
        if i_j_pivot >= 0:
            i_a = steel_model.joints_art[i_j_pivot]
            if steel_dyn.arts_mass_update_mask[i_a, i_b] != 0:
                n = steel_model.arts_n_dofs[i_a]
                i_H = steel_model.arts_H_offset[i_a]
                dof_start = steel_model.arts_dof_start[i_a]
                i_j_start = steel_model.arts_joint_start[i_a]
                i_l = steel_model.joints_link[i_j_pivot]
                F_lin, F_ang = func_mul_spatial_inertia(
                    steel_dyn.links_Ic_mass[i_l, i_b],
                    steel_dyn.links_Ic_h[i_l, i_b],
                    steel_dyn.links_Ic_o[i_l, i_b],
                    steel_dyn.dofs_S_lin[i_d, i_b],
                    steel_dyn.dofs_S_ang[i_d, i_b],
                )
                col = i_d - dof_start
                i_j = i_j_pivot
                for _ in range(steel_model.arts_joint_start[i_a + 1] - i_j_start):
                    if i_j >= i_j_start:
                        i_d_row_start = steel_model.joints_dof_start[i_j]
                        for k in range(steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]):
                            row = i_d_row_start + k - dof_start
                            if i_j != i_j_pivot or row >= col:
                                value = steel_dyn.dofs_S_lin[i_d_row_start + k, i_b].dot(F_lin) + steel_dyn.dofs_S_ang[
                                    i_d_row_start + k, i_b
                                ].dot(F_ang)
                                steel_dyn.H[i_H + qd.max(row, col) * n + qd.min(row, col), i_b] = value
                        i_j = steel_model.joints_ancestor[i_j]

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        if steel_dyn.arts_mass_update_mask[i_a, i_b] != 0:
            n = steel_model.arts_n_dofs[i_a]
            i_H = steel_model.arts_H_offset[i_a]
            dof_start = steel_model.arts_dof_start[i_a]
            for i_r_ in range(steel_dyn.arts_aug_count[i_a, i_b]):
                i_r = i_a * max_art_dofs + i_r_
                local = steel_dyn.aug_dof[i_r, i_b] - dof_start
                K = steel_dyn.aug_K[i_r, i_b]
                if local >= 0 and local < n and K > 0.0:
                    steel_dyn.H[i_H + local * n + local, i_b] = steel_dyn.H[i_H + local * n + local, i_b] + K
            # Cholesky factorization L L^T = H + diag(armature), column by column.
            for j in range(n):
                I_d = [dof_start + j, i_b] if qd.static(steel_config.batch_dofs_info) else dof_start + j
                s = steel_dyn.H[i_H + j * n + j, i_b] + dyn_info.dofs.armature[I_d]
                for k in range(j):
                    r = steel_dyn.L[i_H + j * n + k, i_b]
                    s = s - r * r
                s = qd.sqrt(s)
                inv_s = 1.0 / s
                steel_dyn.L[i_H + j * n + j, i_b] = s
                for i in range(j + 1, n):
                    v = steel_dyn.H[i_H + i * n + j, i_b]
                    for k in range(j):
                        v = v - steel_dyn.L[i_H + i * n + k, i_b] * steel_dyn.L[i_H + j * n + k, i_b]
                    steel_dyn.L[i_H + i * n + j, i_b] = v * inv_s


@qd.func
def func_gyroscopic_velocity(
    inertia: qd.types.matrix(3, 3),
    effective_inertia: qd.types.matrix(3, 3),
    omega: qd.types.vector(3),
    predicted: qd.types.vector(3),
    dt: float,
    steel_info: array_class.SteelInfo,
):
    """Replace the explicit gyroscopic kick of a free body by energy-preserving Cayley updates.

    Every update solves (A - S) w = (A + S) u with A symmetric positive definite and S skew, so w.A w = u.A u whatever
    the fixed-point convergence. S is updated from the midpoint, which approximates the implicit midpoint rule. The
    number of microsteps follows an energy bound of the angular speed.
    """
    scale = qd.max(effective_inertia[0, 0], qd.max(effective_inertia[1, 1], effective_inertia[2, 2]))
    a = effective_inertia / scale
    physical = inertia / scale
    inverse = a.inverse()
    u = predicted + dt * (inverse @ omega.cross(physical @ omega))
    speed_bound = qd.sqrt(qd.max(u.dot(a @ u) * inverse.trace(), 0.0))
    n_microsteps = qd.cast(
        qd.math.clamp(qd.ceil(2.0 * qd.abs(dt) * speed_bound), 1.0, steel_info.gyroscopic_max_microsteps[None]), qd.i32
    )
    h = dt / n_microsteps
    w = u
    for _ in range(n_microsteps):
        u = w
        energy = u.dot(a @ u)
        for _iter in range(steel_info.gyroscopic_max_iterations[None]):
            s = (0.5 * h) * func_skew(physical @ (0.5 * (u + w)))
            candidate = (a - s).inverse() @ ((a + s) @ u)
            delta = candidate - w
            w = candidate
            if delta.dot(a @ delta) <= steel_info.gyroscopic_tolerance[None] * energy:
                break
    return w


@qd.func
def func_steel_unconstrained(
    dt: float,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Solve the unconstrained acceleration and predict the velocity the integrator realizes (stage 3).

    The linear acceleration of a free root is that of the solve origin, a material point of the root, whose classical
    acceleration adds omega x v: the integrator realizes it, so the predictor does too. A free rigid body also replaces
    the explicit gyroscopic kick of its prediction by an energy-preserving update.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        is_active = True
        if qd.static(steel_config.sleep_skip_constraints):
            is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
        if is_active:
            n = steel_model.arts_n_dofs[i_a]
            i_H = steel_model.arts_H_offset[i_a]
            dof_start = steel_model.arts_dof_start[i_a]
            for i in range(n):
                v = steel_dyn.dofs_tau[dof_start + i, i_b]
                for k in range(i):
                    v = v - steel_dyn.L[i_H + i * n + k, i_b] * steel_state.qdd[dof_start + k, i_b]
                L_ii = steel_dyn.L[i_H + i * n + i, i_b]
                steel_state.qdd[dof_start + i, i_b] = v / L_ii if L_ii != 0.0 else 0.0
            for i_ in range(n):
                i = n - 1 - i_
                v = steel_state.qdd[dof_start + i, i_b]
                for k in range(i + 1, n):
                    v = v - steel_dyn.L[i_H + k * n + i, i_b] * steel_state.qdd[dof_start + k, i_b]
                L_ii = steel_dyn.L[i_H + i * n + i, i_b]
                steel_state.qdd[dof_start + i, i_b] = v / L_ii if L_ii != 0.0 else 0.0
            for i in range(n):
                i_d = dof_start + i
                steel_dyn.v_hat[i_d, i_b] = steel_dyn.dofs_qd_work[i_d, i_b] + steel_state.qdd[i_d, i_b] * dt

    if qd.static(steel_config.has_free_roots):
        n_roots = steel_model.free_root_joints.shape[0]
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_r, i_b in qd.ndrange(n_roots, _B):
            i_j = steel_model.free_root_joints[i_r]
            i_a = steel_model.joints_art[i_j]
            is_active = True
            if qd.static(steel_config.sleep_skip_constraints):
                is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
            if is_active:
                i_d = steel_model.joints_dof_start[i_j]
                v = qd.Vector([steel_dyn.dofs_qd_work[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
                w = qd.Vector([steel_dyn.dofs_qd_work[i_d + 3 + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
                c = w.cross(v)
                for k in qd.static(range(3)):
                    steel_dyn.v_hat[i_d + k, i_b] = steel_dyn.v_hat[i_d + k, i_b] + c[k] * dt

                i_a = steel_model.joints_art[i_j]
                if steel_model.arts_is_free_rigid[i_a] != 0:
                    i_l = steel_model.joints_link[i_j]
                    predicted_world = qd.Vector(
                        [steel_dyn.v_hat[i_d + 3 + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float
                    )
                    _, _, inertia = func_body_inertia(i_l, i_b, dyn_info, rigid_info, steel_config)
                    is_stationary = (w == 0.0).all() and (predicted_world == 0.0).all()
                    # Isotropic inertia has an identically zero gyroscopic bias, so the predictor stays exact.
                    is_isotropic = (
                        inertia[0, 0] == inertia[1, 1]
                        and inertia[1, 1] == inertia[2, 2]
                        and inertia[0, 1] == 0.0
                        and inertia[0, 2] == 0.0
                        and inertia[1, 0] == 0.0
                        and inertia[1, 2] == 0.0
                        and inertia[2, 0] == 0.0
                        and inertia[2, 1] == 0.0
                    )
                    if not is_stationary and not is_isotropic:
                        # At the root center of mass translation and rotation decouple: the angular block of the factor
                        # holds the rotational inertia and the armature.
                        i_H = steel_model.arts_H_offset[i_a]
                        lower = qd.Matrix.zero(gs.qd_float, 3, 3)
                        for r in qd.static(range(3)):
                            for c_ in qd.static(range(r + 1)):
                                lower[r, c_] = steel_dyn.L[i_H + (r + 3) * 6 + c_ + 3, i_b]
                        quat = steel_state.links_quat[i_l, i_b]
                        basis = gu.qd_quat_to_R(quat, rigid_info.EPS[None])
                        effective = basis.transpose() @ (lower @ lower.transpose()) @ basis
                        omega = gu.qd_inv_transform_by_quat(w, quat)
                        predicted = gu.qd_inv_transform_by_quat(predicted_world, quat)
                        corrected = gu.qd_transform_by_quat(
                            func_gyroscopic_velocity(inertia, effective, omega, predicted, dt, steel_info), quat
                        )
                        for k in qd.static(range(3)):
                            steel_dyn.v_hat[i_d + 3 + k, i_b] = corrected[k]


@qd.func
def func_integrate_joint(
    i_j: int,
    i_b: int,
    dt: float,
    steel_state: array_class.SteelState,
    dyn_info: array_class.DynInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Advance the coordinates and velocities of joint i_j by one step of symplectic Euler.

    A free root integrates its center of mass exactly and places its anchor at x_com - R' r_ac, so a force-free center
    of mass stays put to roundoff. Its linear acceleration is that of the origin, which adds the transport term
    omega x v. Angular damping scales its angular velocity after the rotation update.
    """
    joint_type = steel_model.joints_type[i_j]
    i_q = steel_model.joints_q_start[i_j]
    i_d = steel_model.joints_dof_start[i_j]
    if (
        joint_type == STEEL_JOINT_TYPE.PRISMATIC
        or joint_type == STEEL_JOINT_TYPE.REVOLUTE
        or joint_type == STEEL_JOINT_TYPE.D6
    ):
        for k in range(steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]):
            qd_new = steel_state.qd[i_d + k, i_b] + steel_state.qdd[i_d + k, i_b] * dt
            steel_state.qd[i_d + k, i_b] = qd_new
            steel_state.q[i_q + k, i_b] = steel_state.q[i_q + k, i_b] + qd_new * dt
    elif joint_type == STEEL_JOINT_TYPE.BALL:
        w = qd.Vector(
            [steel_state.qd[i_d + k, i_b] + steel_state.qdd[i_d + k, i_b] * dt for k in qd.static(range(3))],
            dt=gs.qd_float,
        )
        r = qd.Vector([steel_state.q[i_q + k, i_b] for k in qd.static(range(4))], dt=gs.qd_float)
        drdt = gu.qd_quat_mul(qd.Vector([0.0, w[0], w[1], w[2]], dt=gs.qd_float), r) * 0.5
        r_new = (r + drdt * dt).normalized()
        for k in qd.static(range(4)):
            steel_state.q[i_q + k, i_b] = r_new[k]
        for k in qd.static(range(3)):
            steel_state.qd[i_d + k, i_b] = w[k]
    elif joint_type == STEEL_JOINT_TYPE.FREE:
        i_l = steel_model.joints_link[i_j]
        a = qd.Vector([steel_state.qdd[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
        alpha = qd.Vector([steel_state.qdd[i_d + 3 + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
        v = qd.Vector([steel_state.qd[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
        w = qd.Vector([steel_state.qd[i_d + 3 + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
        w_prev = w
        w = w + alpha * dt
        v = v + (a + w_prev.cross(v)) * dt
        p = qd.Vector([steel_state.q[i_q + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
        r = qd.Vector([steel_state.q[i_q + 3 + k, i_b] for k in qd.static(range(4))], dt=gs.qd_float)
        # The coordinates track the child anchor frame, whose lever to the center of mass goes through its offset.
        I_l = [i_l, i_b] if qd.static(steel_config.batch_links_info) else i_l
        r_ac = dyn_info.links.inertial_pos[I_l] - func_joint_anchor(i_j, i_b, dyn_info, steel_model, steel_config)
        drdt = gu.qd_quat_mul(qd.Vector([0.0, w[0], w[1], w[2]], dt=gs.qd_float), r) * 0.5
        r_new = (r + drdt * dt).normalized()
        x_com_new = p + gu.qd_transform_by_quat(r_ac, r) + v * dt
        p_new = x_com_new - gu.qd_transform_by_quat(r_ac, r_new)
        w = w * (1.0 - steel_info.links_angular_damping[i_l, i_b] * dt)
        for k in qd.static(range(3)):
            steel_state.q[i_q + k, i_b] = p_new[k]
            steel_state.qd[i_d + k, i_b] = v[k]
            steel_state.qd[i_d + 3 + k, i_b] = w[k]
        for k in qd.static(range(4)):
            steel_state.q[i_q + 3 + k, i_b] = r_new[k]


@qd.func
def func_joint_velocity_in_anchor_frame(
    i_j: int, i_b: int, steel_state: array_class.SteelState, steel_model: array_class.SteelModelInfo
):
    """Return the twist across joint i_j in its parent anchor axes (child center of mass twist for a free joint)."""
    joint_type = steel_model.joints_type[i_j]
    i_d = steel_model.joints_dof_start[i_j]
    v_lin = qd.Vector.zero(gs.qd_float, 3)
    v_ang = qd.Vector.zero(gs.qd_float, 3)
    if joint_type == STEEL_JOINT_TYPE.PRISMATIC:
        v_lin = steel_model.dofs_axis[i_d] * steel_state.qd[i_d, i_b]
    elif joint_type == STEEL_JOINT_TYPE.REVOLUTE:
        v_ang = steel_model.dofs_axis[i_d] * steel_state.qd[i_d, i_b]
    elif joint_type == STEEL_JOINT_TYPE.BALL:
        v_ang = qd.Vector([steel_state.qd[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
    elif joint_type == STEEL_JOINT_TYPE.FREE:
        v_lin = qd.Vector([steel_state.qd[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
        v_ang = qd.Vector([steel_state.qd[i_d + 3 + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
    elif joint_type == STEEL_JOINT_TYPE.D6:
        n_lin = steel_model.joints_n_lin[i_j]
        for k in range(n_lin):
            v_lin = v_lin + steel_model.dofs_axis[i_d + k] * steel_state.qd[i_d + k, i_b]
        axes, _ = func_d6_angular_axes(i_j, i_b, steel_state, steel_model)
        for k in range(steel_model.joints_n_ang[i_j]):
            v_ang = v_ang + axes[k, :] * steel_state.qd[i_d + n_lin + k, i_b]
    return v_lin, v_ang


@qd.func
def func_eval_fk_articulation(
    i_a: int,
    i_b: int,
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    dyn_info: array_class.DynInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Evaluate the poses and center of mass twists of the links of articulation i_a from its joint state.

    The twists recur from parent to child through the link origins: the child origin moves with the parent at that
    point plus the joint twist, whose angular part also sweeps the offset between the child anchor and origin. The
    velocity of a free joint is that of the child center of mass.
    """
    for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
        i_l = steel_model.joints_link[i_j]
        i_p = steel_model.joints_parent_link[i_j]
        func_link_transform(i_j, i_b, dyn_state, steel_state, steel_dyn, dyn_info, steel_model, steel_config)
        wpj_pos, wpj_quat = func_parent_anchor_frame(
            i_j, i_b, dyn_state, steel_state, dyn_info, steel_model, steel_config
        )
        x_child_origin = steel_state.links_pos[i_l, i_b]
        v_parent_origin = qd.Vector.zero(gs.qd_float, 3)
        w_parent = qd.Vector.zero(gs.qd_float, 3)
        if i_p >= 0:
            w_parent = steel_state.links_ang[i_p, i_b]
            v_parent_origin = steel_state.links_vel[i_p, i_b] + w_parent.cross(
                x_child_origin - steel_dyn.links_com_pos[i_p, i_b]
            )
        vj_lin, vj_ang = func_joint_velocity_in_anchor_frame(i_j, i_b, steel_state, steel_model)
        linear_joint_world = gu.qd_transform_by_quat(vj_lin, wpj_quat)
        angular_joint_world = gu.qd_transform_by_quat(vj_ang, wpj_quat)
        linear_joint_origin = qd.Vector.zero(gs.qd_float, 3)
        if steel_model.joints_type[i_j] == STEEL_JOINT_TYPE.FREE:
            linear_joint_origin = linear_joint_world - angular_joint_world.cross(
                steel_dyn.links_com_pos[i_l, i_b] - x_child_origin
            )
        else:
            anchor = func_joint_anchor(i_j, i_b, dyn_info, steel_model, steel_config)
            child_anchor = x_child_origin + gu.qd_transform_by_quat(anchor, steel_state.links_quat[i_l, i_b])
            linear_joint_origin = linear_joint_world + angular_joint_world.cross(x_child_origin - child_anchor)
        w_child = w_parent + angular_joint_world
        steel_state.links_ang[i_l, i_b] = w_child
        steel_state.links_vel[i_l, i_b] = (
            v_parent_origin + linear_joint_origin + w_child.cross(steel_dyn.links_com_pos[i_l, i_b] - x_child_origin)
        )


@qd.func
def func_steel_inputs(
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    dyn_info: array_class.DynInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Gather the external wrenches and the force controls of the step in Steel conventions.

    A Genesis external wrench is stored negated and about the center of mass of the kinematic root of its link, Steel
    takes the force at the link center of mass and the torque. A force control of a spherical joint is a torque in the
    child frame, Steel's in the parent anchor axes. A force control of a free joint is a force and a body-frame torque
    dual to the velocity of the link origin, Steel's is dual to the velocity of the center of mass.
    """
    n_links = dyn_state.links.pos.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        force = -(dyn_state.links.cfrc_applied_vel[i_l, i_b] + dyn_state.links.cfrc_coupling_vel[i_l, i_b])
        torque = -(dyn_state.links.cfrc_applied_ang[i_l, i_b] + dyn_state.links.cfrc_coupling_ang[i_l, i_b])
        steel_dyn.links_f_ext_lin[i_l, i_b] = force
        steel_dyn.links_f_ext_ang[i_l, i_b] = torque - dyn_state.links.i_pos[i_l, i_b].cross(force)

    n_joints = steel_model.joints_link.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_b in qd.ndrange(n_joints, _B):
        joint_type = steel_model.joints_type[i_j]
        i_d = steel_model.joints_dof_start[i_j]
        i_q = steel_model.joints_q_start[i_j]
        n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
        for k in range(n_dofs):
            I_d = [i_d + k, i_b] if qd.static(steel_config.batch_dofs_info) else i_d + k
            force_range = dyn_info.dofs.force_range[I_d]
            joint_f = gs.qd_float(0.0)
            if dyn_state.dofs.ctrl_mode[i_d + k, i_b] == gs.CTRL_MODE.FORCE:
                joint_f = qd.math.clamp(dyn_state.dofs.ctrl_force[i_d + k, i_b], force_range[0], force_range[1])
            steel_dyn.dofs_tau[i_d + k, i_b] = joint_f
            steel_dyn.dofs_qd_prev[i_d + k, i_b] = steel_state.qd[i_d + k, i_b]
            steel_dyn.dofs_qd_work[i_d + k, i_b] = steel_state.qd[i_d + k, i_b]
        if joint_type == STEEL_JOINT_TYPE.BALL:
            quat = qd.Vector([steel_state.q[i_q + k, i_b] for k in qd.static(range(4))], dt=gs.qd_float)
            torque = qd.Vector([steel_dyn.dofs_tau[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
            torque = gu.qd_transform_by_quat(torque, quat)
            for k in qd.static(range(3)):
                steel_dyn.dofs_tau[i_d + k, i_b] = torque[k]
        elif joint_type == STEEL_JOINT_TYPE.FREE:
            i_l = steel_model.joints_link[i_j]
            I_l = [i_l, i_b] if qd.static(steel_config.batch_links_info) else i_l
            quat = qd.Vector([steel_state.q[i_q + 3 + k, i_b] for k in qd.static(range(4))], dt=gs.qd_float)
            force = qd.Vector([steel_dyn.dofs_tau[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
            torque = qd.Vector([steel_dyn.dofs_tau[i_d + 3 + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
            com_arm = gu.qd_transform_by_quat(dyn_info.links.inertial_pos[I_l], quat)
            torque = gu.qd_transform_by_quat(torque, quat) - com_arm.cross(force)
            for k in qd.static(range(3)):
                steel_dyn.dofs_tau[i_d + 3 + k, i_b] = torque[k]


@qd.func
def func_steel_integrate(
    dt: float,
    velocity: qd.Tensor,
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    dyn_info: array_class.DynInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Integrate the joint state from the constrained velocity held in velocity (stage 7).

    qdd = (velocity - qd) / dt, minus the transport term of free roots, makes the integrator reproduce the constrained
    velocity exactly.
    """
    n_dofs = steel_state.qd.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        i_j = steel_model.dofs_joint[i_d]
        i_a = steel_model.joints_art[i_j]
        is_active = True
        if qd.static(steel_config.sleep_skip_constraints):
            is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
        if is_active:
            steel_state.qdd[i_d, i_b] = (velocity[i_d, i_b] - steel_state.qd[i_d, i_b]) * (1.0 / dt)

    if qd.static(steel_config.has_free_roots):
        n_roots = steel_model.free_root_joints.shape[0]
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_r, i_b in qd.ndrange(n_roots, _B):
            i_j = steel_model.free_root_joints[i_r]
            i_d = steel_model.joints_dof_start[i_j]
            i_a = steel_model.joints_art[i_j]
            is_active = True
            if qd.static(steel_config.sleep_skip_constraints):
                is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
            if is_active:
                v = qd.Vector([steel_state.qd[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
                w = qd.Vector([steel_state.qd[i_d + 3 + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
                c = w.cross(v)
                for k in qd.static(range(3)):
                    steel_state.qdd[i_d + k, i_b] = steel_state.qdd[i_d + k, i_b] - c[k]

    n_joints = steel_model.joints_link.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_b in qd.ndrange(n_joints, _B):
        i_a = steel_model.joints_art[i_j]
        is_active = True
        if qd.static(steel_config.sleep_skip_constraints):
            is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
        if is_active:
            func_integrate_joint(i_j, i_b, dt, steel_state, dyn_info, steel_info, steel_model, steel_config)


@qd.func
def func_steel_publish_refined_velocity(
    dt: float,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Publish the velocity refined by the velocity-only sweeps, the positions following the position solve.

    The acceleration is the velocity increment of the step over dt (the free-root transport term not removed), and the
    angular velocity of free roots decays as the integrator damped that of the position solve.
    """
    n_dofs = steel_state.qd.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        steel_state.qdd[i_d, i_b] = (steel_dyn.v_out[i_d, i_b] - steel_dyn.dofs_qd_prev[i_d, i_b]) * (1.0 / dt)

    if qd.static(steel_config.has_free_roots):
        n_roots = steel_model.free_root_joints.shape[0]
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_r, i_b in qd.ndrange(n_roots, _B):
            i_j = steel_model.free_root_joints[i_r]
            i_d = steel_model.joints_dof_start[i_j]
            i_l = steel_model.joints_link[i_j]
            scale = 1.0 - steel_info.links_angular_damping[i_l, i_b] * dt
            for k in qd.static(range(3, 6)):
                steel_dyn.v_out[i_d + k, i_b] = steel_dyn.v_out[i_d + k, i_b] * scale

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        steel_state.qd[i_d, i_b] = steel_dyn.v_out[i_d, i_b]


@qd.func
def func_steel_forward_kinematics(
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    dyn_info: array_class.DynInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Evaluate the link poses and twists of the new joint state, keeping those of the start of the step."""
    n_arts = steel_model.arts_dof_start.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        is_active = True
        if qd.static(steel_config.sleep_skip_constraints):
            is_active = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
        if is_active:
            for i_j in range(steel_model.arts_joint_start[i_a], steel_model.arts_joint_start[i_a + 1]):
                i_l = steel_model.joints_link[i_j]
                steel_state.links_pos_prev[i_l, i_b] = steel_state.links_pos[i_l, i_b]
                steel_state.links_quat_prev[i_l, i_b] = steel_state.links_quat[i_l, i_b]
            func_eval_fk_articulation(i_a, i_b, dyn_state, steel_state, steel_dyn, dyn_info, steel_model, steel_config)


@qd.func
def func_steel_publish(
    dt: float,
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Write the joint state of the step to the Genesis coordinates, velocities and accelerations.

    Scalar coordinates add the neutral configuration back. A spherical joint velocity is expressed in the child frame.
    A free joint velocity is the velocity of the link origin and the angular velocity in the link frame. Accelerations
    are the velocity increments of the step over dt.
    """
    n_joints = steel_model.joints_link.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_b in qd.ndrange(n_joints, _B):
        joint_type = steel_model.joints_type[i_j]
        i_d = steel_model.joints_dof_start[i_j]
        i_q = steel_model.joints_q_start[i_j]
        n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
        if joint_type == STEEL_JOINT_TYPE.BALL or joint_type == STEEL_JOINT_TYPE.FREE:
            i_q_quat = i_q + 3 if joint_type == STEEL_JOINT_TYPE.FREE else i_q
            quat = qd.Vector([steel_state.q[i_q_quat + k, i_b] for k in qd.static(range(4))], dt=gs.qd_float)
            for k in range(i_q_quat + 4 - i_q):
                rigid_info.qpos[i_q + k, i_b] = steel_state.q[i_q + k, i_b]
            i_d_ang = i_d + 3 if joint_type == STEEL_JOINT_TYPE.FREE else i_d
            ang = qd.Vector([steel_state.qd[i_d_ang + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
            ang_child = gu.qd_inv_transform_by_quat(ang, quat)
            if joint_type == STEEL_JOINT_TYPE.FREE:
                i_l = steel_model.joints_link[i_j]
                I_l = [i_l, i_b] if qd.static(steel_config.batch_links_info) else i_l
                v_com = qd.Vector([steel_state.qd[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
                v_origin = v_com - ang.cross(gu.qd_transform_by_quat(dyn_info.links.inertial_pos[I_l], quat))
                for k in qd.static(range(3)):
                    dyn_state.dofs.acc[i_d + k, i_b] = (v_origin[k] - dyn_state.dofs.vel[i_d + k, i_b]) / dt
                    dyn_state.dofs.vel[i_d + k, i_b] = v_origin[k]
            for k in qd.static(range(3)):
                dyn_state.dofs.acc[i_d_ang + k, i_b] = (ang_child[k] - dyn_state.dofs.vel[i_d_ang + k, i_b]) / dt
                dyn_state.dofs.vel[i_d_ang + k, i_b] = ang_child[k]
        else:
            for k in range(n_dofs):
                rigid_info.qpos[i_q + k, i_b] = steel_state.q[i_q + k, i_b] + rigid_info.qpos0[i_q + k, i_b]
                vel = steel_state.qd[i_d + k, i_b]
                dyn_state.dofs.acc[i_d + k, i_b] = (vel - dyn_state.dofs.vel[i_d + k, i_b]) / dt
                dyn_state.dofs.vel[i_d + k, i_b] = vel


@qd.kernel(fastcache=True)
def kernel_steel_pull(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
):
    """Load the joint state of the selected environments from the Genesis coordinates and velocities.

    The inverse of func_steel_publish. The link poses and twists are evaluated from it.
    """
    n_joints = steel_model.joints_link.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_b_ in qd.ndrange(n_joints, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        joint_type = steel_model.joints_type[i_j]
        i_d = steel_model.joints_dof_start[i_j]
        i_q = steel_model.joints_q_start[i_j]
        n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
        if joint_type == STEEL_JOINT_TYPE.BALL or joint_type == STEEL_JOINT_TYPE.FREE:
            i_q_quat = i_q + 3 if joint_type == STEEL_JOINT_TYPE.FREE else i_q
            quat = qd.Vector([rigid_info.qpos[i_q_quat + k, i_b] for k in qd.static(range(4))], dt=gs.qd_float)
            quat = quat.normalized()
            for k in range(i_q_quat - i_q):
                steel_state.q[i_q + k, i_b] = rigid_info.qpos[i_q + k, i_b]
            for k in qd.static(range(4)):
                steel_state.q[i_q_quat + k, i_b] = quat[k]
            i_d_ang = i_d + 3 if joint_type == STEEL_JOINT_TYPE.FREE else i_d
            ang_child = qd.Vector([dyn_state.dofs.vel[i_d_ang + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
            ang = gu.qd_transform_by_quat(ang_child, quat)
            if joint_type == STEEL_JOINT_TYPE.FREE:
                i_l = steel_model.joints_link[i_j]
                I_l = [i_l, i_b] if qd.static(steel_config.batch_links_info) else i_l
                v_origin = qd.Vector([dyn_state.dofs.vel[i_d + k, i_b] for k in qd.static(range(3))], dt=gs.qd_float)
                v_com = v_origin + ang.cross(gu.qd_transform_by_quat(dyn_info.links.inertial_pos[I_l], quat))
                for k in qd.static(range(3)):
                    steel_state.qd[i_d + k, i_b] = v_com[k]
            for k in qd.static(range(3)):
                steel_state.qd[i_d_ang + k, i_b] = ang[k]
        else:
            for k in range(n_dofs):
                steel_state.q[i_q + k, i_b] = rigid_info.qpos[i_q + k, i_b] - rigid_info.qpos0[i_q + k, i_b]
                steel_state.qd[i_d + k, i_b] = dyn_state.dofs.vel[i_d + k, i_b]

    n_arts = steel_model.arts_dof_start.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b_ in qd.ndrange(n_arts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        func_eval_fk_articulation(i_a, i_b, dyn_state, steel_state, steel_dyn, dyn_info, steel_model, steel_config)
