"""Passive-island sleeping of the Steel solver (option 'enable_sleeping').

Every step, the articulations of each environment are joined into islands by the contacts between their links, with a
union-find whose roots are the smallest articulation of each island, so the islands do not depend on the contact order.
Static geometry supports an island without joining it, as does the fixed root joint of an articulation. A supported
island whose links all stay below the speed thresholds for 'sleep_quiet_time' falls asleep: its published state is
frozen (coordinates of the step start, zero velocities, the last parent wrench) and, with 'sleep_skip_constraints',
its rows and articulated dynamics are skipped. A sleeping island wakes on a wake event: a non-zero or non-finite
external wrench or force control, a drive, a contact with an awake island, lost support, dropped rows or contacts,
a mimic joint or loop closure, or a state or model change notified by a Genesis setter (see
'SteelSolver._on_state_change' and 'SteelSolver._on_model_change'). A wake crosses the islands of the previous step:
an asleep articulation whose previous island holds a woken member wakes too, even when their contact is gone.

Friction patches carry the history of the contact pairs between links asleep through the current and the previous
step (links_is_frozen, see SteelSleep in array_class.py), whose contacts repeat, and rebuild the history of every other
pair. A Genesis state change of an environment wakes all its islands, so the history of their pairs is rebuilt from
the next step on.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

from .dynamics import func_drive_params
from .model import STEEL_JOINT_TYPE
from .rows import func_contact_points_world, func_is_scalar_axis_joint


@qd.func
def func_island_root(i_a: int, i_b: int, arts_parent: qd.Tensor):
    """Return the root of the island of articulation i_a in the union-find forest arts_parent."""
    i_r = i_a
    # A path visits each articulation once at most.
    for _ in range(arts_parent.shape[0]):
        if arts_parent[i_r, i_b] == i_r:
            break
        i_r = arts_parent[i_r, i_b]
    return i_r


@qd.func
def func_island_union(i_a0: int, i_a1: int, i_b: int, arts_parent: qd.Tensor):
    """Join the islands of two articulations, the larger root pointing to the smaller one."""
    i_r0 = func_island_root(i_a0, i_b, arts_parent)
    i_r1 = func_island_root(i_a1, i_b, arts_parent)
    if i_r0 < i_r1:
        arts_parent[i_r1, i_b] = i_r0
    elif i_r1 < i_r0:
        arts_parent[i_r0, i_b] = i_r1


@qd.func
def func_steel_sleep_begin(
    dyn_state: array_class.DynState,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    steel_status: array_class.SteelStatus,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    steel_info: array_class.SteelInfo,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
    errno: qd.Tensor,
):
    """Build the islands of the step from its contacts, propagate the wake events and select the sleeping islands.

    Runs after the contacts and the external inputs of the step are gathered and before its dynamics. An island wakes
    when it is vetoed (a wake event touches one of its members) or when it joins an awake articulation, and every
    island ends up all awake or all asleep. The articulations asleep for the step skip their dynamics and rows when
    'sleep_skip_constraints' is set, an articulation woken from sleep refreshing its mass matrix.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    n_links = steel_model.links_art.shape[0]
    n_dofs = steel_model.dofs_joint.shape[0]
    n_qs = steel_sleep.q_start.shape[0]
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_state.q.shape[1]

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q, i_b in qd.ndrange(n_qs, _B):
        steel_sleep.q_start[i_q, i_b] = steel_state.q[i_q, i_b]

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        steel_sleep.arts_parent[i_a, i_b] = i_a
        steel_sleep.roots_has_awake[i_a, i_b] = 0
        steel_sleep.roots_is_vetoed[i_a, i_b] = 0
        steel_sleep.roots_is_supported[i_a, i_b] = 0
        steel_sleep.roots_is_ready[i_a, i_b] = 1
        steel_sleep.roots_has_old_wake[i_a, i_b] = 0

    # Collision errors of an environment whose contacts are incomplete: candidate pairs or contacts beyond the buffers.
    contact_overflow_errno = (
        array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS
        | array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS
        | array_class.ErrorCode.OVERFLOW_CONTACTS
    )
    # The union-find of an environment is serial, so it is deterministic and needs no atomic.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        steel_sleep.envs_is_incomplete[i_b] = 1 if (errno[i_b] & contact_overflow_errno) != 0 else 0
        for i_c in range(steel_contacts.n[i_b]):
            i_l0 = steel_contacts.body0[i_c, i_b]
            i_l1 = steel_contacts.body1[i_c, i_b]
            if i_l0 >= 0 and i_l1 >= 0:
                i_a0 = steel_model.links_art[i_l0]
                i_a1 = steel_model.links_art[i_l1]
                func_island_union(i_a0, i_a1, i_b, steel_sleep.arts_parent)
        for i_a in range(n_arts):
            steel_sleep.arts_parent[i_a, i_b] = func_island_root(i_a, i_b, steel_sleep.arts_parent)
        # The wake forest also joins every asleep articulation to its island of the previous step.
        for i_a in range(n_arts):
            steel_sleep.arts_wake_parent[i_a, i_b] = steel_sleep.arts_parent[i_a, i_b]
        for i_a in range(n_arts):
            if steel_sleep.arts_is_awake[i_a, i_b] == 0:
                i_p = steel_sleep.arts_previous_root[i_a, i_b]
                func_island_union(i_a, i_p, i_b, steel_sleep.arts_wake_parent)
        for i_a in range(n_arts):
            steel_sleep.arts_wake_parent[i_a, i_b] = func_island_root(i_a, i_b, steel_sleep.arts_wake_parent)

    # Pending state and model changes are consumed here, mimic joints and loop closures keep their island awake.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        i_r = steel_sleep.arts_parent[i_a, i_b]
        qd.atomic_max(steel_sleep.roots_has_awake[i_r, i_b], steel_sleep.arts_is_awake[i_a, i_b])
        qd.atomic_max(steel_sleep.roots_is_supported[i_r, i_b], steel_sleep.arts_is_anchored[i_a])
        if steel_sleep.arts_is_pinned[i_a] != 0 or steel_sleep.arts_is_changed[i_a, i_b] != 0:
            qd.atomic_max(steel_sleep.roots_is_vetoed[i_r, i_b], 1)
        steel_sleep.arts_is_changed[i_a, i_b] = 0

    # A non-zero or non-finite external wrench wakes the island of its link.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        i_a = steel_model.links_art[i_l]
        if i_a >= 0:
            norm_sq = steel_dyn.links_f_ext_lin[i_l, i_b].norm_sqr() + steel_dyn.links_f_ext_ang[i_l, i_b].norm_sqr()
            if qd.math.isnan(norm_sq) or qd.math.isinf(norm_sq) or norm_sq > 0.0:
                i_r = steel_sleep.arts_parent[i_a, i_b]
                qd.atomic_max(steel_sleep.roots_is_vetoed[i_r, i_b], 1)

    # A non-zero force control (non-finite included) or a drive keeps the island of its degree of freedom awake.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        i_j = steel_model.dofs_joint[i_d]
        if i_j >= 0:
            is_vetoed = False
            if dyn_state.dofs.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
                is_vetoed = dyn_state.dofs.ctrl_force[i_d, i_b] != 0.0
            if func_is_scalar_axis_joint(steel_model.joints_type[i_j]):
                ke, kd, _, _ = func_drive_params(i_d, i_b, dyn_state, dyn_info, rigid_info, steel_config)
                is_vetoed = is_vetoed or ke != 0.0 or kd != 0.0
            if is_vetoed:
                i_a = steel_model.joints_art[i_j]
                i_r = steel_sleep.arts_parent[i_a, i_b]
                qd.atomic_max(steel_sleep.roots_is_vetoed[i_r, i_b], 1)

    # A touching contact between a link and static geometry supports the island of the link. Every Steel body belongs
    # to an articulation, so the other side of a contact with a single Steel body is static geometry, which only moves
    # through a Genesis setter, itself a wake event.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        if i_c < steel_contacts.n[i_b]:
            i_l0 = steel_contacts.body0[i_c, i_b]
            i_l1 = steel_contacts.body1[i_c, i_b]
            if (i_l0 >= 0) != (i_l1 >= 0):
                _, _, _, phi = func_contact_points_world(i_c, i_b, steel_state, steel_contacts)
                if phi <= steel_info.support_gap[None]:
                    i_l = i_l0 if i_l0 >= 0 else i_l1
                    i_a = steel_model.links_art[i_l]
                    i_r = steel_sleep.arts_parent[i_a, i_b]
                    qd.atomic_max(steel_sleep.roots_is_supported[i_r, i_b], 1)

    # Unsupported islands, and every island of an environment whose contacts or rows are incomplete, stay awake.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        i_r = steel_sleep.arts_parent[i_a, i_b]
        if (
            steel_sleep.roots_is_supported[i_r, i_b] == 0
            or steel_sleep.envs_is_incomplete[i_b] != 0
            or steel_status.envs_overflow[i_b] != 0
        ):
            qd.atomic_max(steel_sleep.roots_is_vetoed[i_r, i_b], 1)

    # A woken or vetoed island wakes the previous islands of its asleep members, and every current island they touch.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        if steel_sleep.arts_is_awake[i_a, i_b] == 0:
            i_r = steel_sleep.arts_parent[i_a, i_b]
            if steel_sleep.roots_has_awake[i_r, i_b] != 0 or steel_sleep.roots_is_vetoed[i_r, i_b] != 0:
                i_w = steel_sleep.arts_wake_parent[i_a, i_b]
                qd.atomic_max(steel_sleep.roots_has_old_wake[i_w, i_b], 1)

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        i_w = steel_sleep.arts_wake_parent[i_a, i_b]
        if steel_sleep.roots_has_old_wake[i_w, i_b] != 0:
            i_r = steel_sleep.arts_parent[i_a, i_b]
            qd.atomic_max(steel_sleep.roots_is_vetoed[i_r, i_b], 1)

    # A woken articulation's mass matrix may predate its frozen pose, so it is refreshed on this step.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        i_r = steel_sleep.arts_parent[i_a, i_b]
        is_asleep = steel_sleep.arts_is_awake[i_a, i_b] == 0
        if steel_sleep.roots_is_vetoed[i_r, i_b] != 0 or (is_asleep and steel_sleep.roots_has_awake[i_r, i_b] != 0):
            steel_sleep.arts_quiet_age[i_a, i_b] = 0.0
            if is_asleep:
                steel_state.arts_mass_update_requested[i_a, i_b] = 1
            steel_sleep.arts_is_awake[i_a, i_b] = 1
        steel_sleep.arts_is_step_asleep[i_a, i_b] = 1 - steel_sleep.arts_is_awake[i_a, i_b]

    if qd.static(steel_config.sleep_skip_constraints):
        qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_l, i_b in qd.ndrange(n_links, _B):
            is_frozen = 0
            i_a = steel_model.links_art[i_l]
            if i_a >= 0:
                if steel_sleep.arts_is_awake[i_a, i_b] == 0 and steel_sleep.arts_asleep_steps[i_a, i_b] >= 1:
                    is_frozen = 1
            steel_sleep.links_is_frozen[i_l, i_b] = is_frozen


@qd.func
def func_steel_sleep_finish(
    dt: float,
    steel_state: array_class.SteelState,
    steel_dyn: array_class.SteelDynamics,
    steel_sleep: array_class.SteelSleep,
    steel_status: array_class.SteelStatus,
    steel_model: array_class.SteelModelInfo,
    steel_config: qd.template(),
    linear_threshold: float,
    angular_threshold: float,
    quiet_time: float,
):
    """Update the quiet time of every island from the outputs of the step, publish the sleeping ones and freeze them.

    Runs after the forward kinematics of the step and before its state is written to Genesis. A link faster than
    linear_threshold (center of mass) or angular_threshold, or a non-finite output, vetoes its island, whose output is
    then published as is. The quiet time of an articulation grows while its island is supported, unvetoed and its
    environment complete, and an island whose members are all quiet for quiet_time sleeps. A frozen articulation,
    asleep after the step or asleep through it without veto, keeps the coordinates of the step start with zero
    velocities and its links keep their pose and the parent wrench of the last step.
    """
    n_arts = steel_model.arts_dof_start.shape[0]
    n_joints = steel_model.joints_link.shape[0]
    n_links = steel_model.links_art.shape[0]
    _B = steel_state.q.shape[1]

    # The outputs of the articulations whose dynamics were skipped are their frozen state, not trial results.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        i_a = steel_model.links_art[i_l]
        if i_a >= 0:
            is_checked = True
            if qd.static(steel_config.sleep_skip_constraints):
                is_checked = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
            if is_checked:
                is_vetoed = False
                pos = steel_state.links_pos[i_l, i_b]
                quat = steel_state.links_quat[i_l, i_b]
                for k in qd.static(range(3)):
                    is_vetoed = is_vetoed or qd.math.isnan(pos[k]) or qd.math.isinf(pos[k])
                for k in qd.static(range(4)):
                    is_vetoed = is_vetoed or qd.math.isnan(quat[k]) or qd.math.isinf(quat[k])
                vel_sq = steel_state.links_vel[i_l, i_b].norm_sqr()
                ang_sq = steel_state.links_ang[i_l, i_b].norm_sqr()
                speed_sq = vel_sq + ang_sq
                if (
                    qd.math.isnan(speed_sq)
                    or qd.math.isinf(speed_sq)
                    or qd.sqrt(vel_sq) > linear_threshold
                    or qd.sqrt(ang_sq) > angular_threshold
                ):
                    is_vetoed = True
                if is_vetoed:
                    i_r = steel_sleep.arts_parent[i_a, i_b]
                    qd.atomic_max(steel_sleep.roots_is_vetoed[i_r, i_b], 1)

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_b in qd.ndrange(n_joints, _B):
        i_a = steel_model.joints_art[i_j]
        is_checked = True
        if qd.static(steel_config.sleep_skip_constraints):
            is_checked = steel_sleep.arts_is_step_asleep[i_a, i_b] == 0
        if is_checked:
            joint_type = steel_model.joints_type[i_j]
            n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
            n_qs = n_dofs
            if joint_type == STEEL_JOINT_TYPE.FREE:
                n_qs = 7
            elif joint_type == STEEL_JOINT_TYPE.BALL:
                n_qs = 4
            i_q_start = steel_model.joints_q_start[i_j]
            i_d_start = steel_model.joints_dof_start[i_j]
            is_vetoed = False
            for i_q in range(i_q_start, i_q_start + n_qs):
                q = steel_state.q[i_q, i_b]
                is_vetoed = is_vetoed or qd.math.isnan(q) or qd.math.isinf(q)
            for i_d in range(i_d_start, i_d_start + n_dofs):
                qd_ = steel_state.qd[i_d, i_b]
                is_vetoed = is_vetoed or qd.math.isnan(qd_) or qd.math.isinf(qd_)
            if is_vetoed:
                i_r = steel_sleep.arts_parent[i_a, i_b]
                qd.atomic_max(steel_sleep.roots_is_vetoed[i_r, i_b], 1)

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        i_r = steel_sleep.arts_parent[i_a, i_b]
        if (
            steel_sleep.roots_is_vetoed[i_r, i_b] != 0
            or steel_sleep.roots_is_supported[i_r, i_b] == 0
            or steel_status.envs_overflow[i_b] != 0
            or steel_sleep.envs_is_incomplete[i_b] != 0
        ):
            steel_sleep.arts_quiet_age[i_a, i_b] = 0.0
        else:
            steel_sleep.arts_quiet_age[i_a, i_b] = qd.min(steel_sleep.arts_quiet_age[i_a, i_b] + dt, quiet_time)
        if steel_sleep.arts_quiet_age[i_a, i_b] < quiet_time:
            qd.atomic_min(steel_sleep.roots_is_ready[i_r, i_b], 0)

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b in qd.ndrange(n_arts, _B):
        i_r = steel_sleep.arts_parent[i_a, i_b]
        is_awake = 1 - steel_sleep.roots_is_ready[i_r, i_b]
        steel_sleep.arts_is_awake[i_a, i_b] = is_awake
        steel_sleep.arts_previous_root[i_a, i_b] = i_r
        if is_awake == 0:
            steel_sleep.arts_asleep_steps[i_a, i_b] = steel_sleep.arts_asleep_steps[i_a, i_b] + 1
        else:
            steel_sleep.arts_asleep_steps[i_a, i_b] = 0

    # An articulation asleep through the step keeps its frozen state even if it wakes now, unless its own output
    # vetoed it: a vetoed trial output is published as the evidence of the wake.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_j, i_b in qd.ndrange(n_joints, _B):
        i_a = steel_model.joints_art[i_j]
        i_r = steel_sleep.arts_parent[i_a, i_b]
        if steel_sleep.arts_is_awake[i_a, i_b] == 0 or (
            steel_sleep.arts_is_step_asleep[i_a, i_b] != 0 and steel_sleep.roots_is_vetoed[i_r, i_b] == 0
        ):
            joint_type = steel_model.joints_type[i_j]
            n_dofs = steel_model.joints_n_lin[i_j] + steel_model.joints_n_ang[i_j]
            n_qs = n_dofs
            if joint_type == STEEL_JOINT_TYPE.FREE:
                n_qs = 7
            elif joint_type == STEEL_JOINT_TYPE.BALL:
                n_qs = 4
            i_q_start = steel_model.joints_q_start[i_j]
            i_d_start = steel_model.joints_dof_start[i_j]
            for i_q in range(i_q_start, i_q_start + n_qs):
                steel_state.q[i_q, i_b] = steel_sleep.q_start[i_q, i_b]
            for i_d in range(i_d_start, i_d_start + n_dofs):
                steel_state.qd[i_d, i_b] = 0.0
                steel_state.qdd[i_d, i_b] = 0.0
                steel_dyn.v_out[i_d, i_b] = 0.0

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b in qd.ndrange(n_links, _B):
        i_a = steel_model.links_art[i_l]
        if i_a >= 0:
            i_r = steel_sleep.arts_parent[i_a, i_b]
            steel_sleep.links_is_awake[i_l, i_b] = steel_sleep.arts_is_awake[i_a, i_b]
            steel_sleep.links_island[i_l, i_b] = i_r
            if steel_sleep.arts_is_awake[i_a, i_b] == 0 or (
                steel_sleep.arts_is_step_asleep[i_a, i_b] != 0 and steel_sleep.roots_is_vetoed[i_r, i_b] == 0
            ):
                steel_state.links_pos[i_l, i_b] = steel_state.links_pos_prev[i_l, i_b]
                steel_state.links_quat[i_l, i_b] = steel_state.links_quat_prev[i_l, i_b]
                steel_state.links_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                steel_state.links_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                steel_state.links_parent_f_lin[i_l, i_b] = steel_sleep.links_parent_f_lin_last[i_l, i_b]
                steel_state.links_parent_f_ang[i_l, i_b] = steel_sleep.links_parent_f_ang_last[i_l, i_b]
            steel_sleep.links_parent_f_lin_last[i_l, i_b] = steel_state.links_parent_f_lin[i_l, i_b]
            steel_sleep.links_parent_f_ang_last[i_l, i_b] = steel_state.links_parent_f_ang[i_l, i_b]


@qd.kernel(fastcache=True)
def kernel_steel_sleep_wake(
    envs_idx: qd.types.ndarray(),
    steel_state: array_class.SteelState,
    steel_sleep: array_class.SteelSleep,
    steel_config: qd.template(),
    mark_changed: bool,
    clear_asleep_steps: bool,
):
    """Wake every island of the selected environments.

    A woken articulation restarts its quiet time and refreshes its mass matrix. mark_changed also vetoes the islands at
    the next step, as a change of their state or model does. clear_asleep_steps clears the sleep count of their
    articulations.
    """
    n_arts = steel_sleep.arts_is_awake.shape[0]
    n_links = steel_sleep.links_is_awake.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_a, i_b_ in qd.ndrange(n_arts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        if steel_sleep.arts_is_awake[i_a, i_b] == 0:
            steel_state.arts_mass_update_requested[i_a, i_b] = 1
        steel_sleep.arts_is_awake[i_a, i_b] = 1
        steel_sleep.arts_quiet_age[i_a, i_b] = 0.0
        if mark_changed:
            steel_sleep.arts_is_changed[i_a, i_b] = 1
        if clear_asleep_steps:
            steel_sleep.arts_asleep_steps[i_a, i_b] = 0

    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_b_ in qd.ndrange(n_links, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        steel_sleep.links_is_awake[i_l, i_b] = 1
