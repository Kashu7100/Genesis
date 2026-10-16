"""Steel model plan: the articulations, joints and degrees of freedom of the rigid scene in Steel's terms.

Steel simulates articulations, each a tree of joints rooted at the world, one joint per body. The plan maps every
Genesis link of a root carrying at least one degree of freedom onto a Steel body with exactly one inbound joint, whose
type, anchor frames and axes reproduce the Genesis kinematics of the link. Links of roots without any degree of
freedom are static geometry (Steel body -1). Every environment shares the topology, so the plan is env-independent.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

import genesis as gs
from genesis.utils.misc import qd_to_numpy


class STEEL_JOINT_TYPE(IntEnum):
    """Steel joint types, numbered as in the reference implementation."""

    PRISMATIC = 0
    REVOLUTE = 1
    BALL = 2
    FIXED = 3
    FREE = 4
    D6 = 6


@dataclass(frozen=True)
class SteelModelPlan:
    """Host-side topology tables of the Steel model.

    Joints are numbered articulation by articulation, parents before children, and joint `i_j` owns the Genesis link
    `joints_link[i_j]`, whose parent is the link `joints_parent_link[i_j]` of the same articulation, or static
    geometry (-1, with the Genesis link `joints_static_parent_link[i_j]` posing the joint, -1 for the world). Steel
    degrees of freedom and coordinates use the Genesis indices. The anchor frames of a joint
    are read live from the Genesis link frame and the position of the Genesis joint `joints_anchor[i_j]` (-1 for a
    joint without anchor offset), so the setters of the Genesis kinematic parameters apply to Steel too.
    """

    arts_joint_start: np.ndarray
    arts_dof_start: np.ndarray
    arts_n_dofs: np.ndarray
    arts_is_free_rigid: np.ndarray
    arts_H_offset: np.ndarray
    joints_link: np.ndarray
    joints_parent_link: np.ndarray
    joints_static_parent_link: np.ndarray
    joints_ancestor: np.ndarray
    joints_art: np.ndarray
    joints_type: np.ndarray
    joints_q_start: np.ndarray
    joints_dof_start: np.ndarray
    joints_n_lin: np.ndarray
    joints_n_ang: np.ndarray
    joints_anchor: np.ndarray
    dofs_axis: np.ndarray
    dofs_joint: np.ndarray
    links_joint: np.ndarray
    links_art: np.ndarray
    links_has_response_dofs: np.ndarray
    links_coupling_group: np.ndarray
    links_single_response_dof: np.ndarray
    free_root_joints: np.ndarray
    mimics_eq: np.ndarray
    mimics_art: np.ndarray
    mimics_dof: np.ndarray
    mimics_leader_dof: np.ndarray
    mimics_q: np.ndarray
    mimics_leader_q: np.ndarray
    connects_eq: np.ndarray
    connects_art: np.ndarray
    connects_link_p: np.ndarray
    connects_static_link_p: np.ndarray
    connects_link_c: np.ndarray
    connects_anchor_p_offset: np.ndarray
    H_size: int
    max_art_dofs: int

    @property
    def n_arts(self) -> int:
        return len(self.arts_dof_start)

    @property
    def n_arts_(self) -> int:
        """Number of articulations, at least one, sizing the per-articulation buffers."""
        return max(self.n_arts, 1)

    @property
    def n_mimics(self) -> int:
        return len(self.mimics_eq)

    @property
    def n_connects(self) -> int:
        return len(self.connects_eq)

    @property
    def has_free_rigid_bodies(self) -> bool:
        return bool(self.arts_is_free_rigid.any())

    @property
    def n_joints(self) -> int:
        return len(self.joints_link)


def build_steel_model_plan(solver) -> SteelModelPlan:
    """Map the links and joints of a built rigid solver onto Steel articulations, joints and degrees of freedom.

    Raises when a link carries a joint combination Steel has no joint type for: a free joint below the root, several
    joints mixing spherical or free types, several joints with distinct anchors, revolute joints followed by prismatic
    ones, or two non-orthogonal revolute axes.
    """
    links_parent_idx = qd_to_numpy(solver.dyn_info.links.parent_idx)
    links_root_idx = qd_to_numpy(solver.dyn_info.links.root_idx)
    links_joint_start = qd_to_numpy(solver.dyn_info.links.joint_start)
    links_joint_end = qd_to_numpy(solver.dyn_info.links.joint_end)
    joints_info_type = qd_to_numpy(solver.dyn_info.joints.type)
    joints_info_pos = qd_to_numpy(solver.dyn_info.joints.pos)
    joints_info_q_start = qd_to_numpy(solver.dyn_info.joints.q_start)
    joints_info_dof_start = qd_to_numpy(solver.dyn_info.joints.dof_start)
    dofs_motion_ang = qd_to_numpy(solver.dyn_info.dofs.motion_ang)
    dofs_motion_vel = qd_to_numpy(solver.dyn_info.dofs.motion_vel)
    # Every environment shares the topology and the joint axes, the anchor positions may differ (read live).
    if dofs_motion_ang.ndim == 3:
        if not np.allclose(dofs_motion_ang, dofs_motion_ang[:, :1], atol=gs.EPS) or not np.allclose(
            dofs_motion_vel, dofs_motion_vel[:, :1], atol=gs.EPS
        ):
            gs.raise_exception("Steel does not support joint axes that differ per environment.")
        dofs_motion_ang, dofs_motion_vel = dofs_motion_ang[:, 0], dofs_motion_vel[:, 0]
    if links_parent_idx.ndim == 2:
        links_parent_idx, links_root_idx = links_parent_idx[:, 0], links_root_idx[:, 0]
        links_joint_start, links_joint_end = links_joint_start[:, 0], links_joint_end[:, 0]
    if joints_info_type.ndim == 2:
        joints_info_type, joints_info_pos = joints_info_type[:, 0], joints_info_pos[:, 0]
        joints_info_q_start, joints_info_dof_start = joints_info_q_start[:, 0], joints_info_dof_start[:, 0]

    n_links = solver.n_links
    links_n_dofs = np.array([link.n_dofs for link in solver.links], dtype=gs.np_int)
    roots_n_dofs = np.bincount(links_root_idx, weights=links_n_dofs, minlength=n_links) if n_links else np.zeros(0)
    roots_idx = [i_r for i_r in range(n_links) if links_root_idx[i_r] == i_r and roots_n_dofs[i_r] > 0]

    arts_joint_start, arts_dof_start, arts_n_dofs, arts_is_free_rigid = [0], [], [], []
    joints_link, joints_parent_link, joints_static_parent_link = [], [], []
    joints_type, joints_q_start, joints_dof_start = [], [], []
    joints_n_lin, joints_n_ang, joints_anchor, joints_art = [], [], [], []
    dofs_axis = np.zeros((max(solver.n_dofs, 1), 3), dtype=gs.np_float)
    links_joint = np.full(n_links, -1, dtype=gs.np_int)
    links_art = np.full(n_links, -1, dtype=gs.np_int)
    for i_a, i_r in enumerate(roots_idx):
        links_idx = np.flatnonzero(links_root_idx == i_r)
        dofs_idx = np.concatenate(
            [np.arange(solver.links[i_l].dof_start, solver.links[i_l].dof_end) for i_l in links_idx]
        )
        dof_start, n_dofs = int(dofs_idx[0]), len(dofs_idx)
        if not np.array_equal(dofs_idx, np.arange(dof_start, dof_start + n_dofs)):
            gs.raise_exception(
                f"Steel requires the degrees of freedom of the kinematic tree of link {i_r} to be contiguous."
            )
        for i_l in links_idx:
            link = solver.links[i_l]
            i_p = links_parent_idx[i_l]
            if i_p >= 0 and links_parent_idx[i_l] > i_l:
                gs.raise_exception("Steel requires every link to follow its parent.")
            genesis_joints = range(links_joint_start[i_l], links_joint_end[i_l])
            genesis_types = [joints_info_type[i_j] for i_j in genesis_joints]
            moving_joints = [i_j for i_j in genesis_joints if joints_info_type[i_j] != gs.JOINT_TYPE.FIXED]
            anchor_joint = moving_joints[0] if moving_joints else -1
            n_lin = n_ang = 0
            if not moving_joints:
                steel_type = STEEL_JOINT_TYPE.FIXED
            elif len(moving_joints) == 1 and genesis_types[genesis_joints.index(moving_joints[0])] in (
                gs.JOINT_TYPE.FREE,
                gs.JOINT_TYPE.SPHERICAL,
            ):
                if joints_info_type[moving_joints[0]] == gs.JOINT_TYPE.FREE:
                    if i_p >= 0:
                        gs.raise_exception(f"Steel supports free joints on root links only (link '{link.name}').")
                    steel_type = STEEL_JOINT_TYPE.FREE
                    # A free joint overwrites the pose of its link with the joint coordinates.
                    anchor_joint = -1
                    n_lin, n_ang = 3, 3
                else:
                    steel_type = STEEL_JOINT_TYPE.BALL
                    n_ang = 3
            else:
                for i_j in moving_joints:
                    if joints_info_type[i_j] not in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                        gs.raise_exception(
                            f"Steel does not support spherical or free joints combined with other joints in link "
                            f"'{link.name}'."
                        )
                    if not np.allclose(joints_info_pos[i_j], joints_info_pos[anchor_joint], atol=gs.EPS):
                        gs.raise_exception(f"Steel requires the joints of link '{link.name}' to share one anchor.")
                is_revolute = [joints_info_type[i_j] == gs.JOINT_TYPE.REVOLUTE for i_j in moving_joints]
                if any(is_revolute[i] and not is_revolute[i + 1] for i in range(len(is_revolute) - 1)):
                    gs.raise_exception(
                        f"Steel requires the prismatic joints of link '{link.name}' to precede its revolute joints."
                    )
                n_ang = sum(is_revolute)
                n_lin = len(moving_joints) - n_ang
                if n_lin > 3 or n_ang > 3:
                    gs.raise_exception(f"Link '{link.name}' has more than three joints of one kind.")
                if len(moving_joints) == 1:
                    steel_type = STEEL_JOINT_TYPE.REVOLUTE if n_ang else STEEL_JOINT_TYPE.PRISMATIC
                else:
                    steel_type = STEEL_JOINT_TYPE.D6
                for i_j, is_rev in zip(moving_joints, is_revolute):
                    i_d = joints_info_dof_start[i_j]
                    dofs_axis[i_d] = dofs_motion_ang[i_d] if is_rev else dofs_motion_vel[i_d]
                if steel_type == STEEL_JOINT_TYPE.D6 and n_ang == 2:
                    axes = dofs_axis[link.dof_start + n_lin : link.dof_start + n_lin + 2]
                    if abs(np.dot(axes[0], axes[1])) > gs.EPS:
                        gs.raise_exception(
                            f"Steel requires the two revolute axes of link '{link.name}' to be orthogonal."
                        )
            if steel_type in (STEEL_JOINT_TYPE.BALL, STEEL_JOINT_TYPE.FREE):
                dofs_axis[link.dof_start : link.dof_end] = 0.0
            links_joint[i_l] = len(joints_link)
            links_art[i_l] = i_a
            joints_link.append(i_l)
            is_static_parent = i_p >= 0 and links_root_idx[i_p] != i_r
            joints_parent_link.append(-1 if is_static_parent else i_p)
            joints_static_parent_link.append(i_p if is_static_parent else -1)
            joints_art.append(i_a)
            joints_type.append(steel_type)
            joints_q_start.append(link.q_start)
            joints_dof_start.append(link.dof_start)
            joints_n_lin.append(n_lin)
            joints_n_ang.append(n_ang)
            joints_anchor.append(anchor_joint)
        arts_joint_start.append(len(joints_link))
        arts_dof_start.append(dof_start)
        arts_n_dofs.append(n_dofs)
        arts_is_free_rigid.append(len(links_idx) == 1 and joints_type[-1] == STEEL_JOINT_TYPE.FREE)

    joints_link = np.array(joints_link, dtype=gs.np_int)
    joints_parent_link = np.array(joints_parent_link, dtype=gs.np_int)
    joints_ancestor = np.array(
        [links_joint[i_p] if i_p >= 0 else -1 for i_p in joints_parent_link], dtype=gs.np_int
    ).reshape(-1)
    arts_n_dofs = np.array(arts_n_dofs, dtype=gs.np_int)
    arts_H_offset = np.concatenate(([0], np.cumsum(arts_n_dofs**2)[:-1])).astype(gs.np_int)
    dofs_joint = np.full(max(solver.n_dofs, 1), -1, dtype=gs.np_int)
    for i_j, (dof_start, n_lin, n_ang) in enumerate(zip(joints_dof_start, joints_n_lin, joints_n_ang)):
        dofs_joint[dof_start : dof_start + n_lin + n_ang] = i_j

    # Every body with an inbound joint whose ancestry carries DOFs responds to constraint impulses. Two bodies couple
    # (nonzero cross response) iff they share the child body of their topmost DOF-carrying ancestor joint.
    links_has_response_dofs = np.zeros(n_links, dtype=gs.np_int)
    links_coupling_group = np.arange(n_links, dtype=gs.np_int)
    links_single_response_dof = np.full(n_links, -1, dtype=gs.np_int)
    joints_n_dofs = np.array(joints_n_lin, dtype=gs.np_int) + np.array(joints_n_ang, dtype=gs.np_int)
    for i_j, i_l in enumerate(joints_link):
        supporting_dofs = []
        i_k = i_j
        while i_k >= 0:
            if joints_n_dofs[i_k] > 0:
                supporting_dofs.extend(range(joints_dof_start[i_k], joints_dof_start[i_k] + joints_n_dofs[i_k]))
                links_coupling_group[i_l] = joints_link[i_k]
            i_k = joints_ancestor[i_k]
        links_has_response_dofs[i_l] = len(supporting_dofs) > 0
        if len(supporting_dofs) == 1:
            links_single_response_dof[i_l] = supporting_dofs[0]

    free_root_joints = np.array(
        [i_j for i_j, steel_type in enumerate(joints_type) if steel_type == STEEL_JOINT_TYPE.FREE],
        dtype=gs.np_int,
    )

    mimics, connects = [], []
    for equality in solver.equalities:
        if equality.type == gs.EQUALITY_TYPE.JOINT:
            # q_follower - q0_follower = c0 + c1 (q_leader - q0_leader), the polynomial coefficients being eq_data[:5].
            if equality.eq_obj2id < 0:
                gs.raise_exception(f"Steel supports joint equality '{equality.name}' with a leader joint only.")
            if not np.allclose(equality.eq_data[2:5], 0.0, atol=gs.EPS):
                gs.raise_exception(f"Steel supports linear joint equalities only ('{equality.name}').")
            joint, leader = solver.joints[equality.eq_obj1id], solver.joints[equality.eq_obj2id]
            for joint_ in (joint, leader):
                if joint_.type not in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                    gs.raise_exception(
                        f"Steel supports joint equalities between scalar joints only ('{equality.name}')."
                    )
            i_a, i_a_leader = links_art[joint.link.idx], links_art[leader.link.idx]
            if i_a < 0 or i_a != i_a_leader:
                gs.raise_exception(f"Steel supports joint equalities within one articulation only ('{equality.name}').")
            if joint.dof_start == leader.dof_start:
                gs.raise_exception(f"Joint equality '{equality.name}' couples a joint with itself.")
            mimics.append((equality.idx, i_a, joint.dof_start, leader.dof_start, joint.q_start, leader.q_start))
        elif equality.type == gs.EQUALITY_TYPE.CONNECT:
            # The child of the closure is the articulated link, its parent the other link, which may be static.
            i_l1, i_l2 = equality.eq_obj1id, equality.eq_obj2id
            i_a1, i_a2 = links_art[i_l1], links_art[i_l2]
            if i_a1 >= 0 and i_a2 >= 0 and i_a1 != i_a2:
                gs.raise_exception(
                    f"Steel does not support connect equality '{equality.name}' between two articulations."
                )
            if i_a1 < 0 and i_a2 < 0:
                continue
            if i_a2 >= 0:
                i_l_p, i_l_c, anchor_p_offset, i_a = i_l1, i_l2, 0, i_a2
            else:
                i_l_p, i_l_c, anchor_p_offset, i_a = i_l2, i_l1, 3, i_a1
            is_static_parent = links_art[i_l_p] < 0
            connects.append(
                (
                    equality.idx,
                    i_a,
                    -1 if is_static_parent else i_l_p,
                    i_l_p if is_static_parent else -1,
                    i_l_c,
                    anchor_p_offset,
                )
            )
        else:
            gs.raise_exception(f"Steel supports connect and joint equalities only ('{equality.name}').")
    # Rows of one articulation are reserved together, in equality order.
    mimics = np.array(mimics, dtype=gs.np_int).reshape(-1, 6)
    mimics = mimics[np.argsort(mimics[:, 1], kind="stable")]
    connects = np.array(connects, dtype=gs.np_int).reshape(-1, 6)
    connects = connects[np.argsort(connects[:, 1], kind="stable")]

    return SteelModelPlan(
        arts_joint_start=np.array(arts_joint_start, dtype=gs.np_int),
        arts_dof_start=np.array(arts_dof_start, dtype=gs.np_int),
        arts_n_dofs=arts_n_dofs,
        arts_is_free_rigid=np.array(arts_is_free_rigid, dtype=gs.np_int),
        arts_H_offset=arts_H_offset,
        joints_link=joints_link,
        joints_parent_link=joints_parent_link,
        joints_static_parent_link=np.array(joints_static_parent_link, dtype=gs.np_int),
        joints_ancestor=joints_ancestor,
        joints_art=np.array(joints_art, dtype=gs.np_int),
        joints_type=np.array(joints_type, dtype=gs.np_int),
        joints_q_start=np.array(joints_q_start, dtype=gs.np_int),
        joints_dof_start=np.array(joints_dof_start, dtype=gs.np_int),
        joints_n_lin=np.array(joints_n_lin, dtype=gs.np_int),
        joints_n_ang=np.array(joints_n_ang, dtype=gs.np_int),
        joints_anchor=np.array(joints_anchor, dtype=gs.np_int),
        dofs_axis=dofs_axis,
        dofs_joint=dofs_joint,
        links_joint=links_joint,
        links_art=links_art,
        links_has_response_dofs=links_has_response_dofs,
        links_coupling_group=links_coupling_group,
        links_single_response_dof=links_single_response_dof,
        free_root_joints=free_root_joints,
        mimics_eq=mimics[:, 0],
        mimics_art=mimics[:, 1],
        mimics_dof=mimics[:, 2],
        mimics_leader_dof=mimics[:, 3],
        mimics_q=mimics[:, 4],
        mimics_leader_q=mimics[:, 5],
        connects_eq=connects[:, 0],
        connects_art=connects[:, 1],
        connects_link_p=connects[:, 2],
        connects_static_link_p=connects[:, 3],
        connects_link_c=connects[:, 4],
        connects_anchor_p_offset=connects[:, 5],
        H_size=int((arts_n_dofs**2).sum()),
        max_art_dofs=int(arts_n_dofs.max()) if len(arts_n_dofs) else 0,
    )
