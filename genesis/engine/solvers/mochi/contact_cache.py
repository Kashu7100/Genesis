"""Contact cache of the MochiSolver.

A contact search records the candidate pairs within the contact range widened by a margin, together with the link
poses and vertex positions it ran at. While no link and no deformable vertex has moved more than half of that margin
since, every pair a new search could find is already among the candidates (both sides of a pair moved at most half the
margin each, so no distance changed by more than the widening), and an assembly re-evaluates the candidates at the
current iterate instead of searching again. This is the conservative displacement bound of offset-geometry contact
methods, used as an exactness certificate for the cached search rather than as a limit on the step.
"""

import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .data import MochiContactState, MochiInfo, MochiSoftState, MochiState
from .newton import func_is_env_active


@qd.func
def func_contact_certificate(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    skip_ls_done,
):
    """Flag the environments whose contact candidates may be stale: a dynamic link whose origin displacement plus its
    rotation angle times its bound radius exceeds half the candidate margin since the last search, or a deformable
    vertex that moved more than that. Then count the searches of the flagged environments and reset their candidate
    lists."""
    n_links = dyn_state.links.pos.shape[0]
    n_soft_verts = soft_state.verts_pos.shape[0]
    _B = mochi_state.is_active.shape[0]
    bound = 0.5 * mochi_info.contact_candidate_margin[None]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if not func_is_env_active(i_b, mochi_state, skip_ls_done):
            continue
        if mochi_state.needs_detect[i_b] != 0 or not mochi_info.links.is_dynamic[i_l]:
            continue
        shift = (dyn_state.links.pos[i_l, i_b] - contact_state.links_pos_det[i_l, i_b]).norm()
        # Rotation angle from the quaternion difference (|q - q_det| = 2 sin(angle / 4) for the closer sign): well
        # conditioned near zero, unlike acos of the dot product, whose rounding alone reads as ~1e-3 rad in fp32.
        quat = dyn_state.links.quat[i_l, i_b]
        quat_det = contact_state.links_quat_det[i_l, i_b]
        dq = qd.min((quat - quat_det).norm(), (quat + quat_det).norm())
        angle = 4.0 * qd.asin(qd.min(0.5 * dq, 1.0))
        if shift + angle * mochi_info.links.bound_radius[i_l] > bound:
            mochi_state.needs_detect[i_b] = 1
    if qd.static(mochi_config.has_soft):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_v, i_slot in (
            qd.ndrange(n_soft_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_soft_verts, 1)
        ):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if not func_is_env_active(i_b, mochi_state, skip_ls_done):
                continue
            if mochi_state.needs_detect[i_b] != 0:
                continue
            if (soft_state.verts_pos[i_v, i_b] - soft_state.verts_pos_det[i_v, i_b]).norm() > bound:
                mochi_state.needs_detect[i_b] = 1
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done) and mochi_state.needs_detect[i_b] != 0:
            mochi_state.n_detections[i_b] += 1
            contact_state.n_cand[i_b] = 0
            if qd.static(mochi_config.has_soft):
                soft_state.n_soft_cand[i_b] = 0
                soft_state.n_sc_cand[i_b] = 0
                soft_state.n_pc_cand[i_b] = 0


@qd.kernel
def kernel_contact_certificate(
    dyn_state: array_class.DynState,
    mochi_info: MochiInfo,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    skip_ls_done: qd.i32,
):
    func_contact_certificate(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        mochi_info,
        mochi_state,
        contact_state,
        soft_state,
        rigid_config,
        mochi_config,
        skip_ls_done,
    )


@qd.func
def func_contact_store_reference(
    i_b_env,
    per_env: qd.template(),
    envs: qd.types.ndarray(),
    n_envs: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    skip_ls_done,
):
    """Record the poses the contact search of the flagged environments ran at, clear their flag, and take the largest
    candidate counts over the environments (the runtime bounds of the batched loops over the candidate lists)."""
    n_links = dyn_state.links.pos.shape[0]
    n_soft_verts = soft_state.verts_pos.shape[0]
    _B = mochi_state.is_active.shape[0]
    max_cand = contact_state.cand_pair.shape[0]
    max_soft_cand = soft_state.soft_cand_pair.shape[0]
    max_sc_cand = soft_state.sc_cand_query.shape[0]
    max_pc_cand = soft_state.pc_cand_query.shape[0]
    if qd.static(not per_env):
        contact_state.n_cand_max[None] = 0
        if qd.static(mochi_config.has_soft):
            soft_state.n_soft_cand_max[None] = 0
            soft_state.n_sc_cand_max[None] = 0
            soft_state.n_pc_cand_max[None] = 0

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_l, i_slot in qd.ndrange(n_links, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_links, 1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done) and mochi_state.needs_detect[i_b] != 0:
            contact_state.links_pos_det[i_l, i_b] = dyn_state.links.pos[i_l, i_b]
            contact_state.links_quat_det[i_l, i_b] = dyn_state.links.quat[i_l, i_b]
    if qd.static(mochi_config.has_soft):
        qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_v, i_slot in (
            qd.ndrange(n_soft_verts, n_envs[None]) if qd.static(not per_env) else qd.ndrange(n_soft_verts, 1)
        ):
            i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
            if func_is_env_active(i_b, mochi_state, skip_ls_done) and mochi_state.needs_detect[i_b] != 0:
                soft_state.verts_pos_det[i_v, i_b] = soft_state.verts_pos[i_v, i_b]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]) if qd.static(not per_env) else range(1):
        i_b = envs[i_slot] if qd.static(not per_env) else i_b_env
        if func_is_env_active(i_b, mochi_state, skip_ls_done):
            mochi_state.needs_detect[i_b] = 0
            if qd.static(not per_env):
                qd.atomic_max(contact_state.n_cand_max[None], qd.min(contact_state.n_cand[i_b], max_cand))
                if qd.static(mochi_config.has_soft):
                    qd.atomic_max(soft_state.n_soft_cand_max[None], qd.min(soft_state.n_soft_cand[i_b], max_soft_cand))
                    qd.atomic_max(soft_state.n_sc_cand_max[None], qd.min(soft_state.n_sc_cand[i_b], max_sc_cand))
                    qd.atomic_max(soft_state.n_pc_cand_max[None], qd.min(soft_state.n_pc_cand[i_b], max_pc_cand))


@qd.kernel
def kernel_contact_store_reference(
    dyn_state: array_class.DynState,
    mochi_state: MochiState,
    contact_state: MochiContactState,
    soft_state: MochiSoftState,
    rigid_config: qd.template(),
    mochi_config: qd.template(),
    skip_ls_done: qd.i32,
):
    func_contact_store_reference(
        0,
        False,
        mochi_state.all_envs,
        mochi_state.n_envs_all,
        dyn_state,
        mochi_state,
        contact_state,
        soft_state,
        rigid_config,
        mochi_config,
        skip_ls_done,
    )
