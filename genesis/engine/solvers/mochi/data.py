# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Data structures of the MochiSolver: build-time model description, per-environment solver state and compile-time
configuration, all following the array_class conventions (frozen dataclasses of quadrants tensors, batch dimension
last)."""

import dataclasses
from enum import IntEnum

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, AutoInitMeta, V


class COLLIDER_TYPE(IntEnum):
    NONE = 0
    PLANE = 1
    SPHERE = 2
    BOX = 3
    GRID = 4


class INTEGRATOR(IntEnum):
    BACKWARD_EULER = 0
    BDF2 = 1


class FRICTION_MODEL(IntEnum):
    C1 = 0
    CINF = 1


class LINESEARCH(IntEnum):
    NONE = 0
    RESIDUAL_NORM = 1
    ARMIJO = 2


class SOLVE_STATUS(IntEnum):
    RUNNING = 0
    CONVERGED = 1
    STOPPED = 2
    DIVERGED = 3


# Number of previous steps kept for multistep integration (BDF2 needs two).
N_HISTORY = 2


@qd.data_oriented
class MochiStaticConfig(metaclass=AutoInitMeta):
    backend: int
    para_level: int
    integrator: int
    use_newton_euler_inertia: bool
    friction_model: int
    linesearch_type: int
    use_fitted_friction_hessian: bool
    friction_with_collider_normal: bool
    fade_friction: bool
    implicit_normal_force_for_dissipation: bool
    use_dense_direct: bool
    has_grid_colliders: bool
    record_contacts: bool
    batch_links_info: bool


# =========================================== build-time info ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiLinksInfo:
    # Whether the link carries the 6 degrees of freedom of a free root and enters the Newton system.
    is_dynamic: qd.Tensor
    has_gravity: qd.Tensor
    mass: qd.Tensor
    # Inertia about the center of mass in the inertial frame, and its second moment 0.5 tr(I) 1 - I, which weights
    # the rotation merit.
    inertia: qd.Tensor
    second_moment: qd.Tensor
    damping: qd.Tensor
    layer: qd.Tensor
    # Contact sample range and the link-frame bounding box of the sample cloud.
    sample_start: qd.Tensor
    sample_end: qd.Tensor
    samples_aabb_min: qd.Tensor
    samples_aabb_max: qd.Tensor


def get_mochi_links_info(solver):
    n_links_ = solver.n_links_
    return MochiLinksInfo(
        is_dynamic=V(dtype=gs.qd_bool, shape=(n_links_,)),
        has_gravity=V(dtype=gs.qd_bool, shape=(n_links_,)),
        mass=V(dtype=gs.qd_float, shape=(n_links_,)),
        inertia=V(dtype=gs.qd_mat3, shape=(n_links_,)),
        second_moment=V(dtype=gs.qd_mat3, shape=(n_links_,)),
        damping=V(dtype=gs.qd_float, shape=(n_links_,)),
        layer=V(dtype=gs.qd_int, shape=(n_links_,)),
        sample_start=V(dtype=gs.qd_int, shape=(n_links_,)),
        sample_end=V(dtype=gs.qd_int, shape=(n_links_,)),
        samples_aabb_min=V(dtype=gs.qd_vec3, shape=(n_links_,)),
        samples_aabb_max=V(dtype=gs.qd_vec3, shape=(n_links_,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiGeomsInfo:
    collider_type: qd.Tensor
    penalty_coefficient: qd.Tensor
    penalty_smoothing_half_distance: qd.Tensor
    penalty_threshold: qd.Tensor
    friction: qd.Tensor
    friction_falloff_vel: qd.Tensor
    viscous_friction: qd.Tensor
    normal_viscous_damping: qd.Tensor
    max_alignment_normals: qd.Tensor


def get_mochi_geoms_info(solver):
    n_geoms_ = solver.n_geoms_
    return MochiGeomsInfo(
        collider_type=V(dtype=gs.qd_int, shape=(n_geoms_,)),
        penalty_coefficient=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        penalty_smoothing_half_distance=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        penalty_threshold=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        friction=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        friction_falloff_vel=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        viscous_friction=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        normal_viscous_damping=V(dtype=gs.qd_float, shape=(n_geoms_,)),
        max_alignment_normals=V(dtype=gs.qd_float, shape=(n_geoms_,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiSamplesInfo:
    # Contact sample points (surface quadrature points of the collision triangles) in the link frame, with their
    # quadrature area weight and outward normal.
    pos: qd.Tensor
    normal: qd.Tensor
    weight: qd.Tensor
    link_idx: qd.Tensor
    geom_idx: qd.Tensor


def get_mochi_samples_info(solver):
    n_samples_ = solver.n_samples_
    return MochiSamplesInfo(
        pos=V(dtype=gs.qd_vec3, shape=(n_samples_,)),
        normal=V(dtype=gs.qd_vec3, shape=(n_samples_,)),
        weight=V(dtype=gs.qd_float, shape=(n_samples_,)),
        link_idx=V(dtype=gs.qd_int, shape=(n_samples_,)),
        geom_idx=V(dtype=gs.qd_int, shape=(n_samples_,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiInfo:
    links: MochiLinksInfo
    geoms: MochiGeomsInfo
    samples: MochiSamplesInfo
    # Whether contact between two links is enabled (layer and entity filters folded in).
    links_pair_enabled: qd.Tensor
    # Runtime constants
    dt: qd.Tensor
    gravity: qd.Tensor
    broadphase_margin: qd.Tensor
    newton_abs_tol: qd.Tensor
    newton_rel_tol: qd.Tensor
    explosion_abs_tol: qd.Tensor
    explosion_rel_tol: qd.Tensor
    linesearch_alpha: qd.Tensor
    linesearch_wolfe1: qd.Tensor
    pcg_rel_tol: qd.Tensor
    n_newton_iterations: qd.Tensor
    EPS: qd.Tensor


def get_mochi_info(solver):
    options = solver._options
    return MochiInfo(
        links=get_mochi_links_info(solver),
        geoms=get_mochi_geoms_info(solver),
        samples=get_mochi_samples_info(solver),
        links_pair_enabled=V(dtype=gs.qd_bool, shape=(solver.n_links_, solver.n_links_)),
        dt=_scalar(gs.qd_float, solver._substep_dt),
        gravity=V(dtype=gs.qd_vec3, shape=(solver._B,)),
        broadphase_margin=_scalar(gs.qd_float, options.broadphase_margin),
        newton_abs_tol=_scalar(gs.qd_float, options.newton_abs_tol),
        newton_rel_tol=_scalar(gs.qd_float, options.newton_rel_tol),
        explosion_abs_tol=_scalar(gs.qd_float, options.explosion_abs_tol if options.explosion_control else 0.0),
        explosion_rel_tol=_scalar(gs.qd_float, options.explosion_rel_tol),
        linesearch_alpha=_scalar(gs.qd_float, options.linesearch_alpha),
        linesearch_wolfe1=_scalar(gs.qd_float, options.linesearch_wolfe1),
        pcg_rel_tol=_scalar(gs.qd_float, options.pcg_rel_tol),
        n_newton_iterations=_scalar(gs.qd_int, options.n_newton_iterations),
        EPS=_scalar(gs.qd_float, gs.EPS),
    )


def _scalar(dtype, value):
    data = V(dtype=dtype, shape=())
    data.fill(value)
    return data


# =========================================== runtime state ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiState:
    # Multistep history of the generalized coordinates, velocities and symmetric rotation-derivative correction:
    # slot 0 is the previous step, slot 1 the one before.
    qpos_prev: qd.Tensor
    dofs_vel_prev: qd.Tensor
    links_vsym_prev: qd.Tensor
    # Symmetric part of the finite-difference rotation derivative of every link at the end of the last step. Together
    # with the angular velocity it reproduces the rotation increment of that step exactly, which the rotation merit
    # of the next step extrapolates from.
    links_vsym: qd.Tensor
    # Step-start extrapolation and stage-start reference of the current solve (identical for single-stage schemes).
    qpos_step_start: qd.Tensor
    qpos_stage_start: qd.Tensor
    dofs_vel_stage_start: qd.Tensor
    links_vsym_stage_start: qd.Tensor
    links_pos_stage_start: qd.Tensor
    links_quat_stage_start: qd.Tensor
    geoms_pos_stage_start: qd.Tensor
    geoms_quat_stage_start: qd.Tensor
    # Reference iterate of the line search (the last accepted iterate).
    qpos_ls_ref: qd.Tensor
    # Number of valid previous steps (multistep schemes fall back to backward Euler while it is short) and the stage
    # time step of the current solve.
    n_hist: qd.Tensor
    dt_stage: qd.Tensor
    # Newton system: residual (gradient of the incremental potential), step, convergence weights, per-link diagonal
    # 6x6 blocks and per-contact-pair off-diagonal 6x6 blocks of the Hessian, and its dense condensation.
    res: qd.Tensor
    dx: qd.Tensor
    conv_w: qd.Tensor
    H_diag: qd.Tensor
    H_off: qd.Tensor
    H_dense: qd.Tensor
    # Per-environment Newton and line search control.
    is_active: qd.Tensor
    status: qd.Tensor
    n_iter: qd.Tensor
    res_norm_sq: qd.Tensor
    res_w_sq: qd.Tensor
    res_norm0: qd.Tensor
    res_norm0_w: qd.Tensor
    ls_alpha: qd.Tensor
    ls_ref_norm_sq: qd.Tensor
    ls_is_done: qd.Tensor
    ls_slope: qd.Tensor
    obj: qd.Tensor
    obj_ref: qd.Tensor
    # Preconditioned conjugate gradient scratch.
    pcg_r: qd.Tensor
    pcg_z: qd.Tensor
    pcg_p: qd.Tensor
    pcg_Ap: qd.Tensor
    pcg_rTz: qd.Tensor
    pcg_rTz_new: qd.Tensor
    pcg_pTAp: qd.Tensor
    pcg_rTr: qd.Tensor
    pcg_rTr0: qd.Tensor
    pcg_is_active: qd.Tensor


def get_mochi_state(solver, max_pairs, use_dense_direct):
    _B = solver._B
    n_qs_, n_dofs_, n_links_, n_geoms_ = solver.n_qs_, solver.n_dofs_, solver.n_links_, solver.n_geoms_
    H_dense_shape = (_B, n_dofs_, n_dofs_) if use_dense_direct else ()
    return MochiState(
        qpos_prev=V(dtype=gs.qd_float, shape=(N_HISTORY, n_qs_, _B)),
        dofs_vel_prev=V(dtype=gs.qd_float, shape=(N_HISTORY, n_dofs_, _B)),
        links_vsym_prev=V(dtype=gs.qd_mat3, shape=(N_HISTORY, n_links_, _B)),
        links_vsym=V(dtype=gs.qd_mat3, shape=(n_links_, _B)),
        qpos_step_start=V(dtype=gs.qd_float, shape=(n_qs_, _B)),
        qpos_stage_start=V(dtype=gs.qd_float, shape=(n_qs_, _B)),
        dofs_vel_stage_start=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        links_vsym_stage_start=V(dtype=gs.qd_mat3, shape=(n_links_, _B)),
        links_pos_stage_start=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_quat_stage_start=V(dtype=gs.qd_vec4, shape=(n_links_, _B)),
        geoms_pos_stage_start=V(dtype=gs.qd_vec3, shape=(n_geoms_, _B)),
        geoms_quat_stage_start=V(dtype=gs.qd_vec4, shape=(n_geoms_, _B)),
        qpos_ls_ref=V(dtype=gs.qd_float, shape=(n_qs_, _B)),
        n_hist=V(dtype=gs.qd_int, shape=(_B,)),
        dt_stage=V(dtype=gs.qd_float, shape=(_B,)),
        res=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        dx=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        conv_w=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        H_diag=V_MAT(n=6, m=6, dtype=gs.qd_float, shape=(n_links_, _B)),
        H_off=V_MAT(n=6, m=6, dtype=gs.qd_float, shape=(max_pairs, _B)),
        H_dense=V(dtype=gs.qd_float, shape=H_dense_shape),
        is_active=V(dtype=gs.qd_bool, shape=(_B,)),
        status=V(dtype=gs.qd_int, shape=(_B,)),
        n_iter=V(dtype=gs.qd_int, shape=(_B,)),
        res_norm_sq=V(dtype=gs.qd_float, shape=(_B,)),
        res_w_sq=V(dtype=gs.qd_float, shape=(_B,)),
        res_norm0=V(dtype=gs.qd_float, shape=(_B,)),
        res_norm0_w=V(dtype=gs.qd_float, shape=(_B,)),
        ls_alpha=V(dtype=gs.qd_float, shape=(_B,)),
        ls_ref_norm_sq=V(dtype=gs.qd_float, shape=(_B,)),
        ls_is_done=V(dtype=gs.qd_bool, shape=(_B,)),
        ls_slope=V(dtype=gs.qd_float, shape=(_B,)),
        obj=V(dtype=gs.qd_float, shape=(_B,)),
        obj_ref=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_r=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        pcg_z=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        pcg_p=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        pcg_Ap=V(dtype=gs.qd_float, shape=(n_dofs_, _B)),
        pcg_rTz=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_rTz_new=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_pTAp=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_rTr=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_rTr0=V(dtype=gs.qd_float, shape=(_B,)),
        pcg_is_active=V(dtype=gs.qd_bool, shape=(_B,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MochiContactState:
    # Candidate (colliding link, collider geom) pairs of the current step and their per-pair accumulators: total
    # force, torque about the collider link origin, and the three 3x3 sums the rigid-rigid Hessian blocks are built
    # from (see kernel_pairs_to_blocks).
    n_pairs: qd.Tensor
    pair_link_a: qd.Tensor
    pair_link_b: qd.Tensor
    pair_geom_b: qd.Tensor
    acc_f: qd.Tensor
    acc_q: qd.Tensor
    acc_D: qd.Tensor
    acc_SD: qd.Tensor
    acc_SDS: qd.Tensor
    acc_obj: qd.Tensor
    n_hits: qd.Tensor
    # Conservative per-step world bounds of every link's sample cloud, and the motion padding of every link.
    links_step_aabb_min: qd.Tensor
    links_step_aabb_max: qd.Tensor
    links_step_pad: qd.Tensor
    # Recorded contact points for readback.
    hit_link_a: qd.Tensor
    hit_geom_a: qd.Tensor
    hit_link_b: qd.Tensor
    hit_geom_b: qd.Tensor
    hit_sample: qd.Tensor
    hit_pos: qd.Tensor
    hit_normal: qd.Tensor
    hit_force: qd.Tensor
    hit_distance: qd.Tensor
    hit_weight: qd.Tensor
    n_hits_total: qd.Tensor


def get_mochi_contact_state(solver, max_pairs, max_hits):
    _B = solver._B
    n_links_ = solver.n_links_
    return MochiContactState(
        n_pairs=V(dtype=gs.qd_int, shape=(_B,)),
        pair_link_a=V(dtype=gs.qd_int, shape=(max_pairs, _B)),
        pair_link_b=V(dtype=gs.qd_int, shape=(max_pairs, _B)),
        pair_geom_b=V(dtype=gs.qd_int, shape=(max_pairs, _B)),
        acc_f=V(dtype=gs.qd_vec3, shape=(max_pairs, _B)),
        acc_q=V(dtype=gs.qd_vec3, shape=(max_pairs, _B)),
        acc_D=V(dtype=gs.qd_mat3, shape=(max_pairs, _B)),
        acc_SD=V(dtype=gs.qd_mat3, shape=(max_pairs, _B)),
        acc_SDS=V(dtype=gs.qd_mat3, shape=(max_pairs, _B)),
        acc_obj=V(dtype=gs.qd_float, shape=(max_pairs, _B)),
        n_hits=V(dtype=gs.qd_int, shape=(max_pairs, _B)),
        links_step_aabb_min=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_step_aabb_max=V(dtype=gs.qd_vec3, shape=(n_links_, _B)),
        links_step_pad=V(dtype=gs.qd_float, shape=(n_links_, _B)),
        hit_link_a=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_geom_a=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_link_b=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_geom_b=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_sample=V(dtype=gs.qd_int, shape=(max_hits, _B)),
        hit_pos=V(dtype=gs.qd_vec3, shape=(max_hits, _B)),
        hit_normal=V(dtype=gs.qd_vec3, shape=(max_hits, _B)),
        hit_force=V(dtype=gs.qd_vec3, shape=(max_hits, _B)),
        hit_distance=V(dtype=gs.qd_float, shape=(max_hits, _B)),
        hit_weight=V(dtype=gs.qd_float, shape=(max_hits, _B)),
        n_hits_total=V(dtype=gs.qd_int, shape=(_B,)),
    )
