"""Tests for AvatarEntity (visualization-only ghost entities)."""

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.options.solvers import AvatarOptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_scene_with_avatar(show_viewer=False):
    """Build a scene containing a ground plane, a physics Go2, and a ghost Go2."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0.5, 0.42)),
    )
    ghost = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, -0.5, 0.42)),
        material=gs.materials.Avatar(),
        surface=gs.surfaces.Default(color=(0.4, 0.7, 1.0), opacity=0.5),
    )
    scene.build()
    return scene, robot, ghost


def _go2_joint_names():
    return [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]


def _to_numpy(t):
    """Convert a tensor-like to numpy, handling both torch tensors and arrays."""
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    return np.asarray(t)


# ---------------------------------------------------------------------------
# AvatarOptions
# ---------------------------------------------------------------------------


class TestAvatarOptions:
    def test_default_dt_is_none(self):
        opts = AvatarOptions()
        assert opts.dt is None

    def test_dt_can_be_set(self):
        opts = AvatarOptions(dt=0.005)
        assert opts.dt == 0.005

    def test_no_collision_fields(self):
        """AvatarOptions should not expose physics/collision parameters."""
        opts = AvatarOptions()
        for field in [
            "enable_collision",
            "enable_self_collision",
            "enable_joint_limit",
            "gravity",
            "iterations",
            "tolerance",
        ]:
            assert not hasattr(opts, field), f"AvatarOptions should not have '{field}'"


# ---------------------------------------------------------------------------
# Entity type and hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_avatar_entity_type():
    """Avatar entities should be instances of AvatarEntity and KinematicEntity."""
    from genesis.engine.entities.avatar_entity import AvatarEntity
    from genesis.engine.entities.rigid_entity import KinematicEntity

    scene, robot, ghost = _build_scene_with_avatar()
    try:
        assert isinstance(ghost, AvatarEntity)
        assert isinstance(ghost, KinematicEntity)
        assert not isinstance(ghost, gs.engine.entities.RigidEntity)
        assert not isinstance(robot, AvatarEntity)
        assert isinstance(robot, gs.engine.entities.RigidEntity)
        assert isinstance(robot, KinematicEntity)
    finally:
        scene.destroy()


@pytest.mark.required
def test_avatar_solver_type():
    """The avatar solver should be a KinematicSolver (base class, not RigidSolver)."""
    from genesis.engine.solvers.kinematic_solver import KinematicSolver, AvatarSolver
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    scene, _, _ = _build_scene_with_avatar()
    try:
        solver = scene.avatar_solver
        assert isinstance(solver, KinematicSolver)
        assert isinstance(solver, AvatarSolver)  # AvatarSolver is alias for KinematicSolver
        assert not isinstance(solver, RigidSolver), "Avatar solver must NOT be a RigidSolver"
        assert solver.is_active
    finally:
        scene.destroy()


def test_avatar_entity_components():
    """Avatar entity should use Avatar-specific link, joint, and geom classes."""
    from genesis.engine.entities.avatar_entity.avatar_geom import AvatarGeom
    from genesis.engine.entities.avatar_entity.avatar_joint import AvatarJoint
    from genesis.engine.entities.avatar_entity.avatar_link import AvatarLink

    scene, _, ghost = _build_scene_with_avatar()
    try:
        for link in ghost.links:
            assert isinstance(link, AvatarLink)
        for joint in ghost.joints:
            assert isinstance(joint, AvatarJoint)
        for geom in ghost.geoms:
            assert isinstance(geom, AvatarGeom)
    finally:
        scene.destroy()


# ---------------------------------------------------------------------------
# Collision is disabled
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_avatar_geoms_no_collision():
    """All avatar geoms should have contype=0 and conaffinity=0."""
    scene, _, ghost = _build_scene_with_avatar()
    try:
        for geom in ghost.geoms:
            assert geom.contype == 0, f"Geom {geom.name} has contype != 0"
            assert geom.conaffinity == 0, f"Geom {geom.name} has conaffinity != 0"
    finally:
        scene.destroy()


def test_avatar_solver_no_collider():
    """The avatar solver should not have a collider."""
    scene, _, _ = _build_scene_with_avatar()
    try:
        assert scene.avatar_solver.collider is None
    finally:
        scene.destroy()


def test_avatar_solver_no_constraint_solver():
    """The avatar solver should not have a constraint solver."""
    scene, _, _ = _build_scene_with_avatar()
    try:
        assert scene.avatar_solver.constraint_solver is None
    finally:
        scene.destroy()


# ---------------------------------------------------------------------------
# Routing: Avatar material -> avatar_solver, Rigid material -> rigid_solver
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_entity_solver_routing():
    """Avatar entities go to avatar_solver; rigid entities go to rigid_solver."""
    scene, robot, ghost = _build_scene_with_avatar()
    try:
        assert ghost in scene.avatar_solver.entities
        assert ghost not in scene.rigid_solver.entities
        assert robot in scene.rigid_solver.entities
        assert robot not in scene.avatar_solver.entities
    finally:
        scene.destroy()


# ---------------------------------------------------------------------------
# set_dofs_position and forward kinematics
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_set_dofs_position_updates_qpos():
    """Setting DOF positions should update the internal qpos state."""
    scene, _, ghost = _build_scene_with_avatar()
    try:
        joint_names = _go2_joint_names()
        dof_idx = np.array([ghost.get_joint(n).dofs_idx_local[0] for n in joint_names])

        target = torch.tensor(
            [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
            dtype=gs.tc_float,
            device=gs.device,
        )
        ghost.set_dofs_position(target, dof_idx)

        # FK should be marked dirty
        assert scene.avatar_solver._fk_dirty

        # Read back the DOF positions
        readback = ghost.get_dofs_position(dof_idx)
        np.testing.assert_allclose(_to_numpy(readback), target.cpu().numpy(), atol=1e-5)
    finally:
        scene.destroy()


def test_forward_kinematics_clears_dirty_flag():
    """Calling forward_kinematics should clear the _fk_dirty flag."""
    scene, _, ghost = _build_scene_with_avatar()
    try:
        joint_names = _go2_joint_names()
        dof_idx = np.array([ghost.get_joint(n).dofs_idx_local[0] for n in joint_names])

        target = torch.zeros(12, dtype=gs.tc_float, device=gs.device)
        ghost.set_dofs_position(target, dof_idx)
        assert scene.avatar_solver._fk_dirty

        scene.avatar_solver.forward_kinematics()
        assert not scene.avatar_solver._fk_dirty
    finally:
        scene.destroy()


def test_fk_skipped_when_not_dirty():
    """forward_kinematics should be a no-op when _fk_dirty is False."""
    scene, _, _ = _build_scene_with_avatar()
    try:
        solver = scene.avatar_solver
        assert not solver._fk_dirty
        # Should return immediately without error
        solver.forward_kinematics()
        assert not solver._fk_dirty
    finally:
        scene.destroy()


# ---------------------------------------------------------------------------
# Simulation stepping doesn't crash and avatar doesn't move
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_simulation_step_with_avatar():
    """Scene with avatar entity should step without errors."""
    scene, robot, ghost = _build_scene_with_avatar()
    try:
        for _ in range(10):
            scene.step()
    finally:
        scene.destroy()


def test_avatar_position_stable_without_updates():
    """Avatar entity should not move when no DOF positions are set (no gravity/physics)."""
    scene, _, ghost = _build_scene_with_avatar()
    try:
        joint_names = _go2_joint_names()
        dof_idx = np.array([ghost.get_joint(n).dofs_idx_local[0] for n in joint_names])

        pos_before = _to_numpy(ghost.get_dofs_position(dof_idx)).copy()

        for _ in range(50):
            scene.step()

        pos_after = _to_numpy(ghost.get_dofs_position(dof_idx))
        np.testing.assert_allclose(pos_after, pos_before, atol=1e-6)
    finally:
        scene.destroy()


# ---------------------------------------------------------------------------
# Physics robot is unaffected by avatar presence
# ---------------------------------------------------------------------------


def test_rigid_entity_unaffected_by_avatar():
    """
    The physics robot should behave identically whether or not an avatar entity exists.
    We compare DOF positions of the physics robot after several steps.
    """
    # --- Run without avatar ---
    scene_no_avatar = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene_no_avatar.add_entity(gs.morphs.Plane())
    robot_alone = scene_no_avatar.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0.5, 0.42)),
    )
    scene_no_avatar.build()
    for _ in range(100):
        scene_no_avatar.step()
    pos_alone = _to_numpy(robot_alone.get_dofs_position()).copy()
    scene_no_avatar.destroy()

    # --- Run with avatar ---
    scene_with_avatar, robot_with, ghost = _build_scene_with_avatar()
    joint_names = _go2_joint_names()
    ghost_dof_idx = np.array([ghost.get_joint(n).dofs_idx_local[0] for n in joint_names])
    some_angles = torch.tensor(
        [0.1, 0.5, -1.0, 0.1, 0.5, -1.0, 0.1, 0.5, -1.0, 0.1, 0.5, -1.0],
        dtype=gs.tc_float,
        device=gs.device,
    )
    for _ in range(100):
        ghost.set_dofs_position(some_angles, ghost_dof_idx)
        scene_with_avatar.step()
    pos_with = _to_numpy(robot_with.get_dofs_position())
    scene_with_avatar.destroy()

    np.testing.assert_allclose(pos_with, pos_alone, atol=1e-5)


# ---------------------------------------------------------------------------
# State get/set
# ---------------------------------------------------------------------------


def test_get_state():
    """get_state should return an AvatarSolverState."""
    from genesis.engine.states.solvers import AvatarSolverState

    scene, _, ghost = _build_scene_with_avatar()
    try:
        scene.step()
        state = scene.avatar_solver.get_state()
        assert isinstance(state, AvatarSolverState)
        assert state.qpos is not None
    finally:
        scene.destroy()


def test_set_and_restore_state():
    """reset(state) should restore the avatar to a previously saved state."""
    scene, _, ghost = _build_scene_with_avatar()
    try:
        joint_names = _go2_joint_names()
        dof_idx = np.array([ghost.get_joint(n).dofs_idx_local[0] for n in joint_names])

        # Set a known pose
        target_a = torch.tensor(
            [0.2, 0.5, -1.0, 0.2, 0.5, -1.0, 0.2, 0.5, -1.0, 0.2, 0.5, -1.0],
            dtype=gs.tc_float,
            device=gs.device,
        )
        ghost.set_dofs_position(target_a, dof_idx)
        scene.step()

        # Save state
        state = scene.get_state()

        # Change to a different pose
        target_b = torch.zeros(12, dtype=gs.tc_float, device=gs.device)
        ghost.set_dofs_position(target_b, dof_idx)
        scene.step()

        # Restore state
        scene.reset(state)

        # Verify restored qpos matches target_a
        readback = _to_numpy(ghost.get_dofs_position(dof_idx))
        np.testing.assert_allclose(readback, target_a.cpu().numpy(), atol=1e-5)
    finally:
        scene.destroy()


# ---------------------------------------------------------------------------
# Scene with only avatar (no rigid entities except ground)
# ---------------------------------------------------------------------------


def test_avatar_only_scene():
    """A scene with only a ground plane and an avatar entity should work."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    ghost = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0, 0.42)),
        material=gs.materials.Avatar(),
    )
    scene.build()
    try:
        joint_names = _go2_joint_names()
        dof_idx = np.array([ghost.get_joint(n).dofs_idx_local[0] for n in joint_names])

        target = torch.tensor(
            [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
            dtype=gs.tc_float,
            device=gs.device,
        )
        ghost.set_dofs_position(target, dof_idx)

        for _ in range(10):
            scene.step()

        readback = _to_numpy(ghost.get_dofs_position(dof_idx))
        np.testing.assert_allclose(readback, target.cpu().numpy(), atol=1e-5)
    finally:
        scene.destroy()


# ---------------------------------------------------------------------------
# Multiple avatar entities
# ---------------------------------------------------------------------------


def test_multiple_avatar_entities():
    """Multiple avatar entities can coexist in one scene."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    ghost1 = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, -1, 0.42)),
        material=gs.materials.Avatar(),
    )
    ghost2 = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 1, 0.42)),
        material=gs.materials.Avatar(),
    )
    scene.build()
    try:
        assert len(scene.avatar_solver.entities) == 2
        assert ghost1 in scene.avatar_solver.entities
        assert ghost2 in scene.avatar_solver.entities

        joint_names = _go2_joint_names()
        dof_idx_1 = np.array([ghost1.get_joint(n).dofs_idx_local[0] for n in joint_names])
        dof_idx_2 = np.array([ghost2.get_joint(n).dofs_idx_local[0] for n in joint_names])

        target1 = torch.tensor(
            [0.1, 0.5, -1.0, 0.1, 0.5, -1.0, 0.1, 0.5, -1.0, 0.1, 0.5, -1.0],
            dtype=gs.tc_float,
            device=gs.device,
        )
        target2 = torch.tensor(
            [-0.1, 0.8, -1.5, -0.1, 0.8, -1.5, -0.1, 0.8, -1.5, -0.1, 0.8, -1.5],
            dtype=gs.tc_float,
            device=gs.device,
        )
        ghost1.set_dofs_position(target1, dof_idx_1)
        ghost2.set_dofs_position(target2, dof_idx_2)

        for _ in range(5):
            scene.step()

        rb1 = _to_numpy(ghost1.get_dofs_position(dof_idx_1))
        rb2 = _to_numpy(ghost2.get_dofs_position(dof_idx_2))

        np.testing.assert_allclose(rb1, target1.cpu().numpy(), atol=1e-5)
        np.testing.assert_allclose(rb2, target2.cpu().numpy(), atol=1e-5)
    finally:
        scene.destroy()


# ---------------------------------------------------------------------------
# Class hierarchy independence
# ---------------------------------------------------------------------------


def test_solver_hierarchy():
    """KinematicSolver is the base; RigidSolver inherits from it; AvatarSolver is an alias."""
    from genesis.engine.solvers.kinematic_solver import KinematicSolver, AvatarSolver
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
    from genesis.engine.solvers.base_solver import Solver

    assert AvatarSolver is KinematicSolver
    assert issubclass(KinematicSolver, Solver)
    assert issubclass(RigidSolver, KinematicSolver)
    assert not issubclass(KinematicSolver, RigidSolver)


def test_avatar_material_not_rigid():
    """Avatar material must not inherit from Rigid -- the classes are completely unrelated."""
    from genesis.engine.materials.base import Material

    assert not issubclass(gs.materials.Avatar, gs.materials.Rigid)
    assert issubclass(gs.materials.Avatar, Material)


def test_entity_hierarchy():
    """KinematicEntity is the base; RigidEntity and AvatarEntity inherit from it."""
    from genesis.engine.entities.avatar_entity import AvatarEntity
    from genesis.engine.entities.rigid_entity import KinematicEntity, RigidEntity

    assert issubclass(RigidEntity, KinematicEntity)
    assert issubclass(AvatarEntity, KinematicEntity)
    assert not issubclass(AvatarEntity, RigidEntity)
    assert not issubclass(RigidEntity, AvatarEntity)
