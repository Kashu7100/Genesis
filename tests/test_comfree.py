"""Tests for the ComFree complementarity-free contact solver.

Verifies that ComFree produces results comparable to Newton for
various contact and constraint scenarios, including the hybrid
path where ComFree handles contacts and the iterative solver
handles joint limits and equality constraints.
"""

import tempfile

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_numpy


def _get_rigid_solver(scene):
    """Find the RigidSolver among scene solvers."""
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    for s in scene.sim.solvers:
        if isinstance(s, RigidSolver):
            return s
    raise RuntimeError("No RigidSolver found")


def _run_comfree_vs_newton(build_scene_fn, n_steps, get_result_fn, *, n_envs=1):
    """Run a scenario with both Newton and ComFree, return both results."""
    results = {}
    for solver_type in [gs.constraint_solver.Newton, gs.constraint_solver.ComFree]:
        scene = build_scene_fn(solver_type)
        scene.build(n_envs=n_envs)
        for _ in range(n_steps):
            scene.step()
        results[solver_type] = get_result_fn(scene)
        scene.destroy()
    return results[gs.constraint_solver.Newton], results[gs.constraint_solver.ComFree]


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_box_on_plane(show_viewer):
    """Box falls onto a plane. ComFree should settle to a similar height as Newton."""

    def build(solver_type):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                dt=0.01,
                enable_collision=True,
                enable_joint_limit=False,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 1.0)))
        return scene

    def get_result(scene):
        return scene.entities[1].get_pos().cpu().numpy().flatten()[2]

    newton_z, comfree_z = _run_comfree_vs_newton(build, 200, get_result)
    assert newton_z < 0.2, f"Newton box did not settle: z={newton_z}"
    assert comfree_z < 0.2, f"ComFree box did not settle: z={comfree_z}"
    assert abs(comfree_z - newton_z) < 0.15, (
        f"ComFree z={comfree_z:.4f} too far from Newton z={newton_z:.4f}"
    )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_pendulum_with_joint_limits(show_viewer):
    """Double pendulum with joint limits hitting the ground.

    Tests the hybrid path: ComFree for contacts, iterative solver for joint limits.
    """
    mjcf = """<mujoco>
      <option gravity="0 0 -9.81" timestep="0.01"/>
      <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body pos="0 0 1.5">
          <joint type="hinge" axis="0 1 0" limited="true" range="-90 90"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5" mass="1"/>
          <body pos="0 0 -0.5">
            <joint type="hinge" axis="0 1 0" limited="true" range="-120 120"/>
            <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5" mass="1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>"""

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(mjcf)
        mjcf_path = f.name

    def build(solver_type):
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
        return scene

    def get_result(scene):
        return scene.entities[0].get_pos().cpu().numpy().flatten()[2]

    newton_z, comfree_z = _run_comfree_vs_newton(build, 300, get_result)
    assert abs(comfree_z - newton_z) < 0.5, (
        f"Pendulum ComFree z={comfree_z:.4f} too far from Newton z={newton_z:.4f}"
    )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_no_contacts_only_joint_limits(show_viewer):
    """Hinge joint with limits, no contacts. ComFree contact path is a no-op;
    the iterative solver handles joint limits alone."""
    mjcf = """<mujoco>
      <option gravity="0 0 -9.81" timestep="0.01"/>
      <worldbody>
        <body pos="0 0 2">
          <joint type="hinge" axis="0 1 0" limited="true" range="-45 45"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1"
                contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>"""

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(mjcf)
        mjcf_path = f.name

    def build(solver_type):
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
        return scene

    def get_result(scene):
        return scene.entities[0].get_qpos().cpu().numpy().flatten()[0]

    newton_q, comfree_q = _run_comfree_vs_newton(build, 200, get_result)
    assert abs(comfree_q - newton_q) < 0.1, (
        f"Joint angle ComFree={comfree_q:.4f} too far from Newton={newton_q:.4f}"
    )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_sphere_on_plane(show_viewer):
    """Sphere on plane — different contact geometry than box."""

    def build(solver_type):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                dt=0.01,
                enable_collision=True,
                enable_joint_limit=False,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0, 0, 1.0)))
        return scene

    def get_result(scene):
        return scene.entities[1].get_pos().cpu().numpy().flatten()[2]

    newton_z, comfree_z = _run_comfree_vs_newton(build, 200, get_result)
    assert newton_z < 0.2, f"Newton sphere did not settle: z={newton_z}"
    assert comfree_z < 0.2, f"ComFree sphere did not settle: z={comfree_z}"
    assert abs(comfree_z - newton_z) < 0.15, (
        f"Sphere ComFree z={comfree_z:.4f} too far from Newton z={newton_z:.4f}"
    )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_contact_force_nonzero(show_viewer):
    """Verify that ComFree reports nonzero contact forces when a box rests on a plane."""

    def build(solver_type):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                dt=0.01,
                enable_collision=True,
                enable_joint_limit=False,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.3)))
        return scene

    scene = build(gs.constraint_solver.ComFree)
    scene.build()
    for _ in range(50):
        scene.step()

    rigid = _get_rigid_solver(scene)
    contact_force = qd_to_numpy(rigid.links_state.contact_force)
    total_force_mag = np.linalg.norm(contact_force, axis=-1).sum()
    scene.destroy()

    assert total_force_mag > 0.1, (
        f"ComFree contact forces are near zero: total magnitude={total_force_mag:.6f}"
    )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_contact_force_direction(show_viewer):
    """Contact force on a resting box should be primarily vertical."""

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.ComFree,
            dt=0.01,
            enable_collision=True,
            enable_joint_limit=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.3)))
    scene.build()

    for _ in range(100):
        scene.step()

    rigid = _get_rigid_solver(scene)
    contact_force = qd_to_numpy(rigid.links_state.contact_force)
    box_link_force = contact_force[1, 0]  # link 1 = box, env 0
    scene.destroy()

    assert abs(box_link_force[2]) > 0.1, (
        f"Box contact force z-component should be nonzero: got {box_link_force}"
    )
    assert abs(box_link_force[0]) < abs(box_link_force[2]), (
        f"Lateral x force should be smaller than vertical: {box_link_force}"
    )
    assert abs(box_link_force[1]) < abs(box_link_force[2]), (
        f"Lateral y force should be smaller than vertical: {box_link_force}"
    )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_stacked_boxes(show_viewer):
    """Two boxes stacked — multi-body contact scenario."""

    def build(solver_type):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                dt=0.01,
                enable_collision=True,
                enable_joint_limit=False,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.2), pos=(0, 0, 0.5)))
        scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 1.2)))
        return scene

    def get_result(scene):
        z1 = scene.entities[1].get_pos().cpu().numpy().flatten()[2]
        z2 = scene.entities[2].get_pos().cpu().numpy().flatten()[2]
        return np.array([z1, z2])

    newton_zs, comfree_zs = _run_comfree_vs_newton(build, 300, get_result)
    for i, (nz, cz) in enumerate(zip(newton_zs, comfree_zs)):
        assert cz < 1.5, f"ComFree box {i} did not settle: z={cz}"
        assert abs(cz - nz) < 0.3, (
            f"Stacked box {i}: ComFree z={cz:.4f} too far from Newton z={nz:.4f}"
        )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_weld_constraint_with_contact(show_viewer):
    """Equality (weld) constraint + contact. Tests the hybrid path where
    ComFree handles contacts and the iterative solver handles the weld."""
    mjcf = """<mujoco>
      <option gravity="0 0 -9.81" timestep="0.01"/>
      <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body name="base" pos="0 0 1.0">
          <joint type="free"/>
          <geom type="box" size="0.1 0.1 0.1" mass="1"/>
          <body name="welded" pos="0.25 0 0">
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
          </body>
        </body>
      </worldbody>
      <equality>
        <weld body1="base" body2="welded"/>
      </equality>
    </mujoco>"""

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(mjcf)
        mjcf_path = f.name

    def build(solver_type):
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.MJCF(file=mjcf_path))
        return scene

    def get_result(scene):
        return scene.entities[0].get_pos().cpu().numpy().flatten()[2]

    newton_z, comfree_z = _run_comfree_vs_newton(build, 200, get_result)
    assert comfree_z < 0.5, f"ComFree welded body did not settle: z={comfree_z}"
    assert abs(comfree_z - newton_z) < 0.3, (
        f"Weld+contact: ComFree z={comfree_z:.4f} too far from Newton z={newton_z:.4f}"
    )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_batched_envs(show_viewer):
    """ComFree with multiple batched environments."""

    def build(solver_type):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                dt=0.01,
                enable_collision=True,
                enable_joint_limit=False,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 1.0)))
        return scene

    def get_result(scene):
        pos = scene.entities[1].get_pos().cpu().numpy()
        return pos[:, 2]  # z for all envs

    newton_zs, comfree_zs = _run_comfree_vs_newton(build, 200, get_result, n_envs=4)
    for i in range(4):
        assert comfree_zs[i] < 0.2, f"ComFree env {i} box did not settle: z={comfree_zs[i]}"
        assert abs(comfree_zs[i] - newton_zs[i]) < 0.15, (
            f"Env {i}: ComFree z={comfree_zs[i]:.4f} too far from Newton z={newton_zs[i]:.4f}"
        )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_energy_stability(show_viewer):
    """Box dropped onto plane should not gain energy over time."""

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.ComFree,
            dt=0.01,
            enable_collision=True,
            enable_joint_limit=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 1.0)))
    scene.build()

    max_z = 1.0
    for _ in range(500):
        scene.step()
        z = box.get_pos().cpu().numpy().flatten()[2]
        max_z = max(max_z, z)

    scene.destroy()
    assert max_z < 1.05, (
        f"Box exceeded initial height — energy not conserved: max_z={max_z:.4f}"
    )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_disable_constraint(show_viewer):
    """ComFree with disable_constraint=True. Tests finalize_no_iterative_solve path.

    The iterative solver is skipped, but ComFree contacts still resolve,
    so the box should settle on the plane (just without joint limits/equalities).
    """

    def build(solver_type):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                dt=0.01,
                enable_collision=True,
                enable_joint_limit=False,
                disable_constraint=True,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 1.0)))
        return scene

    scene = build(gs.constraint_solver.ComFree)
    scene.build()
    for _ in range(200):
        scene.step()

    z = scene.entities[1].get_pos().cpu().numpy().flatten()[2]
    scene.destroy()

    assert z < 0.3, (
        f"ComFree disable_constraint: box did not settle: z={z:.4f}"
    )
    assert z > -0.5, (
        f"ComFree disable_constraint: box fell through plane: z={z:.4f}"
    )


@pytest.mark.parametrize("backend", [gs.gpu])
def test_comfree_no_collision(show_viewer):
    """ComFree with enable_collision=False. Contact path is a no-op,
    so the box should free-fall."""

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.ComFree,
            dt=0.01,
            enable_collision=False,
            enable_joint_limit=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 1.0)))
    scene.build()

    for _ in range(200):
        scene.step()

    z = box.get_pos().cpu().numpy().flatten()[2]
    scene.destroy()

    assert z < -5.0, (
        f"Box should free-fall with enable_collision=False: z={z:.4f}"
    )
