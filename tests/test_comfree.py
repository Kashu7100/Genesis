"""Tests for the ComFree complementarity-free contact solver.

Verifies that ComFree produces results comparable to Newton for
various contact and constraint scenarios, including the hybrid
path where ComFree handles contacts and the iterative solver
handles joint limits and equality constraints.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_numpy


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
    mjcf = """
    <mujoco>
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
    </mujoco>
    """

    def build(solver_type):
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.MJCF(string=mjcf))
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
    mjcf = """
    <mujoco>
      <option gravity="0 0 -9.81" timestep="0.01"/>
      <worldbody>
        <body pos="0 0 2">
          <joint type="hinge" axis="0 1 0" limited="true" range="-45 45"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1"
                contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>
    """

    def build(solver_type):
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                constraint_solver=solver_type,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.MJCF(string=mjcf))
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

    contact_force = qd_to_numpy(scene.sim.solvers[0].links_state.contact_force)
    total_force_mag = np.linalg.norm(contact_force, axis=-1).sum()
    scene.destroy()

    assert total_force_mag > 0.1, (
        f"ComFree contact forces are near zero: total magnitude={total_force_mag:.6f}"
    )
