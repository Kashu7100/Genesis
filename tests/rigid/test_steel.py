import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize(
    "steel_kwargs",
    [
        {},
        {"friction_anchor_beta": 0.0},
        {"solve_mode": "split", "enable_restitution": False},
        {"articulated_contact_response": "propagation", "contact_warmstart": True},
    ],
)
@pytest.mark.parametrize("n_envs", [0, 2])
def test_contact_dynamics(steel_kwargs, n_envs, show_viewer):
    # A box sliding at V0 decelerates at MU * g, a resting box is held by a contact force equal to its weight, and a
    # sphere far above the ground falls freely under semi-implicit Euler: z_n = z_0 - g dt^2 n (n + 1) / 2.
    GRAVITY = 9.81
    DT = 0.005
    MU = 0.5
    V0 = 2.0
    HALF_SIZE = 0.05
    Z0 = 3.0
    N_STEPS = 40

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.SteelOptions(
            **steel_kwargs,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -2.5, 1.0),
            camera_lookat=(0.5, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.Rigid(
        friction=MU,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=material,
    )
    box_slide = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, HALF_SIZE),
            size=(2 * HALF_SIZE, 2 * HALF_SIZE, 2 * HALF_SIZE),
        ),
        material=material,
    )
    box_rest = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(1.0, 0.0, HALF_SIZE),
            size=(2 * HALF_SIZE, 2 * HALF_SIZE, 2 * HALF_SIZE),
        ),
        material=material,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, 1.0, Z0),
            radius=HALF_SIZE,
        ),
        material=material,
    )
    scene.build(n_envs=n_envs)

    box_slide.set_dofs_velocity([V0, 0.0, 0.0], dofs_idx_local=slice(0, 3))
    support_forces = []
    for _ in range(N_STEPS):
        scene.step()
        support_forces.append(box_rest.get_links_net_contact_force()[..., 0, 2])

    t = N_STEPS * DT
    assert_allclose(box_slide.get_dofs_velocity()[..., 0], V0 - MU * GRAVITY * t, atol=2e-3)
    assert_allclose(box_slide.get_dofs_velocity()[..., 1:3], 0.0, atol=2e-3)
    assert_allclose(box_rest.get_dofs_velocity(), 0.0, atol=2e-3)
    # The contact force of a resting body fluctuates around its weight from one step to the next.
    assert_allclose(torch.stack(support_forces[N_STEPS // 2 :]).mean(dim=0), box_rest.get_mass() * GRAVITY, rtol=0.01)
    z_free_fall = Z0 - GRAVITY * DT**2 * N_STEPS * (N_STEPS + 1) / 2
    assert_allclose(sphere.get_pos()[..., 2], z_free_fall, tol=gs.EPS * 10.0 + 1e-5)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_restitution(n_envs, show_viewer):
    # A sphere hitting the ground faster than the restitution threshold leaves it at RESTITUTION times its incident
    # speed, the incident speed including the gravity of the impact step.
    GRAVITY = 9.81
    DT = 0.005
    RESTITUTION = 0.5
    RADIUS = 0.05

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.SteelOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.5, 0.5),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.Rigid(
        restitution=RESTITUTION,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=material,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.3),
            radius=RADIUS,
        ),
        material=material,
    )
    scene.build(n_envs=n_envs)

    vz_prev = sphere.get_dofs_velocity()[..., 2]
    for _ in range(60):
        scene.step()
        vz = sphere.get_dofs_velocity()[..., 2]
        if (vz > 0.0).all():
            break
        vz_prev = vz
    assert (vz > 0.0).all()
    assert_allclose(vz, -RESTITUTION * (vz_prev - GRAVITY * DT), rtol=0.02)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_joint_constraints(mimic_hinges, four_bar_linkage, pendulum, n_envs, show_viewer):
    # A position drive carries joint2 of the mimic hinges toward a target beyond its limit: the limit holds it at 45
    # degrees and the mimic equality keeps joint1 equal to it. A velocity limit bounds the speed of a pendulum falling
    # from upright. The loop closure of the four-bar linkage keeps the rocker tip on its anchor of the base, (0.2, 0, 0)
    # in the base frame, while the crank spins.
    LIMIT = np.deg2rad(45.0)
    VELOCITY_LIMIT = 1.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        rigid_options=gs.options.SteelOptions(
            enable_joint_limit=True,
            enable_joint_velocity_limits=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -4.0, 1.5),
            camera_lookat=(1.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    mimic = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ET.tostring(mimic_hinges, encoding="unicode"),
        ),
    )
    linkage = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ET.tostring(four_bar_linkage, encoding="unicode"),
            pos=(2.0, 0.0, 0.0),
        ),
    )
    pendulum_entity = scene.add_entity(
        morph=gs.morphs.URDF(
            file=ET.tostring(pendulum, encoding="unicode"),
            pos=(-1.0, 0.0, 0.0),
            fixed=True,
        ),
    )
    scene.build(n_envs=n_envs)

    mimic.set_dofs_kp([0.0, 200.0])
    mimic.set_dofs_kv([0.0, 20.0])
    scene.sim.rigid_solver.set_dofs_velocity_limit(
        VELOCITY_LIMIT, range(pendulum_entity.dof_start, pendulum_entity.dof_start + pendulum_entity.n_dofs)
    )
    pendulum_entity.set_dofs_position([0.3])
    linkage.set_dofs_velocity([3.0, -3.0, 3.0])
    rocker = linkage.get_link("rocker")
    for _ in range(80):
        mimic.control_dofs_position([0.0, 1.5])
        scene.step()
        assert (torch.abs(pendulum_entity.get_dofs_velocity()) <= VELOCITY_LIMIT + 1e-3).all()
    assert (torch.abs(pendulum_entity.get_dofs_velocity()) > 0.9 * VELOCITY_LIMIT).all()
    q = mimic.get_dofs_position()
    assert_allclose(q[..., 1], LIMIT, atol=2e-3)
    assert_allclose(q[..., 0], q[..., 1], atol=2e-3)
    rocker_tip = rocker.get_pos() + gu.transform_by_quat(
        torch.tensor([0.0, 0.0, 0.2], dtype=gs.tc_float, device=gs.device), rocker.get_quat()
    )
    assert_allclose(rocker_tip, (2.2, 0.0, 1.0), atol=2e-3)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_sleeping(n_envs, show_viewer):
    # A box resting on the ground falls asleep once quiet for 'sleep_quiet_time': its state is frozen exactly. Setting
    # the velocity of one environment wakes its island alone.
    DT = 0.005
    QUIET_TIME = 0.1

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        rigid_options=gs.options.SteelOptions(
            enable_sleeping=True,
            sleep_quiet_time=QUIET_TIME,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.5, 0.5),
            camera_lookat=(0.0, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.1, 0.1, 0.1),
        ),
    )
    scene.build(n_envs=n_envs)

    for _ in range(round(QUIET_TIME / DT) + 10):
        scene.step()
    pos_asleep = box.get_pos()
    for _ in range(5):
        scene.step()
        assert_equal(box.get_dofs_velocity(), 0.0)
        assert_equal(box.get_pos(), pos_asleep)

    box.set_dofs_velocity([0.5, 0.0, 0.0], dofs_idx_local=slice(0, 3), envs_idx=[0] if n_envs else None)
    for _ in range(5):
        scene.step()
    displacement = (box.get_pos() - pos_asleep)[..., 0]
    if n_envs:
        assert (displacement[0] > 0.005).all()
        assert_equal(displacement[1:], 0.0)
    else:
        assert displacement > 0.005


@pytest.mark.required
@pytest.mark.parametrize("feature", ["torsion", "compliance"])
def test_contact_models(feature, yaw_puck, show_viewer):
    # With torsional friction, a puck under a yaw torque below MU * m * g * RADIUS keeps still and spins above it: its
    # footprint is too small for the friction of its corners alone to hold half that torque. With contact compliance, a
    # resting box sinks by its weight over the stiffness of its four corner contacts, each the series of two springs.
    GRAVITY = 9.81
    MU = 0.5
    RADIUS = 0.02
    STIFFNESS = 2.0e4
    HALF_SIZE = 0.05

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.SteelOptions(
            enable_torsional_friction=feature == "torsion",
            friction_anchor_beta=0.0,
            contact_compliance=feature == "compliance",
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -1.0, 0.4),
            camera_lookat=(0.5, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    if feature == "torsion":
        material = gs.materials.Rigid(
            friction=MU,
            friction_torsional=RADIUS,
        )
    else:
        material = gs.materials.Rigid(
            friction=MU,
            contact_stiffness=STIFFNESS,
            contact_damping=100.0,
        )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=material,
    )
    bodies = []
    for i_e in range(2):
        if feature == "torsion":
            morph = gs.morphs.MJCF(
                file=ET.tostring(yaw_puck, encoding="unicode"),
                pos=(1.0 * i_e, 0.0, 0.0),
            )
        else:
            morph = gs.morphs.Box(
                pos=(1.0 * i_e, 0.0, HALF_SIZE),
                size=(2 * HALF_SIZE, 2 * HALF_SIZE, 2 * HALF_SIZE),
            )
        bodies.append(
            scene.add_entity(
                morph=morph,
                material=material,
            )
        )
    scene.build()

    weight = bodies[0].get_mass() * GRAVITY
    for _ in range(60):
        if feature == "torsion":
            for body, factor in zip(bodies, (0.5, 2.0)):
                body.control_dofs_force([0.0, 0.0, 0.0, factor * MU * weight * RADIUS])
        scene.step()
    if feature == "torsion":
        assert_allclose(bodies[0].get_dofs_velocity()[3], 0.0, atol=1e-3)
        assert bodies[1].get_dofs_velocity()[3] > 0.5
    else:
        penetration = HALF_SIZE - bodies[0].get_pos()[2]
        assert_allclose(penetration, weight / (4 * 0.5 * STIFFNESS), rtol=0.02)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_constraint_capacity(n_envs, show_viewer):
    # Four boxes resting on the ground need 3 rows per corner contact, far more than the 8 free-body rows allowed: the
    # rows beyond the capacity are dropped and reported, until a reset clears the status.
    scene = gs.Scene(
        rigid_options=gs.options.SteelOptions(
            free_body_constraint_max=8,
            row_watermark=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -2.0, 1.0),
            camera_lookat=(1.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    for i_b in range(4):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.5 * i_b, 0.0, 0.049),
                size=(0.1, 0.1, 0.1),
            ),
        )
    scene.build(n_envs=n_envs)

    rigid_solver = scene.sim.rigid_solver
    assert not rigid_solver.constraint_overflow.any()
    for _ in range(2):
        scene.step()
    assert rigid_solver.constraint_overflow.all()
    with pytest.raises(gs.GenesisException):
        rigid_solver.check_constraint_capacity()
    watermarks = rigid_solver.constraint_row_watermarks
    assert watermarks.free_body.high_water <= 8 < watermarks.free_body.raw_high_water
    assert watermarks.free_body.overflow_env_steps > 0
    assert watermarks.contact_high_water >= 16
    scene.reset()
    assert not rigid_solver.constraint_overflow.any()


@pytest.mark.required
def test_invalid_options():
    with pytest.raises(gs.GenesisException):
        gs.options.SteelOptions(constraint_solver=gs.constraint_solver.Newton)
    for steel_kwargs in (
        dict(solve_mode="split", contact_warmstart=True),
        dict(solve_mode="split", enable_torsional_friction=True),
        dict(contact_compliance=True, friction_anchor_beta=0.2),
        dict(friction_mode="bisection", friction_anchor_beta=0.2),
        dict(enable_sleeping=True, contact_warmstart=True),
        dict(enable_sleeping=True, articulated_contact_response="propagation"),
        dict(contact_warmstart=True, contact_friction_position_iterations=2),
    ):
        with pytest.raises(gs.GenesisException):
            scene = gs.Scene(rigid_options=gs.options.SteelOptions(**steel_kwargs), show_viewer=False)
            scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
            scene.build()
