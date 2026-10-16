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
def test_material_inputs(n_envs, show_viewer):
    # A sphere hitting the ground faster than the restitution threshold leaves it at the restitution of the contact
    # times its incident speed, the incident speed including the gravity of the impact step. The restitution of a
    # contact is the mean of those of its two geoms, set per environment after the build for the second sphere. A
    # falling sphere whose speed is bounded falls at the bound, the spin of a sphere decays by 1 - damping * DT per
    # step, and a sphere released deep in the ground rises at the bound of its penetration correction velocity.
    GRAVITY = 9.81
    DT = 0.005
    RESTITUTION = 0.5
    RADIUS = 0.05
    MAX_LINEAR_VELOCITY = 0.5
    SPIN = 10.0
    MAX_DEPENETRATION_VELOCITY = 0.1
    DEPTH = 0.02

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.SteelOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -2.5, 0.8),
            camera_lookat=(1.0, 0.0, 0.2),
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
    spheres = []
    for pos, sphere_material in (
        ((0.0, 0.0, 0.3), material),
        ((0.5, 0.0, 0.3), gs.materials.Rigid()),
        ((1.0, 0.0, 1.0), gs.materials.Rigid()),
        ((1.5, 0.0, 1.0), gs.materials.Rigid()),
        ((2.0, 0.0, RADIUS - DEPTH), gs.materials.Rigid()),
    ):
        spheres.append(
            scene.add_entity(
                morph=gs.morphs.Sphere(
                    pos=pos,
                    radius=RADIUS,
                ),
                material=sphere_material,
            )
        )
    scene.build(n_envs=n_envs)
    sphere, sphere_set, sphere_capped, sphere_spun, sphere_deep = spheres

    restitution_set = [0.1, 0.3] if n_envs else 0.1
    sphere_set.set_geoms_restitution(restitution_set)
    assert_allclose(sphere_set.get_geoms_restitution()[..., 0], restitution_set, tol=gs.EPS)
    sphere_capped.set_links_max_linear_velocity(MAX_LINEAR_VELOCITY)
    angular_damping = [1.0, 4.0] if n_envs else 1.0
    sphere_spun.set_links_angular_damping(angular_damping)
    sphere_spun.set_dofs_velocity([SPIN], dofs_idx_local=[5])
    sphere_deep.set_links_max_depenetration_velocity(MAX_DEPENETRATION_VELOCITY)

    scene.step()
    assert_allclose(sphere_deep.get_dofs_velocity()[..., 2], MAX_DEPENETRATION_VELOCITY, tol=1e-4)
    n_steps = 1
    restitution_contact = 0.5 * (RESTITUTION + torch.tensor(restitution_set, dtype=gs.tc_float, device=gs.device))
    vz_prev = torch.stack([sphere.get_dofs_velocity()[..., 2], sphere_set.get_dofs_velocity()[..., 2]])
    vz_rebound = torch.zeros_like(vz_prev)
    for _ in range(60):
        scene.step()
        n_steps += 1
        vz = torch.stack([sphere.get_dofs_velocity()[..., 2], sphere_set.get_dofs_velocity()[..., 2]])
        is_rebound = (vz > 0.0) & (vz_rebound == 0.0)
        vz_rebound = torch.where(is_rebound, vz / (GRAVITY * DT - vz_prev), vz_rebound)
        if (vz_rebound > 0.0).all():
            break
        vz_prev = vz
    assert_allclose(vz_rebound[0], RESTITUTION, rtol=0.02)
    assert_allclose(vz_rebound[1], restitution_contact, rtol=0.02)
    assert_allclose(sphere_capped.get_dofs_velocity()[..., 2], -MAX_LINEAR_VELOCITY, tol=1e-5)
    spin_decay = (1.0 - torch.tensor(angular_damping, dtype=gs.tc_float, device=gs.device) * DT) ** n_steps
    assert_allclose(sphere_spun.get_dofs_velocity()[..., 5], SPIN * spin_decay, rtol=1e-4)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_step_options(n_envs, show_viewer):
    # The step options change through the solver after the build. Without position correction a box released into
    # the ground keeps its penetration, which 'baumgarte' then reduces by that fraction in one step. A restitution
    # threshold above the impact speed of a bouncy sphere keeps it on the ground. The options themselves are fixed.
    DT = 0.005
    HALF_SIZE = 0.05
    DEPTH = 0.01
    BAUMGARTE = 0.2
    RADIUS = 0.05

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        rigid_options=gs.options.SteelOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, -1.5, 0.5),
            camera_lookat=(0.25, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, HALF_SIZE - DEPTH),
            size=(2 * HALF_SIZE, 2 * HALF_SIZE, 2 * HALF_SIZE),
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, 0.0, RADIUS + 0.1),
            radius=RADIUS,
        ),
        material=gs.materials.Rigid(
            restitution=1.0,
        ),
    )
    scene.build(n_envs=n_envs)

    rigid_solver = scene.rigid_solver
    rigid_solver.baumgarte = 0.0
    rigid_solver.restitution_velocity_threshold = 100.0
    for _ in range(40):
        scene.step()
    assert_allclose(HALF_SIZE - box.get_pos()[..., 2], DEPTH, tol=1e-4)
    assert_allclose(sphere.get_dofs_velocity()[..., 2], 0.0, tol=1e-3)

    rigid_solver.baumgarte = BAUMGARTE
    scene.step()
    assert_allclose(HALF_SIZE - box.get_pos()[..., 2], DEPTH * (1.0 - BAUMGARTE), tol=1e-5)

    with pytest.raises(gs.GenesisException):
        scene.options.rigid.baumgarte = BAUMGARTE
    with pytest.raises(gs.GenesisException):
        rigid_solver.contact_warmstart_decay = -1.0
    with pytest.raises(gs.GenesisException):
        rigid_solver.relaxation = float("nan")
    with pytest.raises(gs.GenesisException):
        rigid_solver.velocity_drive_mode = "solve"
    with pytest.raises(gs.GenesisException):
        rigid_solver.contact_friction_gap_max = 0.01


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_joint_constraints(mimic_hinges, four_bar_linkage, pendulum, n_envs, show_viewer):
    # A position drive carries joint2 of the mimic hinges toward a target beyond its limit: the limit holds it at 45
    # degrees and the mimic equality keeps joint1 equal to it. A velocity limit bounds the speed of a pendulum falling
    # from upright. The loop closure of the four-bar linkage keeps the rocker tip on its anchor of the base, (0.2, 0, 0)
    # in the base frame, while the crank spins. At rest, the inbound joint of every link of the linkage carries the
    # weight of the subtree of the link, whose moment about the center of mass of the link it balances.
    GRAVITY = 9.81
    LIMIT = np.deg2rad(45.0)
    VELOCITY_LIMIT = 1.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, 0.0, -GRAVITY),
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

    links_pos = linkage.get_links_pos()
    links_com = links_pos + gu.transform_by_quat(linkage.get_links_COM().expand_as(links_pos), linkage.get_links_quat())
    links_weight = linkage.get_links_mass() * GRAVITY
    scene.step()
    links_parent_wrench = linkage.get_links_parent_wrench()
    for i_l in range(linkage.get_link("base").idx_local, linkage.n_links):
        subtree_weight = links_weight[i_l:]
        lever = links_com[..., i_l:, :] - links_com[..., i_l : i_l + 1, :]
        torque = torch.stack((lever[..., 1], -lever[..., 0]), dim=-1) * subtree_weight[:, None]
        assert_allclose(links_parent_wrench[..., i_l, :2], 0.0, tol=1e-4)
        assert_allclose(links_parent_wrench[..., i_l, 2], subtree_weight.sum(), tol=1e-4)
        assert_allclose(links_parent_wrench[..., i_l, 3:5], torque.sum(dim=-2), tol=1e-4)
        assert_allclose(links_parent_wrench[..., i_l, 5], 0.0, tol=1e-4)

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
    # resting box sinks by its weight over the stiffness of its four corner contacts, each the series of two springs,
    # the second box with the stiffness its geom is set to after the build.
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
    if feature == "compliance":
        bodies[1].set_geoms_contact_stiffness(2.0 * STIFFNESS)

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
        for body, stiffness in zip(bodies, (0.5 * STIFFNESS, 2.0 / 3.0 * STIFFNESS)):
            penetration = HALF_SIZE - body.get_pos()[2]
            assert_allclose(penetration, weight / (4 * stiffness), rtol=0.02)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_constraint_capacity(n_envs, show_viewer):
    # Four boxes resting on the ground need 3 rows per corner contact, far more than the 8 free-body rows allowed: the
    # rows beyond the capacity are dropped and reported, until a reset of the environment clears the status. A setter of
    # the state keeps it.
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
    boxes = []
    for i_b in range(4):
        boxes.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(0.5 * i_b, 0.0, 0.049),
                    size=(0.1, 0.1, 0.1),
                ),
            )
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
    boxes[0].set_dofs_velocity([0.1], dofs_idx_local=[0])
    assert rigid_solver.constraint_overflow.all()
    if n_envs:
        rigid_solver.reset(envs_idx=[0])
        assert_equal(rigid_solver.constraint_overflow, [False, True])
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
