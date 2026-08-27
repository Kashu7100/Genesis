"""Generate the mochi reference trajectories. Run with the superdex environment:

    SUPERDEX_PRECISION=double python -m tests.mochi.reference.gen_reference [case ...]

Scenes are Y-up (mochi convention); positions are stored as recorded.
"""

import sys
from pathlib import Path

import numpy as np

CUBE_COORDINATES = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)
CUBE_CONNECTIVITY = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
    ],
    dtype=np.int32,
)


def _cube_shape(physics, size):
    coords = (0.5 * size * CUBE_COORDINATES).astype(np.float64).reshape((-1,))
    return physics.create_tri_mesh_shape(coordinates=coords, connectivity=CUBE_CONNECTIVITY.reshape((-1,)))


def _contact_params(physics, friction=0.5, normal_damping=0.0):
    params = physics.ContactParams()
    params.coulomb_friction_coefficient = friction
    params.normal_viscous_damping_coefficient = normal_damping
    return params


def _set_solver(physics, scene, *, max_iter, integrator="backward_euler", implicit_normal_force=False):
    params = scene.get_solver_params()
    params.non_linear_solver.max_iter = max_iter
    params.integration_method = (
        physics.IntegrationMethod.BDF2 if integrator == "bdf2" else physics.IntegrationMethod.BACKWARD_EULER
    )
    params.experimental_eval.implicit_normal_force_for_dissipation = implicit_normal_force
    scene.set_solver_params(params)


def _actor_state(actor):
    transform = actor.get_center_of_mass_transform()
    return (
        np.array(transform.translation.tolist() if hasattr(transform.translation, "tolist") else transform.translation),
        np.array(actor.get_linear_velocity()),
        np.array(actor.get_angular_velocity()),
    )


def case_box_drop_plane(physics):
    """Unit cube (density 1000) dropped from height 1 onto a plane, g = 10, dt = 0.05, backward Euler, 40 steps."""
    dt, n_steps = 0.05, 40
    scene = physics.create_scene("box_drop_plane")
    scene.set_gravity([0.0, -10.0, 0.0])
    _set_solver(physics, scene, max_iter=4)
    plane = scene.create_rigid_actor(
        name="plane",
        shape=physics.create_plane_shape(normal=[0.0, 1.0, 0.0], distance=0.0),
        is_static=True,
        contact=_contact_params(physics),
    )
    box = scene.create_rigid_actor(
        name="box",
        shape=_cube_shape(physics, 1.0),
        density=1000.0,
        world_from_local=physics.TransformRT(translation=[0.0, 1.0, 0.0]),
        collider_type=physics.ColliderType.BOX,
        contact=_contact_params(physics),
    )
    box.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
    pos, vel, force = [], [], []
    for _ in range(n_steps):
        scene.step(dt)
        p, v, _ = _actor_state(box)
        pos.append(p)
        vel.append(v)
        force.append(np.array(box.get_contact_force_world()))
    return {"dt": dt, "gravity": 10.0, "pos": np.array(pos), "vel": np.array(vel), "contact_force": np.array(force)}


def case_two_boxes_stack(physics):
    """Cube of side 1 resting on the plane, cube of side 0.5 dropped from 0.05 above it, dt = 0.05, 40 steps."""
    dt, n_steps = 0.05, 40
    scene = physics.create_scene("two_boxes_stack")
    scene.set_gravity([0.0, -10.0, 0.0])
    _set_solver(physics, scene, max_iter=4)
    scene.create_rigid_actor(
        name="plane",
        shape=physics.create_plane_shape(normal=[0.0, 1.0, 0.0], distance=0.0),
        is_static=True,
        contact=_contact_params(physics),
    )
    bottom = scene.create_rigid_actor(
        name="bottom",
        shape=_cube_shape(physics, 1.0),
        density=1000.0,
        world_from_local=physics.TransformRT(translation=[0.0, 0.5, 0.0]),
        collider_type=physics.ColliderType.BOX,
        contact=_contact_params(physics),
    )
    top = scene.create_rigid_actor(
        name="top",
        shape=_cube_shape(physics, 0.5),
        density=1000.0,
        world_from_local=physics.TransformRT(translation=[0.0, 1.3, 0.0]),
        collider_type=physics.ColliderType.BOX,
        contact=_contact_params(physics),
    )
    top.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
    bottom.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
    pos_bottom, pos_top, force_top, force_bottom = [], [], [], []
    for _ in range(n_steps):
        scene.step(dt)
        pos_bottom.append(_actor_state(bottom)[0])
        pos_top.append(_actor_state(top)[0])
        force_top.append(np.array(top.get_contact_force_world()))
        force_bottom.append(np.array(bottom.get_contact_force_world()))
    return {
        "dt": dt,
        "pos_bottom": np.array(pos_bottom),
        "pos_top": np.array(pos_top),
        "contact_force_top": np.array(force_top),
        "contact_force_bottom": np.array(force_bottom),
    }


def case_box_slide_friction(physics):
    """Flat box (0.4 x 0.05 x 0.4) resting on the plane, launched at 2 m/s along x, mu = 0.5, dt = 0.01."""
    dt, n_settle, n_steps = 0.01, 20, 60
    scene = physics.create_scene("box_slide_friction")
    scene.set_gravity([0.0, -10.0, 0.0])
    _set_solver(physics, scene, max_iter=12)
    scene.create_rigid_actor(
        name="plane",
        shape=physics.create_plane_shape(normal=[0.0, 1.0, 0.0], distance=0.0),
        is_static=True,
        contact=_contact_params(physics),
    )
    coords = (CUBE_COORDINATES * np.array([0.2, 0.025, 0.2])).reshape((-1,))
    shape = physics.create_tri_mesh_shape(coordinates=coords, connectivity=CUBE_CONNECTIVITY.reshape((-1,)))
    box = scene.create_rigid_actor(
        name="box",
        shape=shape,
        density=1000.0,
        world_from_local=physics.TransformRT(translation=[0.0, 0.025, 0.0]),
        collider_type=physics.ColliderType.BOX,
        contact=_contact_params(physics),
    )
    for _ in range(n_settle):
        scene.step(dt)
    box.set_velocity([2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    pos, vel = [], []
    for _ in range(n_steps):
        scene.step(dt)
        p, v, _ = _actor_state(box)
        pos.append(p)
        vel.append(v)
    return {"dt": dt, "pos": np.array(pos), "vel": np.array(vel)}


CASES = {
    "box_drop_plane": case_box_drop_plane,
    "two_boxes_stack": case_two_boxes_stack,
    "box_slide_friction": case_box_slide_friction,
}


def generate(name, data_dir):
    from superdex import physics

    physics.initialize(num_worker_threads=0)
    try:
        data = CASES[name](physics)
    finally:
        physics.shutdown()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / f"{name}.npz", **data)
    return data


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    for name in names:
        data = generate(name, Path(__file__).parent / "data")
        print(name, {key: np.asarray(value).shape for key, value in data.items()})
