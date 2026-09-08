import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array

# The contact cache re-evaluates the candidates of the last contact search while no link or vertex moved past the
# certificate bound: the trajectory, the Newton iteration counts and the statuses must match a search at every assembly,
# and the searches must stop once the bodies rest.


def _cube_obj(path, size):
    h = size / 2
    verts = [(sx * h, sy * h, sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    faces = [
        (1, 3, 2),
        (2, 3, 4),  # -x
        (5, 6, 7),
        (6, 8, 7),  # +x
        (1, 2, 5),
        (2, 6, 5),  # -y
        (3, 7, 4),
        (4, 7, 8),  # +y
        (1, 5, 3),
        (3, 5, 7),  # -z
        (2, 4, 6),
        (4, 8, 6),  # +z
    ]
    # outward winding (the loader estimates the wall thickness by casting rays along the face normals)
    lines = [f"v {x} {y} {z}" for x, y, z in verts] + [f"f {a} {c} {b}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _run_rigid(tmp_path, show_viewer, n_steps, **mochi_kwargs):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=4, **mochi_kwargs),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    slab = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0.0, 0.0, 0.052)), material=gs.materials.Mochi.Rigid()
    )
    # a mesh collider (grid signed distance field) and analytic colliders
    cube = scene.add_entity(
        gs.morphs.Mesh(file=_cube_obj(tmp_path / "cube.obj", 0.08), pos=(0.05, 0.0, 0.16)),
        material=gs.materials.Mochi.Rigid(),
    )
    ball = scene.add_entity(gs.morphs.Sphere(radius=0.03, pos=(-0.08, 0.05, 0.2)), material=gs.materials.Mochi.Rigid())
    scene.build(n_envs=4)
    # different drop heights per environment so that the Newton solves of the batch are heterogeneous
    cube.set_pos(np.array([[0.05, 0.0, 0.16 + 0.04 * i_b] for i_b in range(4)]))
    n_iter, status, n_detections = [], [], []
    for _ in range(n_steps):
        scene.step()
        info = scene.mochi_solver.get_convergence_info()
        n_iter.append(tensor_to_array(info["n_iter"]))
        status.append(tensor_to_array(info["status"]))
        n_detections.append(tensor_to_array(info["n_detections"]))
    state = (
        tensor_to_array(slab.get_pos()),
        tensor_to_array(cube.get_pos()),
        tensor_to_array(cube.get_quat()),
        tensor_to_array(ball.get_pos()),
    )
    return state, np.array(n_iter), np.array(status), np.array(n_detections)


@pytest.mark.precision("64")
def test_contact_cache_matches_search_rigid(tmp_path, show_viewer):
    n_steps = 90
    reference = _run_rigid(tmp_path, show_viewer, n_steps, contact_cache=False)
    assert reference[1].max() >= 1
    assert (reference[3] == 0).all()
    for step_kernel, margin in (("monolith", 2e-3), ("pipeline", 2e-3), ("monolith", 1e-2)):
        cached = _run_rigid(
            tmp_path, show_viewer, n_steps, contact_cache=True, contact_candidate_margin=margin, step_kernel=step_kernel
        )
        for a, b in zip(reference[0], cached[0]):
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)
        np.testing.assert_array_equal(reference[1], cached[1])
        np.testing.assert_array_equal(reference[2], cached[2])
        # one search at the start of every step, then at most one per line-search trial that moved past the bound
        n_iter, n_detections = cached[1], cached[3]
        assert (n_detections >= 1).all()
        assert (n_detections <= 1 + 4 * n_iter).all()
        # the bodies rest at the end: nothing moves past the bound, the start-of-step search is the only one
        assert (n_detections[-15:] == 1).all()
        # falling bodies move more than the bound between the start of a step and the first trial
        assert (n_detections[:5] >= 2).all()


def _strip_obj(path, n_x, n_y, length, width):
    xs = np.linspace(-0.5 * length, 0.5 * length, n_x + 1)
    ys = np.linspace(-0.5 * width, 0.5 * width, n_y + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_x):
        for j in range(n_y):
            a = i * (n_y + 1) + j
            faces.append([a, a + 1, a + n_y + 2])
            faces.append([a, a + n_y + 2, a + n_y + 1])
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return str(path)


def _record(scene, n_steps, callback=None):
    n_iter, status, n_detections = [], [], []
    for i_step in range(n_steps):
        if callback is not None:
            callback(i_step)
        scene.step()
        info = scene.mochi_solver.get_convergence_info()
        n_iter.append(tensor_to_array(info["n_iter"]))
        status.append(tensor_to_array(info["status"]))
        n_detections.append(tensor_to_array(info["n_detections"]))
    return np.array(n_iter), np.array(status), np.array(n_detections)


def _run_soft(show_viewer, n_steps, **mochi_kwargs):
    # deformable samples against rigid colliders (plane, box) and against the tetrahedra of another solid
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=4, **mochi_kwargs),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    slab = scene.add_entity(
        gs.morphs.Box(size=(0.4, 0.4, 0.1), pos=(0.0, 0.0, 0.05), maxvolume=0.001, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=2e5, nu=0.4, rho=1000.0),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.12, 0.12, 0.1), pos=(0.0, 0.0, 0.2)), material=gs.materials.Mochi.Rigid()
    )
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.08, 0.08, 0.08), pos=(0.12, -0.1, 0.16), maxvolume=0.0002, nobisect=False),
        material=gs.materials.Mochi.Elastic(E=1e5, nu=0.4, rho=800.0),
    )
    scene.build(n_envs=3)
    box.set_pos(np.array([[0.0, 0.0, 0.2], [0.05, 0.0, 0.25], [-0.05, 0.03, 0.3]]))
    n_iter, status, n_detections = _record(scene, n_steps)
    state = (
        tensor_to_array(slab.get_vertices_position()),
        tensor_to_array(cube.get_vertices_position()),
        tensor_to_array(box.get_pos()),
        tensor_to_array(box.get_quat()),
    )
    return state, n_iter, status, n_detections


def _run_cloth(tmp_path, show_viewer, n_steps, **mochi_kwargs):
    # a shell strip folded onto itself: self-contact through the collider spheres, plus the ground
    length, width, n_x, n_y, radius = 0.6, 0.15, 12, 3, 0.02
    obj_path = _strip_obj(tmp_path / "strip.obj", n_x, n_y, length, width)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8, **mochi_kwargs),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid(penalty_coefficient=1e7))
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 1e-3)),
        material=gs.materials.Mochi.Shell(
            E=1e5,
            nu=0.3,
            rho=200.0,
            thickness=1e-3,
            collider_radius=radius,
            penalty_coefficient=1e7,
            mass_damping=5.0,
            self_contact=True,
        ),
    )
    scene.build()
    rest = tensor_to_array(cloth.get_vertices_position())
    pinned = np.flatnonzero(rest[:, 0] < -0.5 * length + 1e-6)
    driven = np.flatnonzero(rest[:, 0] > 0.5 * length - 1e-6)
    cloth.set_vertices_fixed(pinned)
    n_fold = 45

    def fold(i_step):
        if i_step >= n_fold:
            return
        angle = np.pi * (i_step + 1) / n_fold
        target = rest[driven].copy()
        target[:, 0] = 0.5 * length * np.cos(angle)
        target[:, 2] = 0.5 * length * np.sin(angle) + 1.5 * radius * (i_step + 1) / n_fold + 1e-3
        cloth.set_vertices_target(driven, target)

    n_iter, status, n_detections = _record(scene, n_steps, fold)
    return (tensor_to_array(cloth.get_vertices_position()),), n_iter, status, n_detections


def _assert_cached_matches(reference, cached, rtol, atol):
    for a, b in zip(reference[0], cached[0]):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(reference[1], cached[1])
    np.testing.assert_array_equal(reference[2], cached[2])
    n_iter, n_detections = cached[1], cached[3]
    assert (n_detections >= 1).all()
    assert (n_detections <= 1 + 4 * n_iter).all()


@pytest.mark.precision("64")
def test_contact_cache_matches_search_soft(show_viewer):
    n_steps = 30
    reference = _run_soft(show_viewer, n_steps, contact_cache=False)
    assert reference[1].max() >= 1
    for step_kernel, margin in (("monolith", 2e-3), ("pipeline", 2e-3), ("monolith", 0.0)):
        cached = _run_soft(
            show_viewer, n_steps, contact_cache=True, step_kernel=step_kernel, contact_candidate_margin=margin
        )
        # the batched assembly and the per-environment assembly sum the same terms in different orders; so do the
        # candidate lists and the search, and the conjugate gradient then differs at rounding level
        _assert_cached_matches(reference, cached, rtol=1e-6, atol=1e-7)
        if margin > 0.0:
            # the searches stop for bodies at rest
            assert (cached[3][-5:] <= 3).all()
        else:
            # a zero margin skips only the assemblies where nothing moved: the Hessian pass after an accepted trial
            assert (cached[3] < 1 + 5 * cached[1]).all()


@pytest.mark.precision("64")
def test_contact_cache_matches_search_cloth(tmp_path, show_viewer):
    n_steps = 75
    reference = _run_cloth(tmp_path, show_viewer, n_steps, contact_cache=False)
    assert reference[1].max() >= 1
    cached = _run_cloth(tmp_path, show_viewer, n_steps, contact_cache=True)
    _assert_cached_matches(reference, cached, rtol=1e-6, atol=1e-7)
