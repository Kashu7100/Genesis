import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


@pytest.fixture(scope="session")
def grid_sheet_path(asset_tmp_path):
    def make(n_x, n_y, size_x, size_y):
        """Write a flat rectangular sheet in the xy-plane, centered at the origin, each cell split along alternating
        diagonals, and return the path of the mesh file."""
        path = asset_tmp_path / f"shell_grid_{n_x}x{n_y}_{size_x}x{size_y}.obj"
        xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, n_x + 1) * size_x, np.linspace(-0.5, 0.5, n_y + 1) * size_y)
        verts = np.stack((xs.T.reshape(-1), ys.T.reshape(-1), np.zeros(xs.size)), axis=-1)
        faces = []
        for i in range(n_x):
            for j in range(n_y):
                v00, v10, v11, v01 = (
                    i * (n_y + 1) + j,
                    (i + 1) * (n_y + 1) + j,
                    (i + 1) * (n_y + 1) + j + 1,
                    i * (n_y + 1) + j + 1,
                )
                faces += [(v00, v10, v11), (v00, v11, v01)] if (i + j) % 2 == 0 else [(v00, v10, v01), (v10, v11, v01)]
        trimesh.Trimesh(verts, np.array(faces), process=False).export(path)
        return str(path)

    return make


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_membrane_stretching_stiffness(n_envs, grid_sheet_path, show_viewer):
    # A strip hanging from its top edge stretches under its own weight by rho * g * L^2 / (2 * E), for a Poisson's
    # ratio of zero which leaves its width free. The Green strain stiffens the strip at finite strain, by 1.5 times
    # the strain at most, so the strain is kept below 1e-3.
    GRAVITY, LENGTH, E, RHO = 9.81, 0.4, 1e7, 1000.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.8),
        ),
        show_viewer=show_viewer,
    )
    strip = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=grid_sheet_path(16, 4, LENGTH, 0.1),
            pos=(0.0, 0.0, 0.8),
            euler=(0.0, 90.0, 0.0),
        ),
        material=gs.materials.Shell(
            rho=RHO,
            E=E,
            nu=0.0,
            damping=0.05,
        ),
    )
    scene.build(n_envs=n_envs)

    init_verts = strip.init_verts
    verts_top = np.flatnonzero(init_verts[:, 2] > init_verts[:, 2].max() - gs.EPS)
    verts_bottom = np.flatnonzero(init_verts[:, 2] < init_verts[:, 2].min() + gs.EPS)
    strip.fix_verts(verts_top)
    for _ in range(80):
        scene.step()

    verts_pos = tensor_to_array(strip.get_verts_pos())
    elongation = init_verts[verts_bottom, 2].mean() - verts_pos[..., verts_bottom, 2].mean(axis=-1)
    assert_allclose(elongation, RHO * GRAVITY * LENGTH**2 / (2.0 * E), rtol=2e-3)
    assert_allclose(strip.get_verts_vel(), 0.0, atol=1e-5)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_tearing_at_tensile_strength(grid_sheet_path, show_viewer):
    # Two strips are pulled apart at their ends, faster in the second environment. Fracture splits a strip once its
    # membrane stress E * G (Green strain G, Poisson's ratio zero) reaches the tensile strength, and the weak strip
    # breaks into separate pieces while the strong one stays whole. Resetting an environment restores its mesh.
    E, TENSILE_STRENGTH, LENGTH = 1e6, 3e4, 0.4
    PULL_SPEEDS = (0.05, 0.1)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.0, 1.0),
            camera_lookat=(0.0, 0.15, 0.5),
        ),
        show_viewer=show_viewer,
    )
    strips = [
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=grid_sheet_path(16, 4, LENGTH, 0.1),
                pos=(0.0, 0.3 * i, 0.5),
            ),
            material=gs.materials.Shell(
                E=E,
                nu=0.0,
                tensile_strength=tensile_strength,
            ),
        )
        for i, tensile_strength in enumerate((TENSILE_STRENGTH, 1e3 * TENSILE_STRENGTH))
    ]
    scene.build(n_envs=2)

    for strip in strips:
        init_verts = strip.init_verts
        verts_left = np.flatnonzero(init_verts[:, 0] < init_verts[:, 0].min() + gs.EPS)
        verts_right = np.flatnonzero(init_verts[:, 0] > init_verts[:, 0].max() - gs.EPS)
        strip.fix_verts(np.concatenate((verts_left, verts_right)))
        for i_b, speed in enumerate(PULL_SPEEDS):
            strip.set_verts_vel((-0.5 * speed, 0.0, 0.0), verts_left, envs_idx=[i_b])
            strip.set_verts_vel((0.5 * speed, 0.0, 0.0), verts_right, envs_idx=[i_b])

    def count_pieces(strip):
        faces = tensor_to_array(strip.get_faces())
        n_pieces = []
        for faces_env in faces:
            adjacency = sp.coo_matrix(
                (np.ones(faces_env.size), (faces_env.reshape(-1), faces_env[:, (1, 2, 0)].reshape(-1))),
                shape=(strip.n_verts_max, strip.n_verts_max),
            )
            _, labels = csgraph.connected_components(adjacency, directed=False)
            n_pieces.append(len(np.unique(labels[faces_env])))
        return np.array(n_pieces)

    # The strip breaks at the strain whose Green strain reaches TENSILE_STRENGTH / E
    strain_break = np.sqrt(1.0 + 2.0 * TENSILE_STRENGTH / E) - 1.0
    steps_break = [strain_break * LENGTH / speed / scene.sim.dt for speed in PULL_SPEEDS]
    for i in range(int(1.15 * max(steps_break))):
        scene.step()
        n_pieces = count_pieces(strips[0])
        for i_b in range(2):
            if i + 1 < 0.9 * steps_break[i_b]:
                assert n_pieces[i_b] == 1
    assert (count_pieces(strips[0]) >= 2).all()
    assert_equal(count_pieces(strips[1]), 1)
    assert_equal(strips[1].get_n_verts(), strips[1].n_verts)

    scene.reset(envs_idx=[1])
    assert_equal(count_pieces(strips[0]), (count_pieces(strips[0])[0], 1))
    assert_equal(strips[0].get_n_verts()[1], strips[0].n_verts)
    assert_equal(strips[0].get_faces()[1], strips[0].init_faces)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rigid_coupling_transmits_weight(n_envs, grid_sheet_path, show_viewer):
    # A sheet dropped on a free box resting on the ground settles half its thickness above the box top, and the
    # ground then carries the weight of both.
    GRAVITY, SIZE, THICKNESS, RHO, BOX_HEIGHT = 9.81, 0.15, 2e-3, 1000.0, 0.05

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.5, 0.4),
            camera_lookat=(0.0, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, BOX_HEIGHT),
            pos=(0.0, 0.0, 0.5 * BOX_HEIGHT),
        ),
        material=gs.materials.Rigid(
            rho=100.0,
        ),
    )
    sheet = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=grid_sheet_path(10, 10, SIZE, SIZE),
            pos=(0.0, 0.0, BOX_HEIGHT + 0.01),
        ),
        material=gs.materials.Shell(
            rho=RHO,
            E=1e6,
            thickness=THICKNESS,
            damping=0.01,
        ),
    )
    scene.build(n_envs=n_envs)

    for _ in range(100):
        scene.step()

    box_top = tensor_to_array(box.get_pos())[..., 2] + 0.5 * BOX_HEIGHT
    verts_height = tensor_to_array(sheet.get_verts_pos())[..., 2] - box_top[..., None]
    assert_allclose(verts_height, 0.5 * THICKNESS, atol=5e-4)
    sheet_mass = RHO * THICKNESS * SIZE**2
    ground_force = tensor_to_array(box.get_links_net_contact_force())[..., 0, 2]
    assert_allclose(ground_force, (box.get_mass() + sheet_mass) * GRAVITY, rtol=2e-3)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_plastic_stretching(grid_sheet_path, show_viewer):
    # Two strips are stretched by 10% and released. The one whose yield stress the stretching exceeds keeps a
    # permanent elongation, the elastic one recovers its length.
    E, LENGTH = 1e6, 0.4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.0, 1.0),
            camera_lookat=(0.0, 0.15, 0.5),
        ),
        show_viewer=show_viewer,
    )
    strips = [
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=grid_sheet_path(16, 4, LENGTH, 0.1),
                pos=(0.0, 0.3 * i, 0.5),
            ),
            material=gs.materials.Shell(
                E=E,
                nu=0.0,
                damping=0.02,
                yield_stress=yield_stress,
            ),
        )
        for i, yield_stress in enumerate((0.02 * E, None))
    ]
    scene.build()

    verts_ends = []
    for strip in strips:
        init_verts = strip.init_verts
        verts_left = np.flatnonzero(init_verts[:, 0] < init_verts[:, 0].min() + gs.EPS)
        verts_right = np.flatnonzero(init_verts[:, 0] > init_verts[:, 0].max() - gs.EPS)
        strip.fix_verts(np.concatenate((verts_left, verts_right)))
        strip.set_verts_vel((-0.2, 0.0, 0.0), verts_left)
        strip.set_verts_vel((0.2, 0.0, 0.0), verts_right)
        verts_ends.append((verts_left, verts_right))
    for _ in range(100):
        scene.step()
    for strip in strips:
        strip.release_verts()
        strip.set_verts_vel(0.0)
    for _ in range(200):
        scene.step()

    lengths = []
    for strip, (verts_left, verts_right) in zip(strips, verts_ends):
        verts_pos = tensor_to_array(strip.get_verts_pos())
        lengths.append(verts_pos[verts_right, 0].mean() - verts_pos[verts_left, 0].mean())
    assert lengths[0] > 1.03 * LENGTH
    assert_allclose(lengths[1], LENGTH, rtol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_unsupported_shell_inputs(grid_sheet_path, asset_tmp_path):
    with pytest.raises(gs.GenesisException, match="Poisson"):
        gs.materials.Shell(nu=0.5)

    # Two triangles sharing a single vertex form two fans around it.
    bowtie_path = asset_tmp_path / "shell_bowtie.obj"
    trimesh.Trimesh(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 0, 0], [-1, -1, 0]], dtype=float),
        np.array([[0, 1, 2], [0, 3, 4]]),
        process=False,
    ).export(bowtie_path)
    scene = gs.Scene()
    with pytest.raises(gs.GenesisException, match="not manifold"):
        scene.add_entity(morph=gs.morphs.Mesh(file=str(bowtie_path)), material=gs.materials.Shell())

    scene = gs.Scene(coupler_options=gs.options.SAPCouplerOptions())
    scene.add_entity(morph=gs.morphs.Mesh(file=grid_sheet_path(2, 2, 0.1, 0.1)), material=gs.materials.Shell())
    with pytest.raises(gs.GenesisException, match="LegacyCouplerOptions"):
        scene.build()
