"""
Custom Visual Mesh in Genesis
==============================

Demonstrates how to use ``set_vverts()`` and ``set_vmesh()`` on a
``KinematicEntity`` to update visual mesh vertices (and optionally faces)
at runtime.

``set_vverts()``
    Fast path — only vertex positions change (topology unchanged).
    Ideal for SMPL-style body models where vertex count and faces stay
    constant across frames.

``set_vmesh()``
    General path — both vertices *and* faces can change between frames.
    The renderer rebuilds GPU mesh objects when topology changes, which
    is slower but supports arbitrary mesh updates.

When ``use_visual_raycasting`` is enabled, depth cameras and lidars see
the custom mesh instead of the collision hull.

Requirements (only for SMPL demo)
----------------------------------
- ``smplx`` package  (``pip install smplx``)
- SMPL model files   (download from https://smpl.is.tue.mpg.de)

Usage
-----
    # Wave-deforming box (no external dependencies)
    python custom_visual_mesh.py -v

    # SMPL body mesh (requires smplx)
    python custom_visual_mesh.py -v --smpl --model_path /path/to/smpl/models

    # Batched
    python custom_visual_mesh.py -v -B 4
"""

import argparse
import os
import tempfile

import numpy as np
import trimesh

import genesis as gs


# ----------------------------- helpers ---------------------------------


def create_box_mesh(size=0.5, subdivisions=3):
    """Create a subdivided box mesh suitable for vertex deformation."""
    mesh = trimesh.creation.box(extents=(size, size, size))
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def wave_deform(verts, t, amplitude=0.1, freq=3.0):
    """Apply a sinusoidal wave deformation along the z-axis."""
    deformed = verts.copy()
    deformed[:, 2] += amplitude * np.sin(freq * verts[:, 0] + t) * np.cos(freq * verts[:, 1] + t * 0.7)
    return deformed


# ----------------------------- main ------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Custom visual mesh in Genesis")
    parser.add_argument("-v", "--vis", action="store_true", help="Open interactive viewer")
    parser.add_argument("-B", "--num_envs", type=int, default=0, help="Number of parallel envs (0 = unbatched)")
    parser.add_argument("--smpl", action="store_true", help="Use SMPL body model (requires smplx)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to SMPL model directory")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
            max_FPS=60,
        ),
        rigid_options=gs.options.RigidOptions(dt=0.01, gravity=(0, 0, 0)),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    if args.smpl:
        # -------- SMPL path (set_vverts) --------
        import smplx
        import torch

        if args.model_path is None:
            print("Error: --model_path is required for SMPL demo")
            return

        B = max(1, args.num_envs)
        smpl = smplx.SMPL(model_path=args.model_path, gender="neutral", batch_size=B)

        # Export T-pose mesh for Genesis to load as rigid entity
        output = smpl()
        t_pose_verts = output.vertices.detach().cpu().numpy().squeeze()
        faces = smpl.faces.astype(np.int32)
        mesh = trimesh.Trimesh(vertices=t_pose_verts, faces=faces, process=False)
        tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        mesh.export(tmp.name)

        entity = scene.add_entity(morph=gs.morphs.Mesh(file=tmp.name, pos=(0, 0, 0), fixed=True))
        entity.use_visual_raycasting = True

        cam = scene.add_camera(res=(640, 480), pos=(0, -3.0, 1.0), lookat=(0, 0, 0.8), fov=45)

        if args.num_envs > 0:
            scene.build(n_envs=args.num_envs)
        else:
            scene.build()

        os.unlink(tmp.name)
        print(f"Entity n_vverts: {entity.n_vverts}, SMPL verts: {smpl.get_num_verts()}")

        for step in range(500):
            t = step * 0.03
            body_pose = torch.zeros(B, 69)
            body_pose[:, 0] = 0.5 * torch.sin(torch.tensor(t))
            body_pose[:, 3] = -0.5 * torch.sin(torch.tensor(t))
            body_pose[:, 9] = 0.3 * torch.clamp(torch.sin(torch.tensor(t)), min=0.0)
            body_pose[:, 12] = 0.3 * torch.clamp(-torch.sin(torch.tensor(t)), min=0.0)

            output = smpl(body_pose=body_pose, return_verts=True)
            verts = output.vertices  # (B, 6890, 3)

            if args.num_envs > 0:
                entity.set_vverts(verts)
            else:
                entity.set_vverts(verts[0])

            scene.step()
            cam.render()

    else:
        # -------- Box path (set_vverts + set_vmesh demo) --------
        base_verts, base_faces = create_box_mesh(size=0.5, subdivisions=3)
        # Lift the box above the ground
        base_verts[:, 2] += 0.5

        mesh = trimesh.Trimesh(vertices=base_verts, faces=base_faces, process=False)
        tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        mesh.export(tmp.name)

        entity = scene.add_entity(morph=gs.morphs.Mesh(file=tmp.name, pos=(0, 0, 0), fixed=True))

        cam = scene.add_camera(res=(640, 480), pos=(0, -2.5, 1.5), lookat=(0, 0, 0.5), fov=45)

        if args.num_envs > 0:
            scene.build(n_envs=args.num_envs)
        else:
            scene.build()

        os.unlink(tmp.name)
        B = max(1, args.num_envs)

        print("\n--- Phase 1: set_vverts (vertex-only deformation) ---")
        for step in range(200):
            t = step * 0.05
            if args.num_envs > 0:
                all_verts = np.stack([wave_deform(base_verts, t + b * 0.5) for b in range(B)], axis=0)
                entity.set_vverts(all_verts)
            else:
                entity.set_vverts(wave_deform(base_verts, t))

            scene.step()
            cam.render()

        print("--- Phase 2: set_vmesh (topology change — switch to icosphere) ---")
        ico = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
        ico_verts = ico.vertices.astype(np.float32)
        ico_verts[:, 2] += 0.5
        ico_faces = ico.faces.astype(np.int32)

        for step in range(200):
            t = step * 0.05
            if args.num_envs > 0:
                all_verts = np.stack([wave_deform(ico_verts, t + b * 0.5) for b in range(B)], axis=0)
                entity.set_vmesh(all_verts, ico_faces)
            else:
                entity.set_vmesh(wave_deform(ico_verts, t), ico_faces)

            scene.step()
            cam.render()

        print("--- Phase 3: clear_custom_vmesh (revert to original) ---")
        entity.clear_custom_vmesh()
        for step in range(100):
            scene.step()
            cam.render()

    print("Done!")


if __name__ == "__main__":
    main()
