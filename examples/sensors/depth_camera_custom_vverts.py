"""
Depth Camera with Deforming Kinematic Entity + Rigid Entity
============================================================

Demonstrates that a raycaster-based depth camera can see both:
- A rigid ground plane and box (collision geometry, RigidSolver)
- A deforming kinematic mesh (visual geometry via ``set_vverts``, KinematicSolver)

Depth frames are written to disk as grayscale PNGs.  By default they go to
``/tmp/depth_out/depth_XXXX.png`` (configurable via ``--out-dir``).  You can
watch them live with an external image viewer, e.g.::

    feh --reload 0.1 /tmp/depth_out
    # or
    eog /tmp/depth_out

Usage
-----
    python depth_camera_custom_vverts.py -v          # with Genesis 3D viewer
    python depth_camera_custom_vverts.py             # headless
    python depth_camera_custom_vverts.py -v -B 4     # batched
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh

import genesis as gs


# ----------------------------- helpers ---------------------------------


def create_sphere_mesh(radius=0.3, subdivisions=3):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def wave_deform(verts, t, amplitude=0.08, freq=4.0):
    deformed = verts.copy()
    deformed[:, 2] += amplitude * np.sin(freq * verts[:, 0] + t) * np.cos(freq * verts[:, 1] + t * 0.7)
    deformed[:, 0] += amplitude * 0.5 * np.sin(freq * verts[:, 2] + t * 1.3)
    return deformed


def save_depth_png(depth_img, out_path, max_range):
    """Save a depth tensor as a grayscale PNG (closer = brighter, no-hit = black)."""
    depth_np = depth_img.detach().cpu().numpy()
    valid = np.isfinite(depth_np) & (depth_np < max_range)
    # Invert so closer objects are brighter
    normalized = np.where(valid, 1.0 - depth_np / max_range, 0.0)
    gray = (normalized * 255).astype(np.uint8)

    try:
        from PIL import Image

        img = Image.fromarray(gray).resize((gray.shape[1] * 4, gray.shape[0] * 4), Image.NEAREST)
        img.save(out_path)
    except ImportError:
        np.save(out_path.with_suffix(".npy"), gray)


# ----------------------------- main ------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Depth camera with deforming kinematic entity")
    parser.add_argument("-v", "--vis", action="store_true", help="Open Genesis 3D viewer")
    parser.add_argument("-B", "--num_envs", type=int, default=0, help="Number of parallel envs (0 = unbatched)")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--save-every", type=int, default=5, help="Save depth PNG every N steps")
    parser.add_argument("--out-dir", default="/tmp/depth_out", help="Directory for saved depth PNGs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous run
    for old in out_dir.glob("depth_*.png"):
        old.unlink()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -2.5, 1.8),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=45,
            max_FPS=60,
        ),
        rigid_options=gs.options.RigidOptions(dt=0.01, gravity=(0, 0, -9.81)),
        show_viewer=args.vis,
    )

    # --- Rigid entities ---
    plane = scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(pos=(0.8, 0.0, 0.15), size=(0.3, 0.3, 0.3), fixed=True),
    )

    # --- Kinematic entity with visual raycasting ---
    sphere_verts, sphere_faces = create_sphere_mesh(radius=0.3, subdivisions=3)
    sphere_verts[:, 2] += 0.5

    mesh = trimesh.Trimesh(vertices=sphere_verts, faces=sphere_faces, process=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    mesh.export(tmp.name)

    deform_entity = scene.add_entity(
        morph=gs.morphs.Mesh(file=tmp.name, pos=(0, 0, 0), fixed=True),
        material=gs.materials.Kinematic(use_visual_raycasting=True),
    )

    # --- Depth camera sensor (third-person view, mounted on the plane) ---
    max_range = 5.0
    depth_camera = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(res=(96, 72), fov_horizontal=60.0),
            entity_idx=plane.idx,
            link_idx_local=0,
            pos_offset=(-1.2, 0.0, 1.2),
            euler_offset=(0.0, 45.0, 0.0),
            max_range=max_range,
            return_world_frame=True,
        ),
    )

    if args.num_envs > 0:
        scene.build(n_envs=args.num_envs)
    else:
        scene.build()

    os.unlink(tmp.name)

    B = max(1, args.num_envs)
    print(f"Rigid entities : plane (idx={plane.idx}), box (idx={box.idx})")
    print(f"Kinematic mesh : deform_entity (idx={deform_entity.idx}, n_vverts={deform_entity.n_vverts})")
    print(f"Depth camera   : res=(96, 72), max_range={max_range}")
    print(f"Output dir     : {out_dir}")
    print(f"  (view live with:  feh --reload 0.1 {out_dir})")
    print()

    try:
        for step in range(args.steps):
            t = step * 0.05

            if args.num_envs > 0:
                all_verts = np.stack([wave_deform(sphere_verts, t + b * 0.5) for b in range(B)], axis=0)
                deform_entity.set_vverts(all_verts)
            else:
                deform_entity.set_vverts(wave_deform(sphere_verts, t))

            scene.step()

            depth_img = depth_camera.read_image()
            img = depth_img[0] if args.num_envs > 0 else depth_img

            if step % args.save_every == 0:
                png_path = out_dir / f"depth_{step:04d}.png"
                save_depth_png(img, png_path, max_range=max_range)

            if step % 50 == 0:
                valid = torch.isfinite(img) & (img < max_range)
                min_val = f"{img[valid].min().item():.3f}" if valid.any() else "n/a"
                print(
                    f"  step {step:4d}: valid hits={valid.sum().item()}, min depth={min_val}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)

    print(f"\nDone! Depth PNGs written to: {out_dir}")


if __name__ == "__main__":
    main()
