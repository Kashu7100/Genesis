"""
Benchmark: measure simulation step time with and without AvatarEntity.

Runs headless (no viewer, no rendering) to isolate simulation overhead.
"""

import math
import time

import numpy as np
import torch

import genesis as gs

N_WARMUP = 200
N_STEPS = 2000


def bench_without_avatar():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0, 0.42)),
    )
    scene.build()

    for _ in range(N_WARMUP):
        scene.step()

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        scene.step()
    elapsed = time.perf_counter() - t0

    scene.destroy()
    return elapsed


def bench_with_avatar():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0.5, 0.42)),
    )
    ghost = scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, -0.5, 0.42)),
        material=gs.materials.Avatar(),
    )
    scene.build()

    joint_names = [
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
    ghost_dof_idx = np.array([ghost.get_joint(n).dofs_idx_local[0] for n in joint_names])
    joint_angles = torch.tensor(
        [0, 0.8, -1.5, 0, 0.8, -1.5, 0, 1.0, -1.5, 0, 1.0, -1.5], dtype=torch.float32, device=gs.device
    )

    for _ in range(N_WARMUP):
        scene.step()

    t_set = 0.0
    t_step = 0.0
    for step in range(N_STEPS):
        offset = 0.3 * math.sin(0.02 * math.pi * step)
        ref = joint_angles.clone()
        ref[1] += offset
        ref[4] += offset
        ref[7] -= offset
        ref[10] -= offset

        t0 = time.perf_counter()
        ghost.set_dofs_position(ref, ghost_dof_idx)
        t1 = time.perf_counter()
        scene.step()
        t2 = time.perf_counter()

        t_set += t1 - t0
        t_step += t2 - t1

    scene.destroy()
    return t_set + t_step, t_set, t_step


def bench_with_avatar_no_update():
    """With avatar entity present but NOT updating its pose each step."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0.5, 0.42)),
    )
    scene.add_entity(
        gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, -0.5, 0.42)),
        material=gs.materials.Avatar(),
    )
    scene.build()

    for _ in range(N_WARMUP):
        scene.step()

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        scene.step()
    elapsed = time.perf_counter() - t0

    scene.destroy()
    return elapsed


def main():
    gs.init(logging_level="warning")

    print(f"Warming up {N_WARMUP} steps, then timing {N_STEPS} steps (headless).\n")

    t_without = bench_without_avatar()
    t_no_update = bench_with_avatar_no_update()
    t_total, t_set, t_step = bench_with_avatar()

    fps_without = N_STEPS / t_without
    fps_no_update = N_STEPS / t_no_update
    fps_total = N_STEPS / t_total
    overhead_no_update = (t_no_update - t_without) / t_without * 100
    overhead_total = (t_total - t_without) / t_without * 100

    print(f"{'Without avatar:':<30} {t_without:.3f}s  ({fps_without:.0f} steps/s)")
    print(
        f"{'With avatar (static):':<30} {t_no_update:.3f}s  ({fps_no_update:.0f} steps/s)  {overhead_no_update:+.1f}%"
    )
    print(f"{'With avatar (updating):':<30} {t_total:.3f}s  ({fps_total:.0f} steps/s)  {overhead_total:+.1f}%")
    print(f"  ├─ set_dofs_position:         {t_set:.3f}s  ({t_set / t_total * 100:.0f}%)")
    print(f"  └─ scene.step():              {t_step:.3f}s  ({t_step / t_total * 100:.0f}%)")


if __name__ == "__main__":
    main()
