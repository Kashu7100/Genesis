"""Benchmark: ComFree vs Newton solver FPS on dense-contact scenes.

Run standalone or via pytest:
    python tests/bench_comfree.py
    pytest tests/bench_comfree.py -v -s --backend gpu
"""

import time

import numpy as np
import pytest

import genesis as gs


WARMUP_STEPS = 100
MEASURE_STEPS = 300


def _build_many_boxes_scene(n_boxes, solver_type, n_envs=1):
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
    spacing = 0.3
    cols = int(np.ceil(np.sqrt(n_boxes)))
    for i in range(n_boxes):
        row, col = divmod(i, cols)
        x = (col - cols / 2) * spacing
        y = (row - cols / 2) * spacing
        z = 0.5 + (i % 3) * 0.3
        scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(x, y, z)))
    scene.build(n_envs=n_envs)
    return scene


def _measure_fps(scene, warmup_steps, measure_steps):
    for _ in range(warmup_steps):
        scene.step()

    start = time.perf_counter()
    for _ in range(measure_steps):
        scene.step()
    elapsed = time.perf_counter() - start

    fps = measure_steps / elapsed
    return fps, elapsed


def _run_benchmark(n_boxes, n_envs=1):
    results = {}
    for solver_name, solver_type in [
        ("Newton", gs.constraint_solver.Newton),
        ("ComFree", gs.constraint_solver.ComFree),
    ]:
        scene = _build_many_boxes_scene(n_boxes, solver_type, n_envs=n_envs)
        fps, elapsed = _measure_fps(scene, WARMUP_STEPS, MEASURE_STEPS)
        results[solver_name] = fps
        scene.destroy()
        print(f"  {solver_name:8s}: {fps:8.1f} FPS  ({elapsed:.2f}s for {MEASURE_STEPS} steps)")

    speedup = results["ComFree"] / results["Newton"]
    print(f"  Speedup: {speedup:.2f}x")
    return results, speedup


@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize("n_boxes", [4, 16, 64])
def test_bench_comfree_boxes(n_boxes, show_viewer):
    """Benchmark ComFree vs Newton with varying box counts."""
    print(f"\n--- {n_boxes} boxes, 1 env ---")
    results, speedup = _run_benchmark(n_boxes, n_envs=1)
    assert results["ComFree"] > 0
    assert results["Newton"] > 0


@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize("n_envs", [1, 16, 64, 256])
def test_bench_comfree_batched(n_envs, show_viewer):
    """Benchmark ComFree vs Newton scaling with batched environments."""
    n_boxes = 4
    print(f"\n--- {n_boxes} boxes, {n_envs} envs ---")
    results, speedup = _run_benchmark(n_boxes, n_envs=n_envs)
    assert results["ComFree"] > 0
    assert results["Newton"] > 0


if __name__ == "__main__":
    gs.init(backend=gs.gpu)

    print("=" * 60)
    print("ComFree vs Newton Benchmark")
    print("=" * 60)

    print("\n=== Scaling with number of boxes (1 env) ===")
    for n_boxes in [4, 16, 64]:
        print(f"\n--- {n_boxes} boxes ---")
        _run_benchmark(n_boxes, n_envs=1)

    print("\n=== Scaling with batched environments (4 boxes) ===")
    for n_envs in [1, 16, 64, 256]:
        print(f"\n--- 4 boxes, {n_envs} envs ---")
        _run_benchmark(4, n_envs=n_envs)

    print("\n=== Scaling with batched environments (16 boxes) ===")
    for n_envs in [1, 16, 64]:
        print(f"\n--- 16 boxes, {n_envs} envs ---")
        _run_benchmark(16, n_envs=n_envs)
