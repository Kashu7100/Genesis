"""Time Genesis' MochiSolver on the benchmark scenes.

    python tests/mochi/benchmark/bench_genesis.py rigid [--backend cpu|gpu] [--precision 64|32] [--n-envs B] [--profile]

Protocol (shared with bench_mochi.py): warm-up steps, then `--n-windows` windows of `--n-steps` steps; the best and the
mean window are reported in ms/step. `--profile` adds the quadrants kernel profiler (launches per step, kernel time, top
kernels). Results are written as JSON into `--out` (default `results/`).
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", choices=sorted(scenes.SCENES))
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--precision", default="64", choices=["32", "64"])
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--tag", default="", help="suffix of the result file name")
    parser.add_argument("--option", action="append", default=[], help="MochiOptions override, e.g. linear_solver=pcg")
    args = parser.parse_args()

    import quadrants as qd

    if args.profile:
        _qd_init = qd.init

        def _init(**kwargs):
            kwargs["kernel_profiler"] = True
            return _qd_init(**kwargs)

        qd.init = _init

    import genesis as gs

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, precision=args.precision, logging_level="warning")

    overrides = {}
    for item in args.option:
        key, value = item.split("=", 1)
        overrides[key] = json.loads(value) if value[0] in '0123456789-[{tfn"' else value
    t0 = time.perf_counter()
    scene, probe = scenes.build_genesis(args.scene, n_envs=args.n_envs, **overrides)
    build_time = time.perf_counter() - t0
    solver = scene.mochi_solver

    t0 = time.perf_counter()
    for _ in range(args.warmup):
        scene.step()
    qd.sync()
    warmup_time = time.perf_counter() - t0

    windows = []
    for _ in range(args.n_windows):
        t0 = time.perf_counter()
        for _ in range(args.n_steps):
            scene.step()
        qd.sync()
        windows.append((time.perf_counter() - t0) / args.n_steps * 1e3)
    info = solver.get_convergence_info()
    result = {
        "engine": "genesis",
        "scene": args.scene,
        "backend": args.backend,
        "precision": args.precision,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "n_windows": args.n_windows,
        "warmup": args.warmup,
        "ms_per_step_best": min(windows),
        "ms_per_step_mean": sum(windows) / len(windows),
        "ms_per_step_windows": windows,
        "build_s": build_time,
        "warmup_s": warmup_time,
        "n_dofs_total": int(solver.n_dofs_total),
        "n_samples": int(solver.n_samples),
        "n_soft_verts": int(solver.n_soft_verts),
        "n_soft_elems": int(solver.n_soft_elems),
        "n_shell_elems": int(solver.n_shell_elems),
        "n_rod_elems": int(solver.n_rod_elems),
        "newton_iterations_last": int(info["n_iter"][0]),
        "probe": probe(),
        "options": overrides,
    }
    if args.profile:
        qd.profiler.clear_kernel_profiler_info()
        t0 = time.perf_counter()
        for _ in range(args.n_steps):
            scene.step()
        qd.sync()
        wall = time.perf_counter() - t0
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            qd.profiler.print_kernel_profiler_info("count")
        table = buffer.getvalue()
        rows = re.findall(r"\[\s*([\d.]+)%\s+([\d.]+) s\s+(\d+)x \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) ms\] (\S+)", table)
        result.update(
            profile_ms_per_step=wall / args.n_steps * 1e3,
            kernel_ms_per_step=qd.profiler.get_kernel_profiler_total_time() / args.n_steps * 1e3,
            launches_per_step=sum(int(r[2]) for r in rows) / args.n_steps,
            top_kernels=[
                {
                    "name": r[6],
                    "percent": float(r[0]),
                    "calls_per_step": int(r[2]) / args.n_steps,
                    "avg_ms": float(r[4]),
                }
                for r in rows[:15]
            ],
        )
    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.out, f"genesis_{args.scene}_{args.backend}_fp{args.precision}_B{args.n_envs}{tag}.json")
    with open(path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(
        f"[genesis] {args.scene} {args.backend} fp{args.precision} B={args.n_envs}: "
        f"best {result['ms_per_step_best']:.3f} ms/step, mean {result['ms_per_step_mean']:.3f} ms/step, "
        f"newton {result['newton_iterations_last']}, probe {result['probe']:.4f}"
        + (
            f", launches/step {result['launches_per_step']:.1f}, kernel {result['kernel_ms_per_step']:.3f} ms/step"
            if args.profile
            else ""
        )
    )
    if args.profile:
        for row in result["top_kernels"][:8]:
            print(
                f"    {row['percent']:5.1f}%  {row['calls_per_step']:7.1f}/step  {row['avg_ms']:.3f} ms  {row['name']}"
            )
    print(f"[genesis] wrote {path}")


if __name__ == "__main__":
    main()
