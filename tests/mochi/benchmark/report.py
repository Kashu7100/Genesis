"""Merge the JSON results of bench_genesis.py and bench_mochi.py into a markdown table.

python tests/mochi/benchmark/report.py [results_dir] [--precision 64] [--threads 0]
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    )
    parser.add_argument("--precision", default="64", help="precision of the CPU comparison")
    parser.add_argument("--threads", type=int, default=0, help="mochi thread setting used as the reference")
    args = parser.parse_args()

    # the most recent result of each configuration wins
    runs = []
    for path in sorted(glob.glob(os.path.join(args.results, "*.json")), key=os.path.getmtime):
        with open(path) as fp:
            runs.append(json.load(fp))
    print(
        f"| scene | mochi fp{args.precision} (T={args.threads}) ms/step | Genesis CPU fp{args.precision} ms/step | ratio "
        "| launches/step | Genesis GPU fp32 ms/step (B) | notes |"
    )
    print("|---|---|---|---|---|---|---|")
    for name in scenes.SCENES:
        mochi = [
            r
            for r in runs
            if r["engine"] == "mochi"
            and r["scene"] == name
            and r["precision"] == args.precision
            and r["threads"] == args.threads
        ]
        gen_cpu = [
            r
            for r in runs
            if r["engine"] == "genesis"
            and r["scene"] == name
            and r["backend"] == "cpu"
            and r["precision"] == args.precision
            and r["n_envs"] == 1
            and not r.get("options")
        ]
        gen_gpu_by_envs = {
            r["n_envs"]: r
            for r in runs
            if r["engine"] == "genesis"
            and r["scene"] == name
            and r["backend"] == "gpu"
            and r["precision"] == "32"
            and not r.get("options")
        }
        gen_gpu = [gen_gpu_by_envs[n_envs] for n_envs in sorted(gen_gpu_by_envs)]
        m = f"{mochi[-1]['ms_per_step_best']:.3f}" if mochi else "-"
        g = f"{gen_cpu[-1]['ms_per_step_best']:.3f}" if gen_cpu else "-"
        ratio = f"{gen_cpu[-1]['ms_per_step_best'] / mochi[-1]['ms_per_step_best']:.1f}x" if mochi and gen_cpu else "-"
        # the launch count comes from the most recent profiled run of the configuration
        profiled = [r for r in gen_cpu if "launches_per_step" in r]
        launches = f"{profiled[-1]['launches_per_step']:.0f}" if profiled else "-"
        gpu = ", ".join(f"{r['ms_per_step_best']:.2f} (B={r['n_envs']})" for r in gen_gpu) or "-"
        notes = []
        if gen_cpu:
            notes.append(f"{gen_cpu[-1]['n_dofs_total']} dofs, newton {gen_cpu[-1]['newton_iterations_last']}")
        if mochi:
            notes.append(f"mochi newton {mochi[-1]['newton_iterations_last']}")
        print(f"| {name} | {m} | {g} | {ratio} | {launches} | {gpu} | {'; '.join(notes)} |")


if __name__ == "__main__":
    main()
