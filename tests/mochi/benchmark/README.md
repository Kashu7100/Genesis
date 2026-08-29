# MochiSolver benchmark

Per-step cost of Genesis' `MochiSolver` against the original mochi engine on six scenes that both engines build from the
same description (`scenes.py`: same meshes, coordinates, materials, time step 1/60 s, Newton caps).

| scene | contents | Newton cap |
|---|---|---|
| `rigid` | sphere (r 0.2) and cube (0.4) dropped onto a static table over the ground | 20 |
| `articulated` | double pendulum on a limited rail (prismatic + revolute + spherical) striking a ball | 20 |
| `equalities` | two links on a compliant world pivot and a compliant spherical joint | 20 |
| `soft_duck` | mochi's tetrahedral duck (1899 nodes, 8608 tets, neo-Hookean) dropped 1 m onto a plane | 20 |
| `cloth_tshirt` | mochi's t-shirt shell (3593 nodes, 7076 triangles) with self-contact falling onto a plane | 2 |
| `rod_helix` | mochi's helical spring (129 nodes) hanging from its first node | 20 |
| `franka` | Franka arm on a plane next to a box (Genesis only, 16k contact samples) | 20 |

The deformable assets are mochi's own, converted once to Z-up by `convert_mochi_assets.py` (`assets/`); the rigid
primitives are meshed by Genesis and exported to `assets/generated/` so that the mochi build carries the same triangles.
Both engines use the residual-norm line search, the same contact parameters (mochi defaults) and stop on the same
residual tolerances (`abs 1e-3`, `rel 1e-6`); the cap only matters when Newton does not converge.

## Protocol

5 warm-up steps, then 3 windows of 30 steps; the best window is the headline number (ms per step), the mean is kept in
the JSON. Both engines therefore time the fall, the impact and the settling of the same trajectory. One environment on
the CPU is the acceptance comparison (Genesis fp64 vs mochi built with `SUPERDEX_PRECISION=double`, mochi single-threaded);
Genesis GPU numbers are tracked in fp32 (mochi's default precision) at several batch sizes.

```bash
# Genesis (worktree venv); --profile adds launches/step, kernel time and the top kernels
python tests/mochi/benchmark/bench_genesis.py rigid --profile
python tests/mochi/benchmark/bench_genesis.py soft_duck --backend gpu --precision 32 --n-envs 8

# original engine (superdex venv); the rigid/articulated scenes need the meshes exported by bench_genesis.py
SUPERDEX_PRECISION=double SUPERDEX_ASSETS_PATH=... python tests/mochi/benchmark/bench_mochi.py rigid --threads 0

# table from results/*.json
python tests/mochi/benchmark/report.py
```

## Results (Ryzen 9 9900X, RTX 3080; Genesis worktree `feat/mochi-solver`)

Filled in by `report.py` at every optimization phase; see the git history of this file for the progression.

### Baseline, 2026-08-28, `2e467e90` (before the optimization phases)

| scene | mochi fp64 (T=0) ms/step | Genesis CPU fp64 ms/step | ratio | launches/step | Genesis GPU fp32 ms/step (B) | notes |
|---|---|---|---|---|---|---|
| rigid | 0.017 | 1.454 | 87.8x | 62 | - | 12 dofs, newton 1; mochi newton 0 |
| articulated | 0.036 | 3.190 | 88.0x | 94 | - | 11 dofs, newton 2; mochi newton 2 |
| equalities | 0.022 | 1.313 | 59.0x | 138 | - | 12 dofs, newton 4; mochi newton 4 |
| soft_duck | 12.371 | 258.935 | 20.9x | 1303 | - | 5697 dofs, newton 3; mochi newton 3 |
| cloth_tshirt | 32.201 | 648.020 | 20.1x | 962 | - | 10779 dofs, newton 2; mochi newton 2 |
| rod_helix | 2.277 | 364.261 | 159.9x | 12037 | - | 515 dofs, newton 20; mochi newton 20 |
| franka | - | 10.509 | - | 268 | - | 15 dofs, newton 8 |

Newton iteration counts agree between the two engines on every scene, so the ratios compare equal work.

### After phases B1-B2, B5-B6, B8 (rods) and C1-C2 (one-kernel step), 2026-08-29

| scene | mochi fp64 (T=0) ms/step | Genesis CPU fp64 ms/step | ratio | launches/step | Genesis GPU fp32 ms/step (B) | notes |
|---|---|---|---|---|---|---|
| rigid | 0.017 | 0.069 | 4.1x | 1 | 2.27 (B=1024) | 12 dofs, newton 1; mochi newton 0 |
| articulated | 0.036 | 0.157 | 4.3x | 1 | 9.33 (B=1024) | 11 dofs, newton 2; mochi newton 2 |
| equalities | 0.022 | 0.049 | 2.2x | 1 | - | 12 dofs, newton 4; mochi newton 4 |
| soft_duck | 12.371 | 100.945 | 8.2x | 1 | - | 5697 dofs, newton 3; mochi newton 3 |
| cloth_tshirt | 32.201 | 266.994 | 8.3x | 744 | - | 10779 dofs, newton 2; mochi newton 2 |
| rod_helix | 2.277 | 5.977 | 2.6x | 1 | - | 515 dofs, newton 20; mochi newton 20 |
| franka | - | 0.075 | - | 1 | - | 15 dofs, newton 8 |

Every scene is within the 10x target of single-threaded mochi on the CPU. The t-shirt (self-contact) still runs the
multi-kernel pipeline because its point-cloud collider is built by host-driven bounding-volume kernels; all other
scenes run as one kernel launch per step. GPU (RTX 3080, fp32, monolith, 1024 environments): rigid 2.2 us and
articulated 9.1 us per environment step.

### After B8 (neo-Hookean PSD oracle) and C3 (one kernel per step on every scene without self-contact), 2026-08-29, `5f628355`

| scene | mochi fp64 (T=0) ms/step | Genesis CPU fp64 ms/step | ratio | launches/step | Genesis GPU fp32 ms/step (B) | notes |
|---|---|---|---|---|---|---|
| rigid | 0.017 | 0.066 | 4.0x | 1 | 1.18 (B=1), 1.61 (B=64), 2.20 (B=1024) | 12 dofs, newton 1; mochi newton 0 |
| articulated | 0.036 | 0.164 | 4.5x | 1 | 5.12 (B=1), 5.70 (B=64), 9.89 (B=1024) | 11 dofs, newton 2; mochi newton 2 |
| equalities | 0.022 | 0.049 | 2.2x | 1 | 0.93 (B=1), 2.89 (B=1024) | 12 dofs, newton 4; mochi newton 4 |
| soft_duck | 12.371 | 88.701 | 7.2x | 1 | 21.48 (B=1), 29.53 (B=8) | 5697 dofs, newton 3; mochi newton 3 |
| cloth_tshirt | 32.201 | 239.807 | 7.4x | 744 | 32.85 (B=1), 47.65 (B=4) | 10779 dofs, newton 2; mochi newton 2 |
| rod_helix | 2.277 | 5.988 | 2.6x | 1 | 154.84 (B=1), 219.68 (B=64) | 515 dofs, newton 20; mochi newton 20 |
| franka | - | 0.075 | - | 1 | 1.59 (B=1), 3.83 (B=1024) | 15 dofs, newton 8 |

CPU fp32 (`--precision 32`) reproduces the fp64 trajectories on every scene (same probes) at the same speed: the CPU
kernels are scalar, so the precision only matters on the GPU.

#### GPU fp32 (RTX 3080), `5f628355`, automatic kernel choice

| scene | B=1 ms/step | B=4-8 ms/step | B=64 ms/step | B=1024 ms/step | us per env-step at the largest B | step kernel | GPU fp64 before (`2e467e90`) |
|---|---|---|---|---|---|---|---|
| rigid | 1.18 | - | 1.61 | 2.20 | 2.1 | monolith | 1.40 (B=1), 2.07 (B=64), 12.4 (B=1024) |
| articulated | 5.12 | - | 5.70 | 9.89 | 9.7 | monolith | 8.5 (B=1024) |
| equalities | 0.93 | - | - | 2.89 | 2.8 | monolith | - |
| franka | 1.59 | - | - | 3.83 | 3.7 | monolith | - |
| soft_duck | 21.5 | 29.5 (B=8) | - | - | 3690 | pipeline | 68.7 (B=1), 173 (B=8) |
| cloth_tshirt | 32.9 | 47.7 (B=4) | - | - | 11900 | pipeline | 82 (B=1) |
| rod_helix | 155 | - | 220 | - | 3430 | pipeline | 1311 (B=1), 2145 (B=64) |

The small rigid/articulated scenes run as one kernel per step on the GPU too: 1024 environments cost 2-10 us per
environment step (a single mochi core needs 17-36 us per step on these scenes), and one environment costs about the
launch latency of the monolith (1-5 ms, mostly the kernel's serial per-environment work on one GPU thread). The
deformables run the multi-kernel pipeline on the GPU (300-750 launches and the host-driven bounding-volume
hierarchies per step): the duck at 8 environments and the t-shirt at 4 already amortize to 3.7 ms and 11.9 ms per
environment step, but at one environment they are launch-bound (21-33 ms, on par with one mochi core on the t-shirt,
1.7x slower on the duck); the helix (515 dofs, 20 Newton iterations, ~120 PCG iterations per step) is the worst case
of the launch-bound pipeline at 155 ms - 26x its CPU time - and the one-kernel step is no better there (310 ms at B=1:
one GPU thread per environment). A graph-launched step kernel with parallel element loops (Phase D of the plan) is the
remaining route to a fast GPU path for deformables at small batch sizes.

Where the remaining time goes (CPU fp64, one environment):

- duck: 107 PCG iterations x 0.365 ms (scalar CSR matvec over 5697 dofs) + 3.7 full assemblies x 9.0 ms (1.05 us per
  tetrahedron incl. the PSD-projected tangent); mochi's element kernels are fp32 8-wide SIMD, which is the constant
  factor that remains.
- t-shirt: PCG 60% (86 iterations x 2.3 ms: shell CSR plus the self-contact hits in the matvec), point-cloud broadphase
  15% (one bounding-volume-hierarchy query per assembly), point-cloud collider evaluation 13%, shell assembly 7%.
  Reusing the broadphase candidates across the line-search trials or an in-kernel spatial hash for the self-contact
  broadphase (which would also let the t-shirt run as one kernel) are the next steps if it needs to be faster.
- rigid/articulated/equalities: one kernel launch (15-25 us) plus 30-140 us of serial work (forward kinematics, contact
  sampling over the culled sample tree, dense Cholesky, line search); mochi's 9-36 us is the same work in C++.
