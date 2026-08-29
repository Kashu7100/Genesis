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

## Plan 2: batched deformables on the GPU, memory per environment (2026-08-29)

Three Genesis-only scenes model batched reinforcement learning with deformables (a Franka arm pressing a 31x31 cloth
with self-contact, a Franka gripper closing on a 4263-tetrahedron cube, the arm pressing a 64-node rope); `sweep.py`
runs a scene over batch sizes and every run records the solver's memory per environment (`memory_report()`), the
process' device memory, the conjugate-gradient iteration counts of both engines and the usage of the bounded contact
lists.

### Baseline before Plan 2 (`10b72f6f`, GPU fp32, host-driven pipeline with bounding-volume hierarchies)

| scene | B=1 ms/step | B=64 ms/step (us/env-step) | B=256 ms/step (us/env-step) | MiB/env | fits on 10 GB |
|---|---|---|---|---|---|
| cloth_arm | 30.0 | 257 (4010) | out of memory | 84.5 | 64 |
| soft_gripper | 44.4 | 294 (4590) | 1347 (5260) | 17.8 | 256 |
| rope_arm | 5.6 | 9.8 (154) | 18.2 (71) | 20.4 | 256 |

### After the spatial hash (Phase H): deformable colliders located by an in-kernel hash

Both deformable collider kinds (collider spheres of shells and rods, deformed tetrahedra of solids) are found through
a per-environment spatial hash rebuilt at every assembly, as in mochi (items inserted once in the bin of their cell, a
query walks the 27 cells around its own; with a cell no smaller than the contact range the candidate set is a superset
of the exact one and `tests/mochi/test_spatial_hash.py` checks that the hits equal a brute-force evaluation). No host
synchronization remains in an assembly, so every scene now runs as one kernel per step on the CPU, and the contact
range follows mochi's (`radius + penalty threshold`; the former `threshold + 2 x smoothing` band only added zero-force
hits: 30k of the 33k hits of the flat cloth+arm scene).

| scene | CPU fp64 ms/step (before -> after) | GPU B=1 ms/step | GPU B=64 ms/step (us/env-step) | GPU B=256 ms/step (us/env-step) | MiB/env |
|---|---|---|---|---|---|
| cloth_arm | 105 -> 8.9 (at rest) | 30.0 -> 14.6 | 257 -> 40.3 (630) | out of memory | 83.8 |
| soft_gripper | 130 -> 145 (see below) | 44.4 -> 32.1 | 294 -> 102 (1590) | 1347 -> 358 (1400) | 16.9 |
| rope_arm | 10.0 -> 2.8 | 5.6 -> 3.1 | 9.8 -> 4.8 (75) | 18.2 -> 7.8 (30) | 20.2 |
| soft_duck | 88.7 -> 86.8 | 21.7 -> 20.8 | 60.1 (940) | 177 (690) | 3.98 |
| cloth_tshirt | 240 -> 148 | 32.9 -> 25.1 | 81.5 (1270) | out of memory | 86.7 |

The GPU still runs the multi-kernel pipeline for deformables (the one-kernel graph step is Phase I); the gain is the
removal of the per-assembly hierarchy rebuilds (a host sync per tree layer) and of the flat shared query buffer. The
gripper on the CPU got slower because a coarse hash cell (the largest tetrahedron) made the cube's own samples walk
thousands of its own tetrahedra before the entity filter; the follow-up skips queries that cannot hit anything and
tests the entity filter before the dedupe. Memory per environment is unchanged: the hit buffers dominate (cloth+arm
84 MiB of which 33.7 MiB point-cloud hit records sized `16 x queries` and 45 MiB contact-readback records) - Phase G.

Conjugate-gradient iterations per step, mochi / Genesis (fp64, single thread): duck 92 / 101, t-shirt 78 / 86, helix
17.5 / ~120 - the tetrahedron and shell counts match (same block-Jacobi preconditioner and adaptive tolerance), the
rod's exact banded solve should bring the helix to mochi's count (Phase K).

### Memory diet (Phase G) and the bounds hash (H2, H3)

Contact points are recorded for readback only on demand (`get_contacts` counts, allocates and gathers; the readback
fields of the hit lists live in a struct allocated at the first readback), the hit lists keep the solver-side fields
only and their capacities follow three options (`max_soft_hits_per_sample`, `max_deformable_collider_hits_per_query`,
`max_point_cloud_hits_per_query`). The hash then inserts every item in the (at most eight) cells its bounds overlap and a
sample walks its own cell only; the tetrahedron cell is twice the median rest extent, the few larger tetrahedra live in
an overflow list every sample scans.

| scene | MiB/env before | MiB/env after | GPU fp32 B=256 ms/step (us/env-step) | GPU fp32 B=1024 ms/step (us/env-step) |
|---|---|---|---|---|
| cloth_arm | 84.5 | 6.3 | 120 (468) | 548 -> 382 (373) |
| soft_gripper | 17.8 | 4.8 | 1347 -> 257-282 (1000-1100) | out of memory -> 970 (947) |
| rope_arm | 20.4 | 2.2 | 8.2 (32) | 16.6 (16) |
| soft_duck | 3.98 | 2.30 | 176 (689) | 648 (633) |
| cloth_tshirt | 86.7 | 16.3 | 266 (1040) | out of memory (16.7 GB) |

With these, 1024 environments of every RL scene fit on the 10 GB card, and 4096 of each on 80 GB by projection (the
t-shirt at 4096 needs ~67 GB; the symmetric Hessian storage planned in Phase K/G7 halves its largest remaining array).
The deformables still run the multi-kernel pipeline on the GPU at this point; the graph step kernel is next.
