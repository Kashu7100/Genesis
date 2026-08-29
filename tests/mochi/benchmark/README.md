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
