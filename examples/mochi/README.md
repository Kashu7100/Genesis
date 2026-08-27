# MochiSolver examples

The `MochiSolver` is a port of the implicit multi-physics solver of Meta's mochi physics engine
(https://github.com/facebookresearch/project_superdex, Apache-2.0). Every substep solves one nonlinear system in which
the inertia of all bodies and a smooth signed-distance penalty contact model are assembled together, so large time
steps (10-25 ms) remain stable, and contact needs no coupler.

Entities are added with `gs.materials.Mochi.Rigid(...)`; solver options live in `gs.options.MochiOptions`. Double
precision (`gs.init(precision="64")`) is recommended: the default contact stiffness of 1e9 Pa/m makes the Newton system
ill-conditioned in single precision.

Contact parameters (per material, combined per pair by geometric mean):

| parameter | default | meaning |
|---|---|---|
| `penalty_coefficient` | 1e9 Pa/m | contact pressure per unit penetration once the ramp is fully active |
| `penalty_threshold` | 1 mm | signed distance at which pressure starts to build |
| `penalty_smoothing_half_distance` | 5 mm | half-width of the smooth pressure ramp (full ramp 10 mm) |
| `friction` | 0.5 | Coulomb friction coefficient |
| `friction_falloff_vel` | 0.01 m/s | sliding speed below which friction is regularized |
| `normal_viscous_damping` | 0 s/m | impact damping; the coefficient of restitution e at impact speed v follows from `c = (1-e)(1+4.5e)/(e(1+8e/3))/v` |

Supported: free and fixed rigid bodies, articulated bodies (URDF/MJCF kinematic trees with fixed, revolute, prismatic,
spherical and free joints; joint damping, armature, stiffness, soft range limits, force/velocity/position drives through
the usual `control_dofs_*` / `set_dofs_kp` / `set_dofs_kv` API), deformable bodies (`gs.materials.Mochi.Elastic`:
linear tetrahedra with a stable neo-Hookean, Saint Venant-Kirchhoff or linear material, mass and stiffness damping,
fixed vertices through `set_vertices_fixed`, tetrahedralized by tetgen from Box/Sphere/Cylinder/Mesh morphs or read from
tetgen `.node`/`.ele` files), plane/sphere/box analytic colliders and grid colliders for meshes, backward Euler and
BDF2 time integration. Contact between the links of one entity is disabled, equality constraints are ignored and drive
forces are not clamped to the force range. Deformable bodies collide through the quadrature samples of their boundary
triangles against the rigid colliders only: they do not act as colliders themselves (no soft-soft contact, and a rigid
body smaller than the boundary triangles of a soft body passes through it), so mesh them finely enough
(`maxvolume=..., nobisect=False`) for the rigid bodies that touch them.

Velocities are recovered by finite differences over the step as in mochi: `get_dofs_velocity()` returns
`sin(dq)/dt` for revolute joints and the sine-based angular velocity `vee(((R - R_prev)/dt) R^T)` for spherical and
free joints, which differ from `dq/dt` by a factor `1 - (dq)^2/6`.

- `rigid_bodies.py`: sphere and cube dropped onto a table.
