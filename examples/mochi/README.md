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
triangles against the rigid colliders, and act as colliders themselves (`collider_type="sdf"`, the default): a signed
distance field of the rest shape is queried through the deformed tetrahedra, so rigid samples and the samples of other
soft bodies are pushed out once they are inside the body. That path has no contact skin (contact only registers for
points inside the body, so a few millimeters of interpenetration remain) and its tangent drops the derivatives of the
mapping, as in mochi; a soft body never collides with itself. Mesh soft bodies reasonably finely
(`maxvolume=..., nobisect=False`) where sharp rigid features touch them.

Thin shells (`gs.materials.Mochi.Shell`) use the surface triangles of a `Mesh` (or primitive) morph as elements: a
Saint Venant-Kirchhoff membrane on the metric of the mid-surface and a Koiter-type bending term on its discrete
curvature, both thickness-integrated from `E`, `nu`, `rho` and `thickness` (each stiffness can be overridden). Shell
samples carry no orientation (the contact normal comes from the collider, so both sides collide), and shells act as
colliders through spheres of radius `collider_radius` placed at their vertices (`collider_type="point_cloud"`); a shell
never collides with itself. Consistently wound meshes are required. Light cloth has little inertia: use a lower
`penalty_coefficient` (1e7 Pa/m) or a larger `explosion_rel_tol` so that the sudden contact residual of an impact is
not mistaken for a divergence.

Rods (`gs.morphs.Rod` polyline + `gs.materials.Mochi.Rod`) are discrete elastic rods: stretching of the segments,
bending and twisting at the interior nodes, with one twist angle per segment as an extra unknown and the material
frames carried along as parallel-transported state (a Kirchhoff rod, no shear). The stiffnesses derive from `E`, `nu`,
`rho` and the radius (`E A`, `E I`, `G J`, `rho A`, `rho J`) and can be overridden; clamp a rod by fixing its first two
nodes. Rods collide through samples on their centerline (no radius offset on the colliding side) and act as
colliders through spheres of the rod radius carried by their nodes (`collider_type="auto"`), so a rigid body resting on
a rope sits one rod radius above the centerline.

Velocities are recovered by finite differences over the step as in mochi: `get_dofs_velocity()` returns
`sin(dq)/dt` for revolute joints and the sine-based angular velocity `vee(((R - R_prev)/dt) R^T)` for spherical and
free joints, which differ from `dq/dt` by a factor `1 - (dq)^2/6`.

- `rigid_bodies.py`: sphere and cube dropped onto a table.
