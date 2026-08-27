from typing import TYPE_CHECKING, Any, Literal

from pydantic import StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from ..rigid import Rigid as RigidMaterial
from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.mochi_entity import MochiEntity

ColliderType = Literal["auto", "plane", "sphere", "box", "sdf", "none"]


class Rigid(Base["MochiEntity"], RigidMaterial):
    """
    Rigid body material simulated by the MochiSolver.

    Contact between two bodies is a smooth penalty on the signed distance of sample points placed on one body's
    collision surface to the other body's signed distance field (SDF). Every pair parameter is combined from the two
    bodies' values by geometric mean, except the smoothing distance, the activation threshold and the alignment
    threshold, which are read from the collider body alone.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        See Kinematic. Default is False.
    rho : float, optional
        Density in kg/m^3 used to derive the mass and inertia from the collision geometry when the asset specifies
        none. Overrides per-geometry densities found in the asset. Default is 1000.
    friction : float, optional
        Coulomb friction coefficient. Default is 0.5.
    penalty_coefficient : float, optional
        Contact stiffness in Pa/m: pressure per unit of penetration once the penalty ramp is fully active. Higher
        values reduce penetration at the cost of a stiffer, more ill-conditioned Newton system. Default is 1e9.
    penalty_smoothing_half_distance : float, optional
        Half-width in meters of the smooth ramp between zero contact pressure and the linear regime. The pressure
        ramps up over twice this distance below the activation threshold, so a wider ramp gives softer, better
        conditioned contact and a narrower one sharper contact onset. Default is 0.005.
    penalty_threshold : float, optional
        Signed distance in meters at which contact pressure starts to build up. Positive values start pushing bodies
        apart before they touch (a contact skin), negative values allow some interpenetration before responding.
        Default is 0.001.
    friction_falloff_vel : float, optional
        Sliding speed in m/s below which the Coulomb friction force is regularized towards zero. Larger values give
        smoother, more robust stick-slip transitions with more creep under static load; smaller values approach exact
        Coulomb friction with a stiffer system. Default is 0.01.
    viscous_friction : float, optional
        Tangential viscous friction coefficient in s/m, multiplying the normal force and the sliding velocity.
        Default is 0.
    normal_viscous_damping : float, optional
        Normal viscous damping coefficient in s/m, multiplying the normal force and the approach velocity. This is the
        mechanism controlling the coefficient of restitution: 0 gives fully elastic penalty contact, larger values
        dissipate more impact energy. Default is 0.
    max_alignment_normals : float, optional
        Cosine between the colliding surface normal and the collider gradient above which the contact is disabled, so
        that a body embedded past the far side of a thin collider can escape rather than being trapped. Default is 0.
    collider_type : str, optional
        Representation of this body used when other bodies' sample points collide against it: "plane", "sphere" and
        "box" are exact analytic distance fields, "sdf" is the precomputed grid of the collision mesh, "none" makes the
        body collide only through its own sample points (it never acts as a collider), and "auto" selects the analytic
        field for plane, sphere and box primitives and the grid otherwise. Default is "auto".
    has_gravity : bool, optional
        Whether gravity acts on this body. Default is True.
    contact_layer : str, optional
        Name of the contact layer of this body. Contact between two layers can be disabled at the solver level.
        Default is "default".
    sdf_cell_size : float, optional
        Cell size in SDF grid in meters. Contact resolves the penalty ramp against this grid, so the cell should stay
        well below the ramp width. Default is 0.0025.
    sdf_min_res : int, optional
        Minimum resolution of the SDF grid. Must be at least 16. Default is 32.
    sdf_max_res : int, optional
        Maximum resolution of the SDF grid. Must be >= sdf_min_res. Default is 128.
    """

    rho: PositiveFloat = 1000.0
    friction: NonNegativeFloat = 0.5
    penalty_coefficient: PositiveFloat = 1e9
    penalty_smoothing_half_distance: NonNegativeFloat = 5e-3
    penalty_threshold: ValidFloat = 1e-3
    friction_falloff_vel: NonNegativeFloat = 1e-2
    viscous_friction: NonNegativeFloat = 0.0
    normal_viscous_damping: NonNegativeFloat = 0.0
    max_alignment_normals: ValidFloat = 0.0
    collider_type: ColliderType = "auto"
    has_gravity: StrictBool = True
    contact_layer: str = "default"
    needs_coup: StrictBool = False
    sdf_cell_size: PositiveFloat = 2.5e-3

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.coup_type is not None or self.coup_links is not None or self.coup_collision_links is not None:
            gs.raise_exception("IPC coupling fields are not supported by Mochi materials.")
        if self.gravity_compensation != 0.0:
            gs.raise_exception("Use `has_gravity` instead of `gravity_compensation` for Mochi materials.")
        if self.needs_coup:
            gs.raise_exception("Mochi materials handle contact internally; `needs_coup` must be False.")
        if not (-1.0 <= self.max_alignment_normals <= 1.0):
            gs.raise_exception("`max_alignment_normals` must be in [-1, 1].")
