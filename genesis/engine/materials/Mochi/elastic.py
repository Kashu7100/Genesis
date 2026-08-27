from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.mochi_entity import MochiSoftEntity

ElasticModel = Literal["stable_neohookean", "stvk", "linear"]
SoftColliderType = Literal["auto", "sdf", "none"]


class Elastic(Base["MochiSoftEntity"]):
    """
    Deformable (tetrahedral finite element) material simulated by the MochiSolver.

    The body is discretized into linear tetrahedra whose vertex positions are unknowns of the same implicit Newton
    solve as the rigid bodies. Contact acts on quadrature samples of the boundary triangles against the rigid bodies'
    signed distance fields, with the same smooth penalty and friction model (see `Mochi.Rigid`); pair parameters are
    combined by geometric mean, except the smoothing distance, the activation threshold and the alignment threshold,
    which are read from the collider body.

    Parameters
    ----------
    E : float, optional
        Young's modulus in Pa. Default is 1e5.
    nu : float, optional
        Poisson's ratio. Default is 0.45.
    rho : float, optional
        Density in kg/m^3. Default is 1000.
    model : str, optional
        Constitutive model: "stable_neohookean" (Smith et al. 2018, robust to inversion), "stvk" (Saint
        Venant-Kirchhoff) or "linear" (small strain). Default is "stable_neohookean".
    mass_damping : float, optional
        Mass-proportional (Rayleigh) damping coefficient in 1/s. Default is 0.
    stiffness_damping : float, optional
        Stiffness-proportional (Kelvin-Voigt) damping coefficient in s, acting through the rest-state elastic tangent
        on the rate of the Green strain. Default is 0.
    friction : float, optional
        Coulomb friction coefficient. Default is 0.5.
    penalty_coefficient : float, optional
        Contact stiffness in Pa/m. Default is 1e9.
    penalty_smoothing_half_distance : float, optional
        Half-width in meters of the smooth ramp of the contact pressure. Default is 0.005.
    penalty_threshold : float, optional
        Signed distance in meters at which contact pressure starts to build up. Default is 0.001.
    friction_falloff_vel : float, optional
        Sliding speed in m/s below which the Coulomb friction is regularized. Default is 0.01.
    viscous_friction : float, optional
        Tangential viscous friction coefficient in s/m. Default is 0.
    normal_viscous_damping : float, optional
        Normal viscous damping coefficient in s/m (controls the restitution of impacts). Default is 0.
    max_alignment_normals : float, optional
        Cosine threshold between the colliding normal and the collider gradient above which contact is disabled.
        Default is 0.
    collider_type : str, optional
        Whether other bodies' sample points collide against this body: "sdf" (and "auto") builds a signed distance
        field of the rest shape that is queried through the deformed tetrahedra (contact only registers for points
        inside the body, the activation threshold is zero), "none" makes the body collide only through its own samples.
        Default is "auto".
    has_gravity : bool, optional
        Whether gravity acts on this body. Default is True.
    contact_layer : str, optional
        Name of the contact layer of this body. Default is "default".
    """

    E: PositiveFloat = 1e5
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.45
    rho: PositiveFloat = 1000.0
    model: ElasticModel = "stable_neohookean"
    mass_damping: NonNegativeFloat = 0.0
    stiffness_damping: NonNegativeFloat = 0.0
    friction: NonNegativeFloat = 0.5
    penalty_coefficient: PositiveFloat = 1e9
    penalty_smoothing_half_distance: NonNegativeFloat = 5e-3
    penalty_threshold: ValidFloat = 1e-3
    friction_falloff_vel: NonNegativeFloat = 1e-2
    viscous_friction: NonNegativeFloat = 0.0
    normal_viscous_damping: NonNegativeFloat = 0.0
    max_alignment_normals: ValidFloat = 0.0
    collider_type: SoftColliderType = "auto"
    has_gravity: StrictBool = True
    contact_layer: str = "default"

    # Lame parameters, derived from E and nu.
    mu: float = Field(default=0.0, init=False, repr=False)
    lam: float = Field(default=0.0, init=False, repr=False)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not (-1.0 <= self.max_alignment_normals <= 1.0):
            gs.raise_exception("`max_alignment_normals` must be in [-1, 1].")
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
