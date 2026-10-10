from typing import TYPE_CHECKING, Any

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.shell_entity import ShellEntity


class Shell(Material["ShellEntity"]):
    """
    A thin elastoplastic sheet that can tear and crack, simulated by the shell solver.

    The sheet is a triangle mesh resisting in-plane stretching (membrane) and out-of-plane bending (hinges). Its
    material yields plastically past a stress or curvature threshold, and it fractures at a vertex whose surrounding
    stress exceeds the tensile strength: the vertex splits along the existing mesh edge that relieves the most stress,
    so the crack paths follow the mesh edges.

    Parameters
    ----------
    rho : float, optional
        Volumetric density, in kg/m^3. The mass of a triangle is rho * thickness * area. Default is 1000.
    E : float, optional
        Young's modulus, in Pa. Default is 1e6.
    nu : float, optional
        Poisson's ratio, in [0, 0.5). Default is 0.3.
    thickness : float, optional
        Thickness of the sheet, in m. It scales the membrane stiffness linearly and the bending stiffness cubically.
        Default is 1e-3.
    bending_scale : float, optional
        Multiplier of the bending stiffness of an isotropic plate, for sheets whose bending is softer (paper fibers,
        knits) or stiffer (rolled metal) than their stretching suggests. Default is 1.0.
    damping : float, optional
        Stiffness-proportional damping, in seconds. Larger values dissipate the vibration of stiff sheets faster at the
        cost of a slower, more viscous motion. Default is 0.0.
    tensile_strength : float or None, optional
        Stress at which the material fractures, in Pa. A lower value tears more easily. None disables fracture.
        Default is None.
    bending_fracture_scale : float, optional
        Weight of the bending strain in the fracture criterion. 1.0 treats the outer layer of a bent sheet as the
        stressed one, which makes brittle sheets (glass, ceramics) crack under bending. 0.0 makes the sheet tear under
        stretching alone, as paper does. Default is 1.0.
    yield_stress : float or None, optional
        Von Mises stress past which the sheet stretches plastically, in Pa. Plastic stretching also thins the sheet,
        conserving its volume. None disables stretching plasticity. Default is None.
    plastic_flow_rate : float, optional
        Rate at which the plastic stretching absorbs the stress above the yield stress, in 1/s. A higher value makes
        the yielding faster and closer to perfect plasticity, a lower one makes the sheet creep. Default is 100.
    yield_curvature : float or None, optional
        Curvature past which the sheet bends plastically, in 1/m. A creased sheet keeps its folds. None disables
        bending plasticity. Default is None.
    """

    rho: PositiveFloat = 1000.0
    E: PositiveFloat = 1e6
    nu: NonNegativeFloat = 0.3
    thickness: PositiveFloat = 1e-3
    bending_scale: NonNegativeFloat = 1.0
    damping: NonNegativeFloat = 0.0
    tensile_strength: PositiveFloat | None = None
    bending_fracture_scale: NonNegativeFloat = 1.0
    yield_stress: PositiveFloat | None = None
    plastic_flow_rate: PositiveFloat = 100.0
    yield_curvature: PositiveFloat | None = None

    def model_post_init(self, context: Any) -> None:
        if self.nu >= 0.5:
            gs.raise_exception(f"Poisson's ratio `nu` must lie in [0, 0.5), got {self.nu}.")

    @property
    def stretching_stiffness(self) -> float:
        """Membrane stiffness of the sheet, E * thickness / (1 - nu^2), in N/m."""
        return self.E * self.thickness / (1.0 - self.nu**2)

    @property
    def bending_stiffness(self) -> float:
        """Bending stiffness of the sheet, bending_scale * E * thickness^3 / (12 * (1 - nu^2)), in N*m."""
        return self.bending_scale * self.E * self.thickness**3 / (12.0 * (1.0 - self.nu**2))
