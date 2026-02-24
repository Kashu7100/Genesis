from .rigid import Rigid


class Avatar(Rigid):
    """
    Visualization-only material for ghost/reference entities.

    Avatar entities are rendered but do not participate in physics simulation, collision detection,
    or constraint solving. Only density is accepted since it is used for inertia computation
    during forward kinematics.

    Parameters
    ----------
    rho : float, optional
        The density of the material (used only for inertia computation during FK). Default is 200.0.
    """

    def __init__(self, rho=200.0):
        super().__init__(
            rho=rho,
            needs_coup=False,
            sdf_max_res=32,
        )
