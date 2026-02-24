import genesis as gs

from .rigid import Rigid


class Avatar(Rigid):
    """
    The Avatar class represents a visualization-only material for ghost/reference entities.

    Avatar entities are rendered but do not participate in physics simulation, collision detection,
    or constraint solving. They are useful for displaying reference motions (e.g., mimic policy targets)
    without affecting simulation speed.

    All parameters are inherited from Rigid but collision-related defaults are overridden since
    Avatar entities never collide.

    Parameters
    ----------
    rho : float, optional
        The density of the material (used only for inertia computation during FK). Default is 200.0.
    """

    def __init__(
        self,
        rho=200.0,
        friction=None,
        needs_coup=False,
        coup_friction=0.0,
        coup_softness=0.0,
        coup_restitution=0.0,
        sdf_cell_size=0.005,
        sdf_min_res=32,
        sdf_max_res=32,
        gravity_compensation=0,
    ):
        super().__init__(
            rho=rho,
            friction=friction,
            needs_coup=needs_coup,
            coup_friction=coup_friction,
            coup_softness=coup_softness,
            coup_restitution=coup_restitution,
            sdf_cell_size=sdf_cell_size,
            sdf_min_res=sdf_min_res,
            sdf_max_res=sdf_max_res,
            gravity_compensation=gravity_compensation,
        )
