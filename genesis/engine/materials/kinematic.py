from .base import Material


class Kinematic(Material):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation, collision detection,
    or constraint solving. Only density is accepted since it is used for inertia computation
    during forward kinematics.

    Parameters
    ----------
    rho : float, optional
        The density of the material (used only for inertia computation during FK). Default is 200.0.
    """

    def __init__(self, rho=200.0):
        super().__init__()
        self._rho = float(rho)

    # Properties required by RigidEntity during morphology loading

    @property
    def rho(self):
        return self._rho

    @property
    def friction(self):
        return None

    @property
    def needs_coup(self):
        return False

    @property
    def coup_friction(self):
        return 0.0

    @property
    def coup_softness(self):
        return 0.0

    @property
    def coup_restitution(self):
        return 0.0

    @property
    def sdf_cell_size(self):
        return 0.005

    @property
    def sdf_min_res(self):
        return 32

    @property
    def sdf_max_res(self):
        return 32

    @property
    def gravity_compensation(self):
        return 0.0

    @property
    def coupling_mode(self):
        return None

    @property
    def coupling_link_filter(self):
        return None
