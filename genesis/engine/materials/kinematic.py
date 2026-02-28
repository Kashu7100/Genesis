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
        self._needs_coup = False
        self._friction = None
        self._coup_friction = 0.0
        self._coup_softness = 0.0
        self._coup_restitution = 0.0
        self._sdf_cell_size = 0.005
        self._sdf_min_res = 32
        self._sdf_max_res = 32
        self._gravity_compensation = 0.0
        self._coupling_mode = None
        self._coupling_link_filter = None

    # Properties required by RigidEntity during morphology loading

    @property
    def rho(self):
        return self._rho

    @property
    def friction(self):
        return self._friction

    @property
    def needs_coup(self):
        return self._needs_coup

    @property
    def coup_friction(self):
        return self._coup_friction

    @property
    def coup_softness(self):
        return self._coup_softness

    @property
    def coup_restitution(self):
        return self._coup_restitution

    @property
    def sdf_cell_size(self):
        return self._sdf_cell_size

    @property
    def sdf_min_res(self):
        return self._sdf_min_res

    @property
    def sdf_max_res(self):
        return self._sdf_max_res

    @property
    def gravity_compensation(self):
        return self._gravity_compensation

    @property
    def coupling_mode(self):
        return self._coupling_mode

    @property
    def coupling_link_filter(self):
        return self._coupling_link_filter
