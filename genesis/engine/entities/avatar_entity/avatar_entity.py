from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity

from .avatar_joint import AvatarJoint
from .avatar_link import AvatarLink


class AvatarEntity(RigidEntity):
    """
    A visualization-only entity that renders an articulated body without participating
    in physics simulation, collision detection, or constraint solving.

    Useful for displaying reference/target motions (e.g., mimic policy targets) alongside
    simulated entities without affecting simulation speed.
    """

    _JointClass = AvatarJoint
    _LinkClass = AvatarLink

    def _init_jac_and_IK(self):
        """No-op: avatar entities do not need Jacobian or IK computation."""
        pass
