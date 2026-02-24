from genesis.engine.entities.rigid_entity.rigid_entity import KinematicEntity

from .avatar_joint import AvatarJoint
from .avatar_link import AvatarLink


class AvatarEntity(KinematicEntity):
    """
    A visualization-only entity that renders an articulated body without participating
    in physics simulation, collision detection, or constraint solving.

    Useful for displaying reference/target motions (e.g., mimic policy targets) alongside
    simulated entities without affecting simulation speed.
    """

    _JointClass = AvatarJoint
    _LinkClass = AvatarLink
