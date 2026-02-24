from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom, RigidVisGeom


class AvatarGeom(RigidGeom):
    """
    Collision geometry for avatar entities.

    Forces contype=0 and conaffinity=0 so the geom never participates in collision detection,
    even if collision infrastructure were somehow present.
    """

    def __init__(self, **kwargs):
        kwargs["contype"] = 0
        kwargs["conaffinity"] = 0
        kwargs["needs_coup"] = False
        super().__init__(**kwargs)


class AvatarVisGeom(RigidVisGeom):
    """Visual geometry for avatar entities. Pass-through subclass of RigidVisGeom."""

    pass
