from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
from .avatar_geom import AvatarGeom


class AvatarLink(RigidLink):
    """
    Link class for avatar entities.

    Uses AvatarGeom (which forces contype=0 / conaffinity=0) via the _GeomClass
    attribute. No method overrides needed — inherits _add_geom and _add_vgeom from RigidLink.
    """

    _GeomClass = AvatarGeom
