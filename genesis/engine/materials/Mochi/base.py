from ..base import EntityT, Material


class Base(Material[EntityT]):
    """
    Base class of the materials simulated by the MochiSolver.

    Note
    ----
    This class should *not* be instantiated directly. It only marks a material for dispatch to the MochiSolver.
    """
