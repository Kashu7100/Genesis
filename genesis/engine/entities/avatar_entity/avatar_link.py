from typing import TYPE_CHECKING

import genesis as gs

from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
from .avatar_geom import AvatarGeom, AvatarVisGeom

if TYPE_CHECKING:
    from .avatar_entity import AvatarEntity
    from .avatar_joint import AvatarJoint


class AvatarLink(RigidLink):
    """
    Link class for avatar entities.

    Overrides _add_geom and _add_vgeom to use Avatar-specific geom subclasses,
    which force contype=0 and conaffinity=0 on all collision geometries.
    """

    def _add_geom(
        self,
        mesh,
        init_pos,
        init_quat,
        type,
        friction,
        sol_params,
        center_init=None,
        needs_coup=False,
        contype=1,
        conaffinity=1,
        data=None,
    ):
        geom = AvatarGeom(
            link=self,
            idx=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            verts_state_start=self.n_verts + self._verts_state_start,
            mesh=mesh,
            init_pos=init_pos,
            init_quat=init_quat,
            type=type,
            friction=friction,
            sol_params=sol_params,
            center_init=center_init,
            needs_coup=needs_coup,
            contype=0,
            conaffinity=0,
            data=data,
        )
        self._geoms.append(geom)

    def _add_vgeom(self, vmesh, init_pos, init_quat):
        vgeom = AvatarVisGeom(
            link=self,
            idx=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            vmesh=vmesh,
            init_pos=init_pos,
            init_quat=init_quat,
        )
        self._vgeoms.append(vgeom)
