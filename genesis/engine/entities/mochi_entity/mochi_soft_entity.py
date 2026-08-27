# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.fem_entity import FEMVisGeom
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.entities import MochiSoftEntityState
from genesis.utils.misc import tensor_to_array

from ..base_entity import Entity

TET_NODE_FORMAT = ".node"


def load_tet_files(node_path):
    """Vertices and tetrahedra of a tetgen '.node' / '.ele' file pair (0- or 1-based indices)."""
    ele_path = os.path.splitext(node_path)[0] + ".ele"
    if not os.path.exists(ele_path):
        gs.raise_exception(f"Tetrahedral mesh element file not found: '{ele_path}'.")

    def read_rows(path, n_cols):
        rows = []
        with open(path) as f:
            header = None
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if header is None:
                    header = [int(float(v)) for v in line.split()]
                    continue
                values = line.split()
                rows.append([float(v) for v in values[: 1 + n_cols]])
        if header is None:
            gs.raise_exception(f"Empty tetrahedral mesh file: '{path}'.")
        rows = np.array(rows, dtype=np.float64).reshape((-1, 1 + n_cols))
        return rows[: header[0]]

    nodes = read_rows(node_path, 3)
    elems = read_rows(ele_path, 4)
    index_base = int(nodes[0, 0]) if len(nodes) > 0 else 0
    verts = nodes[:, 1:4]
    tets = elems[:, 1:5].astype(np.int64) - index_base
    if tets.min(initial=0) < 0 or tets.max(initial=-1) >= len(verts):
        gs.raise_exception(f"Tetrahedral mesh '{ele_path}' references vertices outside '{node_path}'.")
    return verts, tets


class MochiSoftEntity(Entity):
    """
    Deformable body simulated by the MochiSolver: a tetrahedral mesh whose vertex positions are unknowns of the
    implicit solve, under the constitutive model of its `gs.materials.Mochi.Elastic` material.

    The simulation mesh is tetrahedralized from the morph surface (tetgen options on the morph), or read directly from
    a tetgen '.node' / '.ele' file pair given as a `gs.morphs.Mesh`. Its boundary triangles carry the contact samples
    and the render mesh.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        idx,
        idx_in_solver,
        v_start,
        el_start,
        s_start,
        vvert_start,
        vface_start,
        name=None,
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)
        self._idx_in_solver = idx_in_solver
        self._v_start = v_start
        self._el_start = el_start
        self._s_start = s_start
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._queried_states = QueriedStates()

        self.sample()

        # Boundary triangles (outward winding) and the tetrahedron owning each of them.
        self._surface_tri_np, self._surface_el_np = self._boundary_triangles(self.elems)

    def _get_morph_identifier(self) -> str:
        morph = self._morph
        if isinstance(morph, gs.morphs.Box):
            return "soft_box"
        if isinstance(morph, gs.morphs.Sphere):
            return "soft_sphere"
        if isinstance(morph, gs.morphs.Cylinder):
            return "soft_cylinder"
        if isinstance(morph, gs.morphs.Mesh):
            return f"soft_{os.path.splitext(os.path.basename(morph.file))[0]}"
        return "soft_body"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def sample(self):
        """Build the render geoms and the tetrahedral simulation mesh from the morph."""
        morph = self._morph
        if isinstance(morph, gs.morphs.Mesh) and morph.file.endswith(TET_NODE_FORMAT):
            verts, elems = load_tet_files(morph.file)
            verts = verts * np.asarray(morph.scale, dtype=np.float64) + np.asarray(morph.pos, dtype=np.float64)
            self.instantiate(verts, elems)
            surface_tri, _ = self._boundary_triangles(self.elems)
            vmesh = gs.Mesh.from_trimesh(
                trimesh.Trimesh(vertices=self.init_positions, faces=surface_tri, process=False),
                surface=self._surface,
            )
            self._vgeoms = gs.List(
                [
                    FEMVisGeom(
                        entity=self,
                        vvert_start=self._vvert_start,
                        vface_start=self._vface_start,
                        vmesh=vmesh,
                        sim_verts_idx=np.arange(self.n_vertices, dtype=gs.np_int),
                    )
                ]
            )
            return

        meshes = gs.Mesh.from_morph_surface(morph, self._surface)
        surface_verts, surface_faces, verts_maps = mu.merge_submeshes(
            [mesh.verts for mesh in meshes], [mesh.faces for mesh in meshes]
        )
        self._vgeoms = gs.List()
        vvert_start, vface_start = self._vvert_start, self._vface_start
        for mesh, verts_idx in zip(meshes, verts_maps):
            self._vgeoms.append(
                FEMVisGeom(
                    entity=self, vvert_start=vvert_start, vface_start=vface_start, vmesh=mesh, sim_verts_idx=verts_idx
                )
            )
            vvert_start += len(mesh.verts)
            vface_start += len(mesh.faces)

        # File meshes are tetrahedralized untranslated so that the result and its cache are shared across placements;
        # primitives keep the position baked in, as the rest state is sensitive to the exact refinement.
        is_mesh_morph = isinstance(morph, gs.morphs.Mesh)
        if not is_mesh_morph:
            surface_verts = surface_verts + morph.pos
        surface_trimesh = trimesh.Trimesh(vertices=surface_verts, faces=surface_faces, process=False)
        verts, elems = eu.mesh_to_elements(surface_trimesh, tet_cfg=mu.generate_tetgen_config_from_morph(morph))
        if is_mesh_morph:
            verts = verts + morph.pos
        self.instantiate(verts, elems)

    @staticmethod
    def _boundary_triangles(elems):
        """Boundary triangles of positively oriented tetrahedra (outward winding) and the tetrahedron owning each."""
        el2tri = np.array(
            [[[v[0], v[1], v[2]], [v[0], v[3], v[1]], [v[1], v[3], v[2]], [v[0], v[2], v[3]]] for v in elems],
            dtype=gs.np_int,
        ).reshape((-1, 3))
        _, unique_idcs, cnt = np.unique(np.sort(el2tri, axis=1), axis=0, return_counts=True, return_index=True)
        return el2tri[unique_idcs][cnt == 1], (unique_idcs // 4)[cnt == 1].astype(gs.np_int)

    def instantiate(self, verts, elems):
        """Set the rest vertex positions (with the morph orientation applied about their centroid and the morph pose
        offset composed on) and the tetrahedra."""
        verts = np.asarray(verts, dtype=gs.np_float)
        elems = np.asarray(elems, dtype=gs.np_int)
        if len(verts) == 0 or len(elems) == 0:
            gs.raise_exception("Entity has no tetrahedra.")
        morph_quat = np.array(self._morph.quat, dtype=gs.np_float)
        init_quat = gu.transform_quat_by_quat(np.array(self._morph.offset_quat, dtype=gs.np_float), morph_quat)
        R = gu.quat_to_R(init_quat)
        verts_COM = verts.mean(axis=0)
        init_positions = (verts - verts_COM) @ R.T + verts_COM
        init_positions = init_positions + gu.transform_by_quat(
            np.array(self._morph.offset_pos, dtype=gs.np_float), morph_quat
        )
        # Positive orientation of every tetrahedron (node 3 as origin).
        D = np.stack([init_positions[elems[:, k]] - init_positions[elems[:, 3]] for k in range(3)], axis=-1)
        is_inverted = np.linalg.det(D) < 0.0
        elems = elems.copy()
        elems[is_inverted, 1], elems[is_inverted, 2] = elems[is_inverted, 2], elems[is_inverted, 1]
        self.init_positions = init_positions.astype(gs.np_float)
        self.elems = elems

    # ------------------------------------------------------------------------------------
    # ---------------------------------------- io ----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self):
        s_global = self._sim.cur_step_global
        if s_global in self._queried_states:
            return self._queried_states[s_global][0]
        state = MochiSoftEntityState(self, s_global)
        self._solver.get_soft_entity_state(self, state.pos, state.vel)
        self._queried_states.append(state)
        return state

    def get_vertices_position(self, envs_idx=None):
        """World positions of the vertices, shape (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        return self._solver.get_soft_vertices_position(self, envs_idx)

    def get_vertices_velocity(self, envs_idx=None):
        """Velocities of the vertices, shape (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        return self._solver.get_soft_vertices_velocity(self, envs_idx)

    def get_vertices_contact_force(self, envs_idx=None):
        """Net contact force applied to each vertex, shape (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        return self._solver.get_soft_vertices_contact_force(self, envs_idx)

    def set_position(self, pos, envs_idx=None):
        """Move the body: a (3,) or (n_envs, 3) offset of the rest positions, or full vertex positions of shape
        (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        pos = torch.as_tensor(tensor_to_array(pos), dtype=gs.tc_float, device=gs.device)
        if pos.shape[-2:] != (self.n_vertices, 3):
            pos = torch.as_tensor(self.init_positions, dtype=gs.tc_float, device=gs.device) + pos[..., None, :]
        self._solver.set_soft_vertices_position(self, pos, envs_idx)

    def set_velocity(self, vel, envs_idx=None):
        """Set the vertex velocities: one (3,) or (n_envs, 3) velocity for all vertices, or per vertex of shape
        (n_vertices, 3) or (n_envs, n_vertices, 3)."""
        vel = torch.as_tensor(tensor_to_array(vel), dtype=gs.tc_float, device=gs.device)
        if vel.shape[-2:] != (self.n_vertices, 3):
            vel = vel[..., None, :].expand(*vel.shape[:-1], self.n_vertices, 3)
        self._solver.set_soft_vertices_velocity(self, vel, envs_idx)

    def set_vertices_fixed(self, verts_idx_local, is_fixed=True, envs_idx=None):
        """Fix the given vertices at their current position (Dirichlet condition), or release them."""
        verts_idx = np.atleast_1d(np.asarray(tensor_to_array(verts_idx_local), dtype=gs.np_int))
        if verts_idx.min(initial=0) < 0 or verts_idx.max(initial=-1) >= self.n_vertices:
            gs.raise_exception("Vertex index out of range.")
        self._solver.set_soft_vertices_fixed(self, verts_idx, is_fixed, envs_idx)

    def set_contact_params(
        self,
        *,
        penalty_coefficient=None,
        friction=None,
        penalty_smoothing_half_distance=None,
        penalty_threshold=None,
        friction_falloff_vel=None,
        viscous_friction=None,
        normal_viscous_damping=None,
    ):
        """Update the contact parameters of this body (see `gs.materials.Mochi.Elastic`)."""
        self._solver.set_soft_entity_contact_params(
            self,
            penalty_coefficient=penalty_coefficient,
            friction=friction,
            penalty_smoothing_half_distance=penalty_smoothing_half_distance,
            penalty_threshold=penalty_threshold,
            friction_falloff_vel=friction_falloff_vel,
            viscous_friction=viscous_friction,
            normal_viscous_damping=normal_viscous_damping,
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def idx_in_solver(self):
        return self._idx_in_solver

    @property
    def n_vertices(self):
        return len(self.init_positions)

    @property
    def n_elements(self):
        return len(self.elems)

    @property
    def n_surfaces(self):
        return len(self._surface_tri_np)

    @property
    def surface_triangles(self):
        return self._surface_tri_np

    @property
    def surface_elements(self):
        return self._surface_el_np

    @property
    def n_dofs(self):
        return 3 * self.n_vertices

    @property
    def v_start(self):
        return self._v_start

    @property
    def v_end(self):
        return self._v_start + self.n_vertices

    @property
    def el_start(self):
        return self._el_start

    @property
    def s_start(self):
        return self._s_start

    @property
    def vgeoms(self):
        return self._vgeoms

    @property
    def n_vgeoms(self):
        return len(self._vgeoms)

    @property
    def n_vverts(self):
        return sum(vgeom.n_vverts for vgeom in self._vgeoms)

    @property
    def n_vfaces(self):
        return sum(vgeom.n_vfaces for vgeom in self._vgeoms)

    @property
    def vvert_start(self):
        return self._vvert_start

    @property
    def vface_start(self):
        return self._vface_start

    @property
    def volume(self):
        D = np.stack(
            [self.init_positions[self.elems[:, k]] - self.init_positions[self.elems[:, 3]] for k in range(3)], -1
        )
        return float(np.abs(np.linalg.det(D)).sum() / 6.0)

    @property
    def mass(self):
        return self.material.rho * self.volume
