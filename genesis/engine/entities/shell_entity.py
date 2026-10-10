from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.base_entity import Entity
from genesis.utils.misc import qd_to_torch

if TYPE_CHECKING:
    from genesis.engine.materials.shell import Shell
    from genesis.engine.solvers.shell_solver import ShellSolver


class ShellTopology(NamedTuple):
    """The rest mesh of a shell entity, as the shell solver consumes it (see ShellInfo in array_class.py).

    Every index is local to the entity: vertices, faces, hinges, and corners (3 * i_f + k).
    """

    faces_mass: np.ndarray
    faces_rest_area: np.ndarray
    faces_Dm: np.ndarray
    faces_basis: np.ndarray
    faces_hinge: np.ndarray
    hinges_corner: np.ndarray
    hinges_opposite_corner: np.ndarray
    hinges_rest_angle: np.ndarray
    hinges_rest_len: np.ndarray
    hinges_rest_area: np.ndarray
    verts_fan_start: np.ndarray
    verts_fan_len: np.ndarray
    verts_is_fan_closed: np.ndarray
    fans_corner: np.ndarray
    fans_next_hinge: np.ndarray


class ShellPatches(NamedTuple):
    """The partition of the vertices of a shell entity into patches, each one spanning the coarse space of the linear
    solve by the displacements affine in its two principal in-plane rest coordinates (see the coarse space in
    shell_solver.py).

    verts_patch is the patch of every vertex, local to the entity, and verts_phi its affine shape functions (1, s1, s2),
    the rest coordinates being centered on the patch and normalized by its radius.
    """

    n_patches: int
    verts_patch: np.ndarray
    verts_phi: np.ndarray


def build_shell_patches(verts: np.ndarray, n_patches: int) -> ShellPatches:
    """Partition the rest vertices of a shell into compact patches by k-means, seeded by farthest point sampling."""
    centers_idx = [0]
    dist_sq = np.square(verts - verts[0]).sum(axis=1)
    for _ in range(n_patches - 1):
        centers_idx.append(int(np.argmax(dist_sq)))
        dist_sq = np.minimum(dist_sq, np.square(verts - verts[centers_idx[-1]]).sum(axis=1))
    centers = verts[centers_idx]
    for _ in range(20):
        verts_patch = np.argmin(np.square(verts[:, None] - centers[None]).sum(axis=-1), axis=1)
        counts = np.bincount(verts_patch, minlength=n_patches)
        sums = np.zeros_like(centers)
        np.add.at(sums, verts_patch, verts)
        centers = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], centers)
    # Patches left empty by the iterations are dropped, renumbering the others contiguously
    patches_used, verts_patch = np.unique(verts_patch, return_inverse=True)

    verts_phi = np.ones((len(verts), 3), dtype=gs.np_float)
    for i_p in range(len(patches_used)):
        verts_mask = verts_patch == i_p
        offsets = verts[verts_mask] - verts[verts_mask].mean(axis=0)
        _, _, axes = np.linalg.svd(offsets, full_matrices=False)
        coords = offsets @ axes[:2].T
        radius = np.sqrt(np.square(coords).sum(axis=1).mean())
        verts_phi[verts_mask, 1:] = coords / max(radius, gs.EPS)
    return ShellPatches(n_patches=len(patches_used), verts_patch=verts_patch.astype(gs.np_int), verts_phi=verts_phi)


def build_shell_topology(verts: np.ndarray, faces: np.ndarray, rho: float, thickness: float) -> ShellTopology:
    """Compute the rest frames, hinges and vertex fans of a consistently oriented, edge- and vertex-manifold mesh.

    Raises if the mesh has an edge shared by more than two faces, two faces of opposite winding across an edge, or a
    vertex whose faces do not form a single fan.
    """
    n_verts, n_faces = len(verts), len(faces)
    corners_next = faces[:, [1, 2, 0]].reshape(-1)
    corners_vert = faces.reshape(-1)
    corners_prev = faces[:, [2, 0, 1]].reshape(-1)

    # Directed edges, one per corner, from the corner vertex to the next vertex of its face
    directed_edges = {}
    for i_c, (i_a, i_b) in enumerate(zip(corners_vert, corners_next)):
        if (i_a, i_b) in directed_edges:
            gs.raise_exception(
                f"Shell mesh holds edge ({i_a}, {i_b}) twice with the same winding: either its faces are not "
                "consistently oriented or the edge is shared by more than two faces."
            )
        directed_edges[(i_a, i_b)] = i_c

    # Hinges, one per interior edge, its first face holding the edge from a to b and its second from b to a
    hinges_corner, hinges_opposite_corner = [], []
    edges_hinge = {}
    for (i_a, i_b), i_c0a in directed_edges.items():
        i_c1b = directed_edges.get((i_b, i_a))
        if i_c1b is None or i_a > i_b:
            continue
        i_f0, i_f1 = i_c0a // 3, i_c1b // 3
        i_c0b = 3 * i_f0 + (i_c0a + 1) % 3
        i_c1a = 3 * i_f1 + (i_c1b + 1) % 3
        edges_hinge[(i_a, i_b)] = edges_hinge[(i_b, i_a)] = len(hinges_corner)
        hinges_corner.append((i_c0a, i_c0b, i_c1a, i_c1b))
        hinges_opposite_corner.append((3 * i_f0 + (i_c0a + 2) % 3, 3 * i_f1 + (i_c1b + 2) % 3))
    hinges_corner = np.array(hinges_corner, dtype=gs.np_int).reshape((-1, 4))
    hinges_opposite_corner = np.array(hinges_opposite_corner, dtype=gs.np_int).reshape((-1, 2))
    faces_hinge = np.array(
        [edges_hinge.get((i_a, i_b), -1) for i_a, i_b in zip(corners_vert, corners_next)], dtype=gs.np_int
    ).reshape((n_faces, 3))

    # Rest frame of every face: an orthonormal basis of its plane, whose cross product is the face normal, and the
    # coordinates of its two edges from the first corner in that basis.
    edges_1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    edges_2 = verts[faces[:, 2]] - verts[faces[:, 0]]
    faces_normal = np.cross(edges_1, edges_2)
    faces_double_area = np.linalg.norm(faces_normal, axis=1)
    if (faces_double_area < gs.EPS).any():
        gs.raise_exception("Shell mesh holds degenerate faces of zero area.")
    faces_normal /= faces_double_area[:, None]
    basis_0 = edges_1 / np.linalg.norm(edges_1, axis=1, keepdims=True)
    basis_1 = np.cross(faces_normal, basis_0)
    faces_basis = np.stack((basis_0, basis_1), axis=-1)
    faces_Dm = np.stack(
        (
            np.stack(((edges_1 * basis_0).sum(1), (edges_2 * basis_0).sum(1)), axis=-1),
            np.stack(((edges_1 * basis_1).sum(1), (edges_2 * basis_1).sum(1)), axis=-1),
        ),
        axis=1,
    )
    faces_rest_area = 0.5 * faces_double_area

    # Rest dihedral angles, signed about the edge running from a to b
    i_va = corners_vert[hinges_corner[:, 0]]
    i_vb = corners_vert[hinges_corner[:, 1]]
    hinges_edge = verts[i_vb] - verts[i_va]
    hinges_rest_len = np.linalg.norm(hinges_edge, axis=1)
    normals_0 = faces_normal[hinges_corner[:, 0] // 3]
    normals_1 = faces_normal[hinges_corner[:, 2] // 3]
    hinges_rest_angle = np.arctan2(
        (hinges_edge / hinges_rest_len[:, None] * np.cross(normals_0, normals_1)).sum(1), (normals_0 * normals_1).sum(1)
    )
    hinges_rest_area = faces_rest_area[hinges_corner[:, 0] // 3] + faces_rest_area[hinges_corner[:, 2] // 3]

    # Counter-clockwise fan of every vertex: the corner after (f, k) is the one whose face holds the edge from the
    # vertex to the previous vertex of f. An open fan starts at the corner whose outgoing edge lies on the boundary.
    verts_corners = [[] for _ in range(n_verts)]
    for i_c, i_v in enumerate(corners_vert):
        verts_corners[i_v].append(i_c)
    verts_fan_start = np.zeros(n_verts, dtype=gs.np_int)
    verts_fan_len = np.zeros(n_verts, dtype=gs.np_int)
    verts_is_fan_closed = np.zeros(n_verts, dtype=np.bool_)
    fans_corner, fans_next_hinge = [], []
    for i_v, corners in enumerate(verts_corners):
        if not corners:
            gs.raise_exception(f"Shell mesh holds vertex {i_v} that belongs to no face.")
        corners_start = [i_c for i_c in corners if (corners_next[i_c], i_v) not in directed_edges]
        if len(corners_start) > 1:
            gs.raise_exception(f"Shell mesh is not manifold at vertex {i_v}: its faces form several fans.")
        i_c = corners_start[0] if corners_start else corners[0]
        fan = [i_c]
        while True:
            i_c = directed_edges.get((i_v, corners_prev[i_c]))
            if i_c is None or i_c == fan[0]:
                break
            fan.append(i_c)
        if len(fan) != len(corners):
            gs.raise_exception(f"Shell mesh is not manifold at vertex {i_v}: its faces form several fans.")
        verts_fan_start[i_v] = len(fans_corner)
        verts_fan_len[i_v] = len(fan)
        verts_is_fan_closed[i_v] = not corners_start
        fans_corner += fan
        fans_next_hinge += [edges_hinge.get((i_v, corners_prev[i_c]), -1) for i_c in fan]

    return ShellTopology(
        faces_mass=rho * thickness * faces_rest_area,
        faces_rest_area=faces_rest_area,
        faces_Dm=faces_Dm,
        faces_basis=faces_basis,
        faces_hinge=faces_hinge,
        hinges_corner=hinges_corner,
        hinges_opposite_corner=hinges_opposite_corner,
        hinges_rest_angle=hinges_rest_angle,
        hinges_rest_len=hinges_rest_len,
        hinges_rest_area=hinges_rest_area,
        verts_fan_start=verts_fan_start,
        verts_fan_len=verts_fan_len,
        verts_is_fan_closed=verts_is_fan_closed,
        fans_corner=np.array(fans_corner, dtype=gs.np_int),
        fans_next_hinge=np.array(fans_next_hinge, dtype=gs.np_int),
    )


class ShellEntity(Entity):
    """
    A thin sheet simulated by the shell solver, which can stretch, bend, yield plastically, and tear.

    The sheet is the surface mesh of its morph, welded into a single mesh that must be manifold and consistently
    oriented. Its vertices are the original vertices of that mesh, followed by the slots that fracture fills, in
    every environment independently, when it splits a vertex in two. An original vertex keeps its index for the whole
    simulation, holding one side of every crack that splits it.
    """

    def __init__(
        self,
        scene,
        solver: "ShellSolver",
        material: "Shell",
        morph,
        surface,
        idx: int,
        vert_start: int,
        face_start: int,
        hinge_start: int,
        name: str | None = None,
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        if not isinstance(morph, (gs.options.morphs.Mesh, gs.options.morphs.Box, gs.options.morphs.Sphere)):
            gs.raise_exception(
                f"Shell entities are created from a 'Mesh', 'Box' or 'Sphere' morph, got {type(morph).__name__}."
            )
        meshes = gs.Mesh.from_morph_surface(morph, surface)
        verts, faces, _ = mu.merge_submeshes([mesh.verts for mesh in meshes], [mesh.faces for mesh in meshes])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.repair.fix_winding(mesh)
        self._surface = meshes[0].surface

        pos, quat = gu.transform_pos_quat_by_trans_quat(
            np.array(morph.offset_pos, dtype=gs.np_float),
            np.array(morph.offset_quat, dtype=gs.np_float),
            np.array(morph.pos, dtype=gs.np_float),
            np.array(morph.quat, dtype=gs.np_float),
        )
        self._init_verts = gu.transform_by_trans_quat(np.asarray(mesh.vertices, dtype=gs.np_float), pos, quat)
        self._faces = np.asarray(mesh.faces, dtype=gs.np_int)
        self._topology = build_shell_topology(self._init_verts, self._faces, material.rho, material.thickness)
        self._patches = None
        if solver.n_coarse_patches > 0:
            n_patches = min(solver.n_coarse_patches, max(1, len(self._init_verts) // 32))
            self._patches = build_shell_patches(self._init_verts, n_patches)

        n_verts = len(self._init_verts)
        n_split_verts = int(np.ceil(solver.fracture_capacity * n_verts)) if material.tensile_strength else 0
        self._idx_in_solver = solver.n_entities
        self._vert_start = vert_start
        self._n_verts = n_verts
        self._n_verts_max = n_verts + n_split_verts
        self._face_start = face_start
        self._hinge_start = hinge_start

    def _get_morph_identifier(self) -> str:
        if isinstance(self._morph, gs.options.morphs.Mesh):
            return f"shell_{self._morph.file}"
        return f"shell_{type(self._morph).__name__.lower()}"

    # ------------------------------------------------------------------------------------
    # ------------------------------------- state ----------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_verts_pos(self, envs_idx=None) -> torch.Tensor:
        """
        Get the position of every vertex slot of the sheet, in m.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        pos : torch.Tensor, shape (n_verts_max, 3) or (n_envs, n_verts_max, 3)
            The original vertices come first. A slot that fracture has not filled yet (see `get_n_verts`) reads zero.
        """
        tensor = qd_to_torch(self._solver.shell_state.verts_pos, envs_idx, transpose=True, copy=True)
        tensor = tensor[:, self._vert_start : self._vert_start + self._n_verts_max]
        return tensor[0] if self._scene.n_envs == 0 else tensor

    @gs.assert_built
    def get_verts_vel(self, envs_idx=None) -> torch.Tensor:
        """
        Get the velocity of every vertex slot of the sheet, in m/s.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        vel : torch.Tensor, shape (n_verts_max, 3) or (n_envs, n_verts_max, 3)
            The original vertices come first. A slot that fracture has not filled yet (see `get_n_verts`) reads zero.
        """
        tensor = qd_to_torch(self._solver.shell_state.verts_vel, envs_idx, transpose=True, copy=True)
        tensor = tensor[:, self._vert_start : self._vert_start + self._n_verts_max]
        return tensor[0] if self._scene.n_envs == 0 else tensor

    @gs.assert_built
    def get_n_verts(self, envs_idx=None) -> torch.Tensor:
        """
        Get the number of vertices of the sheet, the original ones plus those fracture created.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        n_verts : torch.Tensor, shape () or (n_envs,)
        """
        tensor = qd_to_torch(self._solver.shell_state.entities_n_verts, envs_idx, transpose=True, copy=True)
        tensor = tensor[:, self._idx_in_solver]
        return tensor[0] if self._scene.n_envs == 0 else tensor

    @gs.assert_built
    def get_faces(self, envs_idx=None) -> torch.Tensor:
        """
        Get the vertices of every triangle of the sheet, which fracture rewires as it splits vertices.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        faces : torch.Tensor, shape (n_faces, 3) or (n_envs, n_faces, 3)
            The indices of the vertices of each triangle, local to the sheet (see `get_verts_pos`).
        """
        tensor = qd_to_torch(self._solver.shell_state.corners_vert, envs_idx, transpose=True, copy=True)
        tensor = tensor[:, 3 * self._face_start : 3 * (self._face_start + self.n_faces)] - self._vert_start
        tensor = tensor.reshape((tensor.shape[0], self.n_faces, 3))
        return tensor[0] if self._scene.n_envs == 0 else tensor

    @gs.assert_built
    def set_verts_pos(self, pos, verts_idx_local=None, envs_idx=None):
        """
        Set the position of some vertices of the sheet, in m.

        Parameters
        ----------
        pos : array_like, shape (3,), (n_verts, 3) or (n_envs, n_verts, 3)
            The positions, broadcast over the vertices and environments selected.
        verts_idx_local : None | array_like, optional
            The indices of the vertices, local to the sheet. If None, all original vertices. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments. Defaults to None.
        """
        self._solver.set_verts_pos(pos, self, verts_idx_local, envs_idx)

    @gs.assert_built
    def set_verts_vel(self, vel, verts_idx_local=None, envs_idx=None):
        """
        Set the velocity of some vertices of the sheet, in m/s.

        A fixed vertex (see `fix_verts`) keeps the velocity set here, which drives it like a handle.

        Parameters
        ----------
        vel : array_like, shape (3,), (n_verts, 3) or (n_envs, n_verts, 3)
            The velocities, broadcast over the vertices and environments selected.
        verts_idx_local : None | array_like, optional
            The indices of the vertices, local to the sheet. If None, all original vertices. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments. Defaults to None.
        """
        self._solver.set_verts_vel(vel, self, verts_idx_local, envs_idx)

    @gs.assert_built
    def fix_verts(self, verts_idx_local=None, envs_idx=None):
        """
        Fix some vertices of the sheet, which then move at the velocity last set, regardless of any force.

        Parameters
        ----------
        verts_idx_local : None | array_like, optional
            The indices of the vertices, local to the sheet. If None, all original vertices. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments. Defaults to None.
        """
        self._solver.set_verts_fixed(True, self, verts_idx_local, envs_idx)

    @gs.assert_built
    def release_verts(self, verts_idx_local=None, envs_idx=None):
        """
        Release some fixed vertices of the sheet, which then move under the forces acting on them again.

        Parameters
        ----------
        verts_idx_local : None | array_like, optional
            The indices of the vertices, local to the sheet. If None, all original vertices. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments. Defaults to None.
        """
        self._solver.set_verts_fixed(False, self, verts_idx_local, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def init_verts(self) -> np.ndarray:
        """The position of the original vertices at creation, in m."""
        return self._init_verts

    @property
    def init_faces(self) -> np.ndarray:
        """The vertices of every triangle at creation, local to the sheet."""
        return self._faces

    @property
    def topology(self) -> ShellTopology:
        """The rest mesh the shell solver consumes."""
        return self._topology

    @property
    def patches(self) -> ShellPatches | None:
        """The patches spanning the coarse space of the linear solve, None if the solver uses none."""
        return self._patches

    @property
    def n_verts(self) -> int:
        """The number of original vertices."""
        return self._n_verts

    @property
    def n_verts_max(self) -> int:
        """The number of vertex slots, the original vertices plus those fracture can create."""
        return self._n_verts_max

    @property
    def n_faces(self) -> int:
        """The number of triangles."""
        return len(self._faces)

    @property
    def n_hinges(self) -> int:
        """The number of interior edges, each resisting the bending of its two triangles."""
        return len(self._topology.hinges_corner)

    @property
    def vert_start(self) -> int:
        return self._vert_start

    @property
    def face_start(self) -> int:
        return self._face_start

    @property
    def hinge_start(self) -> int:
        return self._hinge_start
