from typing import TYPE_CHECKING

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.avatar_entity import AvatarEntity
from genesis.engine.states.solvers import AvatarSolverState
from genesis.options.solvers import RigidOptions

from .rigid.rigid_solver import RigidSolver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator
    from genesis.options.solvers import AvatarOptions


class AvatarSolver(RigidSolver):
    """
    A visualization-only solver for ghost/reference articulated entities.

    Inherits from RigidSolver to reuse FK, kinematic tree, and render transform infrastructure.
    All physics-related operations (collision, constraint solving, integration) are disabled.
    The solver only computes forward kinematics for visualization purposes.
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: "AvatarOptions") -> None:
        # Build a full RigidOptions with physics disabled so all RigidSolver infrastructure works.
        internal_options = RigidOptions(
            dt=options.dt,
            enable_collision=False,
            enable_joint_limit=False,
            enable_self_collision=False,
            enable_neutral_collision=False,
            enable_adjacent_collision=False,
            disable_constraint=True,
            max_collision_pairs=0,
            enable_multi_contact=False,
            enable_mujoco_compatibility=False,
            use_contact_island=False,
            use_hibernation=False,
            max_dynamic_constraints=0,
            iterations=0,
        )
        super().__init__(scene, sim, internal_options)
        self._enable_collision = False

    def add_entity(self, idx, material, morph, surface, visualize_contact=False, name=None):
        morph_heterogeneous = []
        if isinstance(morph, (tuple, list)):
            morph, *morph_heterogeneous = morph

        morph._enable_mujoco_compatibility = self._enable_mujoco_compatibility

        entity = AvatarEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            idx_in_solver=self.n_entities,
            link_start=self.n_links,
            joint_start=self.n_joints,
            q_start=self.n_qs,
            dof_start=self.n_dofs,
            geom_start=self.n_geoms,
            cell_start=self.n_cells,
            vert_start=self.n_verts,
            free_verts_state_start=self.n_free_verts,
            fixed_verts_state_start=self.n_fixed_verts,
            face_start=self.n_faces,
            edge_start=self.n_edges,
            vgeom_start=self.n_vgeoms,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            visualize_contact=False,
            morph_heterogeneous=morph_heterogeneous,
            name=name,
        )
        assert isinstance(entity, RigidEntity)
        self._entities.append(entity)
        return entity

    def _init_mass_mat(self):
        """No-op: avatar entities have no physics so no mass matrix or gravity needed."""
        pass

    def _init_invweight_and_meaninertia(self, envs_idx=None, *, force_update=True):
        """No-op: avatar entities do not need inverse weight computation."""
        pass

    def _init_collider(self):
        """No-op: avatar entities do not participate in collision."""
        self.collider = None

    def _init_constraint_solver(self):
        """No-op: avatar entities do not need constraint solving."""
        self.constraint_solver = None

    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        from genesis.utils.misc import qd_to_torch
        from genesis.engine.solvers.rigid.rigid_solver import (
            kernel_set_dofs_position,
            kernel_set_zero,
            kernel_forward_kinematics_links_geoms,
        )

        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]
        kernel_set_dofs_position(
            position,
            dofs_idx,
            envs_idx,
            self.dofs_state,
            self.links_info,
            self.joints_info,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

        if gs.use_zerocopy:
            errno = qd_to_torch(self._errno, copy=False)
            errno[envs_idx] = 0
        else:
            kernel_set_zero(envs_idx, self._errno)

        # Skip collider.reset() and constraint_solver.reset() — avatar has neither.

        kernel_forward_kinematics_links_geoms(
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def substep(self, f):
        """No-op: avatar entities are not simulated."""
        pass

    def substep_pre_coupling(self, f):
        """No-op: avatar entities do not participate in coupling."""
        pass

    def substep_post_coupling(self, f):
        """No-op: avatar entities do not participate in coupling."""
        pass

    def check_errno(self):
        """No-op: avatar solver has no error conditions to check."""
        pass

    def clear_external_force(self):
        """No-op: avatar entities have no external forces."""
        pass

    def process_input(self, in_backward=False):
        """Process input for avatar entities (set qpos from user commands)."""
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def get_state(self, f=None):
        if self.is_active:
            s_global = self.sim.cur_step_global
            if s_global in self._queried_states:
                return self._queried_states[s_global][0]

            state = AvatarSolverState(self._scene, s_global)

            from genesis.engine.solvers.rigid.rigid_solver import kernel_get_state

            kernel_get_state(
                qpos=state.qpos,
                vel=state.dofs_vel,
                acc=state.dofs_acc,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                mass_shift=state.mass_shift,
                friction_ratio=state.friction_ratio,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._queried_states.append(state)
        else:
            state = None
        return state

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

            from genesis.utils.misc import qd_to_torch
            from genesis.engine.solvers.rigid.rigid_solver import (
                kernel_set_zero,
                kernel_set_state,
                kernel_forward_kinematics_links_geoms,
            )

            if gs.use_zerocopy:
                errno = qd_to_torch(self._errno, copy=False)
                errno[envs_idx] = 0
            else:
                kernel_set_zero(envs_idx, self._errno)

            kernel_set_state(
                qpos=state.qpos,
                dofs_vel=state.dofs_vel,
                dofs_acc=state.dofs_acc,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                mass_shift=state.mass_shift,
                friction_ratio=state.friction_ratio,
                envs_idx=envs_idx,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            kernel_forward_kinematics_links_geoms(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
