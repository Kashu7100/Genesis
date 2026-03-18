"""Complementarity-free analytical contact solver.

Implements the ComFree-Sim approach (arXiv:2603.12185): closed-form contact
force computation via dual-cone impedance, replacing the iterative CG/Newton
solver for contact constraints.

Contact forces decouple across pairs, making the computation embarrassingly
parallel with O(n) scaling in contact count.

In hybrid mode, ComFree handles contacts while the existing iterative solver
handles equality constraints, joint limits, and frictionloss.
"""

from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd import func_solve_mass_batch

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


class ComFreeSolver:
    def __init__(self, rigid_solver: "RigidSolver"):
        self._solver = rigid_solver
        self._collider = rigid_solver.collider
        self._B = rigid_solver._B

        self._qfrc_contact = qd.ndarray(gs.qd_float, shape=(rigid_solver.n_dofs, self._B))

    def resolve_contacts(self):
        """Compute closed-form contact forces and fold into acc_smooth.

        After this call, dofs_state.acc_smooth and dofs_state.force include
        contact effects, so the iterative solver sees post-contact state.
        The contact generalized forces are saved in self._qfrc_contact for
        later accumulation.
        """
        kernel_comfree_resolve(
            self._solver.links_info,
            self._solver.links_state,
            self._solver.dofs_state,
            self._solver.dofs_info,
            self._solver.entities_info,
            self._collider._collider_state,
            self._qfrc_contact,
            self._solver.constraint_solver.constraint_state,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

    def finalize_no_iterative_solve(self):
        """Set dofs_state.acc when the iterative solver was skipped.

        When disable_constraint is True, the iterative solver doesn't run,
        so func_update_qacc never sets dofs_state.acc. We set it from the
        updated acc_smooth (which includes contact effects).
        """
        kernel_comfree_set_acc_from_acc_smooth(
            self._solver.dofs_state,
            self._qfrc_contact,
            self._solver._static_rigid_sim_config,
            self._solver._errno,
        )

    def post_iterative_solve(self):
        """Add contact forces to the iterative solver's output.

        The iterative solver's func_update_qacc wrote qfrc_constraint and
        force for non-contact constraints only. This adds the contact
        contribution so the totals are correct.
        """
        kernel_comfree_post_solve(
            self._solver.dofs_state,
            self._qfrc_contact,
            self._solver._static_rigid_sim_config,
        )

    def update_contact_force(self):
        """Write per-link 3D contact forces."""
        func_comfree_update_contact_force(
            self._solver.links_state,
            self._collider._collider_state,
            self._solver._static_rigid_sim_config,
        )


# =============================================================================
# Core ComFree kernel
# =============================================================================


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_comfree_resolve(
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    collider_state: array_class.ColliderState,
    qfrc_contact: array_class.V_ANNOTATION,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Closed-form contact force computation.

    Computes per-contact forces and folds them into acc_smooth so that
    the subsequent iterative solver for non-contact constraints sees
    post-contact state.
    """
    EPS = rigid_global_info.EPS[None]
    n_dofs = dofs_state.vel.shape[0]
    _B = dofs_state.vel.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_d in range(n_dofs):
            qfrc_contact[i_d, i_b] = gs.qd_float(0.0)

        for i_col in range(collider_state.n_contacts[i_b]):
            contact_data_normal = collider_state.contact_data.normal[i_col, i_b]
            contact_data_pos = collider_state.contact_data.pos[i_col, i_b]
            contact_data_penetration = collider_state.contact_data.penetration[i_col, i_b]
            contact_data_friction = collider_state.contact_data.friction[i_col, i_b]
            contact_data_sol_params = collider_state.contact_data.sol_params[i_col, i_b]
            link_a = collider_state.contact_data.link_a[i_col, i_b]
            link_b = collider_state.contact_data.link_b[i_col, i_b]

            link_a_maybe_batch = (
                [link_a, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_a
            )
            link_b_maybe_batch = (
                [link_b, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else link_b
            )

            d1, d2 = gu.qd_orthogonals(contact_data_normal)

            invweight = links_info.invweight[link_a_maybe_batch][0]
            if link_b > -1:
                invweight = invweight + links_info.invweight[link_b_maybe_batch][0]

            contact_force_3d = qd.Vector.zero(gs.qd_float, 3)

            for i_dir in range(4):
                d = (2 * (i_dir % 2) - 1) * (d1 if i_dir < 2 else d2)
                n_tilde = d * contact_data_friction - contact_data_normal

                jac_qvel = gs.qd_float(0.0)
                jac_qacc_smooth = gs.qd_float(0.0)

                for i_ab in range(2):
                    sign = gs.qd_float(-1.0)
                    link = link_a
                    if i_ab == 1:
                        sign = gs.qd_float(1.0)
                        link = link_b

                    while link > -1:
                        link_maybe_batch = (
                            [link, i_b]
                            if qd.static(static_rigid_sim_config.batch_links_info)
                            else link
                        )

                        for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                            i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                            cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                            cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                            t_quat = gu.qd_identity_quat()
                            t_pos = contact_data_pos - links_state.root_COM[link, i_b]
                            _, vel = gu.qd_transform_motion_by_trans_quat(
                                cdof_ang, cdot_vel, t_pos, t_quat
                            )

                            diff = sign * vel
                            jac = diff @ n_tilde

                            jac_qvel = jac_qvel + jac * dofs_state.vel[i_d, i_b]
                            jac_qacc_smooth = (
                                jac_qacc_smooth + jac * dofs_state.acc_smooth[i_d, i_b]
                            )

                        link = links_info.parent_idx[link_maybe_batch]

                imp, aref = gu.imp_aref(
                    contact_data_sol_params,
                    -contact_data_penetration,
                    jac_qvel,
                    -contact_data_penetration,
                )

                mu2 = contact_data_friction * contact_data_friction
                JMinvJT = invweight * (1.0 + mu2) * 16.0
                JMinvJT = qd.max(JMinvJT, EPS)

                Jaref = jac_qacc_smooth - aref
                efc_force = qd.max(gs.qd_float(0.0), -Jaref / JMinvJT)

                contact_force_3d = contact_force_3d + n_tilde * efc_force

                if efc_force > 0.0:
                    for i_ab in range(2):
                        sign = gs.qd_float(-1.0)
                        link = link_a
                        if i_ab == 1:
                            sign = gs.qd_float(1.0)
                            link = link_b

                        while link > -1:
                            link_maybe_batch = (
                                [link, i_b]
                                if qd.static(static_rigid_sim_config.batch_links_info)
                                else link
                            )

                            for i_d_ in range(links_info.n_dofs[link_maybe_batch]):
                                i_d = links_info.dof_end[link_maybe_batch] - 1 - i_d_

                                cdof_ang = dofs_state.cdof_ang[i_d, i_b]
                                cdot_vel = dofs_state.cdof_vel[i_d, i_b]

                                t_quat = gu.qd_identity_quat()
                                t_pos = contact_data_pos - links_state.root_COM[link, i_b]
                                _, vel = gu.qd_transform_motion_by_trans_quat(
                                    cdof_ang, cdot_vel, t_pos, t_quat
                                )

                                diff = sign * vel
                                jac = diff @ n_tilde

                                qfrc_contact[i_d, i_b] = (
                                    qfrc_contact[i_d, i_b] + jac * efc_force
                                )

                            link = links_info.parent_idx[link_maybe_batch]

            collider_state.contact_data.force[i_col, i_b] = contact_force_3d

        # Fold contact forces into acc_smooth so the iterative solver
        # for non-contact constraints sees post-contact state.
        # acc_contact = M⁻¹ · qfrc_contact (stored temporarily in constraint_state.qacc)
        func_solve_mass_batch(
            i_b,
            qfrc_contact,
            constraint_state.qacc,
            array_class.PLACEHOLDER,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=False,
        )
        for i_d in range(n_dofs):
            dofs_state.acc_smooth[i_d, i_b] = (
                dofs_state.acc_smooth[i_d, i_b] + constraint_state.qacc[i_d, i_b]
            )
            dofs_state.force[i_d, i_b] = (
                dofs_state.force[i_d, i_b] + qfrc_contact[i_d, i_b]
            )


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_comfree_set_acc_from_acc_smooth(
    dofs_state: array_class.DofsState,
    qfrc_contact: array_class.V_ANNOTATION,
    static_rigid_sim_config: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    """Set acc and qf_constraint when the iterative solver was skipped."""
    n_dofs = dofs_state.acc.shape[0]
    _B = dofs_state.acc.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dofs_state.acc[i_d, i_b] = dofs_state.acc_smooth[i_d, i_b]
        dofs_state.qf_constraint[i_d, i_b] = qfrc_contact[i_d, i_b]
        dofs_state.force[i_d, i_b] = dofs_state.qf_smooth[i_d, i_b] + qfrc_contact[i_d, i_b]
        if qd.math.isnan(dofs_state.acc_smooth[i_d, i_b]):
            errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_FORCE_NAN


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_comfree_post_solve(
    dofs_state: array_class.DofsState,
    qfrc_contact: array_class.V_ANNOTATION,
    static_rigid_sim_config: qd.template(),
):
    """Add ComFree contact forces to the iterative solver's output.

    After the iterative solver's func_update_qacc, qf_constraint and force
    only reflect non-contact constraints. This adds the contact contribution.
    Note: dofs_state.acc is already correct because the iterative solver
    started from the modified acc_smooth (which includes contact effects).
    """
    n_dofs = dofs_state.acc.shape[0]
    _B = dofs_state.acc.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dofs_state.qf_constraint[i_d, i_b] = (
            dofs_state.qf_constraint[i_d, i_b] + qfrc_contact[i_d, i_b]
        )
        dofs_state.force[i_d, i_b] = (
            dofs_state.force[i_d, i_b] + qfrc_contact[i_d, i_b]
        )


@qd.kernel(fastcache=gs.use_fastcache)
def func_comfree_update_contact_force(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    static_rigid_sim_config: qd.template(),
):
    """Write per-link 3D contact forces from ComFree results.

    The iterative solver's func_update_contact_force would have written zeros
    (since no contact constraints were added). This overwrites with the actual
    ComFree contact forces stored in contact_data.force during resolve.
    """
    n_links = links_state.contact_force.shape[0]
    _B = links_state.contact_force.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in qd.ndrange(n_links, _B):
        links_state.contact_force[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_c in range(collider_state.n_contacts[i_b]):
            force = collider_state.contact_data.force[i_c, i_b]
            link_a = collider_state.contact_data.link_a[i_c, i_b]
            link_b = collider_state.contact_data.link_b[i_c, i_b]

            links_state.contact_force[link_a, i_b] = (
                links_state.contact_force[link_a, i_b] - force
            )
            if link_b > -1:
                links_state.contact_force[link_b, i_b] = (
                    links_state.contact_force[link_b, i_b] + force
                )
