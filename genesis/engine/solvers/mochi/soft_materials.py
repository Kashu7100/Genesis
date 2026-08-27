# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Constitutive models of the deformable bodies: energy density, first Piola-Kirchhoff stress and the
positive-semidefinite projected tangent d2Psi/dF2 of linear tetrahedra, plus the Kelvin-Voigt stiffness damping.

Tangents are 9x9 matrices over the row-major flattening of F (index 3 i + j for F[i, j]). The projection clamps the
eigenvalues of the analytic eigensystem of the tangent (Smith et al. 2019): three "scaling" modes u_i v_i^T mixed by a
symmetric 3x3 matrix, and the "twist" u_j v_k^T - u_k v_j^T and "flip" u_j v_k^T + u_k v_j^T modes, whose unnormalized
matrices carry a factor 1/2 on their eigenvalue.
"""

from enum import IntEnum

import quadrants as qd

import genesis as gs

from .lie import sym_eig3


class ELASTIC_MODEL(IntEnum):
    STABLE_NEOHOOKEAN = 0
    STVK = 1
    LINEAR = 2


ELASTIC_MODEL_BY_NAME = {
    "stable_neohookean": ELASTIC_MODEL.STABLE_NEOHOOKEAN,
    "stvk": ELASTIC_MODEL.STVK,
    "linear": ELASTIC_MODEL.LINEAR,
}

# (j, k) pairs of the twist and flip modes n = 0, 1, 2.
_MODE_PAIRS = ((1, 2), (0, 2), (0, 1))


@qd.func
def func_flatten(M):
    """Row-major 9-vector of a 3x3 matrix."""
    v = qd.Vector.zero(gs.qd_float, 9)
    for i, j in qd.static(qd.ndrange(3, 3)):
        v[3 * i + j] = M[i, j]
    return v


@qd.func
def func_cofactor(F):
    """Cofactor matrix dJ/dF of F."""
    f0 = qd.Vector([F[0, 0], F[1, 0], F[2, 0]], dt=gs.qd_float)
    f1 = qd.Vector([F[0, 1], F[1, 1], F[2, 1]], dt=gs.qd_float)
    f2 = qd.Vector([F[0, 2], F[1, 2], F[2, 2]], dt=gs.qd_float)
    return qd.Matrix.cols([f1.cross(f2), f2.cross(f0), f0.cross(f1)])


@qd.func
def func_rotation_variant_svd(F):
    """SVD F = U diag(sigma) V^T with det(U) >= 0 and det(V) >= 0, so that sigma[2] < 0 exactly when F is inverted."""
    U, S, V = qd.svd(F)
    sigma = qd.Vector([S[0, 0], S[1, 1], S[2, 2]], dt=gs.qd_float)
    if U.determinant() < 0.0:
        for i in qd.static(range(3)):
            U[i, 2] = -U[i, 2]
        sigma[2] = -sigma[2]
    if V.determinant() < 0.0:
        for i in qd.static(range(3)):
            V[i, 2] = -V[i, 2]
        sigma[2] = -sigma[2]
    return U, sigma, V


@qd.func
def func_tangent_from_eigensystem(U, V, A, twist, flip, eps, project: qd.template()):
    """Assemble the 9x9 tangent from the scaling matrix A (in the u_i v_i^T basis) and the twist and flip eigenvalues,
    clamping every eigenvalue to at least eps when projecting."""
    evals, Q = sym_eig3(A, n_sweeps=8)
    C = qd.Matrix.zero(gs.qd_float, 9, 9)
    for n in qd.static(range(3)):
        lam_n = evals[n]
        if qd.static(project):
            lam_n = qd.max(lam_n, eps)
        M = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i in qd.static(range(3)):
            u_i = qd.Vector([U[0, i], U[1, i], U[2, i]], dt=gs.qd_float)
            v_i = qd.Vector([V[0, i], V[1, i], V[2, i]], dt=gs.qd_float)
            M += Q[i, n] * u_i.outer_product(v_i)
        m = func_flatten(M)
        C += lam_n * m.outer_product(m)
    for n in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[n])
        u_j = qd.Vector([U[0, j], U[1, j], U[2, j]], dt=gs.qd_float)
        u_k = qd.Vector([U[0, k], U[1, k], U[2, k]], dt=gs.qd_float)
        v_j = qd.Vector([V[0, j], V[1, j], V[2, j]], dt=gs.qd_float)
        v_k = qd.Vector([V[0, k], V[1, k], V[2, k]], dt=gs.qd_float)
        t = func_flatten(u_j.outer_product(v_k) - u_k.outer_product(v_j))
        fl = func_flatten(u_j.outer_product(v_k) + u_k.outer_product(v_j))
        lam_t = twist[n]
        lam_f = flip[n]
        if qd.static(project):
            lam_t = qd.max(lam_t, eps)
            lam_f = qd.max(lam_f, eps)
        C += (0.5 * lam_t) * t.outer_product(t)
        C += (0.5 * lam_f) * fl.outer_product(fl)
    return C


# ------------------------------------------------------------------------------------
# Stable neo-Hookean (Smith et al. 2018), in the reparametrization mu' = 4/3 mu, lambda' = lambda + 5/6 mu.
# ------------------------------------------------------------------------------------


@qd.func
def func_smith_params(mu, lam):
    mu_hat = 4.0 / 3.0 * mu
    lam_hat = lam + 5.0 / 6.0 * mu
    alpha = 1.0 + 0.75 * mu_hat / lam_hat
    return mu_hat, lam_hat, alpha


@qd.func
def func_smith_nh_energy(F, mu, lam):
    mu_hat, lam_hat, alpha = func_smith_params(mu, lam)
    Ic = F.norm_sqr()
    J = F.determinant()
    return 0.5 * mu_hat * (Ic - 3.0) + 0.5 * lam_hat * (J - alpha) ** 2 - 0.5 * mu_hat * qd.log(Ic + 1.0)


@qd.func
def func_smith_nh_stress(F, mu, lam):
    mu_hat, lam_hat, alpha = func_smith_params(mu, lam)
    Ic = F.norm_sqr()
    J = F.determinant()
    return mu_hat * (1.0 - 1.0 / (Ic + 1.0)) * F + lam_hat * (J - alpha) * func_cofactor(F)


@qd.func
def func_smith_nh_tangent(F, mu, lam, eps, project: qd.template()):
    mu_hat, lam_hat, alpha = func_smith_params(mu, lam)
    U, sigma, V = func_rotation_variant_svd(F)
    Ic = sigma.norm_sqr()
    J = sigma[0] * sigma[1] * sigma[2]
    Ic1 = Ic + 1.0
    coeff0 = mu_hat * (1.0 - 1.0 / Ic1)
    coeff1 = lam_hat * (J - alpha)
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[i])
        A[i, i] = (
            (2.0 * mu_hat * sigma[i] ** 2 - mu_hat * Ic1) / (Ic1 * Ic1)
            + lam_hat * sigma[j] ** 2 * sigma[k] ** 2
            + mu_hat
        )
    for i in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[i])
        # The off-diagonal (j, k) couples the modes other than i through sigma_i.
        A[j, k] = (2.0 * J - alpha) * lam_hat * sigma[i] + 2.0 * mu_hat * sigma[j] * sigma[k] / (Ic1 * Ic1)
        A[k, j] = A[j, k]
    twist = qd.Vector.zero(gs.qd_float, 3)
    flip = qd.Vector.zero(gs.qd_float, 3)
    for n in qd.static(range(3)):
        twist[n] = coeff1 * sigma[n] + coeff0
        flip[n] = -coeff1 * sigma[n] + coeff0
    return func_tangent_from_eigensystem(U, V, A, twist, flip, eps, project)


# ------------------------------------------------------------------------------------
# Saint Venant-Kirchhoff
# ------------------------------------------------------------------------------------


@qd.func
def func_green_strain(F):
    return 0.5 * (F.transpose() @ F - qd.Matrix.identity(gs.qd_float, 3))


@qd.func
def func_stvk_energy(F, mu, lam):
    G = func_green_strain(F)
    return mu * G.norm_sqr() + 0.5 * lam * G.trace() ** 2


@qd.func
def func_stvk_stress(F, mu, lam):
    G = func_green_strain(F)
    S = 2.0 * mu * G + lam * G.trace() * qd.Matrix.identity(gs.qd_float, 3)
    return F @ S


@qd.func
def func_stvk_tangent(F, mu, lam, eps, project: qd.template()):
    U, sigma, V = func_rotation_variant_svd(F)
    I2 = sigma.norm_sqr()
    base = -mu + 0.5 * lam * (I2 - 3.0)
    A = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        A[i, i] = base + (lam + 3.0 * mu) * sigma[i] ** 2
    for i in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[i])
        A[j, k] = lam * sigma[j] * sigma[k]
        A[k, j] = A[j, k]
    twist = qd.Vector.zero(gs.qd_float, 3)
    flip = qd.Vector.zero(gs.qd_float, 3)
    for n in qd.static(range(3)):
        j, k = qd.static(_MODE_PAIRS[n])
        twist[n] = base + mu * (sigma[j] ** 2 + sigma[k] ** 2 - sigma[j] * sigma[k])
        flip[n] = base + mu * (sigma[j] ** 2 + sigma[k] ** 2 + sigma[j] * sigma[k])
    return func_tangent_from_eigensystem(U, V, A, twist, flip, eps, project)


# ------------------------------------------------------------------------------------
# Linear elasticity (small strain)
# ------------------------------------------------------------------------------------


@qd.func
def func_linear_strain(F):
    return 0.5 * (F + F.transpose()) - qd.Matrix.identity(gs.qd_float, 3)


@qd.func
def func_linear_energy(F, mu, lam):
    e = func_linear_strain(F)
    return mu * e.norm_sqr() + 0.5 * lam * e.trace() ** 2


@qd.func
def func_linear_stress(F, mu, lam):
    e = func_linear_strain(F)
    return 2.0 * mu * e + lam * e.trace() * qd.Matrix.identity(gs.qd_float, 3)


@qd.func
def func_linear_tangent(mu, lam):
    """C_ijkl = lambda d_ij d_kl + mu (d_ik d_jl + d_il d_jk), constant and positive definite."""
    C = qd.Matrix.zero(gs.qd_float, 9, 9)
    for i, j in qd.static(qd.ndrange(3, 3)):
        C[3 * i + i, 3 * j + j] += lam
        C[3 * i + j, 3 * i + j] += mu
        C[3 * i + j, 3 * j + i] += mu
    return C


# ------------------------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------------------------


@qd.func
def func_elastic_energy(model, F, mu, lam):
    energy = gs.qd_float(0.0)
    if model == ELASTIC_MODEL.STABLE_NEOHOOKEAN:
        energy = func_smith_nh_energy(F, mu, lam)
    elif model == ELASTIC_MODEL.STVK:
        energy = func_stvk_energy(F, mu, lam)
    else:
        energy = func_linear_energy(F, mu, lam)
    return energy


@qd.func
def func_elastic_stress(model, F, mu, lam):
    P = qd.Matrix.zero(gs.qd_float, 3, 3)
    if model == ELASTIC_MODEL.STABLE_NEOHOOKEAN:
        P = func_smith_nh_stress(F, mu, lam)
    elif model == ELASTIC_MODEL.STVK:
        P = func_stvk_stress(F, mu, lam)
    else:
        P = func_linear_stress(F, mu, lam)
    return P


@qd.func
def func_elastic_tangent(model, F, mu, lam, eps, project: qd.template()):
    C = qd.Matrix.zero(gs.qd_float, 9, 9)
    if model == ELASTIC_MODEL.STABLE_NEOHOOKEAN:
        C = func_smith_nh_tangent(F, mu, lam, eps, project)
    elif model == ELASTIC_MODEL.STVK:
        C = func_stvk_tangent(F, mu, lam, eps, project)
    else:
        C = func_linear_tangent(mu, lam)
    return C


# ------------------------------------------------------------------------------------
# Kelvin-Voigt stiffness damping
# ------------------------------------------------------------------------------------


@qd.func
def func_stiffness_damping_stress(F, F_start, mu, lam, kappa):
    """Viscous second Piola-Kirchhoff stress kappa C0 : (E(F) - E(F_start)) with the rest-state isotropic tangent C0
    (Lame parameters lambda, mu) acting on the Green strain increment of the stage, and the corresponding energy
    density 1/2 dE : S."""
    dE = func_green_strain(F) - func_green_strain(F_start)
    S = kappa * (lam * dE.trace() * qd.Matrix.identity(gs.qd_float, 3) + 2.0 * mu * dE)
    energy = 0.5 * (dE * S).sum()
    return energy, S


@qd.func
def func_stiffness_damping_block(F, g_f, g_g, mu, lam, kappa):
    """Material tangent block d(F S_visc g_f)/d x_g = kappa [lambda (F g_f)(F g_g)^T + mu (g_f . g_g) F F^T
    + mu (F g_g)(F g_f)^T] (the geometric part is dropped, as mochi does by default)."""
    Fg_f = F @ g_f
    Fg_g = F @ g_g
    return kappa * (
        lam * Fg_f.outer_product(Fg_g) + mu * g_f.dot(g_g) * (F @ F.transpose()) + mu * Fg_g.outer_product(Fg_f)
    )
