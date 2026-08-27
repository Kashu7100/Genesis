# Portions of this file are derived from Meta Platforms, Inc. and affiliates' "mochi" physics library
# (mochi_core / mochi_physics), licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Signed distance and gradient of a query point against a collider geom, in the geom frame."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class
from genesis.utils.sdf import (
    sdf_func_is_outside_sdf_grid,
    sdf_func_true_grad_consistent,
    sdf_func_true_sdf,
)

from .data import COLLIDER_TYPE, MochiGeomsInfo


@qd.func
def query_collider(
    i_g,
    pos_geom,
    geoms_info: array_class.GeomsInfo,
    mochi_geoms_info: MochiGeomsInfo,
    sdf_info: array_class.SDFInfo,
    mochi_config: qd.template(),
):
    """Signed distance of a point (geom frame) to the collider geom and the gradient of the distance field.

    Returns whether the query is inside the region where the field is defined (a grid collider only answers inside
    its grid; outside, the point is far from the surface by construction of the grid padding), the signed distance and
    its gradient in the geom frame. Analytic colliders return a unit gradient; the grid gradient is the exact gradient
    of the trilinear interpolant, whose norm is close to but not exactly one.
    """
    is_valid = True
    sd = gs.qd_float(0.0)
    grad = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    collider_type = mochi_geoms_info.collider_type[i_g]
    geom_data = geoms_info.data[i_g]

    if collider_type == COLLIDER_TYPE.PLANE:
        normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        sd = pos_geom.dot(normal)
        grad = normal
    elif collider_type == COLLIDER_TYPE.SPHERE:
        norm = pos_geom.norm()
        sd = norm - geom_data[0]
        if norm > 0.0:
            grad = pos_geom / norm
    elif collider_type == COLLIDER_TYPE.BOX:
        half = 0.5 * gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        q = qd.abs(pos_geom) - half
        if q.max() <= 0.0:
            # Inside: the distance to the closest face, along the axis of that face.
            i_max = 0
            if q[1] > q[i_max]:
                i_max = 1
            if q[2] > q[i_max]:
                i_max = 2
            sd = q[i_max]
            grad = qd.Vector.zero(gs.qd_float, 3)
            grad[i_max] = 1.0 if pos_geom[i_max] >= 0.0 else -1.0
        else:
            q_pos = qd.max(q, 0.0)
            sd = q_pos.norm()
            grad = q_pos / sd
            for k in qd.static(range(3)):
                if pos_geom[k] < 0.0:
                    grad[k] = -grad[k]
    else:
        if qd.static(mochi_config.has_grid_colliders):
            pos_sdf = gu.qd_transform_by_T(pos_geom, sdf_info.geoms_info.T_mesh_to_sdf[i_g])
            if sdf_func_is_outside_sdf_grid(i_g, pos_sdf, sdf_info):
                is_valid = False
            else:
                sd = sdf_func_true_sdf(i_g, pos_sdf, sdf_info)
                grad = sdf_func_true_grad_consistent(i_g, pos_sdf, sdf_info)
        else:
            is_valid = False

    return is_valid, sd, grad
