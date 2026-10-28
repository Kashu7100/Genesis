"""Persistent friction patches of the Steel solver: friction regions, their anchors and the history of the anchors.

Normal contact rows are never reduced. The contacts of a body pair are grouped into regions of compatible contacts:
same friction coefficients, nearly aligned normals (about six degrees), nearby contact planes and bounding spheres of
the geoms touching each other on each body, a conservative test across convex seams. Each region selects up to two
anchors carrying its friction, each with an equal share of the total normal impulse of the region. The anchors sit at
the extremes of the footprint along its principal axis, with the transverse offset and the height of the members near
each end, which avoids a diagonal friction couple on symmetric narrow footprints. An isotropic footprint keeps a
canonical farthest pair. Length tolerances scale with the bounding radius of the smaller body.

The tangential displacement and the friction impulse of an anchor persist across steps, the anchors following the
current region. The displacement accumulates the rigid motion of the bodies at the anchor, evaluated with the
logarithm of the pose increment rather than the chord of a rotating material point, so rolling adds no slip. The
history is matched by body pair and geometric compatibility, without contact identities. Unloaded regions and
saturated friction slipping against its impulse release their history, saturation alone does not.

A step runs func_steel_prepare_patches and func_steel_build_patches before the row allocation, which gives
friction rows to the anchors alone, func_steel_link_patch_rows after the row setup, which chains the normal rows of
every region into a ring and shares the friction coefficient among its anchors, func_steel_seed_patch_impulses with
contact warm start, then after the integration func_steel_finish_patch_impulses and func_steel_store_patches.
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu

from .pgs import func_pooled_normal_impulse
from .rows import func_contact_tangent_basis


@qd.func
def func_world_point(i_l: int, i_b: int, point: qd.types.vector(3), links_pos: qd.Tensor, links_quat: qd.Tensor):
    """Return the world position of a point given in the frame of a link, the point itself for static geometry."""
    result = point
    if i_l >= 0:
        result = links_pos[i_l, i_b] + gu.qd_transform_by_quat(point, links_quat[i_l, i_b])
    return result


@qd.func
def func_local_point(i_l: int, i_b: int, point: qd.types.vector(3), links_pos: qd.Tensor, links_quat: qd.Tensor):
    """Return a world point in the frame of a link, the point itself for static geometry."""
    result = point
    if i_l >= 0:
        result = gu.qd_inv_transform_by_quat(point - links_pos[i_l, i_b], links_quat[i_l, i_b])
    return result


@qd.func
def func_rotate(i_l: int, i_b: int, vector: qd.types.vector(3), links_quat: qd.Tensor):
    """Return a vector given in the axes of a link in world axes, the vector itself for static geometry."""
    result = vector
    if i_l >= 0:
        result = gu.qd_transform_by_quat(vector, links_quat[i_l, i_b])
    return result


@qd.func
def func_inv_rotate(i_l: int, i_b: int, vector: qd.types.vector(3), links_quat: qd.Tensor):
    """Return a world vector in the axes of a link, the vector itself for static geometry."""
    result = vector
    if i_l >= 0:
        result = gu.qd_inv_transform_by_quat(vector, links_quat[i_l, i_b])
    return result


@qd.func
def func_normalize_or_zero(v):
    """Return a vector normalized, or zero for a zero vector."""
    norm = v.norm()
    result = v * 0.0
    if norm > 0.0:
        result = v / norm
    return result


@qd.func
def func_key_less(key, other):
    """Return whether a body-pair key precedes another one."""
    return key[0] < other[0] or (key[0] == other[0] and key[1] < other[1])


@qd.func
def func_key_equal(key, other):
    """Return whether two body-pair keys are equal."""
    return key[0] == other[0] and key[1] == other[1]


@qd.func
def func_shapes_adjacent(
    i_g: int, j_g: int, steel_patch_geometry: array_class.SteelPatchGeometry, steel_info: array_class.SteelInfo
):
    """Return whether the bounding spheres of two geoms of one link overlap, a geom being adjacent to itself."""
    is_adjacent = i_g == j_g
    if not is_adjacent and i_g >= 0 and j_g >= 0:
        radius = steel_patch_geometry.geoms_radius[i_g] + steel_patch_geometry.geoms_radius[j_g]
        delta = steel_patch_geometry.geoms_offset[i_g] - steel_patch_geometry.geoms_offset[j_g]
        is_adjacent = delta.norm_sqr() <= (1.0 + steel_info.adjacency_slack[None]) * radius * radius
    return is_adjacent


@qd.func
def func_geometry_adjacent(
    i_ga: int,
    i_gb: int,
    j_ga: int,
    j_gb: int,
    steel_patch_geometry: array_class.SteelPatchGeometry,
    steel_info: array_class.SteelInfo,
):
    """Return whether two contacts of a body pair touch adjacent geoms on both bodies, convex seams included."""
    return func_shapes_adjacent(i_ga, j_ga, steel_patch_geometry, steel_info) and func_shapes_adjacent(
        i_gb, j_gb, steel_patch_geometry, steel_info
    )


@qd.func
def func_pose_motion(
    i_l: int,
    i_b: int,
    point: qd.types.vector(3),
    steel_state: array_class.SteelState,
    steel_info: array_class.SteelInfo,
):
    """Return the displacement of a world point under the pose increment of a link over the last step.

    The increment is evaluated as its SE(3) logarithm about the midpoint of the two origins, which avoids subtracting
    large moments about the world origin and is exact for a constant screw motion: a fixed pivot has no displacement
    and rolling does not mistake the chord of a material point for slip. Static geometry (-1) does not move.
    """
    motion = qd.Vector.zero(gs.qd_float, 3)
    if i_l >= 0:
        x0 = steel_state.links_pos_prev[i_l, i_b]
        x1 = steel_state.links_pos[i_l, i_b]
        r0 = steel_state.links_quat_prev[i_l, i_b]
        r1 = steel_state.links_quat[i_l, i_b]
        # An unchanged pose has no motion. The quaternion product below is not exactly the identity in floating point,
        # and its residual times the lever arm would accumulate into the displacement of a resting body every step,
        # hence the exact comparison.
        is_unchanged = (
            x0[0] == x1[0]
            and x0[1] == x1[1]
            and x0[2] == x1[2]
            and r0[0] == r1[0]
            and r0[1] == r1[1]
            and r0[2] == r1[2]
            and r0[3] == r1[3]
        )
        if not is_unchanged:
            rotation = gu.qd_quat_mul(r1, gu.qd_inv_quat(r0))
            if rotation[0] < 0.0:
                rotation = -rotation
            vector = qd.Vector([rotation[1], rotation[2], rotation[3]], dt=gs.qd_float)
            sine = vector.norm()
            angular = 2.0 * vector
            translation = x1 - x0
            linear = translation
            if sine > steel_info.pose_motion_sine[None]:
                half_angle = qd.atan2(sine, rotation[0])
                axis = vector / sine
                angular = (2.0 * half_angle) * axis
                scale = half_angle * rotation[0] / sine
                linear = scale * translation + (1.0 - scale) * axis.dot(translation) * axis
            motion = linear + angular.cross(point - 0.5 * (x0 + x1))
    return motion


@qd.func
def func_carried_displacement(
    i_p: int,
    i_b: int,
    normal: qd.types.vector(3),
    steel_state: array_class.SteelState,
    steel_patch_history: array_class.SteelPatchHistory,
):
    """Return the stored displacement of a history entry in world axes, turned into the plane of the current normal.

    The parallel transport by the minimal rotation from the stored normal to the current one preserves the length of
    the displacement, where a projection would dissipate the history at every small rocking motion.
    """
    i_la = steel_patch_history.body_a[i_p, i_b]
    error = func_rotate(i_la, i_b, steel_patch_history.displacement[i_p, i_b], steel_state.links_quat)
    old_normal = func_rotate(i_la, i_b, steel_patch_history.normal[i_p, i_b], steel_state.links_quat)
    cosine = old_normal.dot(normal)
    if cosine > 0.0:
        axis = old_normal.cross(normal)
        cross_error = axis.cross(error)
        error = error + cross_error + axis.cross(cross_error) / (1.0 + cosine)
    return error


@qd.func
def func_patch_compatible(
    i_seed: int, i_c: int, i_b: int, steel_patch_frame: array_class.SteelPatchFrame, steel_info: array_class.SteelInfo
):
    """Return whether a contact may join the region of a seed contact.

    It needs a normal aligned with the seed's, the same friction coefficients, a nearby contact plane and a bounded
    distance to the seed.
    The friction coefficients compare exactly: they are copies of the coefficients of the geoms, and a region shares
    one Coulomb budget, which needs one coefficient.
    """
    d = steel_patch_frame.center[i_c, i_b] - steel_patch_frame.center[i_seed, i_b]
    r = steel_patch_frame.radius[i_seed, i_b]
    mu_seed = steel_patch_frame.mu[i_seed, i_b]
    mu_c = steel_patch_frame.mu[i_c, i_b]
    normal_seed = steel_patch_frame.normal[i_seed, i_b]
    return (
        normal_seed.dot(steel_patch_frame.normal[i_c, i_b]) >= steel_info.patch_normal_cosine_min[None]
        and mu_seed[0] == mu_c[0]
        and mu_seed[1] == mu_c[1]
        and qd.abs(d.dot(normal_seed)) <= steel_info.region_plane_distance[None] * r
        and d.norm_sqr() <= steel_info.region_extent_sq[None] * r * r
    )


@qd.func
def func_center_lex_less(p: qd.types.vector(3), q: qd.types.vector(3)):
    """Return whether a point precedes another one in the lexicographic order of their coordinates."""
    return p[0] < q[0] or (p[0] == q[0] and (p[1] < q[1] or (p[1] == q[1] and p[2] < q[2])))


@qd.func
def func_steel_prepare_patches(
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_patch_frame: array_class.SteelPatchFrame,
    steel_patch_geometry: array_class.SteelPatchGeometry,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Record the geometry of every contact in the frame of its body pair, then sort the contacts by body-pair key.

    The key of a body pair is (1 + link) for a link and (n_links + 1 + geom) for a static geom, so every static geom is
    a body of its own sorting after the links, and side a is the side of smaller key. A contact is eligible for friction
    rows when its gap does not exceed contact_gap_limit and friction_gap_limit. The counting sort is stable, so the
    contacts of one pair follow in ascending contact order, which fixes the seeds and every tie of the regions.
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_state.q.shape[1]
    n_links = steel_state.links_pos.shape[0]
    n_keys = steel_patch_frame.sort_count.shape[0]
    contact_gap_limit = steel_info.contact_gap_limit[None]
    friction_gap_limit = steel_info.friction_gap_limit[None]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        # Every reader of the patch fields indexes them by a contact of the step, so the slots past the contact count
        # are left as they are: resetting the whole capacity would write it all at every step.
        if i_c < steel_contacts.n[i_b]:
            steel_patch_frame.indices[i_c, i_b] = i_c
            steel_patch_frame.owner[i_c, i_b] = -1
            steel_patch_frame.valid[i_c, i_b] = 0
            steel_patch_frame.source[i_c, i_b] = -1
            steel_patch_frame.tangent_impulse[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            steel_contacts.patch_weight[i_c, i_b] = 0.0
            steel_contacts.patch_next[i_c, i_b] = -1
            steel_contacts.patch_phi[i_c, i_b] = qd.Vector.zero(gs.qd_float, 2)
            i_ga = steel_contacts.shape0[i_c, i_b]
            i_gb = steel_contacts.shape1[i_c, i_b]
            i_la = steel_contacts.body0[i_c, i_b]
            i_lb = steel_contacts.body1[i_c, i_b]
            key_a = i_la + 1 if i_la >= 0 else n_links + i_ga + 1
            key_b = i_lb + 1 if i_lb >= 0 else n_links + i_gb + 1
            n_c = -steel_contacts.normal[i_c, i_b]
            p_a = func_world_point(
                i_la, i_b, steel_contacts.point0[i_c, i_b], steel_state.links_pos, steel_state.links_quat
            )
            p_b = func_world_point(
                i_lb, i_b, steel_contacts.point1[i_c, i_b], steel_state.links_pos, steel_state.links_quat
            )
            gap = n_c.dot(p_a - p_b)
            steel_patch_frame.eligible[i_c, i_b] = 1 if gap <= contact_gap_limit and gap <= friction_gap_limit else 0
            is_flipped = key_a > key_b
            if is_flipped:
                i_la, i_lb = i_lb, i_la
                key_a, key_b = key_b, key_a
                i_ga, i_gb = i_gb, i_ga
                n_c = -n_c
                p_a, p_b = p_b, p_a
            steel_patch_frame.key[i_c, i_b] = qd.Vector([key_a, key_b], dt=gs.qd_int)
            steel_patch_frame.body_a[i_c, i_b] = i_la
            steel_patch_frame.body_b[i_c, i_b] = i_lb
            steel_patch_frame.shape_a[i_c, i_b] = i_ga
            steel_patch_frame.shape_b[i_c, i_b] = i_gb
            steel_patch_frame.flipped[i_c, i_b] = 1 if is_flipped else 0
            steel_patch_frame.surface_a[i_c, i_b] = func_local_point(
                i_la, i_b, p_a, steel_state.links_pos, steel_state.links_quat
            )
            steel_patch_frame.surface_b[i_c, i_b] = func_local_point(
                i_lb, i_b, p_b, steel_state.links_pos, steel_state.links_quat
            )
            steel_patch_frame.center[i_c, i_b] = 0.5 * (p_a + p_b)
            steel_patch_frame.normal[i_c, i_b] = n_c
            steel_patch_frame.mu[i_c, i_b] = qd.Vector(
                [steel_info.geoms_mu[i_ga, i_b], steel_info.geoms_mu[i_gb, i_b]], dt=gs.qd_float
            )
            r = gs.qd_float(steel_info.radius_ceiling[None])
            if i_la >= 0:
                r = qd.min(r, steel_patch_geometry.links_radius[i_la])
            if i_lb >= 0:
                r = qd.min(r, steel_patch_geometry.links_radius[i_lb])
            steel_patch_frame.radius[i_c, i_b] = qd.max(r, steel_info.radius_floor[None])

    # Stable least-significant-digit counting sort of the keys: by side b, then by side a.
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        n = steel_contacts.n[i_b]
        for i_digit_ in qd.static(range(2)):
            i_digit = qd.static(1 - i_digit_)
            for k in range(n_keys):
                steel_patch_frame.sort_count[k, i_b] = 0
            for i_s in range(n):
                i_c = i_s if qd.static(i_digit_ == 0) else steel_patch_frame.sort_order[i_s, i_b]
                k = steel_patch_frame.key[i_c, i_b][i_digit]
                steel_patch_frame.sort_count[k, i_b] = steel_patch_frame.sort_count[k, i_b] + 1
            offset = 0
            for k in range(n_keys):
                count = steel_patch_frame.sort_count[k, i_b]
                steel_patch_frame.sort_count[k, i_b] = offset
                offset = offset + count
            for i_s in range(n):
                i_c = i_s if qd.static(i_digit_ == 0) else steel_patch_frame.sort_order[i_s, i_b]
                k = steel_patch_frame.key[i_c, i_b][i_digit]
                i_dst = steel_patch_frame.sort_count[k, i_b]
                steel_patch_frame.sort_count[k, i_b] = i_dst + 1
                if qd.static(i_digit_ == 0):
                    steel_patch_frame.sort_order[i_dst, i_b] = i_c
                else:
                    steel_patch_frame.indices[i_dst, i_b] = i_c


@qd.func
def func_carry_frozen_pair(
    i_start: int,
    i_stop: int,
    i_prev_start: int,
    i_prev_stop: int,
    i_b: int,
    steel_state: array_class.SteelState,
    steel_patch_frame: array_class.SteelPatchFrame,
    steel_patch_history: array_class.SteelPatchHistory,
    steel_sleep: array_class.SteelSleep,
):
    """Carry the history of a body pair with a frozen link unchanged, returning whether the pair was carried.

    The poses and the contacts of a frozen pair repeat and it has no rows, so its anchors, regions and carried
    displacements stay as they were. The pair is carried when it has as many contacts as its history, with the same
    geoms in sorted order, the normal of an anchor going back to world axes with the pose of its body a.
    """
    i_c_start = steel_patch_frame.indices[i_start, i_b]
    i_la = steel_patch_frame.body_a[i_c_start, i_b]
    i_lb = steel_patch_frame.body_b[i_c_start, i_b]
    is_frozen = False
    if i_la >= 0:
        is_frozen = steel_sleep.links_is_frozen[i_la, i_b] != 0
    if i_lb >= 0:
        is_frozen = is_frozen or steel_sleep.links_is_frozen[i_lb, i_b] != 0
    is_carried = is_frozen and i_prev_stop - i_prev_start == i_stop - i_start
    if is_carried:
        for k in range(i_stop - i_start):
            i_c = steel_patch_frame.indices[i_start + k, i_b]
            i_p = steel_patch_history.indices[i_prev_start + k, i_b]
            if (
                steel_patch_frame.shape_a[i_c, i_b] != steel_patch_history.shape_a[i_p, i_b]
                or steel_patch_frame.shape_b[i_c, i_b] != steel_patch_history.shape_b[i_p, i_b]
            ):
                is_carried = False
    if is_carried:
        for k in range(i_stop - i_start):
            i_c = steel_patch_frame.indices[i_start + k, i_b]
            i_p = steel_patch_history.indices[i_prev_start + k, i_b]
            steel_patch_frame.source[i_c, i_b] = i_p
            steel_patch_frame.valid[i_c, i_b] = steel_patch_history.valid[i_p, i_b]
            steel_patch_frame.displacement[i_c, i_b] = steel_patch_history.displacement[i_p, i_b]
            steel_patch_frame.tangent_impulse[i_c, i_b] = steel_patch_history.tangent_impulse[i_p, i_b]
            steel_patch_frame.anchor_a[i_c, i_b] = steel_patch_history.anchor_a[i_p, i_b]
            steel_patch_frame.anchor_b[i_c, i_b] = steel_patch_history.anchor_b[i_p, i_b]
            if steel_patch_history.valid[i_p, i_b] != 0:
                steel_patch_frame.surface_a[i_c, i_b] = steel_patch_history.surface_a[i_p, i_b]
                steel_patch_frame.surface_b[i_c, i_b] = steel_patch_history.surface_b[i_p, i_b]
                steel_patch_frame.normal[i_c, i_b] = func_rotate(
                    i_la, i_b, steel_patch_history.normal[i_p, i_b], steel_state.links_quat
                )
            i_owner = steel_patch_history.owner[i_p, i_b]
            for j in range(i_stop - i_start):
                if steel_patch_history.indices[i_prev_start + j, i_b] == i_owner:
                    steel_patch_frame.owner[i_c, i_b] = steel_patch_frame.indices[i_start + j, i_b]
    return is_carried


@qd.func
def func_build_pair_patches(
    i_start: int,
    i_b: int,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_patch_frame: array_class.SteelPatchFrame,
    steel_patch_history: array_class.SteelPatchHistory,
    steel_sleep: array_class.SteelSleep,
    steel_patch_geometry: array_class.SteelPatchGeometry,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Build the friction regions of one body pair and continue the history of their anchors.

    The sorted contacts of the pair start at i_start. Each unassigned contact in sorted order seeds a region, which
    floods through the later compatible contacts whose geoms touch those of a member already in the region,
    compatibility being tested against the seed and adjacency against the member being expanded, so a region grows
    along chains of touching convex pieces. The members of a
    region form a ring through patch_next in acceptance order. The first anchor is the lexicographically smallest
    eligible member, the second the eligible member farthest from it, both refined along the principal axis of an
    anisotropic footprint. A region without eligible member (gap-filtered friction) keeps carrying the history of its
    members without friction rows and never starts a new one.

    Each anchor continues the nearest valid unused history entry of the pair with the same friction coefficients, an
    aligned normal, nearby anchor points, a supported gap, a bounded tangential slip and adjacent geometry, carrying
    its displacement (twist included when its region had two anchors) and its tangent impulse.
    """
    n = steel_contacts.n[i_b]
    i_c_start = steel_patch_frame.indices[i_start, i_b]
    key = steel_patch_frame.key[i_c_start, i_b]
    i_stop = i_start + 1
    for i_s in range(i_start + 1, n):
        i_c = steel_patch_frame.indices[i_s, i_b]
        if not func_key_equal(steel_patch_frame.key[i_c, i_b], key):
            break
        i_stop = i_s + 1

    # Lower bound of the key in the sorted history, then its valid entries in sorted order.
    n_prev = steel_patch_history.n[i_b]
    lo = 0
    hi = n_prev
    for _ in range(steel_info.binary_search_iterations[None]):
        if lo < hi:
            mid = (lo + hi) // 2
            i_p = steel_patch_history.indices[mid, i_b]
            if func_key_less(steel_patch_history.key[i_p, i_b], key):
                lo = mid + 1
            else:
                hi = mid
    i_prev_start = lo
    i_prev_stop = lo
    i_prev_anchor_stop = lo
    for i_s in range(i_prev_start, n_prev):
        i_p = steel_patch_history.indices[i_s, i_b]
        if not func_key_equal(steel_patch_history.key[i_p, i_b], key):
            break
        steel_patch_history.used[i_p, i_b] = 0
        if steel_patch_history.valid[i_p, i_b] != 0:
            steel_patch_history.members[i_prev_anchor_stop, i_b] = i_p
            i_prev_anchor_stop = i_prev_anchor_stop + 1
        i_prev_stop = i_s + 1
    # With row-free sleeping, the pairs of the frozen links carry their history unchanged instead of being rebuilt.
    is_carried = False
    if qd.static(steel_config.sleep_skip_constraints):
        is_carried = func_carry_frozen_pair(
            i_start,
            i_stop,
            i_prev_start,
            i_prev_stop,
            i_b,
            steel_state,
            steel_patch_frame,
            steel_patch_history,
            steel_sleep,
        )
    if not is_carried:
        member_stop = i_start
        remaining = i_stop - i_start
        for i_index in range(i_start, i_stop):
            i_seed = steel_patch_frame.indices[i_index, i_b]
            if steel_patch_frame.owner[i_seed, i_b] < 0:
                steel_patch_frame.owner[i_seed, i_b] = i_seed
                remaining = remaining - 1
                i_tail = i_seed
                i_first = i_seed if steel_patch_frame.eligible[i_seed, i_b] != 0 else -1
                i_cursor = i_seed
                expanded_a = -2
                expanded_b = -2
                # The ring under construction is the work list: every member is expanded once, in acceptance order.
                for _ in range(i_stop - i_start):
                    if i_cursor < 0 or remaining <= 0:
                        break
                    i_ga = steel_patch_frame.shape_a[i_cursor, i_b]
                    i_gb = steel_patch_frame.shape_b[i_cursor, i_b]
                    if i_ga != expanded_a or i_gb != expanded_b:
                        for i_s in range(i_index + 1, i_stop):
                            i_c = steel_patch_frame.indices[i_s, i_b]
                            if (
                                steel_patch_frame.owner[i_c, i_b] < 0
                                and func_patch_compatible(i_seed, i_c, i_b, steel_patch_frame, steel_info)
                                and func_geometry_adjacent(
                                    i_ga,
                                    i_gb,
                                    steel_patch_frame.shape_a[i_c, i_b],
                                    steel_patch_frame.shape_b[i_c, i_b],
                                    steel_patch_geometry,
                                    steel_info,
                                )
                            ):
                                steel_patch_frame.owner[i_c, i_b] = i_seed
                                remaining = remaining - 1
                                steel_contacts.patch_next[i_tail, i_b] = i_c
                                i_tail = i_c
                                # The first anchor is the lexicographically smallest eligible member. The branches stay
                                # apart so the center of the current first anchor is read only once it exists.
                                if steel_patch_frame.eligible[i_c, i_b] != 0:
                                    if i_first < 0:
                                        i_first = i_c
                                    elif func_center_lex_less(
                                        steel_patch_frame.center[i_c, i_b], steel_patch_frame.center[i_first, i_b]
                                    ):
                                        i_first = i_c
                        expanded_a = i_ga
                        expanded_b = i_gb
                    i_cursor = steel_contacts.patch_next[i_cursor, i_b]
                steel_contacts.patch_next[i_tail, i_b] = i_seed

                is_carry_only = i_first < 0
                if is_carry_only:
                    i_first = i_seed
                i_second = i_first
                separation = gs.qd_float(0.0)
                member_start = member_stop
                # Members in sorted order, which fixes the footprint sums and their ties.
                for i_s in range(i_index, i_stop):
                    i_c = steel_patch_frame.indices[i_s, i_b]
                    if steel_patch_frame.owner[i_c, i_b] == i_seed:
                        steel_patch_frame.members[member_stop, i_b] = i_c
                        member_stop = member_stop + 1
                        if is_carry_only or steel_patch_frame.eligible[i_c, i_b] != 0:
                            distance = (
                                steel_patch_frame.center[i_c, i_b] - steel_patch_frame.center[i_first, i_b]
                            ).norm_sqr()
                            if distance > separation:
                                separation = distance
                                i_second = i_c
                r_seed = steel_patch_frame.radius[i_seed, i_b]
                n_anchors = 2 if separation > steel_info.two_anchor_separation[None] * r_seed * r_seed else 1
                location0 = steel_patch_frame.center[i_first, i_b]
                location1 = steel_patch_frame.center[i_second, i_b]
                if n_anchors == 2 and not is_carry_only:
                    # The moments of the footprint accumulate in double precision, so that the roundoff of the member
                    # order cannot invent a principal axis of an isotropic footprint.
                    origin = location0
                    t0, t1 = func_contact_tangent_basis(steel_patch_frame.normal[i_seed, i_b])
                    sum_x = qd.cast(0.0, qd.f64)
                    sum_y = qd.cast(0.0, qd.f64)
                    sum_xx = qd.cast(0.0, qd.f64)
                    sum_xy = qd.cast(0.0, qd.f64)
                    sum_yy = qd.cast(0.0, qd.f64)
                    sum_center = qd.Vector.zero(qd.f64, 3)
                    n_members = 0
                    for i_s in range(member_start, member_stop):
                        i_c = steel_patch_frame.members[i_s, i_b]
                        if steel_patch_frame.eligible[i_c, i_b] != 0:
                            point = steel_patch_frame.center[i_c, i_b]
                            offset = point - origin
                            x = qd.cast(offset.dot(t0), qd.f64)
                            y = qd.cast(offset.dot(t1), qd.f64)
                            sum_x = sum_x + x
                            sum_y = sum_y + y
                            sum_xx = sum_xx + x * x
                            sum_xy = sum_xy + x * y
                            sum_yy = sum_yy + y * y
                            sum_center = sum_center + qd.cast(point, qd.f64)
                            n_members = n_members + 1
                    member_count = qd.cast(n_members, qd.f64)
                    xx = qd.cast(sum_xx - sum_x * sum_x / member_count, gs.qd_float)
                    xy = qd.cast(sum_xy - sum_x * sum_y / member_count, gs.qd_float)
                    yy = qd.cast(sum_yy - sum_y * sum_y / member_count, gs.qd_float)
                    spread = qd.sqrt((xx - yy) * (xx - yy) + 4.0 * xy * xy)
                    # An isotropic footprint has no distinguished axis and keeps its canonical farthest pair.
                    if spread > steel_info.isotropy_tolerance[None] * (xx + yy):
                        largest = 0.5 * (xx + yy + spread)
                        axis2 = qd.Vector([xy, largest - xx], dt=gs.qd_float)
                        if xx >= yy:
                            axis2 = qd.Vector([largest - yy, xy], dt=gs.qd_float)
                        axis2 = func_normalize_or_zero(axis2)
                        axis = axis2[0] * t0 + axis2[1] * t1
                        low = gs.qd_float(steel_info.extreme_projection[None])
                        high = gs.qd_float(-steel_info.extreme_projection[None])
                        for i_s in range(member_start, member_stop):
                            i_c = steel_patch_frame.members[i_s, i_b]
                            if steel_patch_frame.eligible[i_c, i_b] != 0:
                                projection = (steel_patch_frame.center[i_c, i_b] - origin).dot(axis)
                                if projection < low:
                                    low = projection
                                    i_first = i_c
                                if projection > high:
                                    high = projection
                                    i_second = i_c
                        centroid = qd.cast(sum_center / member_count, gs.qd_float)
                        location0 = centroid + (steel_patch_frame.center[i_first, i_b] - centroid).dot(axis) * axis
                        location1 = centroid + (steel_patch_frame.center[i_second, i_b] - centroid).dot(axis) * axis
                        normal = steel_patch_frame.normal[i_seed, i_b]
                        across = normal.cross(axis)
                        across = func_normalize_or_zero(across - axis * across.dot(axis))
                        sum0 = qd.Vector.zero(qd.f64, 3)
                        sum1 = qd.Vector.zero(qd.f64, 3)
                        count0 = 0
                        count1 = 0
                        edge_tolerance = qd.max(
                            steel_info.support_edge_floor[None] * r_seed,
                            steel_info.support_edge_fraction[None] * (high - low),
                        )
                        for i_s in range(member_start, member_stop):
                            i_c = steel_patch_frame.members[i_s, i_b]
                            if steel_patch_frame.eligible[i_c, i_b] != 0:
                                point = steel_patch_frame.center[i_c, i_b]
                                projection = (point - origin).dot(axis)
                                if projection <= low + edge_tolerance:
                                    sum0 = sum0 + qd.cast(point, qd.f64)
                                    count0 = count0 + 1
                                if projection >= high - edge_tolerance:
                                    sum1 = sum1 + qd.cast(point, qd.f64)
                                    count1 = count1 + 1
                        end0 = qd.cast(sum0 / qd.cast(count0, qd.f64), gs.qd_float)
                        end1 = qd.cast(sum1 / qd.cast(count1, qd.f64), gs.qd_float)
                        location0 = location0 + (end0 - centroid).dot(across) * across
                        location1 = location1 + (end1 - centroid).dot(across) * across
                        location0 = location0 + (end0 - centroid).dot(normal) * normal
                        location1 = location1 + (end1 - centroid).dot(normal) * normal

                for i_anchor in range(n_anchors):
                    i_c = i_first if i_anchor == 0 else i_second
                    location = location0 if i_anchor == 0 else location1
                    i_la = steel_patch_frame.body_a[i_c, i_b]
                    i_lb = steel_patch_frame.body_b[i_c, i_b]
                    n_c = steel_patch_frame.normal[i_c, i_b]
                    r = steel_patch_frame.radius[i_c, i_b]
                    mu = steel_patch_frame.mu[i_c, i_b]
                    surface_a = steel_patch_frame.surface_a[i_c, i_b]
                    surface_b = steel_patch_frame.surface_b[i_c, i_b]
                    current_gap = (
                        func_world_point(i_la, i_b, surface_a, steel_state.links_pos, steel_state.links_quat)
                        - func_world_point(i_lb, i_b, surface_b, steel_state.links_pos, steel_state.links_quat)
                    ).dot(n_c)
                    # Genesis geoms detect no contact beyond touching: the support gap envelope is the current gap.
                    gap_limit = qd.max(current_gap, 0.0) + steel_info.history_gap_slack[None] * r
                    i_chosen = -1
                    nearest = gs.qd_float(steel_info.extreme_projection[None])
                    motion = qd.Vector.zero(gs.qd_float, 3)
                    if i_prev_start < i_prev_stop:
                        motion = func_pose_motion(i_la, i_b, location, steel_state, steel_info) - func_pose_motion(
                            i_lb, i_b, location, steel_state, steel_info
                        )
                    for i_s in range(i_prev_start, i_prev_anchor_stop):
                        i_p = steel_patch_history.members[i_s, i_b]
                        mu_p = steel_patch_history.mu[i_p, i_b]
                        # Exact coefficients, as for the regions (see func_patch_compatible).
                        is_candidate = (
                            steel_patch_history.valid[i_p, i_b] != 0
                            and steel_patch_history.used[i_p, i_b] == 0
                            and mu_p[0] == mu[0]
                            and mu_p[1] == mu[1]
                        )
                        if is_candidate:
                            old_normal = func_rotate(
                                i_la, i_b, steel_patch_history.normal[i_p, i_b], steel_state.links_quat
                            )
                            is_candidate = old_normal.dot(n_c) >= steel_info.patch_normal_cosine_min[None]
                        distance = gs.qd_float(0.0)
                        if is_candidate:
                            p_a = func_world_point(
                                i_la,
                                i_b,
                                steel_patch_history.anchor_a[i_p, i_b],
                                steel_state.links_pos,
                                steel_state.links_quat,
                            )
                            p_b = func_world_point(
                                i_lb,
                                i_b,
                                steel_patch_history.anchor_b[i_p, i_b],
                                steel_state.links_pos,
                                steel_state.links_quat,
                            )
                            offset = 0.5 * (p_a + p_b) - location
                            distance = offset.norm_sqr()
                            is_candidate = (
                                distance < nearest
                                and distance <= steel_info.history_distance_sq[None] * r * r
                                and qd.abs((p_a - p_b).dot(n_c)) <= steel_info.history_separation[None] * r
                                and qd.abs(offset.dot(n_c)) <= steel_info.history_height[None] * r
                            )
                        if is_candidate:
                            # The surface witnesses keep the penetration during decompression and reject rocking beyond
                            # the envelope of fresh contacts.
                            support_gap = (
                                func_world_point(
                                    i_la,
                                    i_b,
                                    steel_patch_history.surface_a[i_p, i_b],
                                    steel_state.links_pos,
                                    steel_state.links_quat,
                                )
                                - func_world_point(
                                    i_lb,
                                    i_b,
                                    steel_patch_history.surface_b[i_p, i_b],
                                    steel_state.links_pos,
                                    steel_state.links_quat,
                                )
                            ).dot(n_c)
                            is_candidate = support_gap <= gap_limit
                        if is_candidate:
                            error = func_carried_displacement(i_p, i_b, n_c, steel_state, steel_patch_history) + motion
                            tangent_delta = error - n_c * error.dot(n_c)
                            is_candidate = tangent_delta.norm_sqr() <= steel_info.history_slip_sq[None] * r * r
                        if is_candidate:
                            i_ga_p = steel_patch_history.shape_a[i_p, i_b]
                            i_gb_p = steel_patch_history.shape_b[i_p, i_b]
                            is_connected = func_geometry_adjacent(
                                i_ga_p,
                                i_gb_p,
                                steel_patch_frame.shape_a[i_c, i_b],
                                steel_patch_frame.shape_b[i_c, i_b],
                                steel_patch_geometry,
                                steel_info,
                            )
                            if not is_connected:
                                for i_m in range(member_start, member_stop):
                                    i_member = steel_patch_frame.members[i_m, i_b]
                                    if func_geometry_adjacent(
                                        i_ga_p,
                                        i_gb_p,
                                        steel_patch_frame.shape_a[i_member, i_b],
                                        steel_patch_frame.shape_b[i_member, i_b],
                                        steel_patch_geometry,
                                        steel_info,
                                    ):
                                        is_connected = True
                                        break
                            if is_connected:
                                i_chosen = i_p
                                nearest = distance

                    carried_impulse = qd.Vector.zero(gs.qd_float, 3)
                    error = qd.Vector.zero(gs.qd_float, 3)
                    if i_chosen >= 0:
                        steel_patch_history.used[i_chosen, i_b] = 1
                        error = func_carried_displacement(i_chosen, i_b, n_c, steel_state, steel_patch_history)
                        old_center = 0.5 * (
                            func_world_point(
                                i_la,
                                i_b,
                                steel_patch_history.anchor_a[i_chosen, i_b],
                                steel_state.links_pos,
                                steel_state.links_quat,
                            )
                            + func_world_point(
                                i_lb,
                                i_b,
                                steel_patch_history.anchor_b[i_chosen, i_b],
                                steel_state.links_pos,
                                steel_state.links_quat,
                            )
                        )
                        # Two anchors of one region describe a planar rigid displacement, translation and twist, which
                        # is evaluated at the new location rather than transferring the lever-arm errors independently.
                        # A region with one anchor carries its translation alone.
                        i_owner = steel_patch_history.owner[i_chosen, i_b]
                        for i_s in range(i_prev_start, i_prev_anchor_stop):
                            i_partner = steel_patch_history.members[i_s, i_b]
                            if (
                                i_partner != i_chosen
                                and steel_patch_history.valid[i_partner, i_b] != 0
                                and steel_patch_history.owner[i_partner, i_b] == i_owner
                            ):
                                other_normal = func_rotate(
                                    i_la, i_b, steel_patch_history.normal[i_partner, i_b], steel_state.links_quat
                                )
                                other_gap = (
                                    func_world_point(
                                        i_la,
                                        i_b,
                                        steel_patch_history.surface_a[i_partner, i_b],
                                        steel_state.links_pos,
                                        steel_state.links_quat,
                                    )
                                    - func_world_point(
                                        i_lb,
                                        i_b,
                                        steel_patch_history.surface_b[i_partner, i_b],
                                        steel_state.links_pos,
                                        steel_state.links_quat,
                                    )
                                ).dot(n_c)
                                if (
                                    other_normal.dot(n_c) >= steel_info.patch_normal_cosine_min[None]
                                    and other_gap <= gap_limit
                                ):
                                    other_center = 0.5 * (
                                        func_world_point(
                                            i_la,
                                            i_b,
                                            steel_patch_history.anchor_a[i_partner, i_b],
                                            steel_state.links_pos,
                                            steel_state.links_quat,
                                        )
                                        + func_world_point(
                                            i_lb,
                                            i_b,
                                            steel_patch_history.anchor_b[i_partner, i_b],
                                            steel_state.links_pos,
                                            steel_state.links_quat,
                                        )
                                    )
                                    span = other_center - old_center
                                    span = span - n_c * span.dot(n_c)
                                    span_sq = span.norm_sqr()
                                    if span_sq > steel_info.partner_span_sq[None] * r * r:
                                        other_error = func_carried_displacement(
                                            i_partner, i_b, n_c, steel_state, steel_patch_history
                                        )
                                        spin = (other_error - error).dot(n_c.cross(span)) / span_sq
                                        error = 0.5 * (error + other_error) + spin * n_c.cross(
                                            location - 0.5 * (old_center + other_center)
                                        )
                                break
                        error = error + motion
                        error = error - n_c * error.dot(n_c)
                        carried_impulse = steel_patch_history.tangent_impulse[i_chosen, i_b]
                    steel_patch_frame.displacement[i_c, i_b] = func_inv_rotate(i_la, i_b, error, steel_state.links_quat)
                    steel_patch_frame.source[i_c, i_b] = i_chosen
                    # Gap-filtered members only carry an existing history: no friction row held the pair this step.
                    if not (is_carry_only and i_chosen < 0):
                        steel_patch_frame.anchor_a[i_c, i_b] = func_local_point(
                            i_la, i_b, location, steel_state.links_pos, steel_state.links_quat
                        )
                        steel_patch_frame.anchor_b[i_c, i_b] = func_local_point(
                            i_lb, i_b, location, steel_state.links_pos, steel_state.links_quat
                        )
                        # Default of an anchor without friction rows this step, solved rows overwrite it at the end.
                        steel_patch_frame.tangent_impulse[i_c, i_b] = carried_impulse
                        steel_patch_frame.valid[i_c, i_b] = 1
                        steel_contacts.patch_weight[i_c, i_b] = 0.0 if is_carry_only else 1.0 / n_anchors
                        steel_contacts.patch_point[i_c, i_b] = location
                        n_row = n_c
                        if steel_patch_frame.flipped[i_c, i_b] != 0:
                            n_row = -n_c
                            error = -error
                        t0, t1 = func_contact_tangent_basis(n_row)
                        steel_contacts.patch_phi[i_c, i_b] = qd.Vector([t0.dot(error), t1.dot(error)], dt=gs.qd_float)


@qd.func
def func_steel_build_patches(
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_patch_frame: array_class.SteelPatchFrame,
    steel_patch_history: array_class.SteelPatchHistory,
    steel_sleep: array_class.SteelSleep,
    steel_patch_geometry: array_class.SteelPatchGeometry,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Build the friction regions and anchors of every body pair, one pair per thread (see func_build_pair_patches).

    An anchor of a region with rows gets the weight 1 / (number of anchors of its region), the other contacts keep 0,
    so only anchors receive friction rows. Its patch_phi is its carried displacement in the tangent basis of the row
    normal.
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_state.q.shape[1]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_s, i_b in qd.ndrange(max_contacts, _B):
        is_pair_start = False
        if i_s < steel_contacts.n[i_b]:
            is_pair_start = True
            if i_s > 0:
                i_c = steel_patch_frame.indices[i_s, i_b]
                i_c_prev = steel_patch_frame.indices[i_s - 1, i_b]
                is_pair_start = not func_key_equal(
                    steel_patch_frame.key[i_c_prev, i_b], steel_patch_frame.key[i_c, i_b]
                )
        if is_pair_start:
            func_build_pair_patches(
                i_s,
                i_b,
                steel_state,
                steel_contacts,
                steel_patch_frame,
                steel_patch_history,
                steel_sleep,
                steel_patch_geometry,
                steel_info,
                steel_config,
            )


@qd.func
def func_steel_link_patch_rows(
    route: int,
    row_parent: qd.Tensor,
    row_mu: qd.Tensor,
    steel_contacts: array_class.SteelContacts,
    steel_config: qd.template(),
):
    """Chain the normal rows of every region of one row family into a ring and share its friction coefficient.

    The parent of the normal row of a contact with rows becomes the normal row of the next member of its ring with rows
    in the family (itself without any), so the pooled normal load of a region walks the ring. Friction rows keep their
    own normal row as parent. The friction coefficient of an anchor with friction rows is divided by the number of
    anchors of its region that received friction rows, so the Coulomb budget of the pooled load is shared once.
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_contacts.n.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        n = steel_contacts.n[i_b]
        slot = -1
        if i_c < n and steel_contacts.path[i_c, i_b] == route:
            slot = steel_contacts.slot[i_c, i_b]
        i_row = -1
        if slot >= 0:
            i_row = steel_contacts.patch_next[i_c, i_b]
        if i_row >= 0:
            next_slot = slot
            is_anchor = steel_contacts.patch_weight[i_c, i_b] > 0.0 and steel_contacts.slots_needed[i_c, i_b] == 3
            n_anchors = 1 if is_anchor else 0
            for _ in range(n):
                if i_row == i_c:
                    break
                if steel_contacts.slot[i_row, i_b] >= 0 and steel_contacts.path[i_row, i_b] == route:
                    if next_slot == slot:
                        next_slot = steel_contacts.slot[i_row, i_b]
                    if steel_contacts.patch_weight[i_row, i_b] > 0.0 and steel_contacts.slots_needed[i_row, i_b] == 3:
                        n_anchors = n_anchors + 1
                    # A contact without friction rows only needs its successor.
                    if steel_contacts.patch_weight[i_c, i_b] <= 0.0:
                        break
                i_row = steel_contacts.patch_next[i_row, i_b]
            row_parent[slot, i_b] = next_slot
            if is_anchor and n_anchors > 0:
                for k in qd.static(range(1, 3)):
                    row_mu[slot + k, i_b] = row_mu[slot + k, i_b] / n_anchors


@qd.func
def func_warmstart_dt_scale(dt: float, history_dt: float):
    """Return the ratio of the step to the last step, 1 without a last step."""
    scale = gs.qd_float(1.0)
    if history_dt > 0.0:
        scale = dt / history_dt
    return scale


@qd.func
def func_steel_seed_patch_impulses(
    route: int,
    dt: float,
    row_parent: qd.Tensor,
    row_mu: qd.Tensor,
    impulses: qd.Tensor,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_patch_frame: array_class.SteelPatchFrame,
    steel_patch_history: array_class.SteelPatchHistory,
    steel_contact_history: array_class.SteelContactHistory,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Start the friction rows of every continued anchor of one row family from the tangent impulse it carries.

    It runs after every contact normal is seeded by the warm start. The carried impulse, scaled by the warm-start decay
    and the ratio of the step to the last one, is expressed in the tangent basis of the row normal and clamped to the
    disk of the pooled normal load of its region. Anchors without history, and every anchor when the decay is zero, keep
    the impulses of the contact warm start.
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_contacts.n.shape[0]
    decay = steel_info.warmstart_decay[None]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        i_src = -1
        if (
            i_c < steel_contacts.n[i_b]
            and steel_contacts.path[i_c, i_b] == route
            and steel_contacts.slot[i_c, i_b] >= 0
            and steel_contacts.slots_needed[i_c, i_b] == 3
            and decay > 0.0
        ):
            i_src = steel_patch_frame.source[i_c, i_b]
        if i_src >= 0:
            tangent = steel_patch_history.tangent_impulse[i_src, i_b] * (
                decay * func_warmstart_dt_scale(dt, steel_contact_history.dt[None])
            )
            tangent = func_rotate(steel_patch_frame.body_a[i_c, i_b], i_b, tangent, steel_state.links_quat)
            n_row = steel_patch_frame.normal[i_c, i_b]
            if steel_patch_frame.flipped[i_c, i_b] != 0:
                n_row = -n_row
                tangent = -tangent
            t0, t1 = func_contact_tangent_basis(n_row)
            value = qd.Vector([tangent.dot(t0), tangent.dot(t1)], dt=gs.qd_float)
            slot = steel_contacts.slot[i_c, i_b]
            radius = qd.max(row_mu[slot + 1, i_b] * func_pooled_normal_impulse(slot, i_b, row_parent, impulses), 0.0)
            magnitude = value.norm()
            if magnitude > radius and magnitude > 0.0:
                value = value * (radius / magnitude)
            impulses[slot + 1, i_b] = value[0]
            impulses[slot + 2, i_b] = value[1]


@qd.func
def func_steel_finish_patch_impulses(
    route: int,
    dt: float,
    row_parent: qd.Tensor,
    row_mu: qd.Tensor,
    impulses: qd.Tensor,
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_patch_frame: array_class.SteelPatchFrame,
    dyn_info: array_class.DynInfo,
    steel_info: array_class.SteelInfo,
    steel_config: qd.template(),
):
    """Cache the solved friction impulse of every anchor of one row family and release the anchors that slip.

    It runs after the integration, the link poses of the start of the step being in links_pos_prev / links_quat_prev
    and those of its end in links_pos / links_quat. A region whose pooled normal load is not positive releases its
    anchors, so an unsupported region cannot accumulate a displacement and apply it when it lands. A gap-filtered
    anchor keeps its history and carried impulse. An anchor whose impulse reaches its friction disk and whose anchor
    points slide against that impulse releases its history. Saturation alone does not, resting at the Coulomb limit
    being no proof of sliding. A contact whose rows were rejected for capacity keeps its history.
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_contacts.n.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        slot = -1
        if (
            i_c < steel_contacts.n[i_b]
            and steel_patch_frame.valid[i_c, i_b] != 0
            and steel_contacts.path[i_c, i_b] == route
        ):
            slot = steel_contacts.slot[i_c, i_b]
        if slot >= 0:
            load = func_pooled_normal_impulse(slot, i_b, row_parent, impulses)
            if load <= 0.0:
                steel_patch_frame.valid[i_c, i_b] = 0
            elif steel_contacts.slots_needed[i_c, i_b] == 3:
                n_row = steel_patch_frame.normal[i_c, i_b]
                is_flipped = steel_patch_frame.flipped[i_c, i_b] != 0
                if is_flipped:
                    n_row = -n_row
                t0, t1 = func_contact_tangent_basis(n_row)
                tangent = impulses[slot + 1, i_b] * t0 + impulses[slot + 2, i_b] * t1
                if is_flipped:
                    tangent = -tangent
                i_la = steel_patch_frame.body_a[i_c, i_b]
                i_lb = steel_patch_frame.body_b[i_c, i_b]
                steel_patch_frame.tangent_impulse[i_c, i_b] = func_inv_rotate(
                    i_la, i_b, tangent, steel_state.links_quat_prev
                )
                radius = qd.max(row_mu[slot + 1, i_b] * load, 0.0)
                speed = qd.Vector.zero(gs.qd_float, 3)
                for side in qd.static(range(2)):
                    i_l = i_la if qd.static(side == 0) else i_lb
                    if i_l >= 0:
                        I_l = [i_l, i_b] if qd.static(steel_config.batch_links_info) else i_l
                        anchor = (
                            steel_patch_frame.anchor_a[i_c, i_b]
                            if qd.static(side == 0)
                            else steel_patch_frame.anchor_b[i_c, i_b]
                        )
                        lever = gu.qd_transform_by_quat(
                            anchor - dyn_info.links.inertial_pos[I_l], steel_state.links_quat[i_l, i_b]
                        )
                        velocity = steel_state.links_vel[i_l, i_b] + steel_state.links_ang[i_l, i_b].cross(lever)
                        speed = speed + velocity if qd.static(side == 0) else speed - velocity
                if (
                    radius > 0.0
                    and tangent.norm() >= (1.0 - steel_info.release_saturation[None]) * radius
                    and speed.dot(tangent)
                    < -radius * (steel_info.release_slip[None] * steel_patch_frame.radius[i_c, i_b] / dt)
                ):
                    steel_patch_frame.valid[i_c, i_b] = 0


@qd.func
def func_steel_store_patches(
    steel_state: array_class.SteelState,
    steel_contacts: array_class.SteelContacts,
    steel_patch_frame: array_class.SteelPatchFrame,
    steel_patch_history: array_class.SteelPatchHistory,
    steel_config: qd.template(),
):
    """Keep the friction anchors of the step as the history of the next one.

    The normal of an anchor is stored in the frame of its body a at the start of the step, and the poses of the start
    of the step stay in links_pos_prev / links_quat_prev for the pose increments of the next one. Only the contacts of
    the step are kept: the next step reads the history through its sorted indices, which address those alone.
    """
    max_contacts = steel_contacts.shape0.shape[0]
    _B = steel_contacts.n.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        if i_c < steel_contacts.n[i_b]:
            normal = steel_patch_frame.normal[i_c, i_b]
            if steel_patch_frame.valid[i_c, i_b] != 0:
                normal = func_inv_rotate(steel_patch_frame.body_a[i_c, i_b], i_b, normal, steel_state.links_quat_prev)
            steel_patch_history.key[i_c, i_b] = steel_patch_frame.key[i_c, i_b]
            steel_patch_history.indices[i_c, i_b] = steel_patch_frame.indices[i_c, i_b]
            steel_patch_history.normal[i_c, i_b] = normal
            steel_patch_history.displacement[i_c, i_b] = steel_patch_frame.displacement[i_c, i_b]
            steel_patch_history.mu[i_c, i_b] = steel_patch_frame.mu[i_c, i_b]
            steel_patch_history.body_a[i_c, i_b] = steel_patch_frame.body_a[i_c, i_b]
            steel_patch_history.body_b[i_c, i_b] = steel_patch_frame.body_b[i_c, i_b]
            steel_patch_history.owner[i_c, i_b] = steel_patch_frame.owner[i_c, i_b]
            steel_patch_history.shape_a[i_c, i_b] = steel_patch_frame.shape_a[i_c, i_b]
            steel_patch_history.shape_b[i_c, i_b] = steel_patch_frame.shape_b[i_c, i_b]
            steel_patch_history.anchor_a[i_c, i_b] = steel_patch_frame.anchor_a[i_c, i_b]
            steel_patch_history.anchor_b[i_c, i_b] = steel_patch_frame.anchor_b[i_c, i_b]
            steel_patch_history.surface_a[i_c, i_b] = steel_patch_frame.surface_a[i_c, i_b]
            steel_patch_history.surface_b[i_c, i_b] = steel_patch_frame.surface_b[i_c, i_b]
            steel_patch_history.valid[i_c, i_b] = steel_patch_frame.valid[i_c, i_b]
            steel_patch_history.tangent_impulse[i_c, i_b] = steel_patch_frame.tangent_impulse[i_c, i_b]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b in range(_B):
        steel_patch_history.n[i_b] = steel_contacts.n[i_b]


@qd.kernel(fastcache=True)
def kernel_steel_reset_patches(
    envs_idx: qd.types.ndarray(),
    steel_patch_history: array_class.SteelPatchHistory,
    steel_config: qd.template(),
):
    """Discard the friction anchors carried by the selected environments."""
    max_contacts = steel_patch_history.valid.shape[0]
    qd.loop_config(serialize=qd.static(steel_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c, i_b_ in qd.ndrange(max_contacts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        steel_patch_history.valid[i_c, i_b] = 0
