import taichi as ti
import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.support_field_decomp as support_field

import genesis.engine.solvers.rigid.gjk.gjk_decomp as GJK


@ti.func
def func_gjk_contact(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Detect (possibly multiple) contact between two geometries using GJK and EPA algorithms.

    We first run the GJK algorithm to find the minimum distance between the two geometries. If the distance is
    smaller than the collision epsilon, we consider the geometries colliding. If they are colliding, we run the EPA
    algorithm to find the exact contact points and normals.

    .. seealso::
    MuJoCo's implementation:
    https://github.com/google-deepmind/mujoco/blob/7dc7a349c5ba2db2d3f8ab50a367d08e2f1afbbc/src/engine/engine_collision_gjk.c#L2259
    """
    # print("Running DGJK contact detection: ", i_ga, i_gb, i_b)
    # Clear the cache to prepare for this GJK-EPA run.
    GJK.clear_cache(gjk_state, i_b)

    # Backup state before local perturbation
    ga_pos, ga_quat = geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
    gb_pos, gb_quat = geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b]

    gjk_state.default_pos1[i_b], gjk_state.default_quat1[i_b] = ga_pos, ga_quat
    gjk_state.default_pos2[i_b], gjk_state.default_quat2[i_b] = gb_pos, gb_quat

    gjk_state.perturbed_pos1[i_b], gjk_state.perturbed_quat1[i_b] = ga_pos, ga_quat
    gjk_state.perturbed_pos2[i_b], gjk_state.perturbed_quat2[i_b] = gb_pos, gb_quat

    gjk_state.num_diff_faces[i_b] = 0
    gjk_state.default_epa_success[i_b] = 0

    # Axis to rotate the geometry for perturbation.
    axis_0 = ti.Vector.zero(gs.ti_float, 3)
    axis_1 = ti.Vector.zero(gs.ti_float, 3)

    for i in range(5):
        # First iteration: Detect contact in original configuration.
        # 2,3,4,5: Detect contact in perturbed configuration. This is required to sample subfaces in Minkowski difference.
        if i > 0:
            # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
            # otherwise it would be more sensitive to ill-conditionning.
            axis = (2 * (i % 2) - 1) * axis_0 + (1 - 2 * ((i // 2) % 2)) * axis_1
            qrot = gu.ti_rotvec_to_quat(1e-2 * axis)
            contact_pos = gjk_state.default_epa_contact_pos[i_b]
            func_rotate_frame(i_ga, contact_pos, qrot, i_b, geoms_state, geoms_info)
            func_rotate_frame(i_gb, contact_pos, gu.ti_inv_quat(qrot), i_b, geoms_state, geoms_info)

            gjk_state.perturbed_pos1[i_b], gjk_state.perturbed_quat1[i_b] = (
                geoms_state.pos[i_ga, i_b],
                geoms_state.quat[i_ga, i_b],
            )
            gjk_state.perturbed_pos2[i_b], gjk_state.perturbed_quat2[i_b] = (
                geoms_state.pos[i_gb, i_b],
                geoms_state.quat[i_gb, i_b],
            )

        # print(f"Running {i}-th iteration of GJK")
        gjk_flag = func_gjk(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
        )
        # print(f"GJK flag: {gjk_flag}")
        if gjk_flag == GJK.GJK_RETURN_CODE.INTERSECT:
            # Initialize polytope
            gjk_state.polytope.nverts[i_b] = 0
            gjk_state.polytope.nfaces[i_b] = 0
            gjk_state.polytope.nfaces_map[i_b] = 0
            gjk_state.polytope.horizon_nedges[i_b] = 0

            # Construct the initial polytope from the GJK simplex
            func_epa_init(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
            )
            func_epa(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
            )

            if i == 0 and gjk_state.default_epa_success[i_b] == 1:
                # Set

                # Set rotation axis for perturbation
                normal = gjk_state.default_epa_normal[i_b]
                axis_0, axis_1 = func_contact_orthogonals(
                    i_ga,
                    i_gb,
                    normal,
                    i_b,
                    links_state,
                    links_info,
                    geoms_state,
                    geoms_info,
                    geoms_init_AABB,
                )

        elif i == 0:
            break

        if i > 0:
            # Reset the frame to the original configuration
            func_to_default_config(geoms_state, gjk_state, i_ga, i_gb, i_b)

    # Compute the final contact points and normals.
    n_contacts = 0
    for i in range(gjk_state.num_diff_faces[i_b]):
        contact_pos, contact_normal, penetration, weight, flag = func_differentiable_contact(
            gjk_static_config,
            gjk_state.diff_faces_vert_localpos1[i_b, i, 0],
            gjk_state.diff_faces_vert_localpos1[i_b, i, 1],
            gjk_state.diff_faces_vert_localpos1[i_b, i, 2],
            gjk_state.diff_faces_vert_localpos2[i_b, i, 0],
            gjk_state.diff_faces_vert_localpos2[i_b, i, 1],
            gjk_state.diff_faces_vert_localpos2[i_b, i, 2],
            gjk_state.diff_faces_boundary_sdist_vert_localpos1[i_b, i],
            gjk_state.diff_faces_boundary_sdist_vert_localpos2[i_b, i],
            geoms_state.pos[i_ga, i_b],
            geoms_state.pos[i_gb, i_b],
            geoms_state.quat[i_ga, i_b],
            geoms_state.quat[i_gb, i_b],
            gjk_state.diff_faces_normal[i_b, i],
            gjk_state.default_epa_penetration[i_b],
        )
        # print(f"Computed {i}-th contact: pos = {contact_pos}, normal = {contact_normal}, penetration = {penetration}, weight = {weight}, flag = {flag}")

        if flag == GJK.RETURN_CODE.SUCCESS:
            # Check if the contact point is already in the list
            found_duplicate = False
            for i_c in range(n_contacts):
                duplicate_pos = (contact_pos - gjk_state.contact_pos[i_b, i_c]).norm() < gs.EPS
                duplicate_normal = contact_normal.dot(gjk_state.normal[i_b, i_c]) > 1 - gs.EPS
                if duplicate_pos and duplicate_normal:
                    found_duplicate = True
                    diff_penetration = penetration * weight
                    if diff_penetration > gjk_state.diff_penetration[i_b, i_c]:
                        # Overwrite the contact point
                        gjk_state.contact_pos[i_b, i_c] = contact_pos
                        gjk_state.normal[i_b, i_c] = contact_normal
                        gjk_state.diff_penetration[i_b, i_c] = penetration * weight
                        break
                    else:
                        # Skip the contact point
                        break

            if not found_duplicate:
                # Add the contact point to the list
                gjk_state.contact_pos[i_b, n_contacts] = contact_pos
                gjk_state.normal[i_b, n_contacts] = contact_normal
                gjk_state.diff_penetration[i_b, n_contacts] = penetration * weight
                n_contacts += 1

                if n_contacts >= gjk_static_config.max_contacts_per_pair:
                    break

    gjk_state.n_contacts[i_b] = n_contacts
    gjk_state.is_col[i_b] = 1 if n_contacts > 0 else 0
    gjk_state.multi_contact_flag[i_b] = 1


@ti.func
def func_contact_orthogonals(
    i_ga,
    i_gb,
    normal: ti.types.vector(3),
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
):
    axis_0 = ti.Vector.zero(gs.ti_float, 3)
    axis_1 = ti.Vector.zero(gs.ti_float, 3)

    # The reference geometry is the one that will have the largest impact on the position of
    # the contact point. Basically, the smallest one between the two, which can be approximated
    # by the volume of their respective bounding box.
    i_g = i_gb
    if geoms_info.type[i_ga] != gs.GEOM_TYPE.PLANE:
        size_ga = geoms_init_AABB[i_ga, 7]
        volume_ga = size_ga[0] * size_ga[1] * size_ga[2]
        size_gb = geoms_init_AABB[i_gb, 7]
        volume_gb = size_gb[0] * size_gb[1] * size_gb[2]
        i_g = i_ga if volume_ga < volume_gb else i_gb

    # Compute orthogonal basis mixing principal inertia axes of geometry with contact normal
    i_l = geoms_info.link_idx[i_g]
    rot = gu.ti_quat_to_R(links_state.i_quat[i_l, i_b])
    axis_idx = gs.ti_int(0)
    axis_angle_max = gs.ti_float(0.0)
    for i in ti.static(range(3)):
        axis_angle = ti.abs(rot[:, i].dot(normal))
        if axis_angle > axis_angle_max:
            axis_angle_max = axis_angle
            axis_idx = i
    axis_idx = (axis_idx + 1) % 3
    axis_0 = rot[:, axis_idx]
    axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()
    axis_1 = normal.cross(axis_0)

    return axis_0, axis_1


@ti.func
def func_rotate_frame(
    i_g,
    contact_pos: ti.types.vector(3),
    qrot: ti.types.vector(4),
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
):
    geoms_state.quat[i_g, i_b] = gu.ti_transform_quat_by_quat(geoms_state.quat[i_g, i_b], qrot)

    rel = contact_pos - geoms_state.pos[i_g, i_b]
    vec = gu.ti_transform_by_quat(rel, qrot)
    vec = vec - rel
    geoms_state.pos[i_g, i_b] = geoms_state.pos[i_g, i_b] - vec


@ti.func
def func_differentiable_contact(
    gjk_static_config: ti.template(),
    # Local positions of the vertices that form the face (from geom 1 and 2)
    local_pos1a: ti.types.vector(3),
    local_pos1b: ti.types.vector(3),
    local_pos1c: ti.types.vector(3),
    local_pos2a: ti.types.vector(3),
    local_pos2b: ti.types.vector(3),
    local_pos2c: ti.types.vector(3),
    # Local positions of the vertices that form signed distance for this face (from geom 1 and 2)
    sdist_local_pos1: ti.types.vector(3),
    sdist_local_pos2: ti.types.vector(3),
    # Transformations of geometries
    trans1: ti.types.vector(3),
    trans2: ti.types.vector(3),
    quat1: ti.types.vector(4),
    quat2: ti.types.vector(4),
    # Normal of the face in the default configuration, used for guiding the normal selection of the face
    normal: ti.types.vector(3),
    default_epa_dist,
):
    """
    Compute the contact normal, penetration, and point for face [f] in the default configuration in a differentiable way.

    The gradients flow through the following variables:
    - trans1
    - trans2
    - quat1
    - quat2
    """
    flag = GJK.RETURN_CODE.SUCCESS
    eps_B = gjk_static_config.diff_contact_eps_B
    eps_D = gjk_static_config.diff_contact_eps_D
    eps_A = gjk_static_config.diff_contact_eps_A

    # Result
    contact_pos = gs.ti_vec3(0.0, 0.0, 0.0)
    contact_normal = gs.ti_vec3(0.0, 0.0, 0.0)
    penetration = gs.ti_float(0.0)
    weight = gs.ti_float(0.0)

    # Compute global positions of the vertices
    pos1a = gu.ti_transform_by_trans_quat(local_pos1a, trans1, quat1)
    pos1b = gu.ti_transform_by_trans_quat(local_pos1b, trans1, quat1)
    pos1c = gu.ti_transform_by_trans_quat(local_pos1c, trans1, quat1)
    pos2a = gu.ti_transform_by_trans_quat(local_pos2a, trans2, quat2)
    pos2b = gu.ti_transform_by_trans_quat(local_pos2b, trans2, quat2)
    pos2c = gu.ti_transform_by_trans_quat(local_pos2c, trans2, quat2)

    # Compute the vertices on the Minkowski difference
    mink1 = pos1a - pos2a
    mink2 = pos1b - pos2b
    mink3 = pos1c - pos2c

    # Project the origin onto the affine plane of the face
    proj_o, _ = GJK.func_project_origin_to_plane(gjk_static_config, mink1, mink2, mink3)
    _lambda = GJK.func_triangle_affine_coords(proj_o, mink1, mink2, mink3)

    # Check validity of affine coordinates through reprojection
    proj_o_lambda = mink1 * _lambda[0] + mink2 * _lambda[1] + mink3 * _lambda[2]
    reprojection_error = (proj_o - proj_o_lambda).norm()

    # Take into account the face magnitude, as the error is relative to the face size.
    max_edge_len_inv = ti.rsqrt(
        max(
            (mink1 - mink2).norm_sqr(),
            (mink2 - mink3).norm_sqr(),
            (mink3 - mink1).norm_sqr(),
            gjk_static_config.FLOAT_MIN_SQ,
        )
    )
    rel_reprojection_error = reprojection_error * max_edge_len_inv
    if rel_reprojection_error > gjk_static_config.polytope_max_reprojection_error:
        flag = GJK.RETURN_CODE.FAIL

    if flag == GJK.RETURN_CODE.SUCCESS:
        # Point on geom 1
        w1 = pos1a * _lambda[0] + pos1b * _lambda[1] + pos1c * _lambda[2]
        # Point on geom 2
        w2 = pos2a * _lambda[0] + pos2b * _lambda[1] + pos2c * _lambda[2]

        # Contact position, normal, and penetration depth
        contact_pos = 0.5 * (w1 + w2)
        contact_normal = w2 - w1
        penetration = contact_normal.norm()
        if penetration > gjk_static_config.FLOAT_MIN:
            contact_normal = contact_normal / penetration

            # Compute weight for the penetration depth ---> Differentiable
            # Boundary weight: Compute boundary signed distance to the face
            _normal, _normal_len, _normal_flag = func_plane_normal(gjk_static_config, mink1, mink2, mink3)
            if _normal.dot(normal) < 0.0:
                _normal = -_normal

            face_center = (mink1 + mink2 + mink3) / 3.0

            sdist_global_pos1 = gu.ti_transform_by_trans_quat(sdist_local_pos1, trans1, quat1)
            sdist_global_pos2 = gu.ti_transform_by_trans_quat(sdist_local_pos2, trans2, quat2)
            sdist_mink = sdist_global_pos1 - sdist_global_pos2

            bsdist = ti.max(sdist_mink.dot(_normal) - face_center.dot(_normal), 0.0)
            boundary_weight = 1.0 - ti.math.clamp(bsdist / eps_B, 0.0, 1.0)

            # Distance weight
            distance_weight = 1.0 - ti.math.clamp((penetration - default_epa_dist) / eps_D, 0.0, 1.0)

            # Affine weight
            affine_weight_0 = 1.0 - ti.math.clamp(ti.max(0.0 - _lambda[0], _lambda[0] - 1.0) / eps_A, 0.0, 1.0)
            affine_weight_1 = 1.0 - ti.math.clamp(ti.max(0.0 - _lambda[1], _lambda[1] - 1.0) / eps_A, 0.0, 1.0)
            affine_weight_2 = 1.0 - ti.math.clamp(ti.max(0.0 - _lambda[2], _lambda[2] - 1.0) / eps_A, 0.0, 1.0)
            affine_weight = ti.min(affine_weight_0, affine_weight_1, affine_weight_2)

            # Compute final weight
            weight = affine_weight * distance_weight * boundary_weight
        else:
            flag = GJK.RETURN_CODE.FAIL

    return contact_pos, contact_normal, penetration, weight, flag


@ti.func
def support_driver(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    direction,
    i_g,
    i_b,
):
    # Support point in the global frame
    v = ti.Vector.zero(gs.ti_float, 3)
    # Support point in the local frame
    v_ = ti.Vector.zero(gs.ti_float, 3)
    vid = -1

    geom_type = geoms_info.type[i_g]
    if geom_type == gs.GEOM_TYPE.SPHERE:
        v, v_, vid = support_field._func_diff_support_sphere(geoms_state, geoms_info, direction, i_g, i_b)
    elif geom_type == gs.GEOM_TYPE.BOX:
        v, v_, vid = support_field._func_diff_support_box(geoms_state, geoms_info, direction, i_g, i_b)
    elif geom_type == gs.GEOM_TYPE.MESH:
        v, v_, vid = support_field._func_diff_support_world(
            geoms_state,
            geoms_info,
            support_field_info,
            support_field_static_config,
            direction,
            i_g,
            i_b,
        )
    return v, v_, vid


@ti.func
def func_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    dir,
):
    """
    Find support points on the two objects using [dir].

    Parameters:
    ----------
    dir: gs.ti_vec3
        The direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    support_point_obj1 = gs.ti_vec3(0, 0, 0)
    support_point_obj2 = gs.ti_vec3(0, 0, 0)
    support_point_id_obj1 = -1
    support_point_id_obj2 = -1
    support_point_localpos1 = gs.ti_vec3(0, 0, 0)
    support_point_localpos2 = gs.ti_vec3(0, 0, 0)

    for i in range(2):
        d = dir if i == 0 else -dir
        i_g = i_ga if i == 0 else i_gb

        sp, sp_, si = support_driver(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            d,
            i_g,
            i_b,
        )
        if i == 0:
            support_point_obj1 = sp
            support_point_id_obj1 = si
            support_point_localpos1 = sp_
        else:
            support_point_obj2 = sp
            support_point_localpos2 = sp_
            support_point_id_obj2 = si
    support_point_minkowski = support_point_obj1 - support_point_obj2

    return (
        support_point_obj1,
        support_point_obj2,
        support_point_id_obj1,
        support_point_id_obj2,
        support_point_localpos1,
        support_point_localpos2,
        support_point_minkowski,
    )


@ti.func
def func_gjk(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Safe GJK algorithm to compute the minimum distance between two convex objects.

    This implementation is safer than the one based on the MuJoCo implementation for the following reasons:
    1) It guarantees that the origin is strictly inside the tetrahedron when the intersection is detected.
    2) It guarantees to generate a non-degenerate tetrahedron if there is no numerical error, which is necessary
    for the following EPA algorithm to work correctly.
    3) When computing the face normals on the simplex, it uses a more robust method than using the origin.

    TODO: This implementation could be improved by using shrink_sphere option as the MuJoCo implementation does.
    TODO: This implementation could be further improved by referencing the follow-up work shown below.

    .. seealso::
    Original paper:
    Gilbert, Elmer G., Daniel W. Johnson, and S. Sathiya Keerthi.
    "A fast procedure for computing the distance between complex objects in three-dimensional space."
    IEEE Journal on Robotics and Automation 4.2 (2002): 193-203.

    Further improvements:
    Cameron, Stephen. "Enhancing GJK: Computing minimum and penetration distances between convex polyhedra."
    Proceedings of international conference on robotics and automation. Vol. 4. IEEE, 1997.
    https://www.cs.ox.ac.uk/people/stephen.cameron/distances/gjk2.4/

    Montaut, Louis, et al. "Collision detection accelerated: An optimization perspective."
    https://arxiv.org/abs/2205.09663
    """
    # Compute the initial tetrahedron using two random directions
    init_flag = GJK.RETURN_CODE.SUCCESS
    gjk_state.simplex.nverts[i_b] = 0
    for i in range(4):
        dir = ti.Vector.zero(gs.ti_float, 3)
        dir[2 - i // 2] = 1.0 - 2.0 * (i % 2)

        obj1, obj2, id1, id2, localpos1, localpos2, minkowski = func_gjk_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            dir,
        )

        # Check if the new vertex would make a valid simplex.
        valid = GJK.func_is_new_simplex_vertex_valid(gjk_state, gjk_static_config, i_b, id1, id2, minkowski)

        # If this is not a valid vertex, fall back to a brute-force routine to find a valid vertex.
        if not valid:
            obj1, obj2, id1, id2, localpos1, localpos2, minkowski, init_flag = func_search_valid_simplex_vertex(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
            )
            # If the brute-force search failed, we cannot proceed with GJK.
            if init_flag == GJK.RETURN_CODE.FAIL:
                break

        gjk_state.simplex_vertex.obj1[i_b, i] = obj1
        gjk_state.simplex_vertex.obj2[i_b, i] = obj2
        gjk_state.simplex_vertex.id1[i_b, i] = id1
        gjk_state.simplex_vertex.id2[i_b, i] = id2
        gjk_state.simplex_vertex.localpos1[i_b, i] = localpos1
        gjk_state.simplex_vertex.localpos2[i_b, i] = localpos2
        gjk_state.simplex_vertex.mink[i_b, i] = minkowski
        gjk_state.simplex.nverts[i_b] += 1

    gjk_flag = GJK.GJK_RETURN_CODE.SEPARATED
    if init_flag == GJK.RETURN_CODE.SUCCESS:
        # Simplex index
        si = ti.Vector([0, 1, 2, 3], dt=gs.ti_int)

        for i in range(gjk_static_config.gjk_max_iterations):
            # Compute normal and signed distance of the triangle faces of the simplex with respect to the origin.
            # These normals are supposed to point outwards from the simplex. If the origin is inside the plane,
            # [sdist] will be positive.
            for j in range(4):
                s0, s1, s2, ap = si[2], si[1], si[3], si[0]
                if j == 1:
                    s0, s1, s2, ap = si[0], si[2], si[3], si[1]
                elif j == 2:
                    s0, s1, s2, ap = si[1], si[0], si[3], si[2]
                elif j == 3:
                    s0, s1, s2, ap = si[0], si[1], si[2], si[3]

                n, s = GJK.func_safe_gjk_triangle_info(gjk_state, i_b, s0, s1, s2, ap)

                gjk_state.simplex_buffer.normal[i_b, j] = n
                gjk_state.simplex_buffer.sdist[i_b, j] = s

            # Find the face with the smallest signed distance. We need to find [min_i] for the next iteration.
            min_i = 0
            for j in ti.static(range(1, 4)):
                if gjk_state.simplex_buffer.sdist[i_b, j] < gjk_state.simplex_buffer.sdist[i_b, min_i]:
                    min_i = j

            min_si = si[min_i]
            min_normal = gjk_state.simplex_buffer.normal[i_b, min_i]
            min_sdist = gjk_state.simplex_buffer.sdist[i_b, min_i]

            # If origin is inside the simplex, the signed distances will all be positive
            if min_sdist >= 0:
                # Origin is inside the simplex, so we can stop
                gjk_flag = GJK.GJK_RETURN_CODE.INTERSECT
                break

            # Check if the new vertex would make a valid simplex.
            gjk_state.simplex.nverts[i_b] = 3
            if min_si != 3:
                gjk_state.simplex_vertex.obj1[i_b, min_si] = gjk_state.simplex_vertex.obj1[i_b, 3]
                gjk_state.simplex_vertex.obj2[i_b, min_si] = gjk_state.simplex_vertex.obj2[i_b, 3]
                gjk_state.simplex_vertex.id1[i_b, min_si] = gjk_state.simplex_vertex.id1[i_b, 3]
                gjk_state.simplex_vertex.id2[i_b, min_si] = gjk_state.simplex_vertex.id2[i_b, 3]
                gjk_state.simplex_vertex.localpos1[i_b, min_si] = gjk_state.simplex_vertex.localpos1[i_b, 3]
                gjk_state.simplex_vertex.localpos2[i_b, min_si] = gjk_state.simplex_vertex.localpos2[i_b, 3]
                gjk_state.simplex_vertex.mink[i_b, min_si] = gjk_state.simplex_vertex.mink[i_b, 3]

            # Find a new candidate vertex to replace the worst vertex (which has the smallest signed distance)
            obj1, obj2, id1, id2, localpos1, localpos2, minkowski = func_gjk_support(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
                min_normal,
            )

            duplicate = GJK.func_is_new_simplex_vertex_duplicate(gjk_state, i_b, id1, id2)
            if duplicate:
                # If the new vertex is a duplicate, it means separation.
                gjk_flag = GJK.GJK_RETURN_CODE.SEPARATED
                break

            degenerate = GJK.func_is_new_simplex_vertex_degenerate(gjk_state, gjk_static_config, i_b, minkowski)
            if degenerate:
                # If the new vertex is degenerate, we cannot proceed with GJK.
                gjk_flag = GJK.GJK_RETURN_CODE.NUM_ERROR
                break

            # Check if the origin is strictly outside of the Minkowski difference (which means there is no collision)
            is_no_collision = minkowski.dot(min_normal) < 0.0
            if is_no_collision:
                gjk_flag = GJK.GJK_RETURN_CODE.SEPARATED
                break

            gjk_state.simplex_vertex.obj1[i_b, 3] = obj1
            gjk_state.simplex_vertex.obj2[i_b, 3] = obj2
            gjk_state.simplex_vertex.id1[i_b, 3] = id1
            gjk_state.simplex_vertex.id2[i_b, 3] = id2
            gjk_state.simplex_vertex.localpos1[i_b, 3] = localpos1
            gjk_state.simplex_vertex.localpos2[i_b, 3] = localpos2
            gjk_state.simplex_vertex.mink[i_b, 3] = minkowski
            gjk_state.simplex.nverts[i_b] = 4

    if gjk_flag == GJK.GJK_RETURN_CODE.INTERSECT:
        gjk_state.distance[i_b] = 0.0
    else:
        gjk_flag = GJK.GJK_RETURN_CODE.SEPARATED
        gjk_state.distance[i_b] = gjk_static_config.FLOAT_MAX

    return gjk_flag


@ti.func
def func_gjk_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    dir,
):
    """
    Find support points on the two objects using [dir] to use in the [safe_gjk] algorithm.

    This is a more robust version of the support function that finds only one pair of support points, because this
    function perturbs the support direction to find the best support points that guarantee non-degenerate simplex
    in the GJK algorithm.

    Parameters:
    ----------
    dir: gs.ti_vec3
        The unit direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    obj1 = gs.ti_vec3(0.0, 0.0, 0.0)
    obj2 = gs.ti_vec3(0.0, 0.0, 0.0)
    id1 = gs.ti_int(-1)
    id2 = gs.ti_int(-1)
    localpos1 = gs.ti_vec3(0.0, 0.0, 0.0)
    localpos2 = gs.ti_vec3(0.0, 0.0, 0.0)
    mink = obj1 - obj2

    for i in range(9):
        n_dir = dir
        if i > 0:
            j = i - 1
            n_dir[0] += -(1.0 - 2.0 * (j & 1)) * gs.EPS
            n_dir[1] += -(1.0 - 2.0 * (j & 2)) * gs.EPS
            n_dir[2] += -(1.0 - 2.0 * (j & 4)) * gs.EPS

        # First order normalization based on Taylor series is accurate enough
        n_dir *= 2.0 - n_dir.dot(dir)

        num_supports = GJK.func_count_support(
            geoms_state,
            geoms_info,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            n_dir,
        )
        if i > 0 and num_supports > 1:
            # If this is a perturbed direction and we have more than one support point, we skip this iteration. If
            # it was the original direction, we continue to find the support points to keep it as the baseline.
            continue

        # Use the current direction to find the support points.
        for j in range(2):
            d = n_dir if j == 0 else -n_dir
            i_g = i_ga if j == 0 else i_gb

            sp, sp_, si = support_driver(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                d,
                i_g,
                i_b,
            )
            if j == 0:
                obj1 = sp
                id1 = si
                localpos1 = sp_
            else:
                obj2 = sp
                id2 = si
                localpos2 = sp_

        mink = obj1 - obj2

        if i == 0:
            if num_supports > 1:
                # If there were multiple valid support points, we move on to the next iteration to perturb the
                # direction and find better support points.
                continue
            else:
                break

        # If it was a perturbed direction, check if the support points have been found before.
        if i == 8:
            # If this was the last iteration, we don't check if it has been found before.
            break

        # Check if the updated simplex would be a degenerate simplex.
        if GJK.func_is_new_simplex_vertex_valid(gjk_state, gjk_static_config, i_b, id1, id2, mink):
            break

    return obj1, obj2, id1, id2, localpos1, localpos2, mink


@ti.func
def func_search_valid_simplex_vertex(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Search for a valid simplex vertex (non-duplicate, non-degenerate) in the Minkowski difference.
    """
    obj1 = gs.ti_vec3(0.0, 0.0, 0.0)
    obj2 = gs.ti_vec3(0.0, 0.0, 0.0)
    id1 = -1
    id2 = -1
    localpos1 = gs.ti_vec3(0.0, 0.0, 0.0)
    localpos2 = gs.ti_vec3(0.0, 0.0, 0.0)
    minkowski = gs.ti_vec3(0.0, 0.0, 0.0)
    flag = GJK.RETURN_CODE.FAIL

    # If both geometries are discrete, we can use a brute-force search to find a valid simplex vertex.
    if GJK.func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b):
        geom_nverts = gs.ti_ivec2(0, 0)
        for i in range(2):
            geom_nverts[i] = GJK.func_num_discrete_geom_vertices(geoms_info, i_ga if i == 0 else i_gb, i_b)

        num_cases = geom_nverts[0] * geom_nverts[1]
        for k in range(num_cases):
            m = (k + gjk_state.last_searched_simplex_vertex_id[i_b]) % num_cases
            i = m // geom_nverts[1]
            j = m % geom_nverts[1]

            id1 = geoms_info.vert_start[i_ga] + i
            id2 = geoms_info.vert_start[i_gb] + j
            for p in range(2):
                obj, obj_ = func_get_discrete_geom_vertex(
                    geoms_state, geoms_info, verts_info, i_ga if p == 0 else i_gb, i_b, i if p == 0 else j
                )
                if p == 0:
                    obj1 = obj
                    localpos1 = obj_
                else:
                    obj2 = obj
                    localpos2 = obj_
            minkowski = obj1 - obj2

            # Check if the new vertex is valid
            if GJK.func_is_new_simplex_vertex_valid(gjk_state, gjk_static_config, i_b, id1, id2, minkowski):
                flag = GJK.RETURN_CODE.SUCCESS
                # Update buffer
                gjk_state.last_searched_simplex_vertex_id[i_b] = (m + 1) % num_cases
                break
    else:
        # Try search direction based on the current simplex.
        nverts = gjk_state.simplex.nverts[i_b]
        if nverts == 3:
            # If we have a triangle, use its normal as the search direction.
            v1 = gjk_state.simplex_vertex.mink[i_b, 0]
            v2 = gjk_state.simplex_vertex.mink[i_b, 1]
            v3 = gjk_state.simplex_vertex.mink[i_b, 2]
            dir = (v3 - v1).cross(v2 - v1).normalized()

            for i in range(2):
                d = dir if i == 0 else -dir
                obj1, obj2, id1, id2, localpos1, localpos2, minkowski = func_gjk_support(
                    geoms_state,
                    geoms_info,
                    verts_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_static_config,
                    gjk_state,
                    gjk_static_config,
                    support_field_info,
                    support_field_static_config,
                    i_ga,
                    i_gb,
                    i_b,
                    d,
                )

                # Check if the new vertex is valid
                if GJK.func_is_new_simplex_vertex_valid(gjk_state, gjk_static_config, i_b, id1, id2, minkowski):
                    flag = GJK.RETURN_CODE.SUCCESS
                    break

    return obj1, obj2, id1, id2, localpos1, localpos2, minkowski, flag


@ti.func
def func_get_discrete_geom_vertex(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    i_g,
    i_b,
    i_v,
):
    """
    Get the discrete vertex of the geometry for the given index [i_v].
    """
    geom_type = geoms_info.type[i_g]
    g_pos = geoms_state.pos[i_g, i_b]
    g_quat = geoms_state.quat[i_g, i_b]

    # Get the vertex position in the local frame of the geometry.
    v = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    if geom_type == gs.GEOM_TYPE.BOX:
        # For the consistency with the [func_support_box] function of [SupportField] class, we handle the box
        # vertex positions in a different way than the general mesh.
        v = ti.Vector(
            [
                (1.0 if (i_v & 1 == 1) else -1.0) * geoms_info.data[i_g][0] * 0.5,
                (1.0 if (i_v & 2 == 2) else -1.0) * geoms_info.data[i_g][1] * 0.5,
                (1.0 if (i_v & 4 == 4) else -1.0) * geoms_info.data[i_g][2] * 0.5,
            ],
            dt=gs.ti_float,
        )
    elif geom_type == gs.GEOM_TYPE.MESH:
        vert_start = geoms_info.vert_start[i_g]
        v = verts_info.init_pos[vert_start + i_v]

    # Transform the vertex position to the world frame
    v_ = v
    v = gu.ti_transform_by_trans_quat(v, g_pos, g_quat)

    return v, v_


@ti.func
def func_epa_init(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Create the polytope for safe EPA from a 3-simplex (tetrahedron).

    Assume the tetrahedron is a non-degenerate simplex.
    """

    # Insert simplex vertices into the polytope
    vi = ti.Vector([0, 0, 0, 0], dt=ti.i32)
    for i in range(4):
        vi[i] = func_epa_insert_vertex_to_polytope(
            gjk_state,
            i_b,
            gjk_state.simplex_vertex.obj1[i_b, i],
            gjk_state.simplex_vertex.obj2[i_b, i],
            gjk_state.simplex_vertex.id1[i_b, i],
            gjk_state.simplex_vertex.id2[i_b, i],
            gjk_state.simplex_vertex.localpos1[i_b, i],
            gjk_state.simplex_vertex.localpos2[i_b, i],
            gjk_state.simplex_vertex.mink[i_b, i],
        )

    for i in range(4):
        # Vertex indices for the faces in the hexahedron
        v1, v2, v3 = vi[0], vi[1], vi[2]
        # Adjacent face indices for the faces in the hexahedron
        a1, a2, a3 = 1, 3, 2
        if i == 1:
            v1, v2, v3 = vi[0], vi[3], vi[1]
            a1, a2, a3 = 2, 3, 0
        elif i == 2:
            v1, v2, v3 = vi[0], vi[2], vi[3]
            a1, a2, a3 = 0, 3, 1
        elif i == 3:
            v1, v2, v3 = vi[3], vi[2], vi[1]
            a1, a2, a3 = 2, 0, 1

        func_attach_face_to_polytope(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            v1,
            v2,
            v3,
            a1,
            a2,
            a3,
        )

    # Initialize face map
    for i in ti.static(range(4)):
        gjk_state.polytope_faces_map[i_b, i] = i
        gjk_state.polytope_faces.map_idx[i_b, i] = i
    gjk_state.polytope.nfaces_map[i_b] = 4


@ti.func
def func_epa(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
):
    """
    Differentiable version of the safe EPA algorithm to find (possibly) multiple contact points and penetration
    depths in a differentiable manner.

    This implementation computes the weight of each possible contact point based on the following three criteria:
    1) Boundary weight: To be a valid contact point, the face must lie at the boundary of the polytope.
    2) Distance weight: To be a valid contact point, the penetration depth should not exceed the original EPA depth.
    3) Affine weight: To be a valid contact point, the affine coordinates of the origin's projection onto the face
    should be in range [0, 1].

    Based on these criteria, we compute the weight of each contact point, and adjust the penetration depth. The
    original contact point where EPA depth occurs is always guaranteed to have a weight of 1.0.

    This algorithm works only for discrete geoms.
    """
    # Tolerance for determining if a face is on the boundary on the Minkowski difference.
    # Unlike the non-differentiable EPA, we use a small tolerance even for the discrete case, because we use signed
    # distance to determine if a face is on the boundary.
    tolerance = gjk_static_config.tolerance

    # Epsilon values for differentiable contact.
    eps_B = gjk_static_config.diff_contact_eps_B
    eps_D = gjk_static_config.diff_contact_eps_D
    eps_A = gjk_static_config.diff_contact_eps_A

    discrete = GJK.func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b)

    k_max = gjk_static_config.epa_max_iterations
    for k in range(k_max):
        # print(f"EPA iteration: {k}")
        # Index of the nearest face
        nearest_i_f = gs.ti_int(-1)
        dir = gs.ti_vec3(0.0, 0.0, 0.0)
        found_valid_dir = False

        # Find the polytope face with the smallest distance to the origin
        while True:
            # print(f"Finding nearest face")
            nearest_i_f = gs.ti_int(-1)
            dist2 = gjk_static_config.FLOAT_MAX_SQ
            for i in range(gjk_state.polytope.nfaces_map[i_b]):
                i_f = gjk_state.polytope_faces_map[i_b, i]
                if gjk_state.polytope_faces.visited[i_b, i_f] == 1:
                    continue

                face_dist2 = gjk_state.polytope_faces.dist2[i_b, i_f]

                if face_dist2 < dist2:
                    dist2 = face_dist2
                    nearest_i_f = i_f

            if nearest_i_f == -1:
                # If every face is visited or there is no face to visit, we can stop the algorithm.
                found_valid_dir = False
                break

            dir = gjk_state.polytope_faces.normal[i_b, nearest_i_f]

            # Determine if this nearest face is a boundary face using its signed distance in the perturbed configuration.
            # print(f"Nearest face's perturb boundary sdist: {gjk_state.polytope_faces.perturb_boundary_sdist[i_b, nearest_i_f]:.20g}")
            is_nearest_face_boundary = gjk_state.polytope_faces.perturb_boundary_sdist[i_b, nearest_i_f] < tolerance

            if is_nearest_face_boundary:
                # print(f"Found boundary face: {nearest_i_f}")
                if gjk_state.default_epa_success[i_b] == 0:
                    if (
                        func_set_default_epa_contact(gjk_state, gjk_static_config, i_b, nearest_i_f)
                        == GJK.RETURN_CODE.SUCCESS
                    ):
                        gjk_state.default_epa_success[i_b] = 1
                        # print(f"Set default EPA contact")
                    else:
                        found_valid_dir = False
                        break

                # Mark this face as visited.
                gjk_state.polytope_faces.visited[i_b, nearest_i_f] = 1
                default_info_valid = gjk_state.polytope_faces.default_info_valid[i_b, nearest_i_f]

                if not default_info_valid:
                    # If this boundary face is degenerate in the default configuration, we do not need to consider it.
                    continue

                # Check if this face would generate non-zero weight contact.
                default_dist = gjk_state.polytope_faces.default_dist[i_b, nearest_i_f]
                default_boundary_sdist = gjk_state.polytope_faces.default_boundary_sdist[i_b, nearest_i_f]
                default_projection_affine_coords = gjk_state.polytope_faces.default_projection_affine_coords[
                    i_b, nearest_i_f
                ]

                satisfy_boundary_constraint = default_boundary_sdist < eps_B
                satisfy_distance_constraint = default_dist < gjk_state.default_epa_penetration[i_b] + eps_D
                satisfy_affine_constraint = (
                    (default_projection_affine_coords[0] > -eps_A)
                    and (default_projection_affine_coords[0] < 1.0 + eps_A)
                    and (default_projection_affine_coords[1] > -eps_A)
                    and (default_projection_affine_coords[1] < 1.0 + eps_A)
                    and (default_projection_affine_coords[2] > -eps_A)
                    and (default_projection_affine_coords[2] < 1.0 + eps_A)
                )

                if satisfy_boundary_constraint and satisfy_distance_constraint and satisfy_affine_constraint:
                    # Save this face to compute contact point later.
                    func_add_differentiable_face(gjk_state, nearest_i_f, i_b)
                elif not satisfy_distance_constraint:
                    # The faces that we will find will have 0 weight because they would not satisfy the distance constraint.
                    found_valid_dir = False
                    break
            else:
                found_valid_dir = True
                break

        if not found_valid_dir:
            break

        # Find a new support point w from the nearest face's normal
        wi = func_epa_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            dir,
            1.0,
        )
        w = gjk_state.polytope_verts.mink[i_b, wi]

        if discrete:
            repeated = False
            for i in range(gjk_state.polytope.nverts[i_b]):
                if i == wi:
                    continue
                elif (
                    gjk_state.polytope_verts.id1[i_b, i] == gjk_state.polytope_verts.id1[i_b, wi]
                    and gjk_state.polytope_verts.id2[i_b, i] == gjk_state.polytope_verts.id2[i_b, wi]
                ):
                    # The vertex w is already in the polytope, so we do not need to add it again.
                    repeated = True
                    break
            if repeated:
                # print(f"Repeated vertex found")
                break

        gjk_state.polytope.horizon_w[i_b] = w

        # Compute horizon
        horizon_flag = GJK.func_epa_horizon(gjk_state, gjk_static_config, i_b, nearest_i_f)

        if horizon_flag:
            # There was an error in the horizon construction, so the horizon edge is not a closed loop.
            # print(f"Error in horizon construction")
            break

        if gjk_state.polytope.horizon_nedges[i_b] < 3:
            # Should not happen, because at least three edges should be in the horizon from one deleted face.
            # print(f"Error in horizon construction")
            break

        # Check if the memory space is enough for attaching new faces
        nfaces = gjk_state.polytope.nfaces[i_b]
        nedges = gjk_state.polytope.horizon_nedges[i_b]
        if nfaces + nedges >= gjk_static_config.polytope_max_faces:
            # If the polytope is full, we cannot insert new faces
            # print(f"Polytope is full")
            break

        # Attach the new faces
        # print("Attaching new faces to the polytope")
        attach_flag = GJK.RETURN_CODE.SUCCESS
        for i in range(nedges):
            # Face id of the current face to attach
            i_f0 = nfaces + i
            # Face id of the next face to attach
            i_f1 = nfaces + (i + 1) % nedges

            horizon_i_f = gjk_state.polytope_horizon_data.face_idx[i_b, i]
            horizon_i_e = gjk_state.polytope_horizon_data.edge_idx[i_b, i]
            horizon_v1 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][horizon_i_e]
            horizon_v2 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][(horizon_i_e + 1) % 3]

            # Change the adjacent face index of the existing face
            gjk_state.polytope_faces.adj_idx[i_b, horizon_i_f][horizon_i_e] = i_f0

            # Attach the new face.
            # If this if the first face, will be adjacent to the face that will be attached last.
            adj_i_f_0 = i_f0 - 1 if (i > 0) else nfaces + nedges - 1
            adj_i_f_1 = horizon_i_f
            adj_i_f_2 = i_f1

            attach_flag = func_attach_face_to_polytope(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
                wi,
                horizon_v2,
                horizon_v1,
                adj_i_f_2,  # Previous face id
                adj_i_f_1,
                adj_i_f_0,  # Next face id
            )
            if attach_flag != GJK.RETURN_CODE.SUCCESS:
                # Unrecoverable numerical issue
                break

            dist2 = gjk_state.polytope_faces.dist2[i_b, gjk_state.polytope.nfaces[i_b] - 1]

            # Store face in the map
            nfaces_map = gjk_state.polytope.nfaces_map[i_b]
            gjk_state.polytope_faces_map[i_b, nfaces_map] = i_f0
            gjk_state.polytope_faces.map_idx[i_b, i_f0] = nfaces_map
            gjk_state.polytope.nfaces_map[i_b] += 1

        if attach_flag != GJK.RETURN_CODE.SUCCESS:
            # print(f"Error in attaching face")
            break

        # Clear the horizon data for the next iteration
        gjk_state.polytope.horizon_nedges[i_b] = 0

        if gjk_state.polytope.nfaces_map[i_b] == 0:
            # No face candidate left
            # print(f"No face candidate left")
            break


@ti.func
def func_set_default_epa_contact(
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    i_b,
    nearest_i_f,
):
    flag = GJK.RETURN_CODE.SUCCESS
    _lambda = gjk_state.polytope_faces.default_projection_affine_coords[i_b, nearest_i_f]
    default_dist = gjk_state.polytope_faces.default_dist[i_b, nearest_i_f]

    # Check validity of affine coordinates through reprojection
    i_v1 = gjk_state.polytope_faces.verts_idx[i_b, nearest_i_f][0]
    i_v2 = gjk_state.polytope_faces.verts_idx[i_b, nearest_i_f][1]
    i_v3 = gjk_state.polytope_faces.verts_idx[i_b, nearest_i_f][2]

    # Because this is default epa, the frame is not perturbed.
    v1 = gjk_state.polytope_verts.mink[i_b, i_v1]
    v2 = gjk_state.polytope_verts.mink[i_b, i_v2]
    v3 = gjk_state.polytope_verts.mink[i_b, i_v3]

    proj_o_lambda = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

    proj_o, _ = GJK.func_project_origin_to_plane(gjk_static_config, v1, v2, v3)
    reprojection_error = (proj_o - proj_o_lambda).norm()

    # Take into account the face magnitude, as the error is relative to the face size.
    max_edge_len_inv = ti.rsqrt(
        max((v1 - v2).norm_sqr(), (v2 - v3).norm_sqr(), (v3 - v1).norm_sqr(), gjk_static_config.FLOAT_MIN_SQ)
    )
    rel_reprojection_error = reprojection_error * max_edge_len_inv
    if rel_reprojection_error > gjk_static_config.polytope_max_reprojection_error:
        flag = GJK.RETURN_CODE.FAIL

    if flag == GJK.RETURN_CODE.SUCCESS:
        # Point on geom 1
        v1 = gjk_state.polytope_verts.obj1[i_b, i_v1]
        v2 = gjk_state.polytope_verts.obj1[i_b, i_v2]
        v3 = gjk_state.polytope_verts.obj1[i_b, i_v3]
        witness1 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        # Point on geom 2
        v1 = gjk_state.polytope_verts.obj2[i_b, i_v1]
        v2 = gjk_state.polytope_verts.obj2[i_b, i_v2]
        v3 = gjk_state.polytope_verts.obj2[i_b, i_v3]
        witness2 = v1 * _lambda[0] + v2 * _lambda[1] + v3 * _lambda[2]

        contact_pos = 0.5 * (witness1 + witness2)
        contact_normal = witness2 - witness1
        contact_normal_len = contact_normal.norm()

        if contact_normal_len > gjk_static_config.FLOAT_MIN:
            contact_normal = contact_normal / contact_normal_len

            gjk_state.default_epa_contact_pos[i_b] = contact_pos
            gjk_state.default_epa_normal[i_b] = contact_normal
            gjk_state.default_epa_penetration[i_b] = default_dist
        else:
            flag = GJK.RETURN_CODE.FAIL

    return flag


@ti.func
def func_epa_insert_vertex_to_polytope(
    gjk_state: array_class.GJKState,
    i_b,
    obj1_point,
    obj2_point,
    obj1_id,
    obj2_id,
    obj1_localpos,
    obj2_localpos,
    minkowski_point,
):
    """
    Copy vertex information into the polytope.
    """
    n = gjk_state.polytope.nverts[i_b]
    gjk_state.polytope_verts.obj1[i_b, n] = obj1_point
    gjk_state.polytope_verts.obj2[i_b, n] = obj2_point
    gjk_state.polytope_verts.id1[i_b, n] = obj1_id
    gjk_state.polytope_verts.id2[i_b, n] = obj2_id
    gjk_state.polytope_verts.localpos1[i_b, n] = obj1_localpos
    gjk_state.polytope_verts.localpos2[i_b, n] = obj2_localpos
    gjk_state.polytope_verts.mink[i_b, n] = minkowski_point
    gjk_state.polytope.nverts[i_b] += 1
    return n


@ti.func
def func_epa_support(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    dir,
    dir_norm,
):
    """
    Find support points on the two objects using [dir] and insert them into the polytope.

    Parameters
    ----------
    dir: gs.ti_vec3
        Vector from [ga] (obj1) to [gb] (obj2).
    """
    d = gs.ti_vec3(1, 0, 0)
    if dir_norm > gjk_static_config.FLOAT_MIN:
        d = dir / dir_norm

    # Insert the support points into the polytope
    v_index = func_epa_insert_vertex_to_polytope(
        gjk_state,
        i_b,
        *func_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            d,
        ),
    )

    return v_index


@ti.func
def func_attach_face_to_polytope(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    i_v1,
    i_v2,
    i_v3,
    i_a1,
    i_a2,
    i_a3,
):
    """
    Attach a face to the polytope.

    While attaching the face, 1) determine its normal direction, and 2) estimate the lower bound of the penetration
    depth in robust manner.

    [i_v1, i_v2, i_v3] are the vertices of the face, [i_a1, i_a2, i_a3] are the adjacent faces.
    """
    n = gjk_state.polytope.nfaces[i_b]
    gjk_state.polytope_faces.verts_idx[i_b, n][0] = i_v1
    gjk_state.polytope_faces.verts_idx[i_b, n][1] = i_v2
    gjk_state.polytope_faces.verts_idx[i_b, n][2] = i_v3
    gjk_state.polytope_faces.adj_idx[i_b, n][0] = i_a1
    gjk_state.polytope_faces.adj_idx[i_b, n][1] = i_a2
    gjk_state.polytope_faces.adj_idx[i_b, n][2] = i_a3
    gjk_state.polytope.nfaces[i_b] += 1

    # When added, the face is not visited yet.
    gjk_state.polytope_faces.visited[i_b, n] = 0

    # Compute the normal of the plane
    normal, normal_length, flag = func_plane_normal(
        gjk_static_config,
        gjk_state.polytope_verts.mink[i_b, i_v3],
        gjk_state.polytope_verts.mink[i_b, i_v2],
        gjk_state.polytope_verts.mink[i_b, i_v1],
    )
    if flag == GJK.RETURN_CODE.SUCCESS:
        face_center = (
            gjk_state.polytope_verts.mink[i_b, i_v1]
            + gjk_state.polytope_verts.mink[i_b, i_v2]
            + gjk_state.polytope_verts.mink[i_b, i_v3]
        ) / 3.0

        # Use origin for initialization
        max_orient = -normal.dot(face_center)
        max_abs_orient = ti.abs(max_orient)

        # Consider other vertices in the polytope to reorient the normal
        nverts = gjk_state.polytope.nverts[i_b]
        for i_v in range(nverts):
            if i_v != i_v1 and i_v != i_v2 and i_v != i_v3:
                diff = gjk_state.polytope_verts.mink[i_b, i_v] - face_center
                orient = normal.dot(diff)
                if ti.abs(orient) > max_abs_orient:
                    max_abs_orient = ti.abs(orient)
                    max_orient = orient

        if max_orient > 0.0:
            normal = -normal

        gjk_state.polytope_faces.normal[i_b, n] = normal
        gjk_state.polytope_faces.dist2[i_b, n] = normal.dot(face_center) ** 2

        # Compute boundary signed distance in the perturbed configuration
        _, _, _, _, _, _, mink = func_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            normal,
        )

        # Compute boundary signed distance in this direction
        gjk_state.polytope_faces.perturb_boundary_sdist[i_b, n] = mink.dot(normal) - face_center.dot(normal)

        # Compute information for differentiable contact.
        default_info_flag = func_face_info_in_default_config(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
            i_v1,
            i_v2,
            i_v3,
        )
        # print(f"Perturb normal: {normal}")
        # print(f"Default normal: {gjk_state.polytope_faces.default_normal[i_b, n]}")
        gjk_state.polytope_faces.default_info_valid[i_b, n] = default_info_flag == GJK.RETURN_CODE.SUCCESS

    return flag


@ti.func
def func_face_info_in_default_config(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    i_v1,
    i_v2,
    i_v3,
):
    n = gjk_state.polytope.nfaces[i_b] - 1
    func_to_default_config(geoms_state, gjk_state, i_ga, i_gb, i_b)

    default_mink1 = gs.ti_vec3(0.0, 0.0, 0.0)
    default_mink2 = gs.ti_vec3(0.0, 0.0, 0.0)
    default_mink3 = gs.ti_vec3(0.0, 0.0, 0.0)

    # Recompute the face vertices in the default configuration
    for i in range(3):
        curr_i_v = i_v1
        if i == 1:
            curr_i_v = i_v2
        elif i == 2:
            curr_i_v = i_v3

        mink = func_compute_minkowski_point(
            gjk_state.default_pos1[i_b],
            gjk_state.default_quat1[i_b],
            gjk_state.default_pos2[i_b],
            gjk_state.default_quat2[i_b],
            gjk_state.polytope_verts.localpos1[i_b, curr_i_v],
            gjk_state.polytope_verts.localpos2[i_b, curr_i_v],
        )
        if i == 0:
            default_mink1 = mink
        elif i == 1:
            default_mink2 = mink
        elif i == 2:
            default_mink3 = mink

    # Recompute the face normal in the default configuration
    normal, normal_length, flag = func_plane_normal(
        gjk_static_config,
        default_mink1,
        default_mink2,
        default_mink3,
    )
    if flag == GJK.RETURN_CODE.SUCCESS:
        face_center = (default_mink1 + default_mink2 + default_mink3) / 3.0

        # Orient normal, so that the face normal points to the other side of the origin.
        orient = normal.dot(face_center)
        check_both_direction = False
        if ti.abs(orient) > gjk_static_config.diff_contact_orient_eps:
            if orient < 0.0:
                normal = -normal
        else:
            check_both_direction = True

        # Compute distance to the origin
        dist = ti.abs(normal.dot(face_center))

        # Compute affine coordinates of the origin projected onto the face.
        proj_o, _ = GJK.func_project_origin_to_plane(gjk_static_config, default_mink1, default_mink2, default_mink3)
        _lambda = GJK.func_triangle_affine_coords(
            proj_o,
            default_mink1,
            default_mink2,
            default_mink3,
        )

        min_boundary_sdist = gjk_static_config.FLOAT_MAX
        min_boundary_sdist_normal = gs.ti_vec3(0.0, 0.0, 0.0)
        min_boundary_sdist_vert_localpos1 = gs.ti_vec3(0.0, 0.0, 0.0)
        min_boundary_sdist_vert_localpos2 = gs.ti_vec3(0.0, 0.0, 0.0)

        # Find the support points in the [normal, -normal] direction
        for i in range(2):
            d = normal
            if i == 1:
                d = -normal

            obj1, obj2, id1, id2, localpos1, localpos2, mink = func_support(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
                d,
            )

            # Compute boundary signed distance in this direction
            bsdist = mink.dot(normal) - face_center.dot(normal)
            if bsdist < min_boundary_sdist:
                min_boundary_sdist = bsdist
                min_boundary_sdist_normal = d
                min_boundary_sdist_vert_localpos1 = localpos1
                min_boundary_sdist_vert_localpos2 = localpos2

            if not check_both_direction:
                break

        gjk_state.polytope_faces.default_dist[i_b, n] = dist
        gjk_state.polytope_faces.default_normal[i_b, n] = min_boundary_sdist_normal
        gjk_state.polytope_faces.default_boundary_sdist[i_b, n] = min_boundary_sdist
        gjk_state.polytope_faces.default_boundary_sdist_vert_localpos1[i_b, n] = min_boundary_sdist_vert_localpos1
        gjk_state.polytope_faces.default_boundary_sdist_vert_localpos2[i_b, n] = min_boundary_sdist_vert_localpos2
        gjk_state.polytope_faces.default_projection_affine_coords[i_b, n] = _lambda

    # Revert back to the perturbed configuration
    func_to_perturbed_config(geoms_state, gjk_state, i_ga, i_gb, i_b)

    return flag


@ti.func
def func_add_differentiable_face(
    gjk_state: array_class.GJKState,
    i_f,
    i_b,
):
    n = gjk_state.num_diff_faces[i_b]

    # Check if there is already a duplicate face.
    localpos1a = gjk_state.polytope_verts.localpos1[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][0]]
    localpos1b = gjk_state.polytope_verts.localpos1[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][1]]
    localpos1c = gjk_state.polytope_verts.localpos1[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][2]]

    id1a = gjk_state.polytope_verts.id1[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][0]]
    id1b = gjk_state.polytope_verts.id1[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][1]]
    id1c = gjk_state.polytope_verts.id1[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][2]]

    localpos2a = gjk_state.polytope_verts.localpos2[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][0]]
    localpos2b = gjk_state.polytope_verts.localpos2[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][1]]
    localpos2c = gjk_state.polytope_verts.localpos2[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][2]]

    id2a = gjk_state.polytope_verts.id2[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][0]]
    id2b = gjk_state.polytope_verts.id2[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][1]]
    id2c = gjk_state.polytope_verts.id2[i_b, gjk_state.polytope_faces.verts_idx[i_b, i_f][2]]

    unique = True
    discrete = id1a != -1 and id1b != -1 and id1c != -1 and id2a != -1 and id2b != -1 and id2c != -1

    if discrete:
        for i in range(n):
            i_id1a = gjk_state.diff_faces_vert_id1[i_b, i][0]
            i_id1b = gjk_state.diff_faces_vert_id1[i_b, i][1]
            i_id1c = gjk_state.diff_faces_vert_id1[i_b, i][2]

            i_id2a = gjk_state.diff_faces_vert_id2[i_b, i][0]
            i_id2b = gjk_state.diff_faces_vert_id2[i_b, i][1]
            i_id2c = gjk_state.diff_faces_vert_id2[i_b, i][2]

            if func_match_discrete_vertex(id1a, id1b, id1c, i_id1a, i_id1b, i_id1c) and func_match_discrete_vertex(
                id2a, id2b, id2c, i_id2a, i_id2b, i_id2c
            ):
                unique = False
                break

    # for i in range(n):

    #     i_localpos1a = gjk_state.diff_faces_vert_local_pos1[i_b, i, 0]
    #     i_localpos1b = gjk_state.diff_faces_vert_local_pos1[i_b, i, 1]
    #     i_localpos1c = gjk_state.diff_faces_vert_local_pos1[i_b, i, 2]

    #     i_localpos2a = gjk_state.diff_faces_vert_local_pos2[i_b, i, 0]
    #     i_localpos2b = gjk_state.diff_faces_vert_local_pos2[i_b, i, 1]
    #     i_localpos2c = gjk_state.diff_faces_vert_local_pos2[i_b, i, 2]

    #     if not func_match_vertex(localpos1a, i_localpos1a, i_localpos1b, i_localpos1c):
    #         continue
    #     if not func_match_vertex(localpos1b, i_localpos1a, i_localpos1b, i_localpos1c):
    #         continue
    #     if not func_match_vertex(localpos1c, i_localpos1a, i_localpos1b, i_localpos1c):
    #         continue

    #     if not func_match_vertex(localpos2a, i_localpos2a, i_localpos2b, i_localpos2c):
    #         continue
    #     if not func_match_vertex(localpos2b, i_localpos2a, i_localpos2b, i_localpos2c):
    #         continue
    #     if not func_match_vertex(localpos2c, i_localpos2a, i_localpos2b, i_localpos2c):
    #         continue

    #     # We found a duplicate face.
    #     unique = False
    #     break

    if unique:
        gjk_state.diff_faces_normal[i_b, n] = gjk_state.polytope_faces.default_normal[i_b, i_f]
        gjk_state.diff_faces_boundary_sdist_vert_localpos1[i_b, n] = (
            gjk_state.polytope_faces.default_boundary_sdist_vert_localpos1[i_b, i_f]
        )
        gjk_state.diff_faces_boundary_sdist_vert_localpos2[i_b, n] = (
            gjk_state.polytope_faces.default_boundary_sdist_vert_localpos2[i_b, i_f]
        )

        for i in range(3):
            i_v = gjk_state.polytope_faces.verts_idx[i_b, i_f][i]
            gjk_state.diff_faces_vert_id1[i_b, n][i] = gjk_state.polytope_verts.id1[i_b, i_v]
            gjk_state.diff_faces_vert_id2[i_b, n][i] = gjk_state.polytope_verts.id2[i_b, i_v]
            gjk_state.diff_faces_vert_localpos1[i_b, n, i] = gjk_state.polytope_verts.localpos1[i_b, i_v]
            gjk_state.diff_faces_vert_localpos2[i_b, n, i] = gjk_state.polytope_verts.localpos2[i_b, i_v]

        gjk_state.num_diff_faces[i_b] += 1

        # print(f"== Added differentiable face: {n}")
        # print(f"Normal: {gjk_state.diff_faces_normal[i_b, n]}")
        # print(f"Boundary sdist vert localpos1: {gjk_state.diff_faces_boundary_sdist_vert_localpos1[i_b, n]}")
        # print(f"Boundary sdist vert localpos2: {gjk_state.diff_faces_boundary_sdist_vert_localpos2[i_b, n]}")
        # print(f"Vert id1: {gjk_state.diff_faces_vert_id1[i_b, n]}")
        # print(f"Vert id2: {gjk_state.diff_faces_vert_id2[i_b, n]}")
        # print(f"Vert localpos1a: {gjk_state.diff_faces_vert_localpos1[i_b, n, 0]}")
        # print(f"Vert localpos1b: {gjk_state.diff_faces_vert_localpos1[i_b, n, 1]}")
        # print(f"Vert localpos1c: {gjk_state.diff_faces_vert_localpos1[i_b, n, 2]}")
        # print(f"Vert localpos2a: {gjk_state.diff_faces_vert_localpos2[i_b, n, 0]}")
        # print(f"Vert localpos2b: {gjk_state.diff_faces_vert_localpos2[i_b, n, 1]}")
        # print(f"Vert localpos2c: {gjk_state.diff_faces_vert_localpos2[i_b, n, 2]}")


@ti.func
def func_match_vertex(v, v1, v2, v3):
    # Compare v with v1, v2, v3 and return True if v is close to any of them.
    found = False
    eps = gs.ti_float(1e-6) ** 2
    if (v - v1).normsq() < eps:
        found = True
    if (v - v2).normsq() < eps:
        found = True
    if (v - v3).normsq() < eps:
        found = True
    return found


@ti.func
def func_match_discrete_vertex(
    from_id1,
    from_id2,
    from_id3,
    to_id1,
    to_id2,
    to_id3,
):
    match = False
    if from_id1 == to_id1 and from_id2 == to_id2 and from_id3 == to_id3:
        match = True
    elif from_id1 == to_id1 and from_id2 == to_id3 and from_id3 == to_id2:
        match = True
    elif from_id1 == to_id2 and from_id2 == to_id1 and from_id3 == to_id3:
        match = True
    elif from_id1 == to_id2 and from_id2 == to_id3 and from_id3 == to_id1:
        match = True
    elif from_id1 == to_id3 and from_id2 == to_id1 and from_id3 == to_id2:
        match = True
    elif from_id1 == to_id3 and from_id2 == to_id2 and from_id3 == to_id1:
        match = True

    return match


@ti.func
def func_to_default_config(
    geoms_state: array_class.GeomsState,
    gjk_state: array_class.GJKState,
    i_ga,
    i_gb,
    i_b,
):
    geoms_state.pos[i_ga, i_b] = gjk_state.default_pos1[i_b]
    geoms_state.quat[i_ga, i_b] = gjk_state.default_quat1[i_b]
    geoms_state.pos[i_gb, i_b] = gjk_state.default_pos2[i_b]
    geoms_state.quat[i_gb, i_b] = gjk_state.default_quat2[i_b]


@ti.func
def func_to_perturbed_config(
    geoms_state: array_class.GeomsState,
    gjk_state: array_class.GJKState,
    i_ga,
    i_gb,
    i_b,
):
    geoms_state.pos[i_ga, i_b] = gjk_state.perturbed_pos1[i_b]
    geoms_state.quat[i_ga, i_b] = gjk_state.perturbed_quat1[i_b]
    geoms_state.pos[i_gb, i_b] = gjk_state.perturbed_pos2[i_b]
    geoms_state.quat[i_gb, i_b] = gjk_state.perturbed_quat2[i_b]


@ti.func
def func_compute_minkowski_point(
    ga_pos: ti.types.vector(3),
    ga_quat: ti.types.vector(4),
    gb_pos: ti.types.vector(3),
    gb_quat: ti.types.vector(4),
    va: ti.types.vector(3),
    vb: ti.types.vector(3),
):
    # Transform the points to the global frame
    va_ = gu.ti_transform_by_trans_quat(va, ga_pos, ga_quat)
    vb_ = gu.ti_transform_by_trans_quat(vb, gb_pos, gb_quat)
    return va_ - vb_


@ti.func
def func_plane_normal(gjk_static_config: ti.template(), v1, v2, v3):
    """
    Compute the reliable normal of the plane defined by three points.
    """
    normal, flag = gs.ti_vec3(0.0, 0.0, 0.0), GJK.RETURN_CODE.SUCCESS

    tv1 = ti.Vector([v1[0], v1[1], v1[2]], dt=ti.f64)
    tv2 = ti.Vector([v2[0], v2[1], v2[2]], dt=ti.f64)
    tv3 = ti.Vector([v3[0], v3[1], v3[2]], dt=ti.f64)

    d21 = tv2 - tv1
    d31 = tv3 - tv1
    d32 = tv3 - tv2

    max_nn = ti.f64(0.0)
    max_normal = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

    for i in ti.static(range(3)):
        n = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)  # gs.ti_vec3(0.0, 0.0, 0.0)
        if i == 0:
            # Normal = (v1 - v2) x (v3 - v2)
            n = d32.cross(d21)
        elif i == 1:
            # Normal = (v2 - v1) x (v3 - v1)
            n = d21.cross(d31)
        else:
            # Normal = (v1 - v3) x (v2 - v3)
            n = d31.cross(d32)
        nn = n.norm()
        if nn == 0:
            # Zero normal, cannot project.
            flag = GJK.RETURN_CODE.FAIL
        if nn > max_nn:
            max_nn = nn
            max_normal = n

    f_max_nn = ti.cast(max_nn, gs.ti_float)
    f_max_normal = ti.cast(max_normal, gs.ti_float)

    if flag == GJK.RETURN_CODE.SUCCESS and max_nn > gjk_static_config.FLOAT_MIN:
        # Normalize the normal vector
        normal = f_max_normal.normalized()
        # print(f"nn: {max_nn:.20g}")
    else:
        flag = GJK.RETURN_CODE.FAIL

    return normal, f_max_nn, flag
