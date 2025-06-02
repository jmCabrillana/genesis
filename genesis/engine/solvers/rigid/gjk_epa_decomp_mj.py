import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu

from .support_field_decomp import SupportField
# @TODO: type checking for float, int, etc.

EPA_P2_NONCONVEX = 2
EPA_P3_BAD_NORMAL = 4
EPA_P3_INVALID_V4 = 5
EPA_P3_INVALID_V5 = 6
EPA_P3_MISSING_ORIGIN = 7
EPA_P3_ORIGIN_ON_FACE = 8
EPA_P4_MISSING_ORIGIN = 9

@ti.data_oriented
class GJKEPA:
    def __init__(self, rigid_solver):
        self._solver = rigid_solver
        self._max_contact_pairs = rigid_solver._max_collision_pairs
        self._B = rigid_solver._B
        self._para_level = rigid_solver._para_level
        
        self.FLOAT_MAX = gs.np_float(1e10)
        self.FLOAT_MAX_SQ = self.FLOAT_MAX * self.FLOAT_MAX
        self.EPS_SQ = gs.EPS
        self.EPS = np.sqrt(gs.EPS)
        
        ### Gjk
        self.gjk_max_iterations = 50
        self.gjk_k_approximate_point_obj1 = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B,))
        self.gjk_k_approximate_point_obj2 = ti.Vector.field(n=3, dtype=gs.ti_float, shape=(self._B,))
        
        struct_simplex_vertex = ti.types.struct(
            # Support points on the two objects
            support_point_obj1=gs.ti_vec3,
            support_point_obj2=gs.ti_vec3,
            # Support point on Minkowski difference
            minkowski=gs.ti_vec3
        )
        struct_simplex = ti.types.struct(
            # Number of vertices in the simplex
            nverts=gs.ti_int,
            dist=gs.ti_float,
        )
        # @TODO: data arrangement?
        self.gjk_simplex_vertex = struct_simplex_vertex.field(shape=(self._B, 4))
        self.gjk_simplex_vertex_intersect = struct_simplex_vertex.field(shape=(self._B, 4))
        self.gjk_simplex = struct_simplex.field(shape=(self._B,))
        
        ### EPA
        self.epa_max_iterations = 50
        struct_polytope_vertex = struct_simplex_vertex
        struct_polytope_face = ti.types.struct(
            # Indices of the vertices forming the face on the polytope
            verts_idx=ti.vector(3, gs.ti_int),
            # Indices of adjacent faces, one for each edge: [v1,v2], [v2,v3], [v3,v1]
            adj_idx=ti.vector(3, gs.ti_int),
            # Projection of the origin onto the face, can be used as face normal
            normal=gs.ti_vec3,
            # Square of 2-norm of the normal vector, negative means deleted face
            dist2=gs.ti_float,
            # Index of the face in the polytope map, -1 for not in the map, -2 for deleted
            map_idx=gs.ti_int
        )
        struct_polytope_horizon_data = ti.types.struct(
            # Indices of faces on horizon
            faces_idx=gs.ti_int,
            # Corresponding edge of each face on the horizon
            edges=gs.ti_int,
        )
        struct_polytope = ti.types.struct(
            # Number of vertices in the polytope
            nverts=gs.ti_int,
            # Number of faces in the polytope
            nfaces=gs.ti_int,
            # Number of edges in the horizon
            horizon_nedges=gs.ti_int,
            # Point where the horizon is created
            horizon_w=gs.ti_vec3,
        )
        self.polytope_max_faces = 6 * self.epa_max_iterations
        
        self.polytope = struct_polytope.field(shape=(self._B,))
        self.polytope_verts = struct_polytope_vertex.field(shape=(self._B, 5 + self.epa_max_iterations))
        self.polytope_faces = struct_polytope_face.field(shape=(self._B, self.polytope_max_faces))
        self.polytope_horizon_data = struct_polytope_horizon_data.field(shape=(self._B, 6 + self.epa_max_iterations))
        
        ### Support field
        self.support_field = SupportField(rigid_solver)
    
    '''
    GJK algorithms
    '''
    @ti.func
    def func_gjk(self, i_ga, i_gb, i_b):
        '''
        GJK algorithm to compute the minimum distance between two convex objects.
        '''
        # Simplex index
        n = 0
        # Final number of simplex vertices
        nsimplex = 0
        nx = 0
        dist = 0
        
        # Set initial guess of support vector using (possibly wrong) witness points
        k_approximate_point_obj1 = self.gjk_k_approximate_point_obj1[i_b]
        k_approximate_point_obj2 = self.gjk_k_approximate_point_obj2[i_b]
        support_vector = k_approximate_point_obj1 - k_approximate_point_obj2
        support_vector_norm = 0
        return_flag = 0
        
        # Since we use GJK mainly for collision detection,
        # we use gjk_intersect when it is available
        backup_gjk = 1
        
        # If we handle smooth geometry, finite convergence is not guaranteed,
        # so we need some epsilon to determine convergence and stop the algorithm
        # @TODO: set non-zero value
        epsilon = 0
        
        for i in range(self.gjk_max_iterations):
            # Compute the current support points
            support_vector_normsq = support_vector.dot(support_vector)
            if support_vector_normsq < self.EPS_SQ:
                break
            support_vector_norm = ti.math.sqrt(support_vector_normsq)
            
            # Dir to compute the support point
            # (pointing from obj1 to obj2)
            dir = -support_vector / support_vector_norm
            
            self.gjk_simplex_vertex[i_b, n].support_point_obj1, \
            self.gjk_simplex_vertex[i_b, n].support_point_obj2, \
            self.gjk_simplex_vertex[i_b, n].minkowski = \
                self.func_support(i_ga, i_gb, i_b, dir)
                
            # @TODO: Implement early stopping if epsilon > 0
            
            # Check if the objects are separated using support vector
            new_minkowski = self.gjk_simplex_vertex[i_b, n].minkowski
            is_seperated = support_vector.dot(new_minkowski) > 0.0
            if is_seperated:
                nsimplex = 0
                nx = 0
                dist = self.FLOAT_MAX
                return_flag = 1
                break
            
            # @TODO: Implement early stopping based on cutoff distance
            
            if n == 3 and backup_gjk:
                # Tetrahedron is generated, try to get contact info
                intersect_flag, intersect_iteration = \
                    self.func_gjk_intersect(i_ga, i_gb, i_b, i)
                if intersect_flag == 0:
                    # No intersection, objects are separated
                    nx = 0
                    dist = self.FLOAT_MAX
                    nsimplex = 0
                    return_flag = 1
                    break
                elif intersect_flag == 1:
                    # Intersection found
                    nx = 0
                    dist = 0
                    nsimplex = 4
                    return_flag = 1
                    break
                else:
                    # Since gjk_intersect failed (e.g. origin is on the simplex face),
                    # we fallback to minimum distance computation
                    i = intersect_iteration
                    backup_gjk = 0
                    
            # Run the distance subalgorithm to compute the barycentric
            # coordinates of the closest point to the origin in the simplex
            _lambda = self.func_gjk_subdistance(i_ga, i_gb, i_b, n + 1)
            
            # Remove vertices from the simplex with zero barycentric coordinates
            # as they are not needed for the next iteration
            n = 0
            for j in range(4):
                if _lambda[j]:
                    self.gjk_simplex_vertex[i_b, n] = self.gjk_simplex_vertex[i_b, j]
                    _lambda[n] = _lambda[j]
                    n += 1
                    
            # Should not occur
            if n < 1:
                nsimplex = 0
                nx = 0
                dist = self.FLOAT_MAX
                return_flag = 1
                break
            
            # Get the next support vector
            next_support_vector = self.func_simplex_vertex_linear_comb(
                i_b, 2, 0, 1, 2, 3, _lambda, n
            )
            
            if self.func_is_equal_vec3(next_support_vector, support_vector):
                # If the next support vector is equal to the previous one,
                # we converged to the minimum distance
                break
            
            support_vector = next_support_vector
            
            if n == 4:
                # We have a tetrahedron containing the origin,
                # so we can return early
                break
            
        if not return_flag:
            # Compute the approximate witness points (points that form the min. distance)
            self.gjk_k_approximate_point_obj1[i_b] = self.func_simplex_vertex_linear_comb(
                i_b, 0, 0, 1, 2, 3, _lambda, n
            )
            self.gjk_k_approximate_point_obj2[i_b] = self.func_simplex_vertex_linear_comb(
                i_b, 1, 0, 1, 2, 3, _lambda, n
            )
            nx = 1
            nsimplex = n
            dist = support_vector_norm
            
        return nx, nsimplex, dist
    
    @ti.func
    def func_gjk_intersect(self, i_ga, i_gb, i_b, iteration):
        
        # copy simplex to temporary storage
        self.gjk_simplex_vertex_intersect[i_b, 0] = self.gjk_simplex_vertex[i_b, 0]
        self.gjk_simplex_vertex_intersect[i_b, 1] = self.gjk_simplex_vertex[i_b, 1]
        self.gjk_simplex_vertex_intersect[i_b, 2] = self.gjk_simplex_vertex[i_b, 2]
        self.gjk_simplex_vertex_intersect[i_b, 3] = self.gjk_simplex_vertex[i_b, 3]
        
        # simplex index
        si = ti.Vector([0, 1, 2, 3], dt=gs.ti_int)
        
        final_iteration = iteration
        flag = -2
        for i in range(iteration, self.gjk_max_iterations):
            
            # Compute normal and signed distance of the triangle faces
            # of the simplex with respect to the origin
            normal_0, sdist_0 = self.func_gjk_triangle_info(i_b, si[2], si[1], si[3])
            normal_1, sdist_1 = self.func_gjk_triangle_info(i_b, si[0], si[2], si[3])
            normal_2, sdist_2 = self.func_gjk_triangle_info(i_b, si[1], si[0], si[3])
            normal_3, sdist_3 = self.func_gjk_triangle_info(i_b, si[0], si[1], si[2])
            
            # If the origin is strictly on any affine hull of the triangle faces, 
            # convergence will fail, so ignore this case
            if (
                sdist_0 == 0.0 or
                sdist_1 == 0.0 or
                sdist_2 == 0.0 or
                sdist_3 == 0.0
            ):
                final_iteration = i
                flag = -1
                break
            
            # Find the face with the smallest signed distance
            min_i0 = 0 if (sdist_0 < sdist_1) else 1
            min_i1 = 2 if (sdist_2 < sdist_3) else 3
            min_i = min_i0 if (sdist_0 < sdist_2) else min_i1
            min_si = si[min_i]
            
            min_normal = normal_0
            min_sdist = sdist_0
            if min_i == 0:
                min_normal = normal_0
                min_sdist = sdist_0
            elif min_i == 1:
                min_normal = normal_1
                min_sdist = sdist_1
            elif min_i == 2:
                min_normal = normal_2
                min_sdist = sdist_2
            else:
                min_normal = normal_3
                min_sdist = sdist_3
            
            # If origin is inside the simplex, the signed distances
            # will all be positive
            if min_sdist > 0:
                # Origin is inside the simplex, so we can stop
                final_iteration = i
                flag = 1
                
                # Copy the temporary simplex to the main simplex
                self.gjk_simplex_vertex[i_b, 0] = self.gjk_simplex_vertex_intersect[i_b, si[0]]
                self.gjk_simplex_vertex[i_b, 1] = self.gjk_simplex_vertex_intersect[i_b, si[1]]
                self.gjk_simplex_vertex[i_b, 2] = self.gjk_simplex_vertex_intersect[i_b, si[2]]
                self.gjk_simplex_vertex[i_b, 3] = self.gjk_simplex_vertex_intersect[i_b, si[3]]
                break
            
            # Replace the worst vertex (which has the smallest signed distance) 
            # with new candidate
            self.gjk_simplex_vertex_intersect[i_b, min_si].support_point_obj1, \
            self.gjk_simplex_vertex_intersect[i_b, min_si].support_point_obj2, \
            self.gjk_simplex_vertex_intersect[i_b, min_si].minkowski = \
                self.func_support(i_ga, i_gb, i_b, min_normal)
                
            # Check if the origin is strictly outside of the Minkowski difference
            # (which means there is no collision)
            new_minkowski = self.gjk_simplex_vertex_intersect[i_b, min_si].minkowski
            is_no_collision = (new_minkowski.dot(min_normal) < 0.0)
            if is_no_collision:
                final_iteration = i
                flag = 0
                break
            
            # Swap vertices in the simplex to retain orientation
            m = (min_i + 1) % 4
            n = (min_i + 2) % 4
            swap = si[m]
            si[m] = si[n]
            si[n] = swap
        
        # Never found origin
        if flag == -2:
            final_iteration = iteration
            flag = -1
        
        return flag, final_iteration
    
    @ti.func
    def func_gjk_triangle_info(self, i_b, i_va, i_vb, i_vc):
        '''
        Compute normal and signed distance of the triangle 
        face on the simplex from the origin.
        '''
        vertex_1 = self.gjk_support_point_minkowski[i_b, i_va]
        vertex_2 = self.gjk_support_point_minkowski[i_b, i_vb]
        vertex_3 = self.gjk_support_point_minkowski[i_b, i_vc]
        
        edge_1 = vertex_3 - vertex_1
        edge_2 = vertex_2 - vertex_1
        normal = edge_1.cross(edge_2)
        
        normal_normsq = normal.dot(normal)
        if (normal_normsq > self.EPS_SQ) and (normal_normsq < self.FLOAT_MAX_SQ):
            normal = normal / ti.math.sqrt(normal_normsq)
            sdist = normal.dot(vertex_1)
        else:
            # if the normal length is unstable, return max distance
            sdist = self.FLOAT_MAX
            
        return normal, sdist
    
    @ti.func
    def func_gjk_subdistance(self, i_ga, i_gb, i_b, n):
        '''
        Compute the barycentric coordinates of the 
        closest point to the origin in the simplex.
        [Montanari et al, ToG 2017]
        '''
        _lambda = ti.math.vec4(1.0, 0.0, 0.0, 0.0)
        
        if n == 4:
            _lambda = self.func_gjk_subdistance_3d(i_ga, i_gb, i_b, 0, 1, 2, 3)
        elif n == 3:
            _lambda = self.func_gjk_subdistance_2d(i_b, 0, 1, 2)
        elif n == 2:
            _lambda = self.func_gjk_subdistance_1d(i_b, 0, 1)
        
        return _lambda
    
    @ti.func
    def func_gjk_subdistance_3d(self, i_b, i_s1, i_s2, i_s3, i_s4):
        
        _lambda = ti.math.vec4()
        
        # Simplex vertices
        s1 = self.gjk_simplex_vertex[i_b, i_s1].minkowski
        s2 = self.gjk_simplex_vertex[i_b, i_s2].minkowski
        s3 = self.gjk_simplex_vertex[i_b, i_s3].minkowski
        s4 = self.gjk_simplex_vertex[i_b, i_s4].minkowski
        
        # Compute the cofactors to find det(M),
        # which corresponds to the signed volume of the tetrahedron
        C1 = -self.func_det3(s2, s3, s4)
        C2 =  self.func_det3(s1, s3, s4)
        C3 = -self.func_det3(s1, s2, s4)
        C4 =  self.func_det3(s1, s2, s3)
        m_det = C1 + C2 + C3 + C4
        
        # Compare sign of the cofactors with the determinant
        sc1 = self.func_compare_sign(C1, m_det)
        sc2 = self.func_compare_sign(C2, m_det)
        sc3 = self.func_compare_sign(C3, m_det)
        sc4 = self.func_compare_sign(C4, m_det)
        
        if (sc1 and sc2 and sc3 and sc4):
            # If all barycentric coordinates are positive,
            # the origin is inside the tetrahedron  
            _lambda[0] = C1 / m_det
            _lambda[1] = C2 / m_det
            _lambda[2] = C3 / m_det
            _lambda[3] = C4 / m_det
        else:
            # Since the origin is outside the tetrahedron,
            # we need to find the closest point to the origin
            dmin = self.FLOAT_MAX
            
            # Project origin onto faces of the tetrahedron
            # of which apex point has negative barycentric coordinate
            if not sc1:
                _lambda2d = self.func_gjk_subdistance_2d(i_b, i_s2, i_s3, i_s4)
                closest_point = self.func_simplex_vertex_linear_comb(
                    i_b, 2, i_s2, i_s3, i_s4, 0, _lambda2d, 3
                )
                d = closest_point.dot(closest_point)
                if d < dmin:
                    dmin = d
                    _lambda[0] = 0.0
                    _lambda[1] = _lambda2d[0]
                    _lambda[2] = _lambda2d[1]
                    _lambda[3] = _lambda2d[2]

            if not sc2:
                _lambda2d = self.func_gjk_subdistance_2d(i_b, i_s1, i_s3, i_s4)
                closest_point = self.func_simplex_vertex_linear_comb(
                    i_b, 2, i_s1, i_s3, i_s4, 0, _lambda2d, 3
                )
                d = closest_point.dot(closest_point)
                if d < dmin:
                    dmin = d
                    _lambda[0] = _lambda2d[0]
                    _lambda[1] = 0.0
                    _lambda[2] = _lambda2d[1]
                    _lambda[3] = _lambda2d[2]
            
            if not sc3:
                _lambda2d = self.func_gjk_subdistance_2d(i_b, i_s1, i_s2, i_s4)
                closest_point = self.func_simplex_vertex_linear_comb(
                    i_b, 2, i_s1, i_s2, i_s4, 0, _lambda2d, 3
                )
                d = closest_point.dot(closest_point)
                if d < dmin:
                    dmin = d
                    _lambda[0] = _lambda2d[0]
                    _lambda[1] = _lambda2d[1]
                    _lambda[2] = 0.0
                    _lambda[3] = _lambda2d[2]
                    
            if not sc4:
                _lambda2d = self.func_gjk_subdistance_2d(i_b, i_s1, i_s2, i_s3)
                closest_point = self.func_simplex_vertex_linear_comb(
                    i_b, 2, i_s1, i_s2, i_s3, 0, _lambda2d, 3
                )
                d = closest_point.dot(closest_point)
                if d < dmin:
                    dmin = d
                    _lambda[0] = _lambda2d[0]
                    _lambda[1] = _lambda2d[1]
                    _lambda[2] = _lambda2d[2]
                    _lambda[3] = 0.0
    
        return _lambda
    
    @ti.func
    def func_gjk_subdistance_2d(self, i_b, i_s1, i_s2, i_s3):
        
        _lambda = ti.math.vec4()
        
        # Project origin onto affine hull of the simplex (triangle)
        proj_o, proj_flag = self.func_project_origin_to_plane(
            i_b, 
            self.gjk_simplex_vertex[i_b, i_s1].minkowski,
            self.gjk_simplex_vertex[i_b, i_s2].minkowski,
            self.gjk_simplex_vertex[i_b, i_s3].minkowski,
        )
        if proj_flag:
            # If projection failed because the zero normal,
            # project on to the first edge of the triangle
            pass
        
        # We should find the barycentric coordinates of the projected point,
        # but the linear system is not square:
        # [ s1.x, s2.x, s3.x ] [ l1 ] = [ proj_o.x ]
        # [ s1.y, s2.y, s3.y ] [ l2 ] = [ proj_o.y ]
        # [ s1.z, s2.z, s3.z ] [ l3 ] = [ proj_o.z ]
        # [ 1,    1,    1,   ] [ ?  ] = [ 1.0 ]
        # So we remove one row before solving the system
        # We exclude the axis with the largest projection of the simplex
        # using the minors of the above linear system.
        s1 = self.gjk_simplex_vertex[i_b, i_s1].minkowski
        s2 = self.gjk_simplex_vertex[i_b, i_s2].minkowski
        s3 = self.gjk_simplex_vertex[i_b, i_s3].minkowski
        
        m1 = (s2[1]*s3[2]-s2[2]*s3[1]) - \
            (s1[1]*s3[2]-s1[2]*s3[1]) + \
            (s1[1]*s2[2]-s1[2]*s2[1])
        m2 = (s2[0]*s3[2]-s2[2]*s3[0]) - \
            (s1[0]*s3[2]-s1[2]*s3[0]) + \
            (s1[0]*s2[2]-s1[2]*s2[0])
        m3 = (s2[0]*s3[1]-s2[1]*s3[0]) - \
            (s1[0]*s3[1]-s1[1]*s3[0]) + \
            (s1[0]*s2[1]-s1[1]*s2[0])
            
        m_max = 0
        absm1, absm2, absm3 = ti.abs(m1), ti.abs(m2), ti.abs(m3)
        s1_2d, s2_2d, s3_2d = gs.ti_vec2(), gs.ti_vec2(), gs.ti_vec2()
        proj_o_2d = gs.ti_vec2()
        
        if absm1 >= absm2 and absm1 >= absm3:
            # Remove first row
            m_max = m1
            s1_2d[0] = s1[1]
            s1_2d[1] = s1[2]
            
            s2_2d[0] = s2[1]
            s2_2d[1] = s2[2]
            
            s3_2d[0] = s3[1]
            s3_2d[1] = s3[2]
            
            proj_o_2d[0] = proj_o[1]
            proj_o_2d[1] = proj_o[2]
        elif absm2 >= absm1 and absm2 >= absm3:
            # Remove second row
            m_max = m2
            s1_2d[0] = s1[0]
            s1_2d[1] = s1[2]
            
            s2_2d[0] = s2[0]
            s2_2d[1] = s2[2]
            
            s3_2d[0] = s3[0]
            s3_2d[1] = s3[2]
            
            proj_o_2d[0] = proj_o[0]
            proj_o_2d[1] = proj_o[2]
        else:
            # Remove third row
            m_max = m3
            s1_2d[0] = s1[0]
            s1_2d[1] = s1[1]
            
            s2_2d[0] = s2[0]
            s2_2d[1] = s2[1]
            
            s3_2d[0] = s3[0]
            s3_2d[1] = s3[1]
            
            proj_o_2d[0] = proj_o[0]
            proj_o_2d[1] = proj_o[1]
            
        # Now we find the barycentric coordinates of the projected point
        # by solving the linear system:
        # [ s1_2d.x, s2_2d.x, s3_2d.x ] [ l1 ] = [ proj_o_2d.x ]
        # [ s1_2d.y, s2_2d.y, s3_2d.y ] [ l2 ] = [ proj_o_2d.y ]
        # [ 1,       1,       1,      ] [ l3 ] = [ 1.0 ]
        
        # C1 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s2_2d, s3_2d)
        C1 = proj_o_2d[0]*s2_2d[1] + proj_o_2d[1]*s3_2d[0] + s2_2d[0]*s3_2d[1] - \
            proj_o_2d[0]*s3_2d[1] - proj_o_2d[1]*s2_2d[0] - s3_2d[0]*s2_2d[1]
            
        # C2 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s3_2d)
        C2 = proj_o_2d[0]*s3_2d[1] + proj_o_2d[1]*s1_2d[0] + s3_2d[0]*s1_2d[1] - \
            proj_o_2d[0]*s1_2d[1] - proj_o_2d[1]*s3_2d[0] - s1_2d[0]*s3_2d[1]
            
        # C3 corresponds to the signed area of 2-simplex (triangle): (proj_o_2d, s1_2d, s2_2d)
        C3 = proj_o_2d[0]*s1_2d[1] + proj_o_2d[1]*s2_2d[0] + s1_2d[0]*s2_2d[1] - \
            proj_o_2d[0]*s2_2d[1] - proj_o_2d[1]*s1_2d[0] - s2_2d[0]*s1_2d[1]
            
        # Compare sign of the cofactors with the determinant
        sc1 = self.func_compare_sign(C1, m_max)
        sc2 = self.func_compare_sign(C2, m_max)
        sc3 = self.func_compare_sign(C3, m_max)
        
        if (sc1 and sc2 and sc3):
            # If all barycentric coordinates are positive,
            # the origin is inside the 2-simplex (triangle)
            _lambda[0] = C1 / m_max
            _lambda[1] = C2 / m_max
            _lambda[2] = C3 / m_max
            _lambda[3] = 0.0
        else:
            # Since the origin is outside the 2-simplex (triangle),
            # we need to find the closest point to the origin
            dmin = self.FLOAT_MAX
            
            # Project origin onto edges of the triangle
            # of which apex point has negative barycentric coordinate
            if not sc1:
                _lambda1d = self.func_gjk_subdistance_1d(i_b, i_s2, i_s3)
                closest_point = self.func_simplex_vertex_linear_comb(
                    i_b, 2, i_s2, i_s3, 0, 0, _lambda1d, 2
                )
                d = closest_point.dot(closest_point)
                if d < dmin:
                    dmin = d
                    _lambda[0] = 0.0
                    _lambda[1] = _lambda1d[0]
                    _lambda[2] = _lambda1d[1]
                    _lambda[3] = 0.0
                    
            if not sc2:
                _lambda1d = self.func_gjk_subdistance_1d(i_b, i_s1, i_s3)
                closest_point = self.func_simplex_vertex_linear_comb(
                    i_b, 2, i_s1, i_s3, 0, 0, _lambda1d, 2
                )
                d = closest_point.dot(closest_point)
                if d < dmin:
                    dmin = d
                    _lambda[0] = _lambda1d[0]
                    _lambda[1] = 0.0
                    _lambda[2] = _lambda1d[1]
                    _lambda[3] = 0.0
            
            if not sc3:
                _lambda1d = self.func_gjk_subdistance_1d(i_b, i_s1, i_s2)
                closest_point = self.func_simplex_vertex_linear_comb(
                    i_b, 2, i_s1, i_s2, 0, 0, _lambda1d, 2
                )
                d = closest_point.dot(closest_point)
                if d < dmin:
                    dmin = d
                    _lambda[0] = _lambda1d[0]
                    _lambda[1] = _lambda1d[1]
                    _lambda[2] = 0.0
                    _lambda[3] = 0.0

        
        return _lambda
    
    @ti.func
    def func_gjk_subdistance_1d(self, i_b, i_s1, i_s2):
        
        _lambda = gs.ti_vec4()
        
        v1 = self.gjk_simplex_vertex[i_b, i_s1].minkowski
        v2 = self.gjk_simplex_vertex[i_b, i_s2].minkowski
        diff = v2 - v1
        nv = -v1.dot(diff)
        nn = diff.dot(diff)
        if nn == 0:
            # If the simplex is degenerate, just select the second vertex
            _lambda[0] = 0.0
            _lambda[1] = 1.0
            _lambda[2] = 0.0
            _lambda[3] = 0.0
        else:
            k = nv / nn
            if k < 0.0:
                _lambda[0] = 1.0
                _lambda[1] = 0.0
            elif k > 1.0:
                _lambda[0] = 0.0
                _lambda[1] = 1.0
            else:
                _lambda[0] = 1.0 - k
                _lambda[1] = k
            _lambda[2] = 0.0
            _lambda[3] = 0.0
            
        return _lambda
    
    '''
    EPA algorithms
    '''
    @ti.func
    def func_epa(self, i_ga, i_gb, i_b):
        pass
    
    @ti.func
    def func_epa_insert_vertex_to_polytope(self, i_b, obj1_point, obj2_point, minkowski_point):
        '''
        Copy vertex information into the polytope.
        '''
        n = self.polytope[i_b].nverts
        self.polytope_verts[i_b, n].support_point_obj1 = obj1_point
        self.polytope_verts[i_b, n].support_point_obj2 = obj2_point
        self.polytope_verts[i_b, n].minkowski = minkowski_point
        self.polytope[i_b].nverts += 1
        return n
    
    @ti.func
    def func_epa_init_polytope_2d(self, i_ga, i_gb, i_b):
        '''
        Create the polytope for EPA from a 1-simplex (line segment).
        Return 0 when successful.
        '''
        flag = 0
        
        # Get the simplex vertices
        v1 = self.gjk_simplex_vertex[i_b, 0].minkowski
        v2 = self.gjk_simplex_vertex[i_b, 1].minkowski
        diff = v2 - v1
        
        # Find the element in [diff] with the smallest magnitude,
        # because it will give us the largest cross product below
        min_val = ti.abs(diff[0])
        min_i = 0
        for i in range(1, 3):
            if ti.abs(diff[i]) < min_val:
                min_val = ti.abs(diff[i])
                min_i = i
                
        # Cross product with the found axis,
        # then rotate it by 120 degrees around the axis [diff]
        # to get three more points spaced 120 degrees apart
        rotmat = self.func_rotmat_120(diff)
        e = gs.ti_vec3(0.0, 0.0, 0.0)
        e[min_i] = 1.0
        d1 = e.cross(diff)
        d2 = rotmat @ d1
        d3 = rotmat @ d2
        
        # Insert the first two vertices into the polytope
        v1i = self.func_epa_insert_vertex_to_polytope(
            i_b,
            self.gjk_simplex_vertex[i_b, 0].support_point_obj1,
            self.gjk_simplex_vertex[i_b, 0].support_point_obj2,
            self.gjk_simplex_vertex[i_b, 0].minkowski
        )
        v2i = self.func_epa_insert_vertex_to_polytope(
            i_b,
            self.gjk_simplex_vertex[i_b, 1].support_point_obj1,
            self.gjk_simplex_vertex[i_b, 1].support_point_obj2,
            self.gjk_simplex_vertex[i_b, 1].minkowski
        )
        
        # Find three more vertices using [d1, d2, d3] as support vectors,
        # and insert them into the polytope
        v3i = self.func_epa_support(i_ga, i_gb, i_b, d1, d1.length())
        v4i = self.func_epa_support(i_ga, i_gb, i_b, d2, d2.length())
        v5i = self.func_epa_support(i_ga, i_gb, i_b, d3, d3.length())
        
        v3 = self.polytope_verts[i_b, v3i].minkowski
        v4 = self.polytope_verts[i_b, v4i].minkowski
        v5 = self.polytope_verts[i_b, v5i].minkowski
        
        # Build hexahedron (6 faces) from the five vertices.
        # * This hexahedron would have line [v1, v2] as the central axis,
        # and the other three vertices would be on the sides of the hexahedron,
        # as they are spaced 120 degrees apart.
        # * We already know the face and adjacent face indices in building this.
        # * While building the hexahedron by attaching faces, if the face is very
        # close to the origin, we replace the 1-simplex with the 2-simplex,
        # and restart from it.
        attach_flag = 0
        
        if self.func_attach_face_to_polytope(i_b, v1i, v3i, v4i, 1, 3, 2) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v1i, v3i, v4i)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
            
        if self.func_attach_face_to_polytope(i_b, v1i, v5i, v3i, 2, 4, 0) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v1i, v5i, v3i)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
            
        if self.func_attach_face_to_polytope(i_b, v1i, v4i, v5i, 0, 5, 1) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v1i, v4i, v5i)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
            
        if self.func_attach_face_to_polytope(i_b, v2i, v4i, v3i, 5, 0, 4) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v2i, v4i, v3i)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
            
        if self.func_attach_face_to_polytope(i_b, v2i, v3i, v5i, 3, 1, 5) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v2i, v3i, v5i)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
            
        if self.func_attach_face_to_polytope(i_b, v2i, v5i, v4i, 4, 2, 3) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v2i, v5i, v4i)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
            
        if not attach_flag:
            if not self.func_ray_triangle_intersection(v1, v2, v3, v4, v5):
                # The hexahedron should be convex by definition,
                # but somehow if it is not, we return non-convex flag
                flag = EPA_P2_NONCONVEX
            
        return flag
    
    @ti.func
    def func_epa_init_polytope_3d(self, i_ga, i_gb, i_b):
        '''
        Create the polytope for EPA from a 2-simplex (triangle).
        Return 0 when successful.
        '''
        flag = 0
        
        # Get the simplex vertices
        v1 = self.gjk_simplex_vertex[i_b, 0].minkowski
        v2 = self.gjk_simplex_vertex[i_b, 1].minkowski
        v3 = self.gjk_simplex_vertex[i_b, 2].minkowski
        
        # Get normal; if it is zero, we cannot proceed
        edge1 = v2 - v1
        edge2 = v3 - v1
        n = edge1.cross(edge2)
        n_norm = n.length()
        if n < self.EPS:
            flag = EPA_P3_BAD_NORMAL
        n_neg = -n
        
        # Save vertices in the polytope
        v1i = self.func_epa_insert_vertex_to_polytope(
            i_b,
            self.gjk_simplex_vertex[i_b, 0].support_point_obj1,
            self.gjk_simplex_vertex[i_b, 0].support_point_obj2,
            self.gjk_simplex_vertex[i_b, 0].minkowski
        )
        v2i = self.func_epa_insert_vertex_to_polytope(
            i_b,
            self.gjk_simplex_vertex[i_b, 1].support_point_obj1,
            self.gjk_simplex_vertex[i_b, 1].support_point_obj2,
            self.gjk_simplex_vertex[i_b, 1].minkowski
        )
        v3i = self.func_epa_insert_vertex_to_polytope(
            i_b,
            self.gjk_simplex_vertex[i_b, 2].support_point_obj1,
            self.gjk_simplex_vertex[i_b, 2].support_point_obj2,
            self.gjk_simplex_vertex[i_b, 2].minkowski
        )
        
        # Find the fourth and fifth vertices using the normal 
        # as the support vector. We form a hexahedron (6 faces)
        # with these five vertices.
        v5i = self.func_epa_support(i_ga, i_gb, i_b, n_neg, n_norm)
        v4i = self.func_epa_support(i_ga, i_gb, i_b, n, n_norm)
        v4 = self.polytope_verts[i_b, v4i].minkowski
        v5 = self.polytope_verts[i_b, v5i].minkowski
        
        # Check if v4 or v5 located inside the triangle.
        # If so, we do not proceed anymore.
        if self.func_point_triangle_intersection(v4, v1, v2, v3):
            flag = EPA_P3_INVALID_V4
        if self.func_point_triangle_intersection(v5, v1, v2, v3):
            flag = EPA_P3_INVALID_V5
        
        if flag == 0:
            # If origin does not lie inside the triangle, we need to
            # check if the hexahedron contains the origin.
            
            # @TODO: It's possible for GJK to return a triangle with
            # origin not contained in it but within tolerance from it.
            # In that case, the hexahedron could possibly be constructed
            # that does ont contain the origin, but there is penetration depth.
            if self.gjk_simplex[i_b].dist > 10 * self.EPS and \
                (not self.func_origin_tetra_intersection(v1, v2, v3, v4)) and \
                (not self.func_origin_tetra_intersection(v1, v2, v3, v5)):
                flag = EPA_P3_MISSING_ORIGIN
            else:
                # Build hexahedron (6 faces) from the five vertices.
                if self.func_attach_face_to_polytope(i_b, v4i, v1i, v2i, 1, 3, 2) < self.EPS_SQ:
                    flag = EPA_P3_ORIGIN_ON_FACE
                
                if self.func_attach_face_to_polytope(i_b, v4i, v3i, v1i, 2, 4, 0) < self.EPS_SQ:
                    flag = EPA_P3_ORIGIN_ON_FACE
                
                if self.func_attach_face_to_polytope(i_b, v4i, v2i, v3i, 0, 5, 1) < self.EPS_SQ:
                    flag = EPA_P3_ORIGIN_ON_FACE
                
                if self.func_attach_face_to_polytope(i_b, v5i, v2i, v1i, 5, 0, 4) < self.EPS_SQ:
                    flag = EPA_P3_ORIGIN_ON_FACE
                    
                if self.func_attach_face_to_polytope(i_b, v5i, v1i, v3i, 3, 1, 5) < self.EPS_SQ:
                    flag = EPA_P3_ORIGIN_ON_FACE
                    
                if self.func_attach_face_to_polytope(i_b, v5i, v3i, v2i, 4, 2, 3) < self.EPS_SQ:
                    flag = EPA_P3_ORIGIN_ON_FACE
    
        return flag
    
    @ti.func
    def func_epa_init_polytope_4d(self, i_ga, i_gb, i_b):
        '''
        Create the polytope for EPA from a 3-simplex (tetrahedron).
        Return 0 when successful.
        '''
        flag = 0
        
        # Insert simplex vertices into the polytope
        v1 = self.func_epa_insert_vertex_to_polytope(
            i_b,
            self.gjk_simplex_vertex[i_b, 0].support_point_obj1,
            self.gjk_simplex_vertex[i_b, 0].support_point_obj2,
            self.gjk_simplex_vertex[i_b, 0].minkowski
        )
        v2 = self.func_epa_insert_vertex_to_polytope(
            i_b,
            self.gjk_simplex_vertex[i_b, 1].support_point_obj1,
            self.gjk_simplex_vertex[i_b, 1].support_point_obj2,
            self.gjk_simplex_vertex[i_b, 1].minkowski
        )
        v3 = self.func_epa_insert_vertex_to_polytope(
            i_b,
            self.gjk_simplex_vertex[i_b, 2].support_point_obj1,
            self.gjk_simplex_vertex[i_b, 2].support_point_obj2,
            self.gjk_simplex_vertex[i_b, 2].minkowski
        )
        v4 = self.func_epa_insert_vertex_to_polytope(
            i_b,
            self.gjk_simplex_vertex[i_b, 3].support_point_obj1,
            self.gjk_simplex_vertex[i_b, 3].support_point_obj2,
            self.gjk_simplex_vertex[i_b, 3].minkowski
        )
        
        # If origin is on any face of the tetrahedron,
        # replace the simplex with a 2-simplex (triangle)
        attach_flag = 0
        if self.func_attach_face_to_polytope(i_b, v1, v2, v3, 1, 3, 2) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v1, v2, v3)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
        
        if self.func_attach_face_to_polytope(i_b, v1, v4, v2, 2, 3, 0) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v1, v4, v2)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
            
        if self.func_attach_face_to_polytope(i_b, v1, v3, v4, 0, 3, 1) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v1, v3, v4)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
            
        if self.func_attach_face_to_polytope(i_b, v4, v3, v2, 2, 0, 1) < self.EPS_SQ:
            self.func_replace_simplex_3(i_b, v4, v3, v2)
            flag = self.func_epa_init_polytope_3d(i_ga, i_gb, i_b)
            attach_flag = 1
            
        if not attach_flag:
            # If the tetrahedron does not contain the origin,
            # we do not proceed anymore.
            if not self.func_origin_tetra_intersection(
                self.polytope_verts[i_b, v1].minkowski,
                self.polytope_verts[i_b, v2].minkowski,
                self.polytope_verts[i_b, v3].minkowski,
                self.polytope_verts[i_b, v4].minkowski
            ):
                flag = EPA_P4_MISSING_ORIGIN
        
        return flag
    
    @ti.func
    def func_epa_support(self, i_ga, i_gb, i_b, dir, dir_norm):
        '''
        Find support points on the two objects using [dir].
        [dir] should be a unit vector from [ga] (obj1) to [gb] (obj2).
        After finding them, insert them into the polytope.
        '''
        d = gs.ti_vec3(1, 0, 0)
        if dir_norm > self.EPS:
            d = dir / dir_norm
            
        support_point_obj1, support_point_obj2, support_point_minkowski = \
            self.func_support(i_ga, i_gb, i_b, d)
            
        # Insert the support points into the polytope
        v_index = self.func_epa_insert_vertex_to_polytope(
            i_b, support_point_obj1, support_point_obj2, support_point_minkowski
        )
        
        return v_index
    
    @ti.func
    def func_attach_face_to_polytope(self, i_b, i_v1, i_v2, i_v3, i_a1, i_a2, i_a3):
        '''
        Attach a face to the polytope.
        [i_v1, i_v2, i_v3] are the vertices of the face,
        [i_a1, i_a2, i_a3] are the adjacent faces.
        
        Also return the squared distance of the face to the origin.
        '''
        flag = 0.0
        
        n = self.polytope[i_b].nfaces
        self.polytope_faces[i_b, n].verts_idx[0] = i_v1
        self.polytope_faces[i_b, n].verts_idx[1] = i_v2
        self.polytope_faces[i_b, n].verts_idx[2] = i_v3
        self.polytope_faces[i_b, n].adj_idx[0] = i_a1
        self.polytope_faces[i_b, n].adj_idx[1] = i_a2
        self.polytope_faces[i_b, n].adj_idx[2] = i_a3
        self.polytope[i_b].nfaces += 1
        
        # Compute the squared distance of the face to the origin
        self.polytope_faces[i_b, n].normal, ret = self.func_project_origin_to_plane(
            i_b,
            self.polytope_verts[i_b, i_v3].minkowski,
            self.polytope_verts[i_b, i_v2].minkowski,
            self.polytope_verts[i_b, i_v1].minkowski
        )
        if not ret:
            normal = self.polytope_faces[i_b, n].normal
            self.polytope_faces[i_b, n].dist2 = normal.dot(normal)
            self.polytope_faces[i_b, n].map_idx = -1  # No map index yet
            flag = self.polytope_faces[i_b, n].dist2
            
        return flag
    
    @ti.func
    def func_replace_simplex_3(self, i_b, i_v1, i_v2, i_v3):
        '''
        Replace the simplex with a 2-simplex (triangle) from polytope vertices.
        [i_v1, i_v2, i_v3] are the vertices that we will use from the polytope.
        '''
        self.gjk_simplex[i_b].nverts = 3
        self.gjk_simplex_vertex[i_b, 0] = self.polytope_verts[i_v1]
        self.gjk_simplex_vertex[i_b, 1] = self.polytope_verts[i_v2]
        self.gjk_simplex_vertex[i_b, 2] = self.polytope_verts[i_v3]
        
        # Reset polytope
        self.polytope[i_b].nverts = 0
        self.polytope[i_b].nfaces = 0
    
    @ti.func
    def func_ray_triangle_intersection(self, ray_v1, ray_v2, tri_v1, tri_v2, tri_v3):
        '''
        Check if the ray intersects the triangle.
        Return Non-Zero value if it does, otherwise return Zero.
        '''
        flag = 0
        
        ray = ray_v2 - ray_v1
        tri_ray_1 = tri_v1 - ray_v1
        tri_ray_2 = tri_v2 - ray_v1
        tri_ray_3 = tri_v3 - ray_v1
        
        # Signed volumes of the tetrahedrons formed by the ray and triangle edges
        vol_1 = self.func_det3(tri_ray_1, tri_ray_2, ray)
        vol_2 = self.func_det3(tri_ray_2, tri_ray_3, ray)
        vol_3 = self.func_det3(tri_ray_3, tri_ray_1, ray)
        
        if vol_1 >= 0 and vol_2 >= 0 and vol_3 >= 0:
            flag = 1
        elif vol_1 <= 0 and vol_2 <= 0 and vol_3 <= 0:
            flag = -1
        else:
            flag = 0
            
        return flag
    
    @ti.func
    def func_point_triangle_intersection(self, point, tri_v1, tri_v2, tri_v3):
        '''
        Check if the point is inside the triangle.
        '''
        flag = 0
        # Compute the affine coordinates of the point with respect to the triangle
        _lambda = self.func_triangle_affine_coords(point, tri_v1, tri_v2, tri_v3)
        
        # If any of the affine coordinates is negative,
        # the point is outside the triangle
        if _lambda[0] < 0 or _lambda[1] < 0 or _lambda[2] < 0:
            flag = 0
        else:
            # Check if the point predicted by the affine coordinates
            # is equal to the point itself
            pred = tri_v1 * _lambda[0] + tri_v2 * _lambda[1] + tri_v3 * _lambda[2]
            diff = pred - point
            flag = 1 if diff.dot(diff) < self.EPS_SQ else 0
        
        return flag
    
    @ti.func
    def func_triangle_affine_coords(self, point, tri_v1, tri_v2, tri_v3):
        '''
        Compute the affine coordinates of the point with respect to the triangle.
        '''
        # Compute minors of the triangle vertices
        m_1 = (tri_v2[1] * tri_v3[2] - tri_v2[2] * tri_v3[1]) - \
            (tri_v1[1] * tri_v3[2] - tri_v1[2] * tri_v3[1]) + \
            (tri_v1[1] * tri_v2[2] - tri_v1[2] * tri_v2[1])
        m_2 = (tri_v2[0] * tri_v3[2] - tri_v2[2] * tri_v3[0]) - \
            (tri_v1[0] * tri_v3[2] - tri_v1[2] * tri_v3[0]) + \
            (tri_v1[0] * tri_v2[2] - tri_v1[2] * tri_v2[0])
        m_3 = (tri_v2[0] * tri_v3[1] - tri_v2[1] * tri_v3[0]) - \
            (tri_v1[0] * tri_v3[1] - tri_v1[1] * tri_v3[0]) + \
            (tri_v1[0] * tri_v2[1] - tri_v1[1] * tri_v2[0])
            
        # Exclude one of the axes with the largest projection of the triangle
        # using the minors of the above linear system.
        m_max = 0
        absm1, absm2, absm3 = ti.abs(m_1), ti.abs(m_2), ti.abs(m_3)
        if absm1 >= absm2 and absm1 >= absm3:
            # Remove first row
            m_max = m_1
            x = 1
            y = 2
        elif absm2 >= absm1 and absm2 >= absm3:
            # Remove second row
            m_max = m_2
            x = 0
            y = 2
        else:
            # Remove third row
            m_max = m_3
            x = 0
            y = 1
        
        # C1 corresponds to the signed area of 2-simplex (triangle): (point, tri_v2, tri_v3)
        C1 = point[x] * tri_v2[y] + point[y] * tri_v3[x] + tri_v2[x] * tri_v3[y] - \
            point[x] * tri_v3[y] - point[y] * tri_v2[x] - tri_v3[x] * tri_v2[y]
        
        # C2 corresponds to the signed area of 2-simplex (triangle): (point, tri_v1, tri_v3)
        C2 = point[x] * tri_v3[y] + point[y] * tri_v1[x] + tri_v3[x] * tri_v1[y] - \
            point[x] * tri_v1[y] - point[y] * tri_v3[x] - tri_v1[x] * tri_v3[y]
            
        # C3 corresponds to the signed area of 2-simplex (triangle): (point, tri_v1, tri_v2)
        C3 = point[x] * tri_v1[y] + point[y] * tri_v2[x] + tri_v1[x] * tri_v2[y] - \
            point[x] * tri_v2[y] - point[y] * tri_v1[x] - tri_v2[x] * tri_v1[y]
            
        # Affine coordinates are computed as:
        # [ l1, l2, l3 ] = [ C1 / m_max, C2 / m_max, C3 / m_max ]
        _lambda = gs.ti_vec3()
        _lambda[0] = C1 / m_max
        _lambda[1] = C2 / m_max
        _lambda[2] = C3 / m_max
        
        return _lambda
    
    @ti.func
    def func_origin_tetra_intersection(self, tet_v1, tet_v2, tet_v3, tet_v4):
        '''
        Check if the origin is inside the tetrahedron.
        '''
        flag = self.func_point_plane_same_side(tet_v1, tet_v2, tet_v3, tet_v4) and \
                self.func_point_plane_same_side(tet_v2, tet_v3, tet_v4, tet_v1) and \
                self.func_point_plane_same_side(tet_v3, tet_v4, tet_v1, tet_v2) and \
                self.func_point_plane_same_side(tet_v4, tet_v1, tet_v2, tet_v3)
        return flag
    
    @ti.func
    def func_point_plane_same_side(self, point, plane_v1, plane_v2, plane_v3):
        '''
        Check if the point is on the same side of the plane as the origin.
        '''
        # Compute the normal of the plane
        edge1 = plane_v2 - plane_v1
        edge2 = plane_v3 - plane_v1
        normal = edge1.cross(edge2)
        
        diff1 = point - plane_v1
        dot1 = normal.dot(diff1)
        
        diff2 = -plane_v1
        dot2 = normal.dot(diff2)
        
        flag = 1 if dot1 * dot2 > 0 else 0
        return flag
    
    '''
    Helpers
    '''
    @ti.func
    def func_support(self, i_ga, i_gb, i_b, dir):
        '''
        Find support points on the two objects using [dir].
        [dir] should be a unit vector from [ga] (obj1) to [gb] (obj2).
        '''
        support_point_obj1 = self.support_driver(dir, i_ga, i_b)
        support_point_obj2 = self.support_driver(-dir, i_gb, i_b)
        support_point_minkowski = support_point_obj1 - support_point_obj2
    
        return support_point_obj1, support_point_obj2, support_point_minkowski
    
    @ti.func
    def func_rotmat_120(self, axis):
        '''
        Rotation matrix for 120 degrees rotation around the given axis.
        '''
        n = axis.length()
        u1 = axis[0] / n
        u2 = axis[1] / n
        u3 = axis[2] / n
        
        # sin and cos of 120 degrees
        sin = 0.86602540378
        cos = -0.5
        
        mat = ti.math.mat3()
        mat[0, 0] = cos + u1 * u1 * (1 - cos)
        mat[0, 1] = u1 * u2 * (1 - cos) - u3 * sin
        mat[0, 2] = u1 * u3 * (1 - cos) + u2 * sin
        mat[1, 0] = u2 * u1 * (1 - cos) + u3 * sin
        mat[1, 1] = cos + u2 * u2 * (1 - cos)
        mat[1, 2] = u2 * u3 * (1 - cos) - u1 * sin
        mat[2, 0] = u3 * u1 * (1 - cos) - u2 * sin
        mat[2, 1] = u3 * u2 * (1 - cos) + u1 * sin
        mat[2, 2] = cos + u3 * u3 * (1 - cos)
        
        return mat
    
    @ti.func
    def func_project_origin_to_plane(self, i_b, v1, v2, v3):
        '''
        Project the origin onto the plane defined by the simplex vertices.
        Find the projected point and return flag with it.
        '''
        point, flag = gs.ti_vec3(), -1
        
        
        d21 = v2 - v1
        d31 = v3 - v1
        d32 = v3 - v2
        
        # Normal = (v1 - v2) x (v3 - v2)
        n = d32.cross(d21)
        nv = n.dot(v2)
        nn = n.dot(n)
        if nn == 0:
            flag = 1
        elif nv != 0 and nn > self.EPS:
            point = n * (nv / nn)
            flag = 0
        
        if flag == -1:
            # If previous attempt was numerically unstable,
            # try use other normal estimations
            
            # Normal = (v2 - v1) x (v3 - v1)
            n = d21.cross(d31)
            nv = n.dot(v1)
            nn = n.dot(n)
            if nn == 0:
                flag = 1
            elif nv != 0 and nn > self.EPS:
                point = n * (nv / nn)
                flag = 0
        
        if flag == -1:
            # Last fallback
            
            # Normal = (v1 - v3) x (v2 - v3)
            n = d31.cross(d32)
            nv = n.dot(v3)
            nn = n.dot(n)
            point = n * (nv / nn)
            flag = 0
    
        return point, flag
    
    @ti.func
    def func_simplex_vertex_linear_comb(self, i_b, i_v, i_s1, i_s2, i_s3, i_s4, _lambda, n):
        '''
        Compute the linear combination of the simplex vertices
        
        @ i_v: Which vertex to use (0: obj1, 1: obj2, 2: minkowski)
        @ n: Number of vertices to combine, combine the first n vertices
        '''
        res = ti.math.vec3()
        
        if i_v == 0:
            s1 = self.gjk_simplex_vertex[i_b, i_s1].support_point_obj1
            s2 = self.gjk_simplex_vertex[i_b, i_s2].support_point_obj1
            s3 = self.gjk_simplex_vertex[i_b, i_s3].support_point_obj1
            s4 = self.gjk_simplex_vertex[i_b, i_s4].support_point_obj1
        elif i_v == 1:
            s1 = self.gjk_simplex_vertex[i_b, i_s1].support_point_obj2
            s2 = self.gjk_simplex_vertex[i_b, i_s2].support_point_obj2
            s3 = self.gjk_simplex_vertex[i_b, i_s3].support_point_obj2
            s4 = self.gjk_simplex_vertex[i_b, i_s4].support_point_obj2
        else:
            s1 = self.gjk_simplex_vertex[i_b, i_s1].minkowski
            s2 = self.gjk_simplex_vertex[i_b, i_s2].minkowski
            s3 = self.gjk_simplex_vertex[i_b, i_s3].minkowski
            s4 = self.gjk_simplex_vertex[i_b, i_s4].minkowski
        
        c1 = _lambda[0]
        c2 = _lambda[1]
        c3 = _lambda[2]
        c4 = _lambda[3]
        
        if n == 1:
            res = s1 * c1
        elif n == 2:
            res = s1 * c1 + s2 * c2
        elif n == 3:
            res = s1 * c1 + s2 * c2 + s3 * c3
        else:
            res = s1 * c1 + s2 * c2 + s3 * c3 + s4 * c4
        return res
    
    @ti.func
    def func_det3(self, v1, v2, v3):
        '''
        Compute the determinant of a 3x3 matrix formed by the vectors v1, v2, v3.
        M = [v1 | v2 | v3]
        '''
        return (
            v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
            v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
            v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
        )
        
    @ti.func
    def func_compare_sign(self, a, b):
        '''
        Compare the sign of two values.
        '''
        ret = 0
        if a > 0 and b > 0:
            ret = 1
        elif a < 0 and b < 0:
            ret = -1
        return ret
    
    @ti.func
    def func_is_equal_vec3(self, a, b):
        '''
        Check if two vectors are equal within a small tolerance.
        '''
        diff = (a - b).abs()
        amax = ti.max(a.abs(), b.abs())
        return ((diff < self.EPS) + (diff < amax * self.EPS)).all()
    
    
    
    @ti.func
    def gjk_support_geom(self, direction, i_g, i_b):
        support_pt = self.support_driver(direction, i_g, i_b)
        dist = ti.math.dot(support_pt, direction)
        return dist, support_pt

    def reset(self):
        pass

    @ti.func
    def support_sphere(self, direction, i_g, i_b):
        sphere_center = self._solver.geoms_state[i_g, i_b].pos
        sphere_radius = self._solver.geoms_info[i_g].data[0]
        return sphere_center + direction * sphere_radius

    @ti.func
    def support_ellipsoid(self, direction, i_g, i_b):
        g_state = self._solver.geoms_state[i_g, i_b]
        ellipsoid_center = g_state.pos
        ellipsoid_scaled_axis = ti.Vector(
            [
                self._solver.geoms_info[i_g].data[0] ** 2,
                self._solver.geoms_info[i_g].data[1] ** 2,
                self._solver.geoms_info[i_g].data[2] ** 2,
            ],
            dt=gs.ti_float,
        )
        ellipsoid_scaled_axis = gu.ti_transform_by_quat(ellipsoid_scaled_axis, g_state.quat)
        dist = ellipsoid_scaled_axis / ti.sqrt(direction.dot(1.0 / ellipsoid_scaled_axis))
        return ellipsoid_center + direction * dist

    @ti.func
    def support_capsule(self, direction, i_g, i_b):
        g_state = self._solver.geoms_state[i_g, i_b]
        capule_center = g_state.pos
        capsule_axis = gu.ti_transform_by_quat(ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float), g_state.quat)
        capule_radius = self._solver.geoms_info[i_g].data[0]
        capule_halflength = 0.5 * self._solver.geoms_info[i_g].data[1]
        capule_endpoint_side = ti.math.sign(direction.dot(capsule_axis))
        capule_endpoint = capule_center + capule_halflength * capule_endpoint_side * capsule_axis
        return capule_endpoint + direction * capule_radius

    # @ti.func
    # def support_prism(self, direction, i_g, i_b):
    #     ibest = 0
    #     best = self._solver.collider.prism[ibest, i_b].dot(direction)
    #     for i in range(1, 6):
    #         dot = self._solver.collider.prism[i, i_b].dot(direction)
    #         if dot > best:
    #             ibest = i
    #             best = dot

    #     return self._solver.collider.prism[ibest, i_b], ibest

    @ti.func
    def support_prism(self, direction, i_g, i_b):
        istart = 3
        if direction[2] < 0:
            istart = 0

        ibest = istart
        best = self._solver.collider.prism[istart, i_b].dot(direction)
        for i in range(istart + 1, istart + 3):
            dot = self._solver.collider.prism[i, i_b].dot(direction)
            if dot > best:
                ibest = i
                best = dot

        return self._solver.collider.prism[ibest, i_b], ibest

    @ti.func
    def support_box(self, direction, i_g, i_b):
        g_state = self._solver.geoms_state[i_g, i_b]
        d_box = gu.ti_transform_by_quat(direction, gu.ti_inv_quat(g_state.quat))

        vid = (d_box[0] > 0) * 4 + (d_box[1] > 0) * 2 + (d_box[2] > 0) * 1
        v_ = ti.Vector(
            [
                ti.math.sign(d_box[0]) * self._solver.geoms_info[i_g].data[0] * 0.5,
                ti.math.sign(d_box[1]) * self._solver.geoms_info[i_g].data[1] * 0.5,
                ti.math.sign(d_box[2]) * self._solver.geoms_info[i_g].data[2] * 0.5,
            ],
            dt=gs.ti_float,
        )
        vid += self._solver.geoms_info[i_g].vert_start
        v = gu.ti_transform_by_trans_quat(v_, g_state.pos, g_state.quat)
        return v, vid

    @ti.func
    def support_driver(self, direction, i_g, i_b):
        v = ti.Vector.zero(gs.ti_float, 3)
        geom_type = self._solver.geoms_info[i_g].type
        if geom_type == gs.GEOM_TYPE.SPHERE:
            v = self.support_sphere(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
            v = self.support_ellipsoid(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.CAPSULE:
            v = self.support_capsule(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.BOX:
            v, _ = self.support_box(direction, i_g, i_b)
        elif geom_type == gs.GEOM_TYPE.TERRAIN:
            if ti.static(self._solver.collider._has_terrain):
                v, _ = self.support_prism(direction, i_g, i_b)
        else:
            v, _ = self.support_field._func_support_world(direction, i_g, i_b)
        return v