from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import gstaichi as ti
import numpy as np
import numpy.typing as npt
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class

@ti.kernel
def kernel_forward_kinematics(
    f: ti.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    max_n_links_per_entity: ti.template(),
    max_n_joints_per_link: ti.template(),
    is_forward: ti.template(),
):
    # n_entities = entities_info.n_links.shape[0]
    # if ti.static(static_rigid_sim_config.use_hibernation):
    #     ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    #     for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
    #         i_e = rigid_global_info.awake_entities[i_e_, i_b]
    #         func_forward_kinematics_entity(
    #             f,
    #             i_e,
    #             i_b,
    #             links_state,
    #             links_info,
    #             joints_state,
    #             joints_info,
    #             dofs_state,
    #             dofs_info,
    #             entities_info,
    #             rigid_global_info,
    #             rigid_adjoint_cache,
    #             static_rigid_sim_config,
    #             max_n_links_per_entity,
    #             max_n_joints_per_link,
    #             is_forward,
    #         )
    # else:
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1]):
        func_forward_kinematics_entity(
            f,
            i_e,
            i_b,
            links_state,
            links_info,
            joints_state,
            joints_info,
            dofs_state,
            dofs_info,
            entities_info,
            rigid_global_info,
            rigid_adjoint_cache,
            static_rigid_sim_config,
            max_n_links_per_entity,
            max_n_joints_per_link,
            is_forward,
        )


@ti.func
def func_forward_kinematics_entity(
    f: ti.int32,
    i_e: ti.int32,
    i_b: ti.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
    max_n_links_per_entity: ti.template(),
    max_n_joints_per_link: ti.template(),
    is_forward: ti.template(),
):
    if ti.static(is_forward):
        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            func_forward_kinematics_link(
                f,
                i_l,
                i_b,
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                rigid_global_info,
                rigid_adjoint_cache,
                static_rigid_sim_config,
                max_n_joints_per_link,
                is_forward,
            )
    else:
        for i_l0 in ti.static(range(max_n_links_per_entity)):
            i_l = entities_info.link_start[i_e] + i_l0
            if i_l < entities_info.link_end[i_e]:
                func_forward_kinematics_link(
                    f,
                    i_l,
                    i_b,
                    links_state,
                    links_info,
                    joints_state,
                    joints_info,
                    dofs_state,
                    dofs_info,
                    rigid_global_info,
                    rigid_adjoint_cache,
                    static_rigid_sim_config,
                    max_n_joints_per_link,
                    is_forward,
                )

@ti.func
def func_forward_kinematics_link(
    f: ti.int32,
    i_l: ti.int32,
    i_b: ti.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
    max_n_joints_per_link: ti.template(),
    is_forward: ti.template(),
):
    I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

    pos = links_info.pos[I_l]
    quat = links_info.quat[I_l]
    if links_info.parent_idx[I_l] != -1:
        parent_pos = links_state.pos[f, links_info.parent_idx[I_l], i_b]
        parent_quat = links_state.quat[f, links_info.parent_idx[I_l], i_b]
        pos = parent_pos + gu.ti_transform_by_quat(pos, parent_quat)
        quat = gu.ti_transform_quat_by_quat(quat, parent_quat)
        
    i_j_start = links_info.joint_start[I_l]
    rigid_adjoint_cache.forward_kinematics_joint_pos_in[f, i_j_start, i_b] = pos
    rigid_adjoint_cache.forward_kinematics_joint_quat_in[f, i_j_start, i_b] = quat

    if ti.static(is_forward):
        for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
            func_forward_kinematics_joint(
                f,
                i_j,
                i_b,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                rigid_global_info,
                rigid_adjoint_cache,
                static_rigid_sim_config,
            )
    else:
        for i_j0 in ti.static(range(max_n_joints_per_link)):
            i_j = links_info.joint_start[I_l] + i_j0
            if i_j < links_info.joint_end[I_l]:
                func_forward_kinematics_joint(
                    f,
                    i_j,
                    i_b,
                    joints_state,
                    joints_info,
                    dofs_state,
                    dofs_info,
                    rigid_global_info,
                    rigid_adjoint_cache,
                    static_rigid_sim_config,
                )

    # Skip link pose update for fixed root links to let users manually overwrite them
    if not (links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]):
        i_j_end = links_info.joint_end[I_l] - 1
        links_state.pos[f, i_l, i_b] = rigid_adjoint_cache.forward_kinematics_joint_pos_out[f, i_j_end, i_b]
        links_state.quat[f, i_l, i_b] = rigid_adjoint_cache.forward_kinematics_joint_quat_out[f, i_j_end, i_b]

@ti.func
def func_forward_kinematics_joint(
    f: ti.int32,
    i_j: ti.int32,
    i_b: ti.int32,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    pos = rigid_adjoint_cache.forward_kinematics_joint_pos_in[f, i_j, i_b]
    quat = rigid_adjoint_cache.forward_kinematics_joint_quat_in[f, i_j, i_b]
    
    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
    joint_type = joints_info.type[I_j]
    q_start = joints_info.q_start[I_j]
    dof_start = joints_info.dof_start[I_j]
    I_d = [dof_start, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else dof_start

    # compute axis and anchor
    if joint_type == gs.JOINT_TYPE.FREE:
        joints_state.xanchor[f, i_j, i_b] = ti.Vector(
            [
                rigid_global_info.qpos[f, q_start, i_b],
                rigid_global_info.qpos[f, q_start + 1, i_b],
                rigid_global_info.qpos[f, q_start + 2, i_b],
            ]
        )
        joints_state.xaxis[f, i_j, i_b] = ti.Vector([0.0, 0.0, 1.0])
    elif joint_type == gs.JOINT_TYPE.FIXED:
        pass
    else:
        axis = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
        if joint_type == gs.JOINT_TYPE.REVOLUTE:
            axis = dofs_info.motion_ang[I_d]
        elif joint_type == gs.JOINT_TYPE.PRISMATIC:
            axis = dofs_info.motion_vel[I_d]

        joints_state.xanchor[f, i_j, i_b] = gu.ti_transform_by_quat(joints_info.pos[I_j], quat) + pos
        joints_state.xaxis[f, i_j, i_b] = gu.ti_transform_by_quat(axis, quat)
        
    n_pos = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    n_quat = ti.Vector([0.0, 0.0, 0.0, 1.0], dt=gs.ti_float)

    if joint_type == gs.JOINT_TYPE.FREE:
        n_pos = ti.Vector(
            [
                rigid_global_info.qpos[f, q_start, i_b],
                rigid_global_info.qpos[f, q_start + 1, i_b],
                rigid_global_info.qpos[f, q_start + 2, i_b],
            ],
            dt=gs.ti_float,
        )
        n_quat = ti.Vector(
            [
                rigid_global_info.qpos[f, q_start + 3, i_b],
                rigid_global_info.qpos[f, q_start + 4, i_b],
                rigid_global_info.qpos[f, q_start + 5, i_b],
                rigid_global_info.qpos[f, q_start + 6, i_b],
            ],
            dt=gs.ti_float,
        )
        xyz = gu.ti_quat_to_xyz(n_quat)
        for i in ti.static(range(3)):
            dofs_state.pos[f, dof_start + i, i_b] = n_pos[i]
            dofs_state.pos[f, dof_start + 3 + i, i_b] = xyz[i]
    elif joint_type == gs.JOINT_TYPE.FIXED:
        pass
    elif joint_type == gs.JOINT_TYPE.SPHERICAL:
        qloc = ti.Vector(
            [
                rigid_global_info.qpos[f, q_start, i_b],
                rigid_global_info.qpos[f, q_start + 1, i_b],
                rigid_global_info.qpos[f, q_start + 2, i_b],
                rigid_global_info.qpos[f, q_start + 3, i_b],
            ],
            dt=gs.ti_float,
        )
        xyz = gu.ti_quat_to_xyz(qloc)
        for i in ti.static(range(3)):
            dofs_state.pos[f, dof_start + i, i_b] = xyz[i]
        n_quat = gu.ti_transform_quat_by_quat(qloc, quat)
        n_pos = joints_state.xanchor[f, i_j, i_b] - gu.ti_transform_by_quat(joints_info.pos[I_j], n_quat)
    elif joint_type == gs.JOINT_TYPE.REVOLUTE:
        axis = dofs_info.motion_ang[I_d]
        dofs_state.pos[f, dof_start, i_b] = (
            rigid_global_info.qpos[f, q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
        )
        qloc = gu.ti_rotvec_to_quat(axis * dofs_state.pos[f, dof_start, i_b])
        n_quat = gu.ti_transform_quat_by_quat(qloc, quat)
        n_pos = joints_state.xanchor[f, i_j, i_b] - gu.ti_transform_by_quat(joints_info.pos[I_j], n_quat)
    else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
        dofs_state.pos[f, dof_start, i_b] = (
            rigid_global_info.qpos[f, q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
        )
        n_pos = pos + joints_state.xaxis[f, i_j, i_b] * dofs_state.pos[f, dof_start, i_b]

    rigid_adjoint_cache.forward_kinematics_joint_pos_out[f, i_j, i_b] = n_pos
    rigid_adjoint_cache.forward_kinematics_joint_quat_out[f, i_j, i_b] = n_quat



@ti.kernel
def func_forward_kinematics(
    f: ti.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    max_n_links_per_entity: ti.template(),
    max_n_joints_per_link: ti.template(),
):
    # TODO: Consider hibernation
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[2]):
        
        for i_l0 in ti.static(range(max_n_links_per_entity)):
            i_l = entities_info.link_start[i_e] + i_l0
        
            if i_l < entities_info.link_end[i_e]:
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                
                # Link's local translation and rotation wrt parent
                pos = links_info.pos[I_l]                   
                quat = links_info.quat[I_l]                 
                
                # Promote local pose to world pose
                if links_info.parent_idx[I_l] != -1:         
                    parent_pos = links_state.pos[f, links_info.parent_idx[I_l], i_b]
                    parent_quat = links_state.quat[f, links_info.parent_idx[I_l], i_b]
                    pos = parent_pos + gu.ti_transform_by_quat(pos, parent_quat)
                    quat = gu.ti_transform_quat_by_quat(quat, parent_quat)
                    
                # Handle joints that have this link as child
                for i_j0 in ti.static(range(max_n_joints_per_link)):
                    i_j = links_info.joint_start[I_l] + i_j0
                    
                    if i_j < links_info.joint_end[I_l]:
                        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                        joint_type = joints_info.type[I_j]
                        q_start = joints_info.q_start[I_j]
                        dof_start = joints_info.dof_start[I_j]
                        I_d = [dof_start, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else dof_start
                        
                        # Compute axis and anchor
                        if joint_type == gs.JOINT_TYPE.FREE:
                            joints_state.xanchor[f, i_j, i_b] = ti.Vector(
                                [
                                    rigid_global_info.qpos[f, q_start, i_b],
                                    rigid_global_info.qpos[f, q_start + 1, i_b],
                                    rigid_global_info.qpos[f, q_start + 2, i_b],
                                ]
                            )
                            joints_state.xaxis[f, i_j, i_b] = ti.Vector([0.0, 0.0, 1.0])
                        elif joint_type == gs.JOINT_TYPE.FIXED:
                            pass
                        else:
                            axis = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
                            if joint_type == gs.JOINT_TYPE.REVOLUTE:
                                axis = dofs_info.motion_ang[I_d]
                            elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                                axis = dofs_info.motion_vel[I_d]

                            joints_state.xanchor[f, i_j, i_b] = gu.ti_transform_by_quat(joints_info.pos[I_j], quat) + pos
                            joints_state.xaxis[f, i_j, i_b] = gu.ti_transform_by_quat(axis, quat)

                        # Compute DOFs pos
                        if joint_type == gs.JOINT_TYPE.FREE:
                            pos = ti.Vector(
                                [
                                    rigid_global_info.qpos[f, q_start, i_b],
                                    rigid_global_info.qpos[f, q_start + 1, i_b],
                                    rigid_global_info.qpos[f, q_start + 2, i_b],
                                ],
                                dt=gs.ti_float,
                            )
                            quat = ti.Vector(
                                [
                                    rigid_global_info.qpos[f, q_start + 3, i_b],
                                    rigid_global_info.qpos[f, q_start + 4, i_b],
                                    rigid_global_info.qpos[f, q_start + 5, i_b],
                                    rigid_global_info.qpos[f, q_start + 6, i_b],
                                ],
                                dt=gs.ti_float,
                            )
                            xyz = gu.ti_quat_to_xyz(quat)
                            for i in ti.static(range(3)):
                                dofs_state.pos[f, dof_start + i, i_b] = pos[i]
                                dofs_state.pos[f, dof_start + 3 + i, i_b] = xyz[i]
                        elif joint_type == gs.JOINT_TYPE.FIXED:
                            pass
                        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                            qloc = ti.Vector(
                                [
                                    rigid_global_info.qpos[f, q_start, i_b],
                                    rigid_global_info.qpos[f, q_start + 1, i_b],
                                    rigid_global_info.qpos[f, q_start + 2, i_b],
                                    rigid_global_info.qpos[f, q_start + 3, i_b],
                                ],
                                dt=gs.ti_float,
                            )
                            xyz = gu.ti_quat_to_xyz(qloc)
                            for i in ti.static(range(3)):
                                dofs_state.pos[f, dof_start + i, i_b] = xyz[i]
                            quat = gu.ti_transform_quat_by_quat(qloc, quat)
                            pos = joints_state.xanchor[f, i_j, i_b] - gu.ti_transform_by_quat(joints_info.pos[I_j], quat)
                        elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                            axis = dofs_info.motion_ang[I_d]
                            dofs_state.pos[f, dof_start, i_b] = (
                                rigid_global_info.qpos[f, q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                            )
                            qloc = gu.ti_rotvec_to_quat(axis * dofs_state.pos[f, dof_start, i_b])
                            quat = gu.ti_transform_quat_by_quat(qloc, quat)
                            pos = joints_state.xanchor[f, i_j, i_b] - gu.ti_transform_by_quat(joints_info.pos[I_j], quat)
                        else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
                            dofs_state.pos[f, dof_start, i_b] = (
                                rigid_global_info.qpos[f, q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                            )
                            pos = pos + joints_state.xaxis[f, i_j, i_b] * dofs_state.pos[f, dof_start, i_b]
        
                # If this link is the root link and is fixed, skip the update
                if not (links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]):
                    links_state.pos[f, i_l, i_b] = pos
                    links_state.quat[f, i_l, i_b] = quat
                            
@ti.kernel
def func_COM_links(
    f: ti.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    static_rigid_sim_config: ti.template(),
    rigid_adjoint_cache: array_class.RigidAdjointCache,
):
    # TODO: Consider hibernation
    
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(links_state.pos.shape[1], links_state.pos.shape[2]):
        links_state.root_COM[f, i_l, i_b].fill(0.0)
        links_state.mass_sum[f, i_l, i_b] = 0.0
    
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(links_state.pos.shape[1], links_state.pos.shape[2]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        mass = links_info.inertial_mass[I_l] + links_state.mass_shift[f, i_l, i_b]
        # Since Taichi's autodiff system does not allow overwriting already read variables, we write to cache
        (
            rigid_adjoint_cache.i_pos[f, i_l, i_b],
            links_state.i_quat[f, i_l, i_b],
        ) = gu.ti_transform_pos_quat_by_trans_quat(
            links_info.inertial_pos[I_l] + links_state.i_pos_shift[f, i_l, i_b],
            links_info.inertial_quat[I_l],
            links_state.pos[f, i_l, i_b],
            links_state.quat[f, i_l, i_b],
        )

        i_r = links_info.root_idx[I_l]
        links_state.mass_sum[f, i_r, i_b] += mass
        rigid_adjoint_cache.root_COM[f, i_r, i_b] += mass * rigid_adjoint_cache.i_pos[f, i_l, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(links_state.pos.shape[1], links_state.pos.shape[2]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        i_r = links_info.root_idx[I_l]
        if i_l == i_r:
            if links_state.mass_sum[f, i_l, i_b] > 0.0:
                links_state.root_COM[f, i_l, i_b] = rigid_adjoint_cache.root_COM[f, i_l, i_b] / links_state.mass_sum[f, i_l, i_b]
                
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(links_state.pos.shape[1], links_state.pos.shape[2]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        i_r = links_info.root_idx[I_l]
        if i_l != i_r:
            links_state.root_COM[f, i_l, i_b] = links_state.root_COM[f, i_r, i_b]
            
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(links_state.pos.shape[1], links_state.pos.shape[2]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        i_r = links_info.root_idx[I_l]
        links_state.i_pos[f, i_l, i_b] = rigid_adjoint_cache.i_pos[f, i_l, i_b] - links_state.root_COM[f, i_l, i_b]

        i_inertial = links_info.inertial_i[I_l]
        i_mass = links_info.inertial_mass[I_l] + links_state.mass_shift[f, i_l, i_b]
        (
            links_state.cinr_inertial[f, i_l, i_b],
            links_state.cinr_pos[f, i_l, i_b],
            links_state.cinr_quat[f, i_l, i_b],
            links_state.cinr_mass[f, i_l, i_b],
        ) = gu.ti_transform_inertia_by_trans_quat(
            i_inertial, i_mass, links_state.i_pos[f, i_l, i_b], links_state.i_quat[f, i_l, i_b]
        )
        
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(links_state.pos.shape[1], links_state.pos.shape[2]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
        
        if links_info.n_dofs[I_l] > 0:
            i_p = links_info.parent_idx[I_l]

            _i_j = links_info.joint_start[I_l]
            _I_j = [_i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else _i_j
            joint_type = joints_info.type[_I_j]

            p_pos = ti.Vector.zero(gs.ti_float, 3)
            p_quat = gu.ti_identity_quat()
            if i_p != -1:
                p_pos = links_state.pos[f, i_p, i_b]
                p_quat = links_state.quat[f, i_p, i_b]

            if joint_type == gs.JOINT_TYPE.FREE or (links_info.is_fixed[I_l] and i_p == -1):
                links_state.j_pos[f, i_l, i_b] = links_state.pos[f, i_l, i_b]
                links_state.j_quat[f, i_l, i_b] = links_state.quat[f, i_l, i_b]
            else:
                (
                    rigid_adjoint_cache.j_pos[f, i_l, i_b],
                    rigid_adjoint_cache.j_quat[f, i_l, i_b],
                ) = gu.ti_transform_pos_quat_by_trans_quat(links_info.pos[I_l], links_info.quat[I_l], p_pos, p_quat)

                # Use last joint info
                i_j = links_info.joint_end[I_l] - 1
                I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                
                (
                    links_state.j_pos[f, i_l, i_b],
                    links_state.j_quat[f, i_l, i_b],
                ) = gu.ti_transform_pos_quat_by_trans_quat(
                    joints_info.pos[I_j],
                    gu.ti_identity_quat(),
                    rigid_adjoint_cache.j_pos[f, i_l, i_b],
                    rigid_adjoint_cache.j_quat[f, i_l, i_b],
                )

@ti.kernel
def func_forward_velocity(
    f: ti.int32,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    max_n_links_per_entity: ti.template(),
    max_n_joints_per_link: ti.template(),
    max_n_dofs_per_joint: ti.template(),
):
    # TODO: Consider hibernation
    for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[2]):
        for i_l0 in ti.static(range(max_n_links_per_entity)):
            i_l = entities_info.link_start[i_e] + i_l0
        
            if i_l < entities_info.link_end[i_e]:
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                cvel_vel = ti.Vector.zero(gs.ti_float, 3)
                cvel_ang = ti.Vector.zero(gs.ti_float, 3)
                if links_info.parent_idx[I_l] != -1:
                    cvel_vel = links_state.cd_vel[f, links_info.parent_idx[I_l], i_b]
                    cvel_ang = links_state.cd_ang[f, links_info.parent_idx[I_l], i_b]

                for i_j0 in ti.static(range(max_n_joints_per_link)):
                    i_j = links_info.joint_start[I_l] + i_j0
                    
                    if i_j < links_info.joint_end[I_l]:
                        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                        joint_type = joints_info.type[I_j]
                        q_start = joints_info.q_start[I_j]
                        dof_start = joints_info.dof_start[I_j]

                        if joint_type == gs.JOINT_TYPE.FREE:
                            for i_3 in ti.static(range(3)):
                                cvel_vel += (
                                    dofs_state.cdof_vel[f, dof_start + i_3, i_b] * dofs_state.vel[f, dof_start + i_3, i_b]
                                )
                                cvel_ang += (
                                    dofs_state.cdof_ang[f, dof_start + i_3, i_b] * dofs_state.vel[f, dof_start + i_3, i_b]
                                )

                            for i_3 in ti.static(range(3)):
                                (
                                    dofs_state.cdofd_ang[f, dof_start + i_3, i_b],
                                    dofs_state.cdofd_vel[f, dof_start + i_3, i_b],
                                ) = ti.Vector.zero(gs.ti_float, 3), ti.Vector.zero(gs.ti_float, 3)

                                (
                                    dofs_state.cdofd_ang[f, dof_start + i_3 + 3, i_b],
                                    dofs_state.cdofd_vel[f, dof_start + i_3 + 3, i_b],
                                ) = gu.motion_cross_motion(
                                    cvel_ang,
                                    cvel_vel,
                                    dofs_state.cdof_ang[f, dof_start + i_3 + 3, i_b],
                                    dofs_state.cdof_vel[f, dof_start + i_3 + 3, i_b],
                                )

                            for i_3 in ti.static(range(3)):
                                cvel_vel += (
                                    dofs_state.cdof_vel[f, dof_start + i_3 + 3, i_b] * dofs_state.vel[f, dof_start + i_3 + 3, i_b]
                                )
                                cvel_ang += (
                                    dofs_state.cdof_ang[f, dof_start + i_3 + 3, i_b] * dofs_state.vel[f, dof_start + i_3 + 3, i_b]
                                )

                        else:
                            for i_d0 in ti.static(range(max_n_dofs_per_joint)):
                                i_d = dof_start + i_d0
                                
                                if i_d < joints_info.dof_end[I_j]:
                                    dofs_state.cdofd_ang[f, i_d, i_b], dofs_state.cdofd_vel[f, i_d, i_b] = gu.motion_cross_motion(
                                        cvel_ang,
                                        cvel_vel,
                                        dofs_state.cdof_ang[f, i_d, i_b],
                                        dofs_state.cdof_vel[f, i_d, i_b],
                                    )
                                    
                            for i_d0 in ti.static(range(max_n_dofs_per_joint)):
                                i_d = dof_start + i_d0
                                
                                if i_d < joints_info.dof_end[I_j]:
                                    cvel_vel += dofs_state.cdof_vel[f, i_d, i_b] * dofs_state.vel[f, i_d, i_b]
                                    cvel_ang += dofs_state.cdof_ang[f, i_d, i_b] * dofs_state.vel[f, i_d, i_b]

                links_state.cd_vel[f, i_l, i_b] = cvel_vel
                links_state.cd_ang[f, i_l, i_b] = cvel_ang
                
@ti.kernel
def func_update_geoms(
    f: ti.int32,
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    # TODO: Consider hibernation
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_g, i_b in ti.ndrange(geoms_info.pos.shape[0], geoms_state.pos.shape[2]):
        (
            geoms_state.pos[f, i_g, i_b],
            geoms_state.quat[f, i_g, i_b],
        ) = gu.ti_transform_pos_quat_by_trans_quat(
            geoms_info.pos[i_g],
            geoms_info.quat[i_g],
            links_state.pos[f, geoms_info.link_idx[i_g], i_b],
            links_state.quat[f, geoms_info.link_idx[i_g], i_b],
        )

        geoms_state.verts_updated[f, i_g, i_b] = 0

@ti.kernel
def func_compute_mass_matrix(
    f: ti.i32,
    implicit_damping: ti.template(),
    # taichi variables
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    max_n_links_per_entity: ti.template(),
    max_n_dofs_per_link: ti.template(),
    max_n_dofs_per_entity: ti.template(),
):
    # crb initialize
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(links_state.pos.shape[1], links_state.pos.shape[2]):
        links_state.crb_inertial[f, i_l, i_b] = links_state.cinr_inertial[f, i_l, i_b]
        links_state.crb_pos[f, i_l, i_b] = links_state.cinr_pos[f, i_l, i_b]
        links_state.crb_quat[f, i_l, i_b] = links_state.cinr_quat[f, i_l, i_b]
        links_state.crb_mass[f, i_l, i_b] = links_state.cinr_mass[f, i_l, i_b]

    # crb
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[2]):
        for i_l0 in ti.static(range(max_n_links_per_entity)):
            i_l = entities_info.link_start[i_e] + i_l0
            
            if i_l < entities_info.link_end[i_e]:
                i_l = entities_info.link_end[i_e] - 1 - i_l0
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                i_p = links_info.parent_idx[I_l]

                if i_p != -1:
                    ti.atomic_add(links_state.crb_inertial[f, i_p, i_b], links_state.crb_inertial[f, i_l, i_b])
                    ti.atomic_add(links_state.crb_mass[f, i_p, i_b], links_state.crb_mass[f, i_l, i_b])
                    ti.atomic_add(links_state.crb_pos[f, i_p, i_b], links_state.crb_pos[f, i_l, i_b])
                    ti.atomic_add(links_state.crb_quat[f, i_p, i_b], links_state.crb_quat[f, i_l, i_b])

    # mass_mat
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(links_state.pos.shape[1], links_state.pos.shape[2]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
        
        for i_d0 in ti.static(range(max_n_dofs_per_link)):
            i_d = links_info.dof_start[I_l] + i_d0
            
            if i_d < links_info.dof_end[I_l]:
                dofs_state.f_ang[f, i_d, i_b], dofs_state.f_vel[f, i_d, i_b] = gu.inertial_mul(
                    links_state.crb_pos[f, i_l, i_b],
                    links_state.crb_inertial[f, i_l, i_b],
                    links_state.crb_mass[f, i_l, i_b],
                    dofs_state.cdof_vel[f, i_d, i_b],
                    dofs_state.cdof_ang[f, i_d, i_b],
                )

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_e, i_b, i_d0, j_d0 in ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[2], max_n_dofs_per_entity, max_n_dofs_per_entity):
        i_d = entities_info.dof_start[i_e] + i_d0
        j_d = entities_info.dof_start[i_e] + j_d0
        
        if i_d < entities_info.dof_end[i_e] and j_d < entities_info.dof_end[i_e]:
            rigid_global_info.mass_mat[f, i_d, j_d, i_b] = (
                dofs_state.f_ang[f, i_d, i_b].dot(dofs_state.cdof_ang[f, j_d, i_b])
                + dofs_state.f_vel[f, i_d, i_b].dot(dofs_state.cdof_vel[f, j_d, i_b])
            ) * rigid_global_info.mass_parent_mask[i_d, j_d]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_e, i_b, i_d0, j_d0 in ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[2], max_n_dofs_per_entity, max_n_dofs_per_entity):
        i_d = entities_info.dof_start[i_e] + i_d0
        j_d = entities_info.dof_start[i_e] + j_d0
        
        if i_d < entities_info.dof_end[i_e] and j_d < entities_info.dof_end[i_e] and j_d > i_d:
            rigid_global_info.mass_mat[f, i_d, j_d, i_b] = rigid_global_info.mass_mat[f, j_d, i_d, i_b]

    # Take into account motor armature
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(dofs_state.f_ang.shape[1], links_state.pos.shape[2]):
        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
        rigid_global_info.mass_mat[f, i_d, i_d, i_b] += dofs_info.armature[I_d]
        
    # Take into account first-order correction terms for implicit integration scheme right away
    if ti.static(implicit_damping):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(dofs_state.f_ang.shape[1], links_state.pos.shape[2]):
            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
            rigid_global_info.mass_mat[f, i_d, i_d, i_b] += dofs_info.damping[I_d] * static_rigid_sim_config.substep_dt
            if (dofs_state.ctrl_mode[f, i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                dofs_state.ctrl_mode[f, i_d, i_b] == gs.CTRL_MODE.VELOCITY
            ):
                # qM += d qfrc_actuator / d qvel
                rigid_global_info.mass_mat[f, i_d, i_d, i_b] += dofs_info.kv[I_d] * static_rigid_sim_config.substep_dt

@ti.kernel
def func_factor_mass(
    f: ti.i32,
    implicit_damping: ti.template(),
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    max_n_dofs_per_entity: ti.template(),
    rigid_adjoint_cache: array_class.RigidAdjointCache,
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[2]):
        if rigid_global_info._mass_mat_mask[f, i_e, i_b] == 1:
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            n_dofs = entities_info.n_dofs[i_e]
            
            for i_d0 in ti.static(range(max_n_dofs_per_entity)):
                if i_d0 < n_dofs:
                    i_d = entity_dof_start + i_d0
                    for j_d in range(entity_dof_start, i_d + 1):
                        rigid_adjoint_cache.mass_mat_L0[f, i_d, j_d, i_b] = rigid_global_info.mass_mat[f, i_d, j_d, i_b]

                    if ti.static(implicit_damping):
                        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                        rigid_adjoint_cache.mass_mat_L0[f, i_d, i_d, i_b] += (
                            dofs_info.damping[I_d] * static_rigid_sim_config.substep_dt
                        )
                        if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                            if (dofs_state.ctrl_mode[f, i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                                dofs_state.ctrl_mode[f, i_d, i_b] == gs.CTRL_MODE.VELOCITY
                            ):
                                rigid_adjoint_cache.mass_mat_L0[f, i_d, i_d, i_b] += (
                                    dofs_info.kv[I_d] * static_rigid_sim_config.substep_dt
                                )

            # Cholesky-Banachiewicz algorithm for autodiff (compute L of LL^T)
            # https://en.wikipedia.org/wiki/Cholesky_decomposition
            for i_d0 in ti.static(range(max_n_dofs_per_entity)):
                for i_d1 in ti.static(range(max_n_dofs_per_entity)):
                    if i_d0 < n_dofs and i_d1 < n_dofs and i_d1 <= i_d0:
                        # Will finalize [f, i_d, j_d, i_b] of mass_mat_L1
                        i_d = entity_dof_start + i_d0
                        j_d = entity_dof_start + i_d1
                        
                        sum = 0.0
                        for i_d2 in ti.static(range(max_n_dofs_per_entity)):
                            if i_d2 < i_d1:
                                # Will read [f, i_d, k_d, i_b] and [f, j_d, k_d, i_b] of mass_mat_L1, which are already
                                # finalized in the previous iterations (k_d < j_d, j_d < i_d), so safe for autodiff
                                k_d = entity_dof_start + i_d2
                                sum += rigid_adjoint_cache.mass_mat_L1[f, i_d, k_d, i_b] * \
                                    rigid_adjoint_cache.mass_mat_L1[f, j_d, k_d, i_b]
                                    
                        if i_d == j_d:
                            rigid_adjoint_cache.mass_mat_L1[f, i_d, j_d, i_b] = \
                                ti.sqrt(rigid_adjoint_cache.mass_mat_L0[f, i_d, j_d, i_b] - sum)
                        else:
                            # It's safe to read [f, j_d, j_d, i_b] of mass_mat_L1, because j_d < i_d
                            rigid_adjoint_cache.mass_mat_L1[f, i_d, j_d, i_b] = \
                                (1.0 / rigid_adjoint_cache.mass_mat_L1[f, j_d, j_d, i_b]) * \
                                (rigid_adjoint_cache.mass_mat_L0[f, i_d, j_d, i_b] - sum)
            
            # Convert to LDL^T
            for i_d0 in ti.static(range(max_n_dofs_per_entity)):
                for i_d1 in ti.static(range(max_n_dofs_per_entity)):
                    if i_d0 < n_dofs and i_d1 < n_dofs and i_d1 <= i_d0:
                        i_d = entity_dof_start + i_d0
                        j_d = entity_dof_start + i_d1
                        
                        rigid_global_info.mass_mat_L[f, i_d, j_d, i_b] = \
                            rigid_adjoint_cache.mass_mat_L1[f, i_d, j_d, i_b] / rigid_adjoint_cache.mass_mat_L1[f, j_d, j_d, i_b]
                            
                        if i_d == j_d:
                            rigid_global_info.mass_mat_D_inv[f, i_d, i_b] = 1.0 / (rigid_adjoint_cache.mass_mat_L1[f, i_d, i_d, i_b] ** 2)

@ti.kernel
def func_torque_and_passive_force(
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
):
    n_entities = entities_info.n_links.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]
    n_dofs = dofs_state.ctrl_mode.shape[0]
    n_links = links_info.root_idx.shape[0]

    # compute force based on each dof's ctrl mode
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_e, i_b in ti.ndrange(n_entities, _B):
        wakeup = False
        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

            i_j = links_info.joint_start[I_l]
            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]

            for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                force = gs.ti_float(0.0)
                if dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
                    force = dofs_state.ctrl_force[i_d, i_b]
                elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
                    force = dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
                elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION and not (
                    joint_type == gs.JOINT_TYPE.FREE and i_d >= links_info.dof_start[I_l] + 3
                ):
                    force = (
                        dofs_info.kp[I_d] * (dofs_state.ctrl_pos[i_d, i_b] - dofs_state.pos[i_d, i_b])
                        - dofs_info.kv[I_d] * dofs_state.vel[i_d, i_b]
                    )

                dofs_state.qf_applied[i_d, i_b] = ti.math.clamp(
                    force,
                    dofs_info.force_range[I_d][0],
                    dofs_info.force_range[I_d][1],
                )

                if ti.abs(force) > gs.EPS:
                    wakeup = True

            dof_start = links_info.dof_start[I_l]
            if joint_type == gs.JOINT_TYPE.FREE and (
                dofs_state.ctrl_mode[dof_start + 3, i_b] == gs.CTRL_MODE.POSITION
                or dofs_state.ctrl_mode[dof_start + 4, i_b] == gs.CTRL_MODE.POSITION
                or dofs_state.ctrl_mode[dof_start + 5, i_b] == gs.CTRL_MODE.POSITION
            ):
                xyz = ti.Vector(
                    [
                        dofs_state.pos[0 + 3 + dof_start, i_b],
                        dofs_state.pos[1 + 3 + dof_start, i_b],
                        dofs_state.pos[2 + 3 + dof_start, i_b],
                    ],
                    dt=gs.ti_float,
                )

                ctrl_xyz = ti.Vector(
                    [
                        dofs_state.ctrl_pos[0 + 3 + dof_start, i_b],
                        dofs_state.ctrl_pos[1 + 3 + dof_start, i_b],
                        dofs_state.ctrl_pos[2 + 3 + dof_start, i_b],
                    ],
                    dt=gs.ti_float,
                )

                quat = gu.ti_xyz_to_quat(xyz)
                ctrl_quat = gu.ti_xyz_to_quat(ctrl_xyz)

                q_diff = gu.ti_transform_quat_by_quat(ctrl_quat, gu.ti_inv_quat(quat))
                rotvec = gu.ti_quat_to_rotvec(q_diff)

                for j in ti.static(range(3)):
                    i_d = dof_start + 3 + j
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    force = dofs_info.kp[I_d] * rotvec[j] - dofs_info.kv[I_d] * dofs_state.vel[i_d, i_b]

                    dofs_state.qf_applied[i_d, i_b] = ti.math.clamp(
                        force, dofs_info.force_range[I_d][0], dofs_info.force_range[I_d][1]
                    )

                    if ti.abs(force) > gs.EPS:
                        wakeup = True



