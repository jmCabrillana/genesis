from typing import TYPE_CHECKING, Literal

import gstaichi as ti
import numpy as np
import numpy.typing as npt
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.entities import AvatarEntity, DroneEntity, RigidEntity
from genesis.engine.entities.base_entity import Entity
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.solvers import RigidSolverState
from genesis.options.solvers import RigidOptions
from genesis.utils import linalg as lu
from genesis.utils.misc import ALLOCATE_TENSOR_WARNING, DeprecationError, ti_to_torch
from genesis.utils.sdf_decomp import SDF

from ..base_solver import Solver
from .collider_decomp import Collider
from .constraint_solver_decomp import ConstraintSolver
from .constraint_solver_decomp_island import ConstraintSolverIsland
from .rigid_solver_decomp_util import func_wakeup_entity_and_its_temp_island


@ti.kernel
def kernel_update_geoms(
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
    is_backward: ti.template(),
):
    """
    NOTE: this only update geom pose, not its verts and else.
    """
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(links_state.pos.shape[1]):
        func_update_geoms(
            i_b,
            entities_info,
            geoms_info,
            geoms_state,
            links_state,
            rigid_global_info,
            static_rigid_sim_config,
            force_update_fixed_geoms,
            is_backward,
        )


@ti.func
def func_update_geoms(
    i_b,
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
    is_backward: ti.template(),
):
    n_geoms = geoms_info.pos.shape[0]
    for i_0 in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(n_geoms)
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(geoms_info.pos.shape[0]))
        )
    ):
        i_e = rigid_global_info.awake_entities[i_0, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 0
        n_geoms = entities_info.geom_end[i_e] - entities_info.geom_start[i_e]

        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(n_geoms)
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_geoms_per_entity))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            i_g = i_1 + entities_info.geom_start[i_e] if ti.static(static_rigid_sim_config.use_hibernation) else i_0
            if i_1 < (n_geoms if ti.static(static_rigid_sim_config.use_hibernation) else 1):
                if force_update_fixed_geoms or not geoms_info.is_fixed[i_g]:
                    (
                        geoms_state.pos[i_g, i_b],
                        geoms_state.quat[i_g, i_b],
                    ) = gu.ti_transform_pos_quat_by_trans_quat(
                        geoms_info.pos[i_g],
                        geoms_info.quat[i_g],
                        links_state.pos[geoms_info.link_idx[i_g], i_b],
                        links_state.quat[geoms_info.link_idx[i_g], i_b],
                    )

                    geoms_state.verts_updated[i_g, i_b] = False


@ti.kernel
def kernel_forward_velocity(
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(links_state.pos.shape[1]):
        func_forward_velocity(
            i_b=i_b,
            entities_info=entities_info,
            links_info=links_info,
            links_state=links_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )


@ti.func
def func_forward_velocity(
    i_b,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    n_entities = entities_info.n_links.shape[0]
    for i_e_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(n_entities)
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(entities_info.n_links.shape[0]))
        )
    ):
        i_e = (
            rigid_global_info.awake_entities[i_e_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_e_
        )
        func_forward_velocity_entity(
            i_e=i_e,
            i_b=i_b,
            entities_info=entities_info,
            links_info=links_info,
            links_state=links_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )


@ti.func
def func_forward_velocity_entity(
    i_e,
    i_b,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not is_backward)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not is_backward) else (i_l_ + entities_info.link_start[i_e])

        if i_l < entities_info.link_end[i_e]:
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

            links_state.cd_vel_bw[i_l, 0, i_b] = ti.Vector.zero(gs.ti_float, 3)
            links_state.cd_ang_bw[i_l, 0, i_b] = ti.Vector.zero(gs.ti_float, 3)

            if links_info.parent_idx[I_l] != -1:
                links_state.cd_vel_bw[i_l, 0, i_b] = links_state.cd_vel[links_info.parent_idx[I_l], i_b]
                links_state.cd_ang_bw[i_l, 0, i_b] = links_state.cd_ang[links_info.parent_idx[I_l], i_b]

            for i_j_ in (
                range(n_joints)
                if ti.static(not is_backward)
                else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
            ):
                i_j = i_j_ + links_info.joint_start[I_l]

                if i_j < links_info.joint_end[I_l]:
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]
                    q_start = joints_info.q_start[I_j]
                    dof_start = joints_info.dof_start[I_j]

                    curr_i_j = 0 if ti.static(not is_backward) else i_j_
                    next_i_j = 0 if ti.static(not is_backward) else i_j_ + 1

                    if joint_type == gs.JOINT_TYPE.FREE:
                        for i_3 in ti.static(range(3)):
                            links_state.cd_vel_bw[i_l, curr_i_j, i_b] += (
                                dofs_state.cdof_vel[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b]
                            )
                            links_state.cd_ang_bw[i_l, curr_i_j, i_b] += (
                                dofs_state.cdof_ang[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b]
                            )

                        for i_3 in ti.static(range(3)):
                            (
                                dofs_state.cdofd_ang[dof_start + i_3, i_b],
                                dofs_state.cdofd_vel[dof_start + i_3, i_b],
                            ) = ti.Vector.zero(gs.ti_float, 3), ti.Vector.zero(gs.ti_float, 3)

                            (
                                dofs_state.cdofd_ang[dof_start + i_3 + 3, i_b],
                                dofs_state.cdofd_vel[dof_start + i_3 + 3, i_b],
                            ) = gu.motion_cross_motion(
                                links_state.cd_ang_bw[i_l, curr_i_j, i_b],
                                links_state.cd_vel_bw[i_l, curr_i_j, i_b],
                                dofs_state.cdof_ang[dof_start + i_3 + 3, i_b],
                                dofs_state.cdof_vel[dof_start + i_3 + 3, i_b],
                            )

                        links_state.cd_vel_bw[i_l, next_i_j, i_b] = links_state.cd_vel_bw[i_l, curr_i_j, i_b]
                        links_state.cd_ang_bw[i_l, next_i_j, i_b] = links_state.cd_ang_bw[i_l, curr_i_j, i_b]

                        for i_3 in ti.static(range(3)):
                            links_state.cd_vel_bw[i_l, next_i_j, i_b] += (
                                dofs_state.cdof_vel[dof_start + i_3 + 3, i_b] * dofs_state.vel[dof_start + i_3 + 3, i_b]
                            )
                            links_state.cd_ang_bw[i_l, next_i_j, i_b] += (
                                dofs_state.cdof_ang[dof_start + i_3 + 3, i_b] * dofs_state.vel[dof_start + i_3 + 3, i_b]
                            )

                    else:
                        for i_d_ in (
                            range(dof_start, joints_info.dof_end[I_j])
                            if ti.static(not is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                        ):
                            i_d = i_d_ if ti.static(not is_backward) else (i_d_ + dof_start)
                            if i_d < joints_info.dof_end[I_j]:
                                dofs_state.cdofd_ang[i_d, i_b], dofs_state.cdofd_vel[i_d, i_b] = gu.motion_cross_motion(
                                    links_state.cd_ang_bw[i_l, curr_i_j, i_b],
                                    links_state.cd_vel_bw[i_l, curr_i_j, i_b],
                                    dofs_state.cdof_ang[i_d, i_b],
                                    dofs_state.cdof_vel[i_d, i_b],
                                )

                        links_state.cd_vel_bw[i_l, next_i_j, i_b] = links_state.cd_vel_bw[i_l, curr_i_j, i_b]
                        links_state.cd_ang_bw[i_l, next_i_j, i_b] = links_state.cd_ang_bw[i_l, curr_i_j, i_b]

                        for i_d_ in (
                            range(dof_start, joints_info.dof_end[I_j])
                            if ti.static(not is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                        ):
                            i_d = i_d_ if ti.static(not is_backward) else (i_d_ + dof_start)
                            if i_d < joints_info.dof_end[I_j]:
                                links_state.cd_vel_bw[i_l, next_i_j, i_b] += (
                                    dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                )
                                links_state.cd_ang_bw[i_l, next_i_j, i_b] += (
                                    dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                )

            i_j_ = 0 if ti.static(not is_backward) else n_joints
            links_state.cd_vel[i_l, i_b] = links_state.cd_vel_bw[i_l, i_j_, i_b]
            links_state.cd_ang[i_l, i_b] = links_state.cd_ang_bw[i_l, i_j_, i_b]
