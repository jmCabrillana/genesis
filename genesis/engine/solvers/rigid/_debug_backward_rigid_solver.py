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


# ================================================== func_update_geoms ==================================================
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


# ================================================== func_forward_velocity ==================================================
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


# ================================================== func_COM_links ==================================================
@ti.kernel
def kernel_COM_links(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(links_state.pos.shape[1]):
        func_COM_links(
            i_b=i_b,
            links_state=links_state,
            links_info=links_info,
            joints_state=joints_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )


@ti.func
def func_COM_links(
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(links_info.root_idx.shape[0]))
        )
    ):
        if i_l_ < (
            rigid_global_info.n_awake_links[i_b]
            if ti.static(static_rigid_sim_config.use_hibernation)
            else links_info.root_idx.shape[0]
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )

            links_state.root_COM_bw[i_l, i_b].fill(0.0)
            links_state.mass_sum[i_l, i_b] = 0.0

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(links_info.root_idx.shape[0]))
        )
    ):
        if i_l_ < (
            rigid_global_info.n_awake_links[i_b]
            if ti.static(static_rigid_sim_config.use_hibernation)
            else links_info.root_idx.shape[0]
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.i_pos_bw[i_l, i_b],
                links_state.i_quat[i_l, i_b],
            ) = gu.ti_transform_pos_quat_by_trans_quat(
                links_info.inertial_pos[I_l] + links_state.i_pos_shift[i_l, i_b],
                links_info.inertial_quat[I_l],
                links_state.pos[i_l, i_b],
                links_state.quat[i_l, i_b],
            )

            i_r = links_info.root_idx[I_l]
            links_state.mass_sum[i_r, i_b] += mass
            links_state.root_COM_bw[i_r, i_b] += mass * links_state.i_pos_bw[i_l, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(links_info.root_idx.shape[0]))
        )
    ):
        if i_l_ < (
            rigid_global_info.n_awake_links[i_b]
            if ti.static(static_rigid_sim_config.use_hibernation)
            else links_info.root_idx.shape[0]
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            if i_l == i_r and links_state.mass_sum[i_l, i_b] > 0.0:
                links_state.root_COM[i_l, i_b] = links_state.root_COM_bw[i_l, i_b] / links_state.mass_sum[i_l, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(links_info.root_idx.shape[0]))
        )
    ):
        if i_l_ < (
            rigid_global_info.n_awake_links[i_b]
            if ti.static(static_rigid_sim_config.use_hibernation)
            else links_info.root_idx.shape[0]
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.root_COM[i_l, i_b] = links_state.root_COM[i_r, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(links_info.root_idx.shape[0]))
        )
    ):
        if i_l_ < (
            rigid_global_info.n_awake_links[i_b]
            if ti.static(static_rigid_sim_config.use_hibernation)
            else links_info.root_idx.shape[0]
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.i_pos[i_l, i_b] = links_state.i_pos_bw[i_l, i_b] - links_state.root_COM[i_l, i_b]

            i_inertial = links_info.inertial_i[I_l]
            i_mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.cinr_inertial[i_l, i_b],
                links_state.cinr_pos[i_l, i_b],
                links_state.cinr_quat[i_l, i_b],
                links_state.cinr_mass[i_l, i_b],
            ) = gu.ti_transform_inertia_by_trans_quat(
                i_inertial, i_mass, links_state.i_pos[i_l, i_b], links_state.i_quat[i_l, i_b]
            )

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(links_info.root_idx.shape[0]))
        )
    ):
        if i_l_ < (
            rigid_global_info.n_awake_links[i_b]
            if ti.static(static_rigid_sim_config.use_hibernation)
            else links_info.root_idx.shape[0]
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            if links_info.n_dofs[I_l] > 0:
                i_p = links_info.parent_idx[I_l]

                _i_j = links_info.joint_start[I_l]
                _I_j = [_i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else _i_j
                joint_type = joints_info.type[_I_j]

                p_pos = ti.Vector.zero(gs.ti_float, 3)
                p_quat = gu.ti_identity_quat()
                if i_p != -1:
                    p_pos = links_state.pos[i_p, i_b]
                    p_quat = links_state.quat[i_p, i_b]

                if joint_type == gs.JOINT_TYPE.FREE or (links_info.is_fixed[I_l] and i_p == -1):
                    links_state.j_pos[i_l, i_b] = links_state.pos[i_l, i_b]
                    links_state.j_quat[i_l, i_b] = links_state.quat[i_l, i_b]
                else:
                    (
                        links_state.j_pos_bw[i_l, 0, i_b],
                        links_state.j_quat_bw[i_l, 0, i_b],
                    ) = gu.ti_transform_pos_quat_by_trans_quat(links_info.pos[I_l], links_info.quat[I_l], p_pos, p_quat)

                    n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

                    for i_j_ in (
                        range(n_joints)
                        if ti.static(not is_backward)
                        else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
                    ):
                        i_j = i_j_ + links_info.joint_start[I_l]

                        curr_i_j = 0 if ti.static(not is_backward) else i_j_
                        next_i_j = 0 if ti.static(not is_backward) else i_j_ + 1

                        if i_j < links_info.joint_end[I_l]:
                            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j

                            (
                                links_state.j_pos_bw[i_l, next_i_j, i_b],
                                links_state.j_quat_bw[i_l, next_i_j, i_b],
                            ) = gu.ti_transform_pos_quat_by_trans_quat(
                                joints_info.pos[I_j],
                                gu.ti_identity_quat(),
                                links_state.j_pos_bw[i_l, curr_i_j, i_b],
                                links_state.j_quat_bw[i_l, curr_i_j, i_b],
                            )

                    i_j_ = 0 if ti.static(not is_backward) else n_joints
                    links_state.j_pos[i_l, i_b] = links_state.j_pos_bw[i_l, i_j_, i_b]
                    links_state.j_quat[i_l, i_b] = links_state.j_quat_bw[i_l, i_j_, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(links_info.root_idx.shape[0]))
        )
    ):
        if i_l_ < (
            rigid_global_info.n_awake_links[i_b]
            if ti.static(static_rigid_sim_config.use_hibernation)
            else links_info.root_idx.shape[0]
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            if links_info.n_dofs[I_l] > 0:
                for i_j_ in (
                    range(links_info.joint_start[I_l], links_info.joint_end[I_l])
                    if ti.static(not is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
                ):
                    i_j = i_j_ if ti.static(not is_backward) else (i_j_ + links_info.joint_start[I_l])

                    if i_j < links_info.joint_end[I_l]:
                        offset_pos = links_state.root_COM[i_l, i_b] - joints_state.xanchor[i_j, i_b]
                        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                        joint_type = joints_info.type[I_j]

                        dof_start = joints_info.dof_start[I_j]

                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            dofs_state.cdof_ang[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                            dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b].cross(offset_pos)
                        elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                            dofs_state.cdof_ang[dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                            dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                            xmat_T = gu.ti_quat_to_R(links_state.quat[i_l, i_b]).transpose()
                            for i in ti.static(range(3)):
                                dofs_state.cdof_ang[i + dof_start, i_b] = xmat_T[i, :]
                                dofs_state.cdof_vel[i + dof_start, i_b] = xmat_T[i, :].cross(offset_pos)
                        elif joint_type == gs.JOINT_TYPE.FREE:
                            for i in ti.static(range(3)):
                                dofs_state.cdof_ang[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                                dofs_state.cdof_vel[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                                dofs_state.cdof_vel[i + dof_start, i_b][i] = 1.0

                            xmat_T = gu.ti_quat_to_R(links_state.quat[i_l, i_b]).transpose()
                            for i in ti.static(range(3)):
                                dofs_state.cdof_ang[i + dof_start + 3, i_b] = xmat_T[i, :]
                                dofs_state.cdof_vel[i + dof_start + 3, i_b] = xmat_T[i, :].cross(offset_pos)

                        for i_d_ in (
                            range(dof_start, joints_info.dof_end[I_j])
                            if ti.static(not is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                        ):
                            i_d = i_d_ if ti.static(not is_backward) else (i_d_ + dof_start)
                            if i_d < joints_info.dof_end[I_j]:
                                dofs_state.cdofvel_ang[i_d, i_b] = (
                                    dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                )
                                dofs_state.cdofvel_vel[i_d, i_b] = (
                                    dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                )


# ================================================== func_forward_kinematics ==================================================
@ti.kernel
def kernel_forward_kinematics(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(links_state.pos.shape[1]):
        func_forward_kinematics(
            i_b=i_b,
            links_state=links_state,
            links_info=links_info,
            joints_state=joints_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )


@ti.func
def func_forward_kinematics(
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    for i_e_ in (
        (
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not is_backward)
        else (
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(entities_info.n_links.shape[0]))
        )
    ):
        if i_e_ < (
            rigid_global_info.n_awake_entities[i_b]
            if ti.static(static_rigid_sim_config.use_hibernation)
            else entities_info.n_links.shape[0]
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_e_
            )

            func_forward_kinematics_entity(
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
                static_rigid_sim_config,
                is_backward,
            )


@ti.func
def func_forward_kinematics_entity(
    i_e,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    # Becomes static loop in backward pass, because we assume this loop is an inner loop
    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not is_backward)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not is_backward) else (i_l_ + entities_info.link_start[i_e])

        if i_l < entities_info.link_end[i_e]:
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            links_state.pos_bw[i_l, 0, i_b] = links_info.pos[I_l]
            links_state.quat_bw[i_l, 0, i_b] = links_info.quat[I_l]
            if links_info.parent_idx[I_l] != -1:
                parent_pos = links_state.pos[links_info.parent_idx[I_l], i_b]
                parent_quat = links_state.quat[links_info.parent_idx[I_l], i_b]
                links_state.pos_bw[i_l, 0, i_b] = parent_pos + gu.ti_transform_by_quat(links_info.pos[I_l], parent_quat)
                links_state.quat_bw[i_l, 0, i_b] = gu.ti_transform_quat_by_quat(links_info.quat[I_l], parent_quat)

            n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

            for i_j_ in (
                range(n_joints)
                if ti.static(not is_backward)
                else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
            ):
                i_j = i_j_ + links_info.joint_start[I_l]

                curr_i_j = 0 if ti.static(not is_backward) else i_j_
                next_i_j = 0 if ti.static(not is_backward) else i_j_ + 1

                if i_j < links_info.joint_end[I_l]:
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]
                    q_start = joints_info.q_start[I_j]
                    dof_start = joints_info.dof_start[I_j]
                    I_d = [dof_start, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else dof_start

                    # compute axis and anchor
                    if joint_type == gs.JOINT_TYPE.FREE:
                        joints_state.xanchor[i_j, i_b] = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ]
                        )
                        joints_state.xaxis[i_j, i_b] = ti.Vector([0.0, 0.0, 1.0])
                    elif joint_type == gs.JOINT_TYPE.FIXED:
                        pass
                    else:
                        axis = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            axis = dofs_info.motion_ang[I_d]
                        elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                            axis = dofs_info.motion_vel[I_d]

                        joints_state.xanchor[i_j, i_b] = (
                            gu.ti_transform_by_quat(joints_info.pos[I_j], links_state.quat_bw[i_l, curr_i_j, i_b])
                            + links_state.pos_bw[i_l, curr_i_j, i_b]
                        )
                        joints_state.xaxis[i_j, i_b] = gu.ti_transform_by_quat(
                            axis, links_state.quat_bw[i_l, curr_i_j, i_b]
                        )

                    if joint_type == gs.JOINT_TYPE.FREE:
                        links_state.pos_bw[i_l, next_i_j, i_b] = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ],
                            dt=gs.ti_float,
                        )
                        links_state.quat_bw[i_l, next_i_j, i_b] = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start + 3, i_b],
                                rigid_global_info.qpos[q_start + 4, i_b],
                                rigid_global_info.qpos[q_start + 5, i_b],
                                rigid_global_info.qpos[q_start + 6, i_b],
                            ],
                            dt=gs.ti_float,
                        )
                        xyz = gu.ti_quat_to_xyz(links_state.quat_bw[i_l, next_i_j, i_b])
                        for j in ti.static(range(3)):
                            dofs_state.pos[dof_start + j, i_b] = links_state.pos_bw[i_l, next_i_j, i_b][j]
                            dofs_state.pos[dof_start + 3 + j, i_b] = xyz[j]
                    elif joint_type == gs.JOINT_TYPE.FIXED:
                        pass
                    elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                        qloc = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                                rigid_global_info.qpos[q_start + 3, i_b],
                            ],
                            dt=gs.ti_float,
                        )
                        xyz = gu.ti_quat_to_xyz(qloc)
                        for j in ti.static(range(3)):
                            dofs_state.pos[dof_start + j, i_b] = xyz[j]
                        links_state.quat_bw[i_l, next_i_j, i_b] = gu.ti_transform_quat_by_quat(
                            qloc, links_state.quat_bw[i_l, curr_i_j, i_b]
                        )
                        links_state.pos_bw[i_l, next_i_j, i_b] = joints_state.xanchor[
                            i_j, i_b
                        ] - gu.ti_transform_by_quat(joints_info.pos[I_j], links_state.quat_bw[i_l, next_i_j, i_b])
                    elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                        axis = dofs_info.motion_ang[I_d]
                        dofs_state.pos[dof_start, i_b] = (
                            rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                        )
                        qloc = gu.ti_rotvec_to_quat(axis * dofs_state.pos[dof_start, i_b])
                        links_state.quat_bw[i_l, next_i_j, i_b] = gu.ti_transform_quat_by_quat(
                            qloc, links_state.quat_bw[i_l, curr_i_j, i_b]
                        )
                        links_state.pos_bw[i_l, next_i_j, i_b] = joints_state.xanchor[
                            i_j, i_b
                        ] - gu.ti_transform_by_quat(joints_info.pos[I_j], links_state.quat_bw[i_l, next_i_j, i_b])
                    else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
                        dofs_state.pos[dof_start, i_b] = (
                            rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                        )
                        links_state.pos_bw[i_l, next_i_j, i_b] = (
                            links_state.pos_bw[i_l, curr_i_j, i_b]
                            + joints_state.xaxis[i_j, i_b] * dofs_state.pos[dof_start, i_b]
                        )

            # Skip link pose update for fixed root links to let users manually overwrite them
            i_j_ = 0 if ti.static(not is_backward) else n_joints
            if not (links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]):
                links_state.pos[i_l, i_b] = links_state.pos_bw[i_l, i_j_, i_b]
                links_state.quat[i_l, i_b] = links_state.quat_bw[i_l, i_j_, i_b]


# ================================================== func_integrate ==================================================
@ti.kernel
def kernel_integrate(
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (ti.ndrange(1, dofs_state.ctrl_mode.shape[1]))
        if ti.static(static_rigid_sim_config.use_hibernation)
        else (ti.ndrange(dofs_state.ctrl_mode.shape[0], dofs_state.ctrl_mode.shape[1]))
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_dofs[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_dofs))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if i_1 < (rigid_global_info.n_awake_dofs[i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 1):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                dofs_state.vel_next[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] + dofs_state.acc[i_d, i_b] * rigid_global_info.substep_dt[i_b]
                )

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (ti.ndrange(1, dofs_state.ctrl_mode.shape[1]))
        if ti.static(static_rigid_sim_config.use_hibernation)
        else (ti.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1]))
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_links))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if i_1 < (
                rigid_global_info.n_awake_links[i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 1
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] > 0:
                    dof_start = links_info.dof_start[I_l]
                    q_start = links_info.q_start[I_l]
                    q_end = links_info.q_end[I_l]

                    i_j = links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    if joint_type == gs.JOINT_TYPE.FREE:
                        pos = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ]
                        )
                        vel = ti.Vector(
                            [
                                dofs_state.vel_next[dof_start, i_b],
                                dofs_state.vel_next[dof_start + 1, i_b],
                                dofs_state.vel_next[dof_start + 2, i_b],
                            ]
                        )
                        pos += vel * rigid_global_info.substep_dt[i_b]
                        for j in ti.static(range(3)):
                            rigid_global_info.qpos_next[q_start + j, i_b] = pos[j]
                    if joint_type == gs.JOINT_TYPE.SPHERICAL or joint_type == gs.JOINT_TYPE.FREE:
                        rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                        rot0 = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start + rot_offset + 0, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 1, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 2, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 3, i_b],
                            ]
                        )
                        ang = (
                            ti.Vector(
                                [
                                    dofs_state.vel_next[dof_start + rot_offset + 0, i_b],
                                    dofs_state.vel_next[dof_start + rot_offset + 1, i_b],
                                    dofs_state.vel_next[dof_start + rot_offset + 2, i_b],
                                ]
                            )
                            * rigid_global_info.substep_dt[i_b]
                        )
                        qrot = gu.ti_rotvec_to_quat(ang)
                        rot = gu.ti_transform_quat_by_quat(qrot, rot0)
                        for j in ti.static(range(4)):
                            rigid_global_info.qpos_next[q_start + j + rot_offset, i_b] = rot[j]
                    else:
                        for j_ in (
                            (range(q_end - q_start))
                            if ti.static(not is_backward)
                            else (ti.static(range(static_rigid_sim_config.max_n_qs_per_link)))
                        ):
                            j = q_start + j_
                            if j < q_end:
                                rigid_global_info.qpos_next[j, i_b] = (
                                    rigid_global_info.qpos[j, i_b]
                                    + dofs_state.vel_next[dof_start + j_, i_b] * rigid_global_info.substep_dt[i_b]
                                )


# ================================================== func_factor_mass ==================================================
@ti.kernel
def kernel_factor_mass(
    implicit_damping: ti.template(),
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    func_factor_mass(
        implicit_damping=implicit_damping,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )


@ti.func
def func_factor_mass(
    implicit_damping: ti.template(),
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    """
    Compute Cholesky decomposition (L^T @ D @ L) of mass matrix.

    TODO: Implement backward pass manually for accelerating its compilation time (because of nested static inner loops).
    """
    if ti.static(not is_backward):
        _B = dofs_state.ctrl_mode.shape[1]
        n_entities = entities_info.n_links.shape[0]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_0, i_b in (
            ti.ndrange(1, _B) if ti.static(static_rigid_sim_config.use_hibernation) else ti.ndrange(n_entities, _B)
        ):
            for i_1 in (
                (
                    # Dynamic inner loop for forward pass
                    range(rigid_global_info.n_awake_entities[i_b])
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else range(1)
                )
                if ti.static(not is_backward)
                else (
                    # Static inner loop for backward pass
                    ti.static(range(static_rigid_sim_config.max_n_awake_entities))
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else ti.static(range(1))
                )
            ):
                if i_1 < (
                    rigid_global_info.n_awake_entities[i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 1
                ):
                    i_e = (
                        rigid_global_info.awake_entities[i_1, i_b]
                        if ti.static(static_rigid_sim_config.use_hibernation)
                        else i_0
                    )

                    if rigid_global_info._mass_mat_mask[i_e, i_b] == 1:
                        entity_dof_start = entities_info.dof_start[i_e]
                        entity_dof_end = entities_info.dof_end[i_e]
                        n_dofs = entities_info.n_dofs[i_e]

                        for i_d_ in (
                            range(entity_dof_start, entity_dof_end)
                            if ti.static(not is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            i_d = i_d_ if ti.static(not is_backward) else entities_info.dof_start[i_e] + i_d_

                            if i_d < entity_dof_end:
                                for j_d_ in (
                                    range(entity_dof_start, i_d + 1)
                                    if ti.static(not is_backward)
                                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                                ):
                                    j_d = j_d_ if ti.static(not is_backward) else entities_info.dof_start[i_e] + j_d_

                                    if j_d < i_d + 1:
                                        rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat[
                                            i_d, j_d, i_b
                                        ]

                                if ti.static(implicit_damping):
                                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                                    rigid_global_info.mass_mat_L[i_d, i_d, i_b] += (
                                        dofs_info.damping[I_d] * rigid_global_info.substep_dt[i_b]
                                    )
                                    if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                        if (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                                            dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                                        ):
                                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] += (
                                                dofs_info.kv[I_d] * rigid_global_info.substep_dt[i_b]
                                            )

                        for i_d_ in (
                            range(n_dofs)
                            if ti.static(not is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            if i_d_ < n_dofs:
                                i_d = entity_dof_end - i_d_ - 1
                                rigid_global_info.mass_mat_D_inv[i_d, i_b] = (
                                    1.0 / rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                                )

                                for j_d_ in (
                                    range(i_d - entity_dof_start)
                                    if ti.static(not is_backward)
                                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                                ):
                                    if j_d_ < i_d - entity_dof_start:
                                        j_d = i_d - j_d_ - 1
                                        a = (
                                            rigid_global_info.mass_mat_L[i_d, j_d, i_b]
                                            * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                                        )

                                        for k_d_ in (
                                            range(entity_dof_start, j_d + 1)
                                            if ti.static(not is_backward)
                                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                                        ):
                                            k_d = (
                                                k_d_
                                                if ti.static(not is_backward)
                                                else entities_info.dof_start[i_e] + k_d_
                                            )
                                            if k_d < j_d + 1:
                                                rigid_global_info.mass_mat_L[j_d, k_d, i_b] -= (
                                                    a * rigid_global_info.mass_mat_L[i_d, k_d, i_b]
                                                )
                                        rigid_global_info.mass_mat_L[i_d, j_d, i_b] = a

                                # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                                rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0

    else:
        # This block is logically equivalent to the above block, but the access pattern has been adjusted to be safe
        # for AD. However, it shows slightly numerical difference in the result, and thus it fails for a unit test
        # ("test_urdf_rope"), while passing all the others. TODO: Investigate if we can fix this and only use this block.

        # Assume this is the outermost loop
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1]):
            if rigid_global_info._mass_mat_mask[i_e, i_b] == 1:
                entity_dof_start = entities_info.dof_start[i_e]
                entity_dof_end = entities_info.dof_end[i_e]
                n_dofs = entities_info.n_dofs[i_e]

                for i_d0 in (
                    range(n_dofs)
                    if ti.static(not is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    if i_d0 < n_dofs:
                        i_d = entity_dof_start + i_d0
                        i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                        for j_d_ in (
                            range(entity_dof_start, i_d + 1)
                            if ti.static(not is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            j_d = j_d_ if ti.static(not is_backward) else (j_d_ + entities_info.dof_start[i_e])
                            j_pr = (entity_dof_start + entity_dof_end - 1) - j_d
                            if j_d < i_d + 1:
                                rigid_global_info.mass_mat_L_bw[0, i_pr, j_pr, i_b] = rigid_global_info.mass_mat[
                                    i_d, j_d, i_b
                                ]
                                rigid_global_info.mass_mat_L_bw[0, j_pr, i_pr, i_b] = rigid_global_info.mass_mat[
                                    i_d, j_d, i_b
                                ]

                        if ti.static(implicit_damping):
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            rigid_global_info.mass_mat_L_bw[0, i_pr, i_pr, i_b] += (
                                dofs_info.damping[I_d] * rigid_global_info.substep_dt[i_b]
                            )
                            if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                                    dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                                ):
                                    rigid_global_info.mass_mat_L_bw[0, i_pr, i_pr, i_b] += (
                                        dofs_info.kv[I_d] * rigid_global_info.substep_dt[i_b]
                                    )

                # Cholesky-Banachiewicz algorithm (in the perturbed indices), access pattern is safe for autodiff
                # https://en.wikipedia.org/wiki/Cholesky_decomposition
                for p_i0 in (
                    range(n_dofs)
                    if ti.static(not is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    for p_j0 in (
                        range(p_i0 + 1)
                        if ti.static(not is_backward)
                        else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                    ):
                        if p_i0 < n_dofs and p_j0 < n_dofs and p_j0 <= p_i0:
                            # j_pr <= i_pr
                            i_pr = entity_dof_start + p_i0
                            j_pr = entity_dof_start + p_j0

                            sum = gs.ti_float(0.0)
                            for p_k0 in (
                                range(p_j0)
                                if ti.static(not is_backward)
                                else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                            ):
                                # k_pr < j_pr
                                if p_k0 < p_j0:
                                    k_pr = entity_dof_start + p_k0
                                    sum += (
                                        rigid_global_info.mass_mat_L_bw[1, i_pr, k_pr, i_b]
                                        * rigid_global_info.mass_mat_L_bw[1, j_pr, k_pr, i_b]
                                    )

                            if p_i0 == p_j0:
                                rigid_global_info.mass_mat_L_bw[1, i_pr, j_pr, i_b] = ti.sqrt(
                                    rigid_global_info.mass_mat_L_bw[0, i_pr, j_pr, i_b] - sum
                                )
                            else:
                                rigid_global_info.mass_mat_L_bw[1, i_pr, j_pr, i_b] = (
                                    rigid_global_info.mass_mat_L_bw[0, i_pr, j_pr, i_b] - sum
                                ) / rigid_global_info.mass_mat_L_bw[1, j_pr, j_pr, i_b]

                for i_d0 in (
                    range(n_dofs)
                    if ti.static(not is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    for i_d1 in (
                        range(i_d0 + 1)
                        if ti.static(not is_backward)
                        else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                    ):
                        if i_d0 < n_dofs and i_d1 < n_dofs and i_d1 <= i_d0:
                            i_d = entity_dof_start + i_d0
                            j_d = entity_dof_start + i_d1
                            i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                            j_pr = (entity_dof_start + entity_dof_end - 1) - j_d

                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = (
                                rigid_global_info.mass_mat_L_bw[1, j_pr, i_pr, i_b]
                                / rigid_global_info.mass_mat_L_bw[1, i_pr, i_pr, i_b]
                            )

                            if i_d == j_d:
                                rigid_global_info.mass_mat_D_inv[i_d, i_b] = 1.0 / (
                                    rigid_global_info.mass_mat_L_bw[1, i_pr, i_pr, i_b] ** 2
                                )


# ================================================== func_compute_mass_mask ==================================================
@ti.kernel
def kernel_compute_mass_mask(
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    func_compute_mass_mask(
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )


@ti.func
def func_compute_mass_mask(
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    # Determine whether the mass matrix must be re-computed to take into account first-order correction terms.
    # Note that avoiding inverting the mass matrix twice would not only speed up simulation but also improving
    # numerical stability as computing post-damping accelerations from forces is not necessary anymore.
    if ti.static(
        not static_rigid_sim_config.enable_mujoco_compatibility
        or static_rigid_sim_config.integrator == gs.integrator.Euler
    ):
        for i_e, i_b in ti.ndrange(entities_info.dof_start.shape[0], dofs_state.ctrl_mode.shape[1]):
            rigid_global_info._mass_mat_mask[i_e, i_b] = 0

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b in ti.ndrange(entities_info.dof_start.shape[0], dofs_state.ctrl_mode.shape[1]):
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            # This part does not need to be differentiable, because mass_mat_mask is integer
            for i_d in range(entity_dof_start, entity_dof_end):
                if i_d < entity_dof_end:
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    if dofs_info.damping[I_d] > gs.EPS:
                        rigid_global_info._mass_mat_mask[i_e, i_b] = 1
                    if ti.static(static_rigid_sim_config.integrator != gs.integrator.Euler):
                        if (
                            (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION)
                            or (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY)
                        ) and dofs_info.kv[I_d] > gs.EPS:
                            rigid_global_info._mass_mat_mask[i_e, i_b] = 1


# ================================================== func_solve_mass_batched ==================================================
@ti.func
def func_solve_mass_batched(
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    out_bw: array_class.V_ANNOTATION,  # Should not be None if backward
    i_b: ti.int32,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    # This loop is considered an inner loop
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_0 in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(entities_info.n_links.shape[0]))
        )
    ):
        n_entities = entities_info.n_links.shape[0]

        if i_0 < (
            rigid_global_info.n_awake_entities[i_b]
            if ti.static(static_rigid_sim_config.use_hibernation)
            else n_entities
        ):
            i_e = (
                rigid_global_info.awake_entities[i_0, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_0
            )

            if rigid_global_info._mass_mat_mask[i_e, i_b] == 1:
                entity_dof_start = entities_info.dof_start[i_e]
                entity_dof_end = entities_info.dof_end[i_e]
                n_dofs = entities_info.n_dofs[i_e]

                # Step 1: Solve w st. L^T @ w = y
                for i_d_ in (
                    range(n_dofs)
                    if ti.static(not is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    if i_d_ < n_dofs:
                        i_d = entity_dof_end - i_d_ - 1
                        if ti.static(is_backward):
                            out_bw[0, i_d, i_b] = vec[i_d, i_b]
                        else:
                            out[i_d, i_b] = vec[i_d, i_b]

                        for j_d_ in (
                            range(i_d + 1, entity_dof_end)
                            if ti.static(not is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            j_d = j_d_ if ti.static(not is_backward) else (j_d_ + entities_info.dof_start[i_e])
                            if j_d >= i_d + 1 and j_d < entity_dof_end:
                                # Since we read out[j_d, i_b], and j_d > i_d, which means that out[j_d, i_b] is already
                                # finalized at this point, we don't need to care about AD mutation rule.
                                if ti.static(is_backward):
                                    out_bw[0, i_d, i_b] += -(
                                        rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out_bw[0, j_d, i_b]
                                    )
                                else:
                                    out[i_d, i_b] += -(rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b])

                # Step 2: z = D^{-1} w
                for i_d_ in (
                    range(entity_dof_start, entity_dof_end)
                    if ti.static(not is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    i_d = i_d_ if ti.static(not is_backward) else (i_d_ + entities_info.dof_start[i_e])
                    if i_d < entity_dof_end:
                        if ti.static(is_backward):
                            out_bw[1, i_d, i_b] = out_bw[0, i_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                        else:
                            out[i_d, i_b] *= rigid_global_info.mass_mat_D_inv[i_d, i_b]

                # Step 3: Solve x st. L @ x = z
                for i_d_ in (
                    range(entity_dof_start, entity_dof_end)
                    if ti.static(not is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    i_d = i_d_ if ti.static(not is_backward) else (i_d_ + entities_info.dof_start[i_e])
                    if i_d < entity_dof_end:
                        curr_out = out[i_d, i_b]
                        if ti.static(is_backward):
                            curr_out = out_bw[1, i_d, i_b]

                        for j_d_ in (
                            range(entity_dof_start, i_d)
                            if ti.static(not is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            j_d = j_d_ if ti.static(not is_backward) else (j_d_ + entities_info.dof_start[i_e])
                            if j_d < i_d:
                                curr_out += -(rigid_global_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b])

                        out[i_d, i_b] = curr_out


@ti.func
def func_solve_mass(
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    out_bw: array_class.V_ANNOTATION,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    # This loop must be the outermost loop to be differentiable
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_b in range(out.shape[1]):
        func_solve_mass_batched(
            vec,
            out,
            out_bw,
            i_b,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )


@ti.kernel
def kernel_solve_mass(
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    out_bw: array_class.V_ANNOTATION,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    is_backward: ti.template(),
):
    func_solve_mass(
        vec=vec,
        out=out,
        out_bw=out_bw,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )
