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
