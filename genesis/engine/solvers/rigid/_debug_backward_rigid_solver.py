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
            i_e = (
                rigid_global_info.awake_entities[i_0, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 0
            )

            for i_1 in (
                (
                    # Dynamic inner loop for forward pass
                    range(entities_info.geom_end[i_e] - entities_info.geom_start[i_e])
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
                if i_g < (entities_info.geom_end[i_e] if ti.static(static_rigid_sim_config.use_hibernation) else 1):
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
