import torch
import genesis as gs
from dataclasses import dataclass
import numpy as np


@dataclass
class DiffrigidLocoConfig:
    dt: float = 0.01
    n_envs: int = 256          
    horizon: int = 64
    num_iter: int = 50

    urdf_path: str = "/home/jm/Development/Genesis/my_scripts/h1/h1.urdf"

    target_forward_dist: float = 1.0    # meters forward over the horizon
    target_height: float = 0.9

    seed: int = 0
    show_viewer: bool = False


class GenesisH1DiffrigidEnv:
    def __init__(self, cfg: DiffrigidLocoConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        gs.init(
            backend=gs.gpu,
            precision="32",
            logging_level="info",
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=cfg.dt,
                substeps=1,
                requires_grad=True,           # <- important
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
                disable_constraint=False,
                use_contact_island=False,  # Disabled due to compilation bug with GsTaichi
                use_hibernation=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, -3.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=40,
                max_FPS=60,
            ),
            show_viewer=cfg.show_viewer,
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=True,
                ambient_light=(0.1, 0.1, 0.1),
            ),
            renderer=gs.renderers.BatchRenderer(use_rasterizer=True),
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())

        self.humanoid = self.scene.add_entity(
            gs.morphs.URDF(
                file=cfg.urdf_path,
                pos=(0.0, 0.0, 1.0),
            )
        )

        # Map joint names to local DOF indices using joints, not a non-existent get_dofs_idx
        DOF_NAMES = [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_joint",

            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_joint",

            "torso_joint",

            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",

            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ]

        actuated_dof_indices = []
        for jname in DOF_NAMES:
            joint = self.humanoid.get_joint(jname)
            idx_local = joint.dofs_idx_local
            if isinstance(idx_local, int):
                actuated_dof_indices.append(idx_local)
            else:
                actuated_dof_indices.extend(list(idx_local))

        self.actuated_dofs = torch.tensor(
            actuated_dof_indices,
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.n_act = self.actuated_dofs.numel()

        self.cam = self.scene.add_camera(
            res=(640, 480),
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=False,
        )

        self.scene.build(n_envs=cfg.n_envs, env_spacing=(2.0, 2.0))

        # Default joint pose (no grad)
        self.default_dof_pos = self.humanoid.get_dofs_position().detach().clone()

    def reset_scene(self):
        """Non-differentiable reset, like in the test."""
        self.scene.reset()

        # Lift robot slightly above the plane to avoid initial collision
        base_pos = gs.tensor([[0.0, 0.0, self.cfg.target_height + 0.2]] * self.cfg.n_envs, requires_grad=False)
        base_quat = gs.tensor([[1.0, 0.0, 0.0, 0.0]] * self.cfg.n_envs, requires_grad=False)

        # For rigid URDFs, set_pos / set_quat are the right calls
        self.humanoid.set_pos(base_pos)
        self.humanoid.set_quat(base_quat)
        self.humanoid.set_dofs_position(self.default_dof_pos)

    def rollout(self, action_vel: gs.Tensor):
        """
        action_vel: genesis.Tensor of shape [n_act], requires_grad=True.
                    We broadcast it over envs and keep it constant over time
                    (like init_pos/init_quat in the test).
        Returns: loss (torch scalar) that is differentiable w.r.t action_vel.
        """
        cfg = self.cfg
        self.reset_scene()

        # Broadcast action to [n_envs, n_act]
        vel_field = action_vel.view(1, -1).expand(cfg.n_envs, self.n_act)

        # Accumulate no loss inside loop; we just care about final state
        frames = []
        for t in range(cfg.horizon):
            self.humanoid.set_dofs_velocity(
                velocity=vel_field,
                dofs_idx_local=self.actuated_dofs,
            )
            self.scene.step()
            # (Optional) you could add per-step penalties here

            rgb, _, _, _ = self.cam.render(rgb=True)  # BatchRenderer output
            # Move to CPU immediately to avoid GPU memory accumulation
            rgb = np.asarray(rgb.cpu())
            frames.append(rgb)

        # Final state: use scene.get_state (root_pos) to avoid guessing entity state layout
        # Use the humanoid entity API instead of SimState
        root_pos = self.humanoid.get_pos()  # [B, 3] genesis tensor

        # We want the base to end up at target_forward_dist in +x, and near target_height in z
        target_x = cfg.target_forward_dist
        target_z = cfg.target_height

        x = root_pos[:, 0]
        z = root_pos[:, 2]


        # Simple squared error in position
        pos_loss = (x - target_x) ** 2 + 0.5 * (z - target_z) ** 2  # [B]

        # Small regularization on joint velocities at the end to avoid crazy spins
        dof_vel = self.humanoid.get_dofs_velocity()
        act_vel = dof_vel[:, self.actuated_dofs]
        reg_loss = 1e-3 * (act_vel ** 2).mean(dim=-1)  # [B]

        total_loss = (pos_loss + reg_loss).mean()  # scalar genesis.Tensor (subclass of torch.Tensor)

        return total_loss, np.asarray(frames)
