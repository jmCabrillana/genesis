import os
from dataclasses import dataclass

import torch
import genesis as gs


# -----------------------------
#  Config
# -----------------------------

@dataclass
class DiffrigidLocoConfig:
    # Sim / env
    dt: float = 0.01
    n_envs: int = 256
    horizon: int = 32            # timesteps per rollout (short horizon for gradients)
    max_steps: int = 2000        # outer training iterations

    # Robot / assets
    humanoidverse_root: str = "./HumanoidVerse"  # path to your HumanoidVerse repo
    urdf_path: str = "/home/jm/Development/Genesis/my_scripts/h1/h1.urdf" 

    # Rewards / commands
    target_lin_vel_x: float = 1.0
    target_lin_vel_y: float = 0.0
    target_yaw_rate: float = 0.0
    target_base_height: float = 0.9

    # Policy / optimization
    obs_dim: int = 64           # will be trimmed or padded
    hidden_dim: int = 256
    lr: float = 3e-4
    seed: int = 0

    # Logging / debug
    show_viewer: bool = False

# -----------------------------
#  Genesis Humanoid Env
# -----------------------------

class GenesisH1LocoEnv:
    """
    Differentiable H1-like humanoid locomotion env in Genesis.

    The important bit: we deliberately feed actions through
    differentiable APIs (set_dofs_velocity) and pull losses
    from Genesis tensors so autograd traverses the sim.
    """

    def __init__(self, cfg: DiffrigidLocoConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # ---- Genesis init: use GPU + float32 + logging on
        gs.init(
            backend=gs.gpu,
            precision="32",
            logging_level="info",
        )

        # ---- Scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=cfg.dt,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, -3.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=40,
                max_FPS=60,
            ),
            show_viewer=cfg.show_viewer,
        )

        # Ground
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        # ---- Humanoid (H1 lower body)
        self.humanoid = self.scene.add_entity(
            gs.morphs.URDF(
                file=cfg.urdf_path,
                pos=(0.0, 0.0, 1.0),
            )
        )

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

        # Map DOF names to local indices
        # self.actuated_dofs = self.humanoid.get_dofs_idx(DOF_NAMES)
        # self.n_act = self.actuated_dofs.shape[0]
        # Build DOF indices from joint names
        actuated_dof_indices = []
        for jname in DOF_NAMES:
            joint = self.humanoid.get_joint(jname)  # RigidJoint
            idx_local = joint.dofs_idx_local        # can be int or a sequence

            if isinstance(idx_local, int):
                actuated_dof_indices.append(idx_local)
            else:
                # For joints with multiple DOFs (e.g. 2-3 DoF joints)
                actuated_dof_indices.extend(list(idx_local))

        # Store as a torch tensor of local DOF indices
        self.actuated_dofs = torch.tensor(
            actuated_dof_indices,
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.n_act = self.actuated_dofs.numel()

        # ---- Build batched envs
        self.scene.build(
            n_envs=cfg.n_envs,
            env_spacing=(2.0, 2.0),  # just for visualization
        )

        # Default pose buffer
        self.default_dof_pos = self.humanoid.get_dofs_position().detach().clone()
        # shape: [n_envs, n_dofs]
        # We only care about subset for action scaling
        self.default_act_pos = self.default_dof_pos[:, self.actuated_dofs]

        # Simple buffers for commands (Genesis-side genesis.Tensor)
        self.command_lin_vel = gs.tensor(
            [cfg.target_lin_vel_x, cfg.target_lin_vel_y], requires_grad=False
        )
        self.command_yaw_rate = gs.tensor(cfg.target_yaw_rate, requires_grad=False)

        # Episode step counter per env (torch, not differentiable)
        self.step_count = torch.zeros(cfg.n_envs, dtype=torch.long)

    # ---------------------
    #  Reset
    # ---------------------
    def reset(self):
        """
        Reset all envs to an upright H1 posture.
        """

        # --- 1) Reset Genesis scene state & its gradient bookkeeping
        # This resets to the scene's registered initial state and clears grad flags.
        self.scene.reset()

        # --- 2) Now put humanoid into your desired initial posture
        base_pos = torch.tensor(
            [[0.0, 0.0, self.cfg.target_base_height]] * self.cfg.n_envs,
            device=gs.device,
            dtype=gs.tc_float,
        )
        base_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * self.cfg.n_envs,  # identity (w, x, y, z)
            device=gs.device,
            dtype=gs.tc_float,
        )

        self.humanoid.set_pos(base_pos)
        self.humanoid.set_quat(base_quat)

        self.humanoid.set_dofs_position(self.default_dof_pos)
        self.humanoid.zero_all_dofs_velocity()

        self.step_count.zero_()

        return self._compute_obs()


    # ---------------------
    #  Step (differentiable)
    # ---------------------
    def step(self, action_torch: torch.Tensor):
        """
        One *differentiable* rollout segment of length horizon.

        action_torch: [B, n_act] PyTorch tensor (requires_grad=True)
                      interpreted as target joint velocities.
        Returns:
            obs_tp1: [B, obs_dim] torch
            reward:  [B] torch
        """
        B = self.cfg.n_envs
        assert action_torch.shape == (B, self.n_act)

        # Convert actions into a Genesis tensor that is *connected*
        # to the PyTorch graph (detach=False).
        act_gs = gs.from_torch(action_torch, detach=False)

        # Short horizon rollout
        total_reward = None
        for t in range(self.cfg.horizon):
            # Apply joint velocities (differentiable w.r.t act_gs)
            self.humanoid.set_dofs_velocity(
                velocity=act_gs,
                dofs_idx_local=self.actuated_dofs,
            )

            # Step simulation once (differentiable through rigid solver)
            self.scene.step()

            # Compute per-timestep reward as Genesis tensor, accumulate
            r_t = self._compute_reward()  # genesis.Tensor
            if total_reward is None:
                total_reward = r_t
            else:
                total_reward = total_reward + r_t

            self.step_count += 1

        obs_tp1 = self._compute_obs()
        return obs_tp1, total_reward

    # ---------------------
    #  Observations
    # ---------------------
    def _compute_obs(self) -> torch.Tensor:
        """
        Build a locomotion observation using *snapshots* of sim state.
        """

        # --- Snapshot base stuff
        base_pos = self.humanoid.get_pos().clone()      # (B, 3)
        base_quat = self.humanoid.get_quat().clone()    # (B, 4)  (w, x, y, z)
        base_lin_vel = self.humanoid.get_vel().clone()  # (B, 3)
        base_ang_vel = self.humanoid.get_ang().clone()  # (B, 3)

        B = base_pos.shape[0]

        # Gravity “projection” (keep it simple for now)
        gravity_world = torch.tensor(
            [0.0, 0.0, -9.81],
            device=base_pos.device,
            dtype=base_pos.dtype,
        )
        proj_grav = gravity_world.view(1, 3).expand(B, 3)

        # --- Snapshot DOFs
        dof_pos = self.humanoid.get_dofs_position().clone()   # (B, n_dofs)
        dof_vel = self.humanoid.get_dofs_velocity().clone()   # (B, n_dofs)

        act_pos = dof_pos[:, self.actuated_dofs]              # (B, n_act)
        act_vel = dof_vel[:, self.actuated_dofs]              # (B, n_act)

        obs = torch.cat(
            [
                base_lin_vel,   # 3
                base_ang_vel,   # 3
                proj_grav,      # 3
                act_pos,        # n_act
                act_vel,        # n_act
            ],
            dim=-1,
        )  # (B, D)

        # --- Pad / trim to cfg.obs_dim
        D = obs.shape[1]
        if D >= self.cfg.obs_dim:
            obs = obs[:, : self.cfg.obs_dim]
        else:
            pad = torch.zeros(
                (B, self.cfg.obs_dim - D),
                device=obs.device,
                dtype=obs.dtype,
            )
            obs = torch.cat([obs, pad], dim=-1)

        return obs


    # ---------------------
    #  Reward
    # ---------------------
    def _compute_reward(self):
        cfg = self.cfg

        # --- Snapshot base kinematics
        base_pos = self.humanoid.get_pos().clone()
        base_lin_vel = self.humanoid.get_vel().clone()
        base_ang_vel = self.humanoid.get_ang().clone()

        target_vx = cfg.target_lin_vel_x
        target_vy = cfg.target_lin_vel_y
        target_yaw = cfg.target_yaw_rate

        vx = base_lin_vel[:, 0]
        vy = base_lin_vel[:, 1]
        yaw_rate = base_ang_vel[:, 2]

        w_lin = 1.0
        w_yaw = 0.2
        w_lin_z = 0.2
        w_height = 0.5

        lin_tracking = -w_lin * ((vx - target_vx) ** 2 + (vy - target_vy) ** 2)
        yaw_tracking = -w_yaw * ((yaw_rate - target_yaw) ** 2)
        lin_z_penalty = -w_lin_z * (base_lin_vel[:, 2] ** 2)

        height_err = base_pos[:, 2] - cfg.target_base_height
        height_term = -w_height * (height_err ** 2)

        # --- Snapshot DOF velocities for regularization
        dof_vel = self.humanoid.get_dofs_velocity().clone()
        act_vel = dof_vel[:, self.actuated_dofs]          # (B, n_act)
        action_rate_pen = -0.01 * (act_vel ** 2).mean(dim=-1)

        reward = (
            lin_tracking
            + yaw_tracking
            + lin_z_penalty
            + height_term
            + action_rate_pen
        )

        # Keep this differentiable through Genesis
        if isinstance(reward, torch.Tensor):
            reward = gs.from_torch(reward, detach=False)

        return reward
