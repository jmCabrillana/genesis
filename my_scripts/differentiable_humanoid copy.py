import torch
import torch.nn as nn
import torch.optim as optim
from H1 import *
import cv2
import os

import genesis as gs

torch.autograd.set_detect_anomaly(True)

# -----------------------------
#  Simple MLP Policy
# -----------------------------

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs):
        # obs: [B, obs_dim]
        return self.net(obs)


# -----------------------------
#  Training loop
# -----------------------------
def main():
    cfg = DiffrigidLocoConfig()
    env = GenesisH1DiffrigidEnv(cfg)

    # Trainable control: one constant velocity vector for all joints
    # across the entire horizon. This is the "init_pos" analogue.
    init_vel = gs.tensor(
        torch.zeros(env.n_act, device=gs.device),
        requires_grad=True,
    )

    optimizer = torch.optim.Adam([init_vel], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_iter,
        eta_min=1e-3,
    )

    video_writer = None
    os.makedirs("runs", exist_ok=True)
    
    for it in range(cfg.num_iter):
        optimizer.zero_grad()

        loss, new_frames = env.rollout(init_vel)   # genesis scalar, differentiable w.r.t init_vel
        
        # Initialize video writer on first frame
        if video_writer is None and len(new_frames) > 0:
            # new_frames is [horizon, n_envs, H, W, 3], extract env 0's first frame
            sample_frame = new_frames[0, 0]  # [H, W, 3]
            h, w = sample_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter('runs/video.mp4', fourcc, 60, (w, h))
        
        # Write frames to video stream (no memory accumulation)
        for frame in new_frames[:,0]:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Optional: clamp or scale velocities if they explode
        with torch.no_grad():
            init_vel.data.clamp_(-5.0, 5.0)

        if it % 10 == 0:
            print(f"[it {it}] loss={loss.item():.4f}")

    print("Final loss:", loss.item())
    if video_writer is not None:
        video_writer.release()
        print("Video saved to runs/video.mp4")



if __name__ == "__main__":
    main()
