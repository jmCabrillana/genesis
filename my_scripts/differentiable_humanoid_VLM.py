import torch
import torch.nn as nn
import torch.optim as optim
from H1 import *
import cv2
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

import genesis as gs

torch.autograd.set_detect_anomaly(True)

# -----------------------------
#  VLM Reward Model (Frozen)
# -----------------------------
class VLMReward:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", device="cuda"):
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Prompt template for reward prediction
        self.prompt_template = """USER: <image>
You are evaluating a humanoid robot's walking performance. 
Based on the image, rate how well the robot is walking forward on a scale of 0-10.
Consider: forward progress, balance, upright posture, and stability.
Respond with ONLY a number between 0 and 10.
ASSISTANT:"""
    
    @torch.no_grad()
    def get_reward(self, frame):
        """
        Args:
            frame: numpy array [H, W, 3] RGB image
        Returns:
            reward: float score 0-10
        """
        image = Image.fromarray(frame.astype('uint8'))
        inputs = self.processor(text=self.prompt_template, images=image, return_tensors="pt").to(self.device)
        
        # Generate
        output_ids = self.model.generate(**inputs, max_new_tokens=10)
        output_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract numeric reward
        try:
            # Extract last number from output
            import re
            numbers = re.findall(r'\d+\.?\d*', output_text)
            reward = float(numbers[-1]) if numbers else 5.0
            reward = max(0.0, min(10.0, reward))  # Clamp to [0, 10]
        except:
            reward = 5.0  # Default middle score
        
        return reward


# -----------------------------
#  Differentiable Surrogate Critic
# -----------------------------
class SurrogateReward(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state):
        # state: [B, state_dim]
        # returns: [B, 1] predicted reward
        return self.net(state)


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
    cfg.n_envs = 16  # Smaller batch for VLM evaluation
    cfg.num_iter = 100
    env = GenesisH1DiffrigidEnv(cfg)

    # Initialize VLM reward model
    print("Loading VLM reward model...")
    vlm_reward = VLMReward()
    
    # State dimension: ALL robot dof positions + velocities + base pose
    # Get actual state to determine dimension
    dummy_state = env.get_state_vector()
    state_dim = dummy_state.shape[1]  # Should be n_dofs*2 + 7
    
    # Initialize surrogate reward network
    surrogate = SurrogateReward(state_dim).cuda()
    surrogate_optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    
    # Trainable control
    init_vel = gs.tensor(
        torch.zeros(env.n_act, device=gs.device),
        requires_grad=True,
    )

    policy_optimizer = torch.optim.Adam([init_vel], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer,
        T_max=cfg.num_iter,
        eta_min=1e-3,
    )

    video_writer = None
    os.makedirs("runs", exist_ok=True)
    
    for it in range(cfg.num_iter):
        # === Phase 1: Collect VLM rewards (non-differentiable) ===
        with torch.no_grad():
            # Run simulation to get frames and states
            loss_physics, new_frames = env.rollout(init_vel)
            
            # Get VLM rewards for a subset of frames (expensive!)
            # Evaluate on final frame of each environment
            vlm_rewards = []
            for env_idx in range(cfg.n_envs):
                final_frame = new_frames[-1, env_idx]  # [H, W, 3]
                reward = vlm_reward.get_reward(final_frame)
                vlm_rewards.append(reward)
            
            vlm_rewards = torch.tensor(vlm_rewards, device=gs.device)  # [n_envs]
            
            # Get final states for surrogate training
            final_states = env.get_state_vector()  # [n_envs, state_dim]
        
        # === Phase 2: Train surrogate to match VLM rewards ===
        surrogate_optimizer.zero_grad()
        predicted_rewards = surrogate(final_states).squeeze(-1)  # [n_envs]
        surrogate_loss = nn.MSELoss()(predicted_rewards, vlm_rewards)
        surrogate_loss.backward()
        surrogate_optimizer.step()
        
        # === Phase 3: Optimize policy using differentiable surrogate ===
        policy_optimizer.zero_grad()
        
        # Run simulation again (this time with gradients)
        _, frames = env.rollout(init_vel)
        final_states_grad = env.get_state_vector()  # [n_envs, state_dim] with gradients
        
        # Get differentiable rewards from surrogate
        predicted_rewards_grad = surrogate(final_states_grad).squeeze(-1)  # [n_envs]
        
        # Maximize predicted reward (minimize negative reward)
        policy_loss = -predicted_rewards_grad.mean()
        
        # Add physics-based regularization
        policy_loss += 0.1 * loss_physics
        
        policy_loss.backward()
        policy_optimizer.step()
        scheduler.step()

        # Clamp velocities
        with torch.no_grad():
            init_vel.data.clamp_(-5.0, 5.0)

        # === Video recording (env 0 only) ===
        if video_writer is None and len(frames) > 0:
            sample_frame = frames[0, 0]
            h, w = sample_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter('runs/video.mp4', fourcc, 60, (w, h))
        
        for frame in frames[:, 0]:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        if it % 5 == 0:
            print(f"[it {it}] VLM reward: {vlm_rewards.mean():.2f} | "
                  f"Surrogate loss: {surrogate_loss.item():.4f} | "
                  f"Policy loss: {policy_loss.item():.4f}")

    print("Training complete!")
    if video_writer is not None:
        video_writer.release()
        print("Video saved to runs/video.mp4")



if __name__ == "__main__":
    main()
