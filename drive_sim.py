"""
Autonomous driving in the DonkeyCar simulator using trained Dreamer models.

Requires: vae_encoder.pth + dreamer_actor.pth (from training pipeline),
          running simulator on SIM_HOST:SIM_PORT
Downstream: None (end of pipeline - inference/deployment)

Loads frozen VAE encoder, RSSM world model, and trained Actor. Runs a real-time loop:
camera image -> VAE latent -> RSSM posterior update (h, z) -> Actor action -> steering/throttle.
RSSM hidden state resets on crash detection.
"""

import time
import torch
import gymnasium as gym
import gym_donkeycar

from core import (
    MODEL_DIR, DeterministicEncoder, Actor, RSSM, process_sim_image,
    VAE_WEIGHTS, ACTOR_WEIGHTS, WORLD_MODEL_WEIGHTS, SIM_HOST, SIM_PORT, SIM_ENV,
    device, HIDDEN_DIM, STOCH_DIM, ACTION_DIM
)

# --- INITIALIZATION ---
print(f"Engaging Autopilot on: {device}")
print(f"\nUsing model directory: {MODEL_DIR}\n")

print("Loading VAE Brain...")
vae = DeterministicEncoder().to(device)
vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
vae.eval()

print("Loading RSSM World Model...")
world_model = RSSM().to(device)
world_model.load_state_dict(torch.load(WORLD_MODEL_WEIGHTS, map_location=device))
world_model.eval()

print("Loading Dreamer Reflexes...")
actor = Actor().to(device)
actor.load_state_dict(torch.load(ACTOR_WEIGHTS, map_location=device))
actor.eval()

# --- SIMULATOR CONNECTION ---
def drive():
    print("Connecting to Simulator...")
    conf = {
        "exe_path": "remote",
        "host": SIM_HOST,
        "port": SIM_PORT,
        "body_style": "donkey",
        "body_rgb": (0, 255, 0),
        "car_name": "Dreamer_AI",
        "font_size": 100
    }

    env = gym.make(SIM_ENV, conf=conf)
    obs, info = env.reset()

    print("\n" + "="*40)
    print("AUTONOMOUS MODE ENGAGED")
    print("Press CTRL+C in this terminal to stop.")
    print("="*40 + "\n")

    def _reset_rssm_state():
        h = torch.zeros(1, HIDDEN_DIM, device=device)
        z = torch.zeros(1, STOCH_DIM, device=device)
        a = torch.zeros(1, ACTION_DIM, device=device)
        return h, z, a

    h_t, z_t, a_prev = _reset_rssm_state()

    try:
        while True:
            img_tensor = torch.tensor(obs.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            img_cropped = process_sim_image(img_tensor)

            with torch.no_grad():
                x_t = vae(img_cropped)
                # Posterior update: incorporate real observation into RSSM state
                z_t, h_t, state = world_model.observe_step(x_t, a_prev, z_t, h_t)
                action = actor(state).squeeze(0).cpu().numpy()

            steering = float(action[0])
            throttle = float(action[1])
            a_prev = torch.tensor([[steering, throttle]], device=device)

            obs, reward, terminated, truncated, info = env.step([steering, throttle])

            if terminated or truncated:
                print("Crash detected! Resetting RSSM state...")
                obs, info = env.reset()
                h_t, z_t, a_prev = _reset_rssm_state()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nDisengaging Autopilot...")
    finally:
        env.close()

if __name__ == "__main__":
    drive()
