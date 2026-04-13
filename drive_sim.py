"""
Autonomous driving in the DonkeyCar simulator using trained RSSM Dreamer models.

Requires: vae_encoder.pth + world_model.pth + dreamer_actor.pth (from training pipeline),
          running simulator on SIM_HOST:SIM_PORT
Downstream: None (end of pipeline - inference/deployment)

Loads frozen VAE encoder, RSSM world model, and trained Actor. Maintains RSSM state
(h_t, s_t) across frames: each frame is processed through observe_step (posterior) to
update the deterministic + stochastic state. Actor sees cat(h_t, s_t) = STATE_DIM.
Resets on crash detection.
"""

import time
import torch
import gymnasium as gym
import gym_donkeycar

from core import (
    MODEL_DIR, DeterministicEncoder, RSSMWorldModel, Actor, process_sim_image,
    VAE_WEIGHTS, WORLD_MODEL_WEIGHTS, ACTOR_WEIGHTS,
    SIM_HOST, SIM_PORT, SIM_ENV, device,
    DETER_DIM, STOCH_DIM, ACTION_DIM
)

# Match the latent noise used during world model training
LATENT_NOISE_STD = 0.005

# --- INITIALIZATION ---
print(f"Engaging Autopilot on: {device}")


print(f"\n\nUsing model directory: {MODEL_DIR}\n\n")


print("Loading VAE Brain...")
vae = DeterministicEncoder().to(device)
vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
vae.eval()

print("Loading RSSM World Model...")
world_model = RSSMWorldModel().to(device)
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

    def reset_rssm_state():
        h = torch.zeros(1, DETER_DIM, device=device)
        s = torch.zeros(1, STOCH_DIM, device=device)
        a = torch.zeros(1, ACTION_DIM, device=device)
        return h, s, a

    h_t, s_t, a_prev = reset_rssm_state()

    try:
        while True:
            img_tensor = torch.tensor(obs.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            img_cropped = process_sim_image(img_tensor)

            with torch.no_grad():
                enc = vae(img_cropped)
                enc = enc + torch.randn_like(enc) * LATENT_NOISE_STD

                # Update RSSM state using posterior (we have a real observation)
                s_t, h_t = world_model.observe_step(s_t, a_prev, h_t, enc)

                state = torch.cat([h_t, s_t], dim=-1)
                action = actor(state).squeeze(0).cpu().numpy()

            steering = float(action[0])
            throttle = float(action[1])

            a_prev = torch.tensor([[steering, throttle]], dtype=torch.float32, device=device)

            obs, reward, terminated, truncated, info = env.step([steering, throttle])

            if terminated or truncated:
                print("Crash detected! The AI is resetting...")
                obs, info = env.reset()
                h_t, s_t, a_prev = reset_rssm_state()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nDisengaging Autopilot...")
    finally:
        env.close()

if __name__ == "__main__":
    drive()
