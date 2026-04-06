"""
Autonomous driving in the DonkeyCar simulator using trained Dreamer models.

Requires: vae_encoder.pth + dreamer_actor.pth (from training pipeline),
          running simulator on SIM_HOST:SIM_PORT
Downstream: None (end of pipeline - inference/deployment)

Loads frozen VAE encoder and trained Actor, connects to the simulator via gym-donkeycar,
and runs a real-time autonomous driving loop. Camera image -> VAE latent -> Actor action
-> steering/throttle. Resets on crash detection.
"""

import time
import torch
import gymnasium as gym
import gym_donkeycar

from core import (
    DeterministicEncoder, Actor, process_sim_image,
    VAE_WEIGHTS, ACTOR_WEIGHTS, SIM_HOST, SIM_PORT, SIM_ENV, device
)

# --- INITIALIZATION ---
print(f"Engaging Autopilot on: {device}")

print("Loading VAE Brain...")
vae = DeterministicEncoder().to(device)
vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
vae.eval()

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

    try:
        while True:
            img_tensor = torch.tensor(obs.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            img_cropped = process_sim_image(img_tensor)

            with torch.no_grad():
                latent_state = vae(img_cropped)
                action = actor(latent_state).squeeze(0).cpu().numpy()

            steering = float(action[0])
            throttle = float(action[1])

            obs, reward, terminated, truncated, info = env.step([steering, throttle])

            if terminated or truncated:
                print("Crash detected! The AI is resetting...")
                obs, info = env.reset()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nDisengaging Autopilot...")
    finally:
        env.close()

if __name__ == "__main__":
    drive()
