import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import gym_donkeycar

# --- 1. ARCHITECTURE DEFINITIONS ---
# We must redefine the exact network structures so PyTorch can load the weights.

class DeterministicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(10240, 32)

    def forward(self, x):
        return self.fc_mu(self.encoder(x))

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, z):
        raw_action = self.net(z)
        steer = torch.tanh(raw_action[:, 0:1]) 
        throttle = torch.sigmoid(raw_action[:, 1:2]) * 0.4 
        return torch.cat([steer, throttle], dim=-1)

# --- 2. INITIALIZATION ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Engaging Autopilot on: {device}")

print("Loading VAE Brain...")
vae = DeterministicEncoder().to(device)
vae.load_state_dict(torch.load("./models/vae/vae_encoder.pth", map_location=device))
vae.eval()

print("Loading Dreamer Reflexes...")
actor = Actor().to(device)
actor.load_state_dict(torch.load("./models/vae/dreamer_actor.pth", map_location=device))
actor.eval()

# --- 3. SIMULATOR CONNECTION ---
def drive():
    print("Connecting to Simulator...")
    conf = {
        "exe_path": "remote", 
        "host": "127.0.0.1",
        "port": 9091,
        "body_style": "donkey",
        "body_rgb": (0, 255, 0), # Green car for Autonomous Mode!
        "car_name": "Dreamer_AI",
        "font_size": 100
    }
    
    env = gym.make("donkey-generated-track-v0", conf=conf)
    obs, info = env.reset()
    
    print("\n" + "="*40)
    print("🏎️ AUTONOMOUS MODE ENGAGED")
    print("Press CTRL+C in this terminal to stop.")
    print("="*40 + "\n")
    
    try:
        while True:
            # 1. Preprocess the image to match the VAE's training data exactly
            # Raw Sim Output: (120, 160, 3) RGB
            # We need: (80, 128, 3) RGB, moved to PyTorch format (1, 3, 80, 128)
            
            # Crop top 40px, and 16px from each side
            cropped_img = obs[40:120, 16:144, :] 
            
            # Convert to PyTorch Tensor, normalize to 0-1, and add batch dimension
            img_tensor = torch.tensor(cropped_img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            
            # 2. Neural Network Inference
            with torch.no_grad():
                # Compress image to 32 numbers
                latent_state = vae(img_tensor)
                
                # Get steering and throttle from the imagined policy
                action = actor(latent_state).squeeze(0).cpu().numpy()
            
            # 3. Apply Action
            steering = float(action[0])
            throttle = float(action[1])
            
            obs, reward, terminated, truncated, info = env.step([steering, throttle])
            
            if terminated or truncated:
                print("Crash detected! The AI is resetting...")
                obs, info = env.reset()
                
            time.sleep(0.05) # Cap framerate to match real-world latency
            
    except KeyboardInterrupt:
        print("\nDisengaging Autopilot...")
    finally:
        env.close()

if __name__ == "__main__":
    drive()