"""
Step 2: Train a GRU-based world model (RSSM) to predict latent dynamics and rewards.

Requires: vae_encoder.pth (from train_vae.py), tub with telemetry.csv and images
Downstream: train_actor_critic.py (uses frozen world_model.pth)

Freezes the VAE encoder and converts all images to latent vectors. Builds episode-aware
sequences (SEQ_LEN=32, ~1.5s memory) that never cross episode boundaries. Trains a
GRUCell(34->256) to predict next latent state and reward. Reward = throttle - |steering|*0.1
with terminal penalty (-10) at crash frames. Exports world_model.pth.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from core import DeterministicEncoder, WorldModel, process_sim_image, TUB_DIR, VAE_WEIGHTS, MODEL_DIR, device, LATENT_DIM, ACTION_DIM, HIDDEN_DIM, BATCH_SIZE_WORLD, PIN_MEMORY, THROTTLE_CAP

# --- CONFIGURATION ---
SEQ_LEN = 32     # UPGRADE: Gives the physics engine a 1.5-second memory of momentum
BATCH_SIZE = BATCH_SIZE_WORLD
EPOCHS = 50
FLIP_EPISODE_OFFSET = 100000  # Offset for flipped episode IDs to prevent cross-contamination
LATENT_NOISE_STD = 0.02       # Gaussian noise injected into latents during training
ACTION_NOISE_STD = 0.01       # Gaussian noise injected into actions during training

# --- TRAJECTORY DATASET ---
class DonkeyTrajectoryDataset(Dataset):
    def __init__(self, tub_dir, seq_len, vae_model):
        self.seq_len = seq_len
        self.transform = transforms.Compose([transforms.ToTensor()])

        telemetry_path = os.path.join(tub_dir, "telemetry.csv")
        if not os.path.exists(telemetry_path):
            raise FileNotFoundError(f"Could not find {telemetry_path}!")

        print(f"Reading telemetry from {telemetry_path}...")

        # Initialize separate lists for original and flipped data
        orig_latents, orig_actions, orig_rewards, orig_episodes = [], [], [], []
        flip_latents, flip_actions, flip_rewards, flip_episodes = [], [], [], []

        with open(telemetry_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            for i, row in enumerate(rows):
                img_path = os.path.join(tub_dir, row['frame'])
                if not os.path.exists(img_path): continue

                steering = float(row['steering'])
                throttle = float(row['throttle'])
                ep_id = int(row.get('episode_id', 0))

                reward = throttle * 1.0 - abs(steering) * 0.1

                is_terminal = False
                if i == len(rows) - 1 or int(rows[i+1].get('episode_id', 0)) != ep_id:
                    is_terminal = True
                    reward = -10.0

                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(device)
                img_cropped = process_sim_image(img_tensor)

                with torch.no_grad():
                    latent = vae_model(img_cropped).squeeze(0).cpu().numpy()
                    img_flipped = TF.hflip(img_cropped)
                    latent_flip = vae_model(img_flipped).squeeze(0).cpu().numpy()

                # Store Original
                orig_latents.append(latent)
                orig_actions.append([steering, throttle])
                orig_rewards.append([reward])
                orig_episodes.append(ep_id)

                # Store Flipped
                flip_latents.append(latent_flip)
                flip_actions.append([-steering, throttle])
                flip_rewards.append([reward])
                flip_episodes.append(ep_id + FLIP_EPISODE_OFFSET)

                if i % 1000 == 0 and i > 0:
                    print(f"Processed {i} frames through the VAE...")

        # Concatenate them cleanly so sequences are continuous
        self.latents = torch.tensor(np.array(orig_latents + flip_latents), dtype=torch.float32)
        self.actions = torch.tensor(np.array(orig_actions + flip_actions), dtype=torch.float32)
        self.rewards = torch.tensor(np.array(orig_rewards + flip_rewards), dtype=torch.float32)
        self.episodes = orig_episodes + flip_episodes

        # --- THE TELEPORTATION FIX ---
        self.valid_indices = []
        for i in range(len(self.latents) - self.seq_len):
            # Only allow sequences that start and end in the exact same episode
            if self.episodes[i] == self.episodes[i + self.seq_len - 1]:
                self.valid_indices.append(i)
                
        print(f"Filtered dataset: {len(self.valid_indices)} safe, unbroken sequences available.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map the random index to a safe, continuous sequence
        safe_idx = self.valid_indices[idx]

        z = self.latents[safe_idx : safe_idx + self.seq_len].clone()
        a = self.actions[safe_idx : safe_idx + self.seq_len].clone()
        r = self.rewards[safe_idx : safe_idx + self.seq_len]

        # Latent noise injection — regularization for small datasets
        z = z + torch.randn_like(z) * LATENT_NOISE_STD

        # Action jitter — smooths policy sensitivity to exact action values
        a = a + torch.randn_like(a) * ACTION_NOISE_STD
        a[:, 0].clamp_(-1.0, 1.0)    # keep steering valid
        a[:, 1].clamp_(0.0, THROTTLE_CAP)  # keep throttle valid

        return z, a, r


if __name__ == '__main__':
    print(f"Using device: {device}")
    print(f"TUB_DIR: {TUB_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"Config: batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, pin_memory={PIN_MEMORY}")

    print("\nLoading Frozen VAE Encoder...")
    vae = DeterministicEncoder().to(device)
    vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
    vae.eval()
    print(f"  Loaded {VAE_WEIGHTS}")

    print("Building Sequential Dataset...")
    dataset = DonkeyTrajectoryDataset(TUB_DIR, SEQ_LEN, vae)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=PIN_MEMORY)
    print(f"Batches per epoch: {len(dataloader)}")

    # --- TRAINING LOOP ---
    world_model = WorldModel().to(device)
    total_params = sum(p.numel() for p in world_model.parameters())
    print(f"World Model parameters: {total_params:,}")
    optimizer = optim.Adam(world_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"\nStarting World Model Training ({EPOCHS} epochs)...")
    print("-" * 80)
    for epoch in range(EPOCHS):
        world_model.train()
        total_loss, dyn_loss_total, rew_loss_total = 0, 0, 0

        for z_seq, a_seq, r_seq in dataloader:
            z_seq, a_seq, r_seq = z_seq.to(device), a_seq.to(device), r_seq.to(device)

            optimizer.zero_grad()
            z_pred, r_pred = world_model(z_seq, a_seq)
            # z_target wants the NEXT state
            z_target = z_seq[:, 1:, :]
            
            # r_target wants the CURRENT reward associated with a_t
            r_target = r_seq[:, :-1, :] 

            dynamics_loss = criterion(z_pred, z_target)
            reward_loss = criterion(r_pred, r_target)
            loss = dynamics_loss + (reward_loss * 5.0)

            loss.backward()
            
            # PROTECT THE OPTIMIZER FROM TERMINAL PENALTY SPIKES
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()
            dyn_loss_total += dynamics_loss.item()
            rew_loss_total += reward_loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {total_loss/len(dataloader):.4f} | Dyn Loss: {dyn_loss_total/len(dataloader):.4f} | Rew Loss: {rew_loss_total/len(dataloader):.4f}")

    print("-" * 80)
    print("Training Complete. Exporting World Model...")
    torch.save(world_model.state_dict(), os.path.join(MODEL_DIR, "world_model.pth"))
    print(f"  Saved world_model.pth to {MODEL_DIR}")
    print("Done! Ready for Actor-Critic dreaming.")