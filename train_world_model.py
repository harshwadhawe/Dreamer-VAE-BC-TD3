import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from core import DeterministicEncoder, WorldModel, process_sim_image, TUB_DIR, VAE_WEIGHTS, MODEL_DIR, device, LATENT_DIM, ACTION_DIM, HIDDEN_DIM

# --- CONFIGURATION ---
SEQ_LEN = 32     # UPGRADE: Gives the physics engine a 1.5-second memory of momentum    
BATCH_SIZE = 64
EPOCHS = 100

# --- TRAJECTORY DATASET ---
class DonkeyTrajectoryDataset(Dataset):
    def __init__(self, tub_dir, seq_len, vae_model):
        self.seq_len = seq_len
        self.transform = transforms.Compose([transforms.ToTensor()])

        telemetry_path = os.path.join(tub_dir, "telemetry.csv")
        if not os.path.exists(telemetry_path):
            raise FileNotFoundError(f"Could not find {telemetry_path}!")

        print(f"Reading telemetry from {telemetry_path}...")

        self.latents = []
        self.actions = []
        self.rewards = []
        self.episodes = [] # NEW: Track episode segments

        with open(telemetry_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader) # Load all rows so we can look ahead
            
            for i, row in enumerate(rows):
                img_path = os.path.join(tub_dir, row['frame'])
                if not os.path.exists(img_path): continue

                steering = float(row['steering'])
                throttle = float(row['throttle'])
                ep_id = int(row.get('episode_id', 0))

                action = [steering, throttle]
                # Extract the true Unity environmental reward from your new CSV column!
                reward = throttle * 1.0 - abs(steering) * 0.1 # Simple reward: encourage forward throttle, penalize sharp steering

                # --- THE TERMINAL PENALTY ---
                # If the next row is a new episode (or we hit the end of the file), 
                # this current frame is the CRASH. Penalize it heavily!
                is_terminal = False
                if i == len(rows) - 1:
                    is_terminal = True
                elif int(rows[i+1].get('episode_id', 0)) != ep_id:
                    is_terminal = True

                if is_terminal:
                    reward = -10.0 

                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(device)
                img_cropped = process_sim_image(img_tensor)

                with torch.no_grad():
                    latent = vae_model(img_cropped).squeeze(0).cpu().numpy()

                self.latents.append(latent)
                self.actions.append(action)
                self.rewards.append([reward])
                self.episodes.append(ep_id)

                if i % 1000 == 0 and i > 0:
                    print(f"Processed {i} frames through the VAE...")

        self.latents = torch.tensor(np.array(self.latents), dtype=torch.float32)
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        self.rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32)

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
        
        z = self.latents[safe_idx : safe_idx + self.seq_len]
        a = self.actions[safe_idx : safe_idx + self.seq_len]
        r = self.rewards[safe_idx : safe_idx + self.seq_len]
        return z, a, r


if __name__ == '__main__':
    print(f"Using Device: {device}")

    print("Loading Frozen VAE Encoder...")
    vae = DeterministicEncoder().to(device)
    vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
    vae.eval()

    print("Building Sequential Dataset...")
    dataset = DonkeyTrajectoryDataset(TUB_DIR, SEQ_LEN, vae)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --- TRAINING LOOP ---
    world_model = WorldModel().to(device)
    optimizer = optim.Adam(world_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("\nStarting World Model Training...")
    for epoch in range(EPOCHS):
        world_model.train()
        total_loss, dyn_loss_total, rew_loss_total = 0, 0, 0

        for z_seq, a_seq, r_seq in dataloader:
            z_seq, a_seq, r_seq = z_seq.to(device), a_seq.to(device), r_seq.to(device)

            optimizer.zero_grad()
            z_pred, r_pred = world_model(z_seq, a_seq)

            z_target = z_seq[:, 1:, :]
            r_target = r_seq[:, 1:, :]

            dynamics_loss = criterion(z_pred, z_target)
            reward_loss = criterion(r_pred, r_target)
            loss = dynamics_loss + (reward_loss * 5.0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            dyn_loss_total += dynamics_loss.item()
            rew_loss_total += reward_loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {total_loss/len(dataloader):.4f} | Dyn Loss: {dyn_loss_total/len(dataloader):.4f} | Rew Loss: {rew_loss_total/len(dataloader):.4f}")

    print("\nExporting the Imagination Engine (World Model)...")
    torch.save(world_model.state_dict(), os.path.join(MODEL_DIR, "world_model.pth"))
    print("Done! We are ready to let the Actor-Critic Dream.")