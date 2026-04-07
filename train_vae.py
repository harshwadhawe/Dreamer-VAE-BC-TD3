"""
Step 1: Train a Beta-VAE to compress camera images into a compact latent space.

Requires: tub with .jpg images (from collect_data.py or add_telemetry_to_tub.py)
Downstream: train_world_model.py (uses frozen vae_encoder.pth)

Trains a convolutional VAE (80x128x3 -> 32-dim latent) with aggressive augmentation
(ColorJitter, RandomGrayscale, GaussianBlur, RandomErasing) to force geometry-only
features. Uses Free Bits KL (0.5 nats/dim) to prevent posterior collapse and beta
warmup (0 -> 0.1 over 3 epochs). Exports DeterministicEncoder (mu-only, no sampling)
as .pth and full VAE as full_vae.pth for reconstruction visualization.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from core import (
    DeterministicEncoder, TUB_DIR, MODEL_DIR, device,
    IMG_HEIGHT, IMG_WIDTH, IMG_CROP_TOP, LATENT_DIM, ENCODER_CHANNELS, ENCODER_FLAT_DIM,
    DATASET_MULTIPLIER, BATCH_SIZE_VAE, NUM_WORKERS, PIN_MEMORY
)

# Beta warm-up: ramps from 0 to BETA_MAX over BETA_WARMUP_EPOCHS
BETA_MAX = 0.1
BETA_WARMUP_EPOCHS = 3

# --- CONFIGURATION ---
EPOCH_IMG_DIR = os.path.join(MODEL_DIR, 'epoch_images_torch')
BATCH_SIZE = BATCH_SIZE_VAE
EPOCHS = 50

os.makedirs(EPOCH_IMG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class DonkeyTubDataset(Dataset):
    def __init__(self, tub_dir):
        self.img_paths = glob.glob(os.path.join(tub_dir, '**', '*.jpg'), recursive=True)
        print(f"Found {len(self.img_paths)} images in {tub_dir}")
        if len(self.img_paths) == 0:
            raise ValueError("No images found! Check your TUB_DIR path.")
        
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.RandomGrayscale(p=0.05), 
            transforms.ToTensor(), 
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        ])

    def __len__(self):
        # 2. Multiply the reported length
        return len(self.img_paths) * DATASET_MULTIPLIER

    def __getitem__(self, idx):
        # 3. The Modulus math: Wrap the index back to the real file limit
        real_idx = idx % len(self.img_paths)
        
        img = Image.open(self.img_paths[real_idx]).convert('RGB')
        img_tensor = self.transform(img)
        
        _, h, w = img_tensor.shape
        start_x = (w - IMG_WIDTH) // 2
        img_cropped = img_tensor[:, IMG_CROP_TOP:IMG_CROP_TOP+IMG_HEIGHT, start_x:start_x+IMG_WIDTH]
        return img_cropped


# --- VAE ARCHITECTURE (Dreamer-Compatible) ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        ch = ENCODER_CHANNELS  # [3, 32, 64, 128, 256]

        # ENCODER — must match DeterministicEncoder in core.py
        self.encoder = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch[1], ch[2], kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch[2], ch[3], kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(ENCODER_FLAT_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(ENCODER_FLAT_DIM, LATENT_DIM)

        # DECODER
        self.fc_decode = nn.Linear(LATENT_DIM, ENCODER_FLAT_DIM)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch[4], ch[3], kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(ch[3], ch[2], kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(ch[2], ch[1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(ch[1], ch[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, ENCODER_CHANNELS[-1], IMG_HEIGHT // 16, IMG_WIDTH // 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


# def vae_loss(recon_x, x, mu, logvar, beta):
#     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
#     kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + (beta * kl_divergence)
def vae_loss(recon_x, x, mu, logvar, beta):
    # 1. Reconstruction Loss (How well does it draw the track?)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # 2. Raw KL Divergence (Calculated per dimension, per item in batch)
    # Shape: [Batch_Size, LATENT_DIM]
    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. FREE BITS: Allow 0.5 "Nats" of free information per dimension.
    # If the KL penalty is below 0.5, we clamp it to 0.5. 
    # This prevents the network from injecting pure static into unused dimensions.
    free_nats = 0.5
    kl_div_clamped = torch.clamp(kl_div, min=free_nats)
    
    # Sum the clamped KL loss
    kl_loss = torch.sum(kl_div_clamped)
    
    return recon_loss + (beta * kl_loss)

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    print(f"Using device: {device}")
    print(f"TUB_DIR: {TUB_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"Config: batch_size={BATCH_SIZE}, workers={NUM_WORKERS}, multiplier={DATASET_MULTIPLIER}, pin_memory={PIN_MEMORY}")
    print(f"VAE: latent_dim={LATENT_DIM}, beta_max={BETA_MAX}, beta_warmup={BETA_WARMUP_EPOCHS}")

    dataset = DonkeyTubDataset(TUB_DIR)
    print(f"Effective dataset size: {len(dataset)} ({len(dataset)//DATASET_MULTIPLIER} images x {DATASET_MULTIPLIER} multiplier)")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print(f"Batches per epoch: {len(dataloader)}")

    model = VAE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VAE parameters: {total_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Grab a fixed batch for visualizing progress
    test_batch = next(iter(dataloader)).to(device)

    print(f"\nStarting VAE Training ({EPOCHS} epochs)...")
    print("-" * 60)
    for epoch in range(EPOCHS):
        # Beta warm-up: ramp from 0 to BETA_MAX over BETA_WARMUP_EPOCHS
        beta = min(BETA_MAX, BETA_MAX * epoch / max(BETA_WARMUP_EPOCHS, 1))

        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar, beta)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} \t Loss: {avg_loss:.4f} \t Beta: {beta:.3f}")

        # Save Epoch Image
        model.eval()
        with torch.no_grad():
            recon_imgs, _, _ = model(test_batch)

            fig, axes = plt.subplots(2, 12, figsize=(25, 5))
            for i in range(12):
                orig = test_batch[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
                recon = recon_imgs[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)

                axes[0, i].imshow(orig)
                axes[0, i].axis('off')
                axes[1, i].imshow(recon)
                axes[1, i].axis('off')

            plt.suptitle(f"Epoch {epoch + 1}")
            plt.savefig(os.path.join(EPOCH_IMG_DIR, f"epoch_{epoch + 1:03d}.png"))
            plt.close(fig)

    # --- EXPORT ---
    print("-" * 60)
    print("Training Complete. Exporting models...")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "full_vae.pth"))
    print(f"  Saved full_vae.pth")

    export_model = DeterministicEncoder().to("cpu")
    export_model.encoder.load_state_dict(model.encoder.state_dict())
    export_model.fc_mu.load_state_dict(model.fc_mu.state_dict())
    export_model.eval()

    torch.save(export_model.state_dict(), os.path.join(MODEL_DIR, "vae_encoder.pth"))
    print(f"  Saved vae_encoder.pth")

    print(f"Done! Encoder exported to {MODEL_DIR}")