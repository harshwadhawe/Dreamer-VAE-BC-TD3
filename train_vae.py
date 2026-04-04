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
    IMG_HEIGHT, IMG_WIDTH, IMG_CROP_TOP, LATENT_DIM, ENCODER_CHANNELS, ENCODER_FLAT_DIM
)

# --- CONFIGURATION ---
EPOCH_IMG_DIR = os.path.join(MODEL_DIR, 'epoch_images_torch')
BATCH_SIZE = 64
EPOCHS = 50

os.makedirs(EPOCH_IMG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- DATASET LOADER ---
class DonkeyTubDataset(Dataset):
    def __init__(self, tub_dir):
        self.img_paths = glob.glob(os.path.join(tub_dir, '**', '*.jpg'), recursive=True)
        print(f"Found {len(self.img_paths)} images in {tub_dir}")
        if len(self.img_paths) == 0:
            raise ValueError("No images found! Check your TUB_DIR path.")
        
        # PyTorch expects Channels-First (C, H, W)
        self.transform = transforms.Compose([
            # Artificially shift lighting/shadows to force the network to learn track geometry
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.ToTensor(), # Converts to [0.0, 1.0] and (C, H, W)
        ])


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img_tensor = self.transform(img)
        # Crop top 40 pixels: tensor shape is [3, 120, 160] (Sim default) or [3, 120, 128]
        # We crop the top 40px and take the center 128px for width
        _, h, w = img_tensor.shape
        start_x = (w - IMG_WIDTH) // 2
        img_cropped = img_tensor[:, IMG_CROP_TOP:IMG_CROP_TOP+IMG_HEIGHT, start_x:start_x+IMG_WIDTH]
        return img_cropped


# --- VAE ARCHITECTURE (Dreamer-Compatible) ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        ch = ENCODER_CHANNELS  # [3, 32, 64, 128, 256]

        # ENCODER
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


def vae_loss(recon_x, x, mu, logvar, beta=0.5):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # The Beta Multiplier prevents the KL term from erasing sharp shadows
    return recon_loss + (beta * kl_divergence)

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # Everything below here is protected from the multiprocessing workers
    
    dataset = DonkeyTubDataset(TUB_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Grab a fixed batch for visualizing progress
    test_batch = next(iter(dataloader)).to(device)

    print("Starting PyTorch Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_loss = train_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} \t Average Loss: {avg_loss:.4f}")
        
        # Save Epoch Image
        model.eval()
        with torch.no_grad():
            recon_imgs, _, _ = model(test_batch)
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 5))
            for i in range(5):
                orig = test_batch[i].cpu().permute(1, 2, 0).numpy()
                recon = recon_imgs[i].cpu().permute(1, 2, 0).numpy()
                
                axes[0, i].imshow(orig)
                axes[0, i].axis('off')
                axes[1, i].imshow(recon)
                axes[1, i].axis('off')
                
            plt.suptitle(f"Epoch {epoch + 1}")
            plt.savefig(os.path.join(EPOCH_IMG_DIR, f"epoch_{epoch + 1:03d}.png"))
            plt.close(fig)

    # --- EXPORT ---
    print("Training Complete. Exporting Deterministic Encoder...")

    # NEW: Save the full brain so we can visualize reconstructions later!
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "full_vae.pth"))

    export_model = DeterministicEncoder().to("cpu")
    
    # Surgically copy only the trained Encoder weights from the VAE
    export_model.encoder.load_state_dict(model.encoder.state_dict())
    export_model.fc_mu.load_state_dict(model.fc_mu.state_dict())
    
    export_model.eval()

    torch.save(export_model.state_dict(), os.path.join(MODEL_DIR, "vae_encoder.pth"))

    dummy_input = torch.randn(1, ENCODER_CHANNELS[0], IMG_HEIGHT, IMG_WIDTH)
    onnx_path = os.path.join(MODEL_DIR, "vae_encoder.onnx")
    torch.onnx.export(
        export_model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        input_names=['input'], 
        output_names=['output']
    )

    print(f"Success! Encoder saved as PyTorch (.pth) and ONNX (.onnx) in {MODEL_DIR}")