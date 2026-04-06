import os
import glob
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Import only the image processor from core
from core import process_sim_image, device

# --- CONFIGURATION ---
TUB_1 = "./tub_sim"       
TUB_2 = "./tub_sim"  
FULL_VAE_WEIGHTS = "./models/transfer/full_vae.pth"
NUM_IMAGES = 5
LATENT_DIM = 128 # Must match your expanded brain!

# --- FULL VAE ARCHITECTURE ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(10240, LATENT_DIM)
        self.fc_logvar = nn.Linear(10240, LATENT_DIM)
        
        # DECODER
        self.fc_decode = nn.Linear(LATENT_DIM, 10240)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
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
        h = h.view(-1, 256, 5, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

def visualize_reconstructions():
    print(f"Loading Full VAE from {FULL_VAE_WEIGHTS}...")
    
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(FULL_VAE_WEIGHTS, map_location=device))
    vae.eval()

    transform = transforms.Compose([transforms.ToTensor()])

    print("Fetching images...")
    tub1_imgs = glob.glob(os.path.join(TUB_1, '**', '*.jpg'), recursive=True)
    tub2_imgs = glob.glob(os.path.join(TUB_2, '**', '*.jpg'), recursive=True)

    sample1 = random.sample(tub1_imgs, NUM_IMAGES)
    sample2 = random.sample(tub2_imgs, NUM_IMAGES)

    fig, axes = plt.subplots(4, NUM_IMAGES, figsize=(15, 10))
    
    all_paths = [sample1, sample2]
    titles = ["Track", "Warehouse"]

    for row_group in range(2):
        for col in range(NUM_IMAGES):
            # Load and process original image
            img = Image.open(all_paths[row_group][col]).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_cropped = process_sim_image(img_tensor)
            
            # Ask the AI to dream the image back
            with torch.no_grad():
                reconstruction = vae(img_cropped)
            
            # Format images for plotting
            orig_plot = img_cropped.squeeze(0).cpu().permute(1, 2, 0).numpy()
            recon_plot = reconstruction.squeeze(0).cpu().permute(1, 2, 0).numpy()
            
            # Plot Original
            axes[row_group * 2, col].imshow(orig_plot)
            axes[row_group * 2, col].axis('off')
            if col == 2: axes[row_group * 2, col].set_title(f"{titles[row_group]} (Original)", fontweight='bold')
            
            # Plot Reconstruction
            axes[row_group * 2 + 1, col].imshow(recon_plot)
            axes[row_group * 2 + 1, col].axis('off')
            if col == 2: axes[row_group * 2 + 1, col].set_title(f"{titles[row_group]} (AI Reconstruction)", fontweight='bold')

    plt.tight_layout()
    save_path = "reconstruction_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Success! Open '{save_path}' to see how the AI views the world.")
    plt.show()

if __name__ == "__main__":
    visualize_reconstructions()