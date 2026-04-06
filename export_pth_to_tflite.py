import torch
import litert_torch
from core import DeterministicEncoder, Actor

# 1. Point to your specific model folder
tub_dir = "models/tub_1_26-04-01" 

print("Loading PyTorch Brains...")
vae = DeterministicEncoder().eval()
vae.load_state_dict(torch.load(f"{tub_dir}/vae_encoder.pth", map_location="cpu"))

actor = Actor().eval()
actor.load_state_dict(torch.load(f"{tub_dir}/dreamer_actor.pth", map_location="cpu"))

# 2. Create the exact dummy inputs (Dynamically sizing the Actor's input)
dummy_img = torch.randn(1, 3, 80, 128) 

with torch.no_grad():
    dummy_latent = vae(dummy_img)

# 3. Direct PyTorch -> TFLite Conversion
print("Converting VAE to TFLite (This may take a minute)...")
# The comma after dummy_img is required because it expects a tuple of inputs
edge_vae = litert_torch.convert(vae, (dummy_img,))
edge_vae.export(f"{tub_dir}/vae_encoder.tflite")

print("Converting Actor to TFLite...")
edge_actor = litert_torch.convert(actor, (dummy_latent,))
edge_actor.export(f"{tub_dir}/dreamer_actor.tflite")

print("\n=== SUCCESS ===")
print("Your native .tflite models are ready for the Pi!")
