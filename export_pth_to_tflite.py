"""
Converts PyTorch .pth models to TFLite for Raspberry Pi deployment.

Requires: vae_encoder.pth + dreamer_actor.pth (from training pipeline)
Downstream: drive_physical_tflite.py (uses .tflite models on Pi)

Uses litert_torch to convert the VAE encoder and Actor from PyTorch to TFLite format.
Traces both models with dummy inputs to generate optimized edge inference graphs.
Outputs vae_encoder.tflite and dreamer_actor.tflite to the same model directory.

Usage:
    python export_pth_to_tflite.py <model_dir>
    python export_pth_to_tflite.py models/tub_1_26-04-01
"""

import os
import sys
import torch
import litert_torch
from core import DeterministicEncoder, Actor, MODEL_DIR

if __name__ == '__main__':
    # Accept CLI arg or fall back to MODEL_DIR env var
    # if len(sys.argv) > 1:
    #     model_dir = sys.argv[1]
    # elif os.getenv('MODEL_DIR'):
    #     model_dir = os.getenv('MODEL_DIR')
    # else:
    #     print("Usage: python export_pth_to_tflite.py <model_dir>")
    #     print("Example: python export_pth_to_tflite.py models/tub_1_26-04-01")
    #     sys.exit(1)

    print(f"Loading PyTorch Brains from {MODEL_DIR}...")
    vae = DeterministicEncoder().eval()
    vae.load_state_dict(torch.load(f"{MODEL_DIR}/vae_encoder.pth", map_location="cpu"))

    actor = Actor().eval()
    actor.load_state_dict(torch.load(f"{MODEL_DIR}/dreamer_actor.pth", map_location="cpu"))

    # Create the exact dummy inputs (Dynamically sizing the Actor's input)
    dummy_img = torch.randn(1, 3, 80, 128)

    with torch.no_grad():
        dummy_latent = vae(dummy_img)

    # Direct PyTorch -> TFLite Conversion
    print("Converting VAE to TFLite (This may take a minute)...")
    edge_vae = litert_torch.convert(vae, (dummy_img,))
    edge_vae.export(f"{MODEL_DIR}/vae_encoder.tflite")

    print("Converting Actor to TFLite...")
    edge_actor = litert_torch.convert(actor, (dummy_latent,))
    edge_actor.export(f"{MODEL_DIR}/dreamer_actor.tflite")

    print(f"\n=== SUCCESS ===")
    print(f"TFLite models saved to {MODEL_DIR}/")
