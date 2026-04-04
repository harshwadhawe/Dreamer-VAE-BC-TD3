import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

from core import DeterministicEncoder, Actor, process_sim_image, TUB_DIR, VAE_WEIGHTS, MODEL_DIR, device


if __name__ == '__main__':
    print(f"Using Device: {device}")

    # --- LOAD VAE & INITIALIZE ACTOR ---
    print("Loading VAE...")
    vae = DeterministicEncoder().to(device)
    vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
    vae.eval()

    actor = Actor().to(device)
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # --- EXTRACT LATENTS & ACTIONS ---
    print("Converting CSV into Latent Training Data...")
    transform = transforms.Compose([transforms.ToTensor()])

    latents = []
    targets = []

    csv_path = os.path.join(TUB_DIR, "telemetry.csv")
    print(f"Opening CSV at: {os.path.abspath(csv_path)}")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            img_path = os.path.join(TUB_DIR, row['frame'])

            if not os.path.exists(img_path):
                print(f"MISSING: Looked for {os.path.abspath(img_path)}")
                continue
            elif i == 0:
                print(f"SUCCESS: Found first image at {os.path.abspath(img_path)}")

            steering = float(row['steering'])
            throttle = float(row['throttle'])
            target_action = [steering, throttle]

            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_cropped = process_sim_image(img_tensor)

            with torch.no_grad():
                latent_state = vae(img_cropped).squeeze(0)

            latents.append(latent_state)
            targets.append(torch.tensor(target_action, dtype=torch.float32).to(device))

    X = torch.stack(latents)
    Y = torch.stack(targets)
    print(f"Ready to train on {len(X)} frames.")

    # --- BEHAVIORAL CLONING TRAINING LOOP ---
    print("\nTraining Latent Mimic Actor...")
    EPOCHS = 250
    BATCH_SIZE = 64

    for epoch in range(EPOCHS):
        permutation = torch.randperm(X.size()[0])
        epoch_loss = 0

        actor.train()
        for i in range(0, X.size()[0], BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = X[indices], Y[indices]

            optimizer.zero_grad()
            
            # THE FIX: Simulate the car drifting slightly off the perfect line
            # This forces the Actor to learn robust, generalized steering
            noise = torch.randn_like(batch_x) * 0.05
            noisy_batch_x = batch_x + noise
            
            predictions = actor(noisy_batch_x)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 25 == 0:
            print(f"Epoch {epoch:03d} | Imitation Loss: {epoch_loss/X.size()[0]:.4f}")

    # --- EXPORT ---
    print("\nExporting the new Actor...")
    torch.save(actor.state_dict(), os.path.join(MODEL_DIR, "dreamer_actor.pth"))
    print("Done! The hallucinating AI has been replaced.")
