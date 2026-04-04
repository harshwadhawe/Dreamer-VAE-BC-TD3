import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# --- THE UNIFIED HUB ---
from core import DeterministicEncoder, Actor, WorldModel, Critic, process_sim_image, TUB_DIR, VAE_WEIGHTS, WORLD_MODEL_WEIGHTS, MODEL_DIR, device, LATENT_DIM, ACTION_DIM, HIDDEN_DIM

# --- CONFIGURATION ---
IMAGINATION_HORIZON = 15
BATCH_SIZE = 128
DREAM_EPOCHS = 1000
GAMMA = 0.99

if __name__ == '__main__':
    print(f"Using Device: {device}")

    print("Loading Models from core.py architecture...")
    vae = DeterministicEncoder().to(device)
    vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
    vae.eval()

    world_model = WorldModel().to(device)
    world_model.load_state_dict(torch.load(WORLD_MODEL_WEIGHTS, map_location=device))
    world_model.eval()

    actor = Actor().to(device)
    critic = Critic().to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

    # --- EXTRACT REAL SEED STATES & ACTIONS (TD3+BC ANCHOR) ---
    print("Extracting REAL seed states from your driving data...")
    transform = transforms.Compose([transforms.ToTensor()])
    real_latents = []
    real_actions = []

    with open(os.path.join(TUB_DIR, "telemetry.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # if i >= 1500: break
            img_path = os.path.join(TUB_DIR, row['frame'])
            if not os.path.exists(img_path): continue

            steering = float(row['steering'])
            throttle = float(row['throttle'])
            real_actions.append([steering, throttle])

            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Using the unified cropper
            img_cropped = process_sim_image(img_tensor)

            with torch.no_grad():
                latent = vae(img_cropped).squeeze(0)
                real_latents.append(latent)

    real_latents = torch.stack(real_latents).to(device)
    real_actions = torch.tensor(real_actions, dtype=torch.float32).to(device)
    print(f"Loaded {len(real_latents)} real track positions. No more void dreaming.")

    def get_seed_states(batch_size):
        idx = torch.randint(0, len(real_latents), (batch_size,))
        return real_latents[idx], real_actions[idx]

    # --- DREAMING LOOP ---
    print("\nInitiating Imagination Engine...")

    for epoch in range(DREAM_EPOCHS):
        z_t, true_a_t = get_seed_states(BATCH_SIZE)
        h_t = torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(device)

        actor_loss = 0
        critic_loss = 0
        bc_loss = 0

        for t in range(IMAGINATION_HORIZON):
            a_t = actor(z_t)
            
            # TD3+BC Anchor: Penalize straying from human driving on step 0
            if t == 0:
                bc_loss = nn.functional.mse_loss(a_t, true_a_t)

            z_next, r_t, h_next = world_model.forward_step(z_t, a_t, h_t)

            v_next = critic(z_next)
            steering_penalty = torch.pow(a_t[:, 0:1], 2) 
            target_return = r_t + GAMMA * v_next - steering_penalty
            actor_loss += -target_return.mean()

            # CRITIC LOSS: Detached targets (fixes the retain_graph memory leak)
            current_v = critic(z_t.detach())
            v_next_detached = critic(z_next.detach())
            critic_target = r_t + GAMMA * v_next_detached - steering_penalty.detach()
            critic_loss += nn.functional.mse_loss(current_v, critic_target.detach())

            z_t = z_next
            h_t = h_next

        actor_loss = actor_loss / IMAGINATION_HORIZON
        critic_loss = critic_loss / IMAGINATION_HORIZON

        # Blend RL Imagination with the Reality Anchor
        total_actor_loss = actor_loss + (bc_loss * 10.0)

        actor_opt.zero_grad()
        total_actor_loss.backward() 
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_opt.step()

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        if epoch % 100 == 0:
            print(f"Dream Epoch {epoch:04d} | RL Loss: {actor_loss.item():.4f} | BC Anchor: {bc_loss.item():.4f}")

    # --- EXPORT ---
    print("\nWaking up. Exporting the trained Actor...")
    torch.save(actor.state_dict(), os.path.join(MODEL_DIR, "dreamer_actor.pth"))
    print("Done! Autopilot updated.")