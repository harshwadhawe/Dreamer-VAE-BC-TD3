"""
Step 3: Dreamer-style imagination training of Actor and Critic using RSSM.

Requires: vae_encoder.pth (from train_vae.py), world_model.pth (from train_world_model.py),
          tub with telemetry.csv and images (for BC anchor seed states)
Downstream: drive_sim.py (uses dreamer_actor.pth)

Freezes VAE + RSSM world model, trains Actor and Critic entirely in imagination.
Seeds dreams by running one observe_step (posterior) on real frames to get (h_0, s_0).
Actor/Critic see full RSSM state: cat(h_t, s_t) = STATE_DIM (288).
TD3+BC behavioral cloning anchor on step 0. Imagination uses prior only (no observation).
Exports dreamer_actor.pth.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

# --- THE UNIFIED HUB ---
from core import DeterministicEncoder, Actor, RSSMWorldModel, Critic, process_sim_image, TUB_DIR, VAE_WEIGHTS, WORLD_MODEL_WEIGHTS, MODEL_DIR, device, LATENT_DIM, ACTION_DIM, HIDDEN_DIM, DETER_DIM, STOCH_DIM, STATE_DIM, BATCH_SIZE_ACTOR, MAX_TRAIN_IMAGES

# --- CONFIGURATION ---
IMAGINATION_HORIZON = 30     # Slightly shorter horizon limits compounding physics errors
BATCH_SIZE = BATCH_SIZE_ACTOR # 512 is great, drop to 256 if CUDA OOM occurs
DREAM_EPOCHS = 1000           # Prevents the Actor from exploiting World Model loopholes
GAMMA = 0.99                 # Prioritizes immediate survival over long-term planning
STEERING_PENALTY = 0.1       # Discourages extreme steering in imagined trajectories
BC_WEIGHT = 5.0              # Behavioral cloning anchor weight (lower = more RL, higher = more imitation)

if __name__ == '__main__':
    print(f"Using device: {device}")
    print(f"TUB_DIR: {TUB_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"Config: batch_size={BATCH_SIZE}, horizon={IMAGINATION_HORIZON}, dream_epochs={DREAM_EPOCHS}, gamma={GAMMA}")

    print("\nLoading Frozen VAE Encoder...")
    vae = DeterministicEncoder().to(device)
    vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
    vae.eval()
    vae.requires_grad_(False)
    print(f"  Loaded {VAE_WEIGHTS}")

    print("Loading Frozen RSSM World Model...")
    world_model = RSSMWorldModel().to(device)
    world_model.load_state_dict(torch.load(WORLD_MODEL_WEIGHTS, map_location=device))
    world_model.eval()
    world_model.requires_grad_(False)
    print(f"  Loaded {WORLD_MODEL_WEIGHTS}")

    actor = Actor().to(device)
    critic = Critic().to(device)
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"Actor parameters: {actor_params:,}")
    print(f"Critic parameters: {critic_params:,}")

    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

    # --- EXTRACT REAL SEED STATES & ACTIONS (TD3+BC ANCHOR) ---
    print("Extracting REAL seed states from your driving data...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 1. Group all rows by episode first
    episodes_data = {}
    with open(os.path.join(TUB_DIR, "telemetry.csv"), 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if MAX_TRAIN_IMAGES is not None:
            rows = rows[:MAX_TRAIN_IMAGES]
        for row in rows:
            img_path = os.path.join(TUB_DIR, row['frame'])
            if not os.path.exists(img_path): continue

            ep_id = int(row.get('episode_id', 0))
            if ep_id not in episodes_data:
                episodes_data[ep_id] = []
            episodes_data[ep_id].append(row)

    real_latents = []
    real_actions = []

    # 2. The Safety Margin
    FRAMES_TO_DROP_AT_END = 30 
    
    # We want at least 10 usable "safe" frames from an episode to consider it valid
    MIN_VALID_EP_LEN = FRAMES_TO_DROP_AT_END + 10 

    skipped_micro_episodes = 0
    valid_episodes = 0

    for ep_id, rows in episodes_data.items():
        # BUG FIX: Explicitly handle and track episodes that are too short
        if len(rows) < MIN_VALID_EP_LEN:
            skipped_micro_episodes += 1
            continue 
            
        valid_episodes += 1
        safe_rows = rows[:-FRAMES_TO_DROP_AT_END]
        
        for row in safe_rows:
            steering = float(row['steering'])
            throttle = float(row['throttle'])

            img_path = os.path.join(TUB_DIR, row['frame'])
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_cropped = process_sim_image(img_tensor)

            with torch.no_grad():
                latent = vae(img_cropped).squeeze(0).cpu()
                # Flipped copy for 2x seed states
                img_flipped = TF.hflip(img_cropped)
                latent_flip = vae(img_flipped).squeeze(0).cpu()

            # Original
            real_latents.append(latent)
            real_actions.append([steering, throttle])

            # Flipped (negated steering)
            real_latents.append(latent_flip)
            real_actions.append([-steering, throttle])

    # 3. FATAL ERROR PREVENTION: Check if we actually have data left
    if len(real_latents) == 0:
        print("\n❌ FATAL ERROR: No valid frames survived the crash filter!")
        print(f"Total episodes found: {len(episodes_data)}")
        print(f"Micro-episodes discarded (< {MIN_VALID_EP_LEN} frames): {skipped_micro_episodes}")
        print("You must record longer, safer driving sessions before training.")
        exit(1)

    real_latents = torch.stack(real_latents)
    real_actions = torch.tensor(real_actions, dtype=torch.float32)

    print(f"Discarded {skipped_micro_episodes} micro-episodes (too short).")
    print(f"Purged {valid_episodes * FRAMES_TO_DROP_AT_END} crash frames from valid episodes.")
    print(f"Loaded {len(real_latents)} purely safe track positions for the BC Anchor.")

    def get_seed_states(batch_size):
        """Sample real frames, run one posterior observe_step to get (h_0, s_0)."""
        idx = torch.randint(0, len(real_latents), (batch_size,))
        enc = real_latents[idx].to(device)
        true_a = real_actions[idx].to(device)

        # Bootstrap RSSM state from one observation (h,s initialized to zeros)
        h_init = torch.zeros(batch_size, DETER_DIM, device=device)
        s_init = torch.zeros(batch_size, STOCH_DIM, device=device)
        a_init = torch.zeros(batch_size, ACTION_DIM, device=device)

        with torch.no_grad():
            s0, h0 = world_model.observe_step(s_init, a_init, h_init, enc)

        return s0, h0, true_a
    
    # --- DREAMING LOOP ---
    print(f"\nInitiating RSSM Imagination Engine ({DREAM_EPOCHS} dream epochs)...")
    print("-" * 80)

    for epoch in range(DREAM_EPOCHS):
        # Seed: get (h_0, s_0) from real observations via posterior
        s_t, h_t, true_a_t = get_seed_states(BATCH_SIZE)

        actor_loss = 0
        critic_loss = 0
        bc_loss = 0

        for t in range(IMAGINATION_HORIZON):
            state_t = torch.cat([h_t, s_t], dim=-1)
            a_t = actor(state_t)

            # TD3+BC Anchor: Penalize straying from human driving on step 0
            if t == 0:
                bc_loss = nn.functional.mse_loss(a_t, true_a_t)

            # forward_step: returns reward for current state, advances to next state
            s_next, r_t, h_next = world_model.forward_step(s_t, a_t, h_t)

            state_next = torch.cat([h_next, s_next], dim=-1)
            v_next = critic(state_next)
            steering_penalty = torch.pow(a_t[:, 0:1], 2) * STEERING_PENALTY
            target_return = r_t + GAMMA * v_next - steering_penalty

            actor_loss += -target_return.mean()

            # Critic loss: bootstrap from next value estimate
            current_v = critic(state_t.detach())
            v_next_detached = critic(state_next.detach())
            critic_target = (r_t + GAMMA * v_next_detached - steering_penalty).detach()
            critic_loss += nn.functional.mse_loss(current_v, critic_target)

            s_t = s_next
            h_t = h_next

        actor_loss = actor_loss / IMAGINATION_HORIZON
        critic_loss = critic_loss / IMAGINATION_HORIZON

        total_actor_loss = actor_loss + (bc_loss * BC_WEIGHT)

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
    print("-" * 80)
    print("Training Complete. Exporting Actor...")
    torch.save(actor.state_dict(), os.path.join(MODEL_DIR, "dreamer_actor.pth"))
    print(f"  Saved dreamer_actor.pth to {MODEL_DIR}")
    print("Done! Autopilot updated.")