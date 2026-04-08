"""
Centralized configuration and model architectures for the Dreamer pipeline.

Requires: Nothing (this is the root dependency)
Downstream: Every other script imports from here.

Contains all shared constants (image dims, latent dim, batch sizes, device selection),
unified image preprocessing (process_sim_image), and all neural network architectures
(DeterministicEncoder, Actor, Critic, WorldModel). Hardware-aware config auto-selects
batch sizes and workers for CUDA / MPS / CPU.

Pipeline Dependency Graph:

    core.py (root - no dependencies)
      |
      +-- collect_data.py --------------+
      |
      +-- add_telemetry_to_tub.py       |  (data sources)
      |                                 |
      +-- train_vae.py <----------------+
      |     | vae_encoder.pth
      +-- train_world_model.py
      |     | world_model.pth
      +-- train_actor_critic.py
      |     | dreamer_actor.pth
      |
      +-- auto_run.py (orchestrates the 3 training steps)
      |
      +-- drive_sim.py (simulator inference)
      +-- export_pth_to_tflite.py -> drive_physical_tflite.py (Pi deployment)
"""

import os
import torch
import torch.nn as nn

# --- UNIFIED CONFIGURATION ---

# Paths
TUB_NAME = "tub_1_26-04-06"  # Default tub name (can be overridden by auto_run.py CLI arg)
TUB_DIR = os.getenv('TUB_DIR', f'./data/{TUB_NAME}')
MODEL_DIR = os.getenv('MODEL_DIR', f'./models/{TUB_NAME}')
VAE_WEIGHTS = os.path.join(MODEL_DIR, 'vae_encoder.pth')
WORLD_MODEL_WEIGHTS = os.path.join(MODEL_DIR, 'world_model.pth')
ACTOR_WEIGHTS = os.path.join(MODEL_DIR, 'dreamer_actor.pth')
# Image preprocessing
IMG_CROP_TOP = 40
IMG_HEIGHT = 80
IMG_WIDTH = 128

# Model dimensions
LATENT_DIM = 32
ACTION_DIM = 2
HIDDEN_DIM = 256
ENCODER_CHANNELS = [3, 32, 64, 128, 256]
ACTOR_HIDDEN = 64
CRITIC_HIDDEN = 64
DYNAMICS_HIDDEN = 128
REWARD_HIDDEN = 64

# After 4 stride-2 convs on (IMG_HEIGHT x IMG_WIDTH), flatten size = 256 * 5 * 8
ENCODER_FLAT_DIM = ENCODER_CHANNELS[-1] * (IMG_HEIGHT // 16) * (IMG_WIDTH // 16)

# Actor output scaling
THROTTLE_CAP = 0.3

# Simulator
SIM_HOST = "127.0.0.1"
SIM_PORT = 9091
SIM_ENV = "donkey-generated-roads-v0"  # Change this to your desired track/environment

# donkey-minimonaco-track-v0
# donkey-generated-roads-v0 
# donkey-generated-track-v0 
# donkey-warehouse-v0 
# donkey-circuit-launch-track-v0 
# donkey-warren-track-v0 
# donkey-roboracingleague-track-v0

# --- UNIVERSAL HARDWARE SELECTOR ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    # CUDA config (RTX 4060 / 8GB+)
    BATCH_SIZE_VAE = 256
    BATCH_SIZE_WORLD = 256
    BATCH_SIZE_ACTOR = 256
    NUM_WORKERS = 4
    DATASET_MULTIPLIER = 5
    PIN_MEMORY = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    # MPS config (Apple Silicon)
    BATCH_SIZE_VAE = 64
    BATCH_SIZE_WORLD = 64
    BATCH_SIZE_ACTOR = 128
    NUM_WORKERS = 2
    DATASET_MULTIPLIER = 1
    PIN_MEMORY = False
else:
    device = torch.device("cpu")
    BATCH_SIZE_VAE = 32
    BATCH_SIZE_WORLD = 32
    BATCH_SIZE_ACTOR = 64
    NUM_WORKERS = 0
    DATASET_MULTIPLIER = 1
    PIN_MEMORY = False

print(f"Using device: {device}")
# --- UNIFIED IMAGE PREPROCESSING ---
def process_sim_image(img_tensor):
    """
    Ensures identical cropping across Data Collection, Training, and Simulation.
    Expects PyTorch Tensor (B, C, H, W). Crops top pixels and center-crops width.
    """
    _, _, h, w = img_tensor.shape
    start_x = (w - IMG_WIDTH) // 2
    return img_tensor[:, :, IMG_CROP_TOP:IMG_CROP_TOP + IMG_HEIGHT, start_x:start_x + IMG_WIDTH]

# --- UNIFIED ARCHITECTURE ---
class DeterministicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = ENCODER_CHANNELS
        self.encoder = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch[1], ch[2], 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch[2], ch[3], 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(ch[3], ch[4], 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(ENCODER_FLAT_DIM, LATENT_DIM)

    def forward(self, x):
        return self.fc_mu(self.encoder(x))

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, ACTOR_HIDDEN), nn.ELU(),
            nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN), nn.ELU(),
            nn.Linear(ACTOR_HIDDEN, ACTION_DIM)
        )

    def forward(self, z):
        raw_action = self.net(z)
        steer = torch.tanh(raw_action[:, 0:1]) # The smooth curve
        throttle = torch.sigmoid(raw_action[:, 1:2]) * THROTTLE_CAP
        return torch.cat([steer, throttle], dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, CRITIC_HIDDEN), nn.ELU(),
            nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN), nn.ELU(),
            nn.Linear(CRITIC_HIDDEN, 1)
        )

    def forward(self, z):
        return self.net(z)

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRUCell(input_size=LATENT_DIM + ACTION_DIM, hidden_size=HIDDEN_DIM)
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(HIDDEN_DIM, DYNAMICS_HIDDEN), nn.ReLU(), nn.Linear(DYNAMICS_HIDDEN, LATENT_DIM)
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(HIDDEN_DIM, REWARD_HIDDEN), nn.ReLU(), nn.Linear(REWARD_HIDDEN, 1)
        )

    def forward(self, z_seq, a_seq):
        """
        Unrolls the RNN over a sequence.
        z_seq: [Batch, Seq_Len, Latent_Dim]
        a_seq: [Batch, Seq_Len, Action_Dim]
        """
        batch_size, seq_len, _ = z_seq.shape
        h = torch.zeros(batch_size, HIDDEN_DIM).to(z_seq.device)

        predicted_latents = []
        predicted_rewards = []

        for t in range(seq_len - 1):
            current_z = z_seq[:, t, :]
            current_a = a_seq[:, t, :]

            rnn_input = torch.cat([current_z, current_a], dim=-1)
            h = self.rnn(rnn_input, h)

            predicted_latents.append(self.dynamics_predictor(h))
            predicted_rewards.append(self.reward_predictor(h))

        return torch.stack(predicted_latents, dim=1), torch.stack(predicted_rewards, dim=1)

    def forward_step(self, z_t, a_t, h_t):
        rnn_input = torch.cat([z_t, a_t], dim=-1)
        h_next = self.rnn(rnn_input, h_t)
        z_next = self.dynamics_predictor(h_next)
        reward = self.reward_predictor(h_next)
        return z_next, reward, h_next
