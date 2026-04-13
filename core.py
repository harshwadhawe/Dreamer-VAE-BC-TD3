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
      +-- collect_sim_data.py ----------+
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

# Simulator
SIM_HOST = "127.0.0.1"
SIM_PORT = 9091
SIM_ENV = "donkey-generated-track-v0"  # Change this to your desired track/environment

# donkey-minimonaco-track-v0
# donkey-generated-roads-v0
# donkey-generated-track-v0
# donkey-warehouse-v0
# donkey-circuit-launch-track-v0
# donkey-warren-track-v0
# donkey-roboracingleague-track-v0

# Paths — derived from SIM_ENV so collect + training always agree
_env_name = SIM_ENV.removeprefix("donkey-").removesuffix("-v0").replace("-", "_")
# TUB_NAME = f"tub_{_env_name}"
TUB_NAME = f"tub_sim"
TUB_DIR = os.getenv('TUB_DIR', f'./data/{TUB_NAME}')
MODEL_DIR = os.getenv('MODEL_DIR', f'./models/{TUB_NAME}')
VAE_WEIGHTS = os.path.join(MODEL_DIR, 'vae_encoder.pth')
WORLD_MODEL_WEIGHTS = os.path.join(MODEL_DIR, 'world_model.pth')
ACTOR_WEIGHTS = os.path.join(MODEL_DIR, 'dreamer_actor.pth')

# Training data cap — set to None to use all images
MAX_TRAIN_IMAGES = None

# Image preprocessing
IMG_CROP_TOP = 40
IMG_HEIGHT = 80
IMG_WIDTH = 128

# Model dimensions
LATENT_DIM = 32          # VAE encoder output (observation encoding)
ACTION_DIM = 2
HIDDEN_DIM = 256
ENCODER_CHANNELS = [3, 32, 64, 128, 256]
ACTOR_HIDDEN = 128
CRITIC_HIDDEN = 128
DYNAMICS_HIDDEN = 128
REWARD_HIDDEN = 64

# RSSM-specific dimensions
STOCH_DIM = 32           # Stochastic RSSM state
DETER_DIM = HIDDEN_DIM   # Deterministic GRU state (256)
STATE_DIM = DETER_DIM + STOCH_DIM  # Full model state for actor/critic (288)

# After 4 stride-2 convs on (IMG_HEIGHT x IMG_WIDTH), flatten size = 256 * 5 * 8
ENCODER_FLAT_DIM = ENCODER_CHANNELS[-1] * (IMG_HEIGHT // 16) * (IMG_WIDTH // 16)

# Actor output scaling
THROTTLE_CAP = 0.3

# --- UNIVERSAL HARDWARE SELECTOR ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    # CUDA config (RTX 4060 / 8GB+)
    BATCH_SIZE_VAE = 256
    BATCH_SIZE_WORLD = 256
    BATCH_SIZE_ACTOR = 256
    NUM_WORKERS = 4
    DATASET_MULTIPLIER = 3
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
            nn.Linear(STATE_DIM, ACTOR_HIDDEN), nn.ELU(),
            nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN), nn.ELU(),
            nn.Linear(ACTOR_HIDDEN, ACTION_DIM)
        )

    def forward(self, state):
        """state: cat(h_t, s_t), shape [B, STATE_DIM]"""
        raw_action = self.net(state)
        steer = torch.tanh(raw_action[:, 0:1])
        throttle = torch.sigmoid(raw_action[:, 1:2]) * THROTTLE_CAP
        return torch.cat([steer, throttle], dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, CRITIC_HIDDEN), nn.ELU(),
            nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN), nn.ELU(),
            nn.Linear(CRITIC_HIDDEN, 1)
        )

    def forward(self, state):
        """state: cat(h_t, s_t), shape [B, STATE_DIM]"""
        return self.net(state)

class RSSMWorldModel(nn.Module):
    """
    Recurrent State Space Model (RSSM) from DreamerV1.

    State = (h_t, s_t) where:
      h_t — deterministic GRU hidden state (DETER_DIM)
      s_t — stochastic latent sampled from prior or posterior (STOCH_DIM)

    Training uses posterior q(s_t | h_t, enc_t) with real observations.
    Imagination uses prior p(s_t | h_t) — no observation needed.
    Actor/Critic see cat(h_t, s_t) = STATE_DIM.
    """
    def __init__(self):
        super().__init__()
        # Recurrent: h_t = GRU(cat(s_{t-1}, a_{t-1}), h_{t-1})
        self.recurrent = nn.GRUCell(STOCH_DIM + ACTION_DIM, DETER_DIM)

        # V2: LayerNorm on GRU output — stabilizes h_t before it feeds into all heads
        self.h_norm = nn.LayerNorm(DETER_DIM)

        # Prior p(s_t | h_t) — for imagination (no observation)
        self.prior_net = nn.Sequential(
            nn.Linear(DETER_DIM, DETER_DIM), nn.ELU(),
            nn.Linear(DETER_DIM, STOCH_DIM * 2)  # mu + log_std
        )

        # Posterior q(s_t | h_t, enc_t) — for training with real observations
        self.posterior_net = nn.Sequential(
            nn.Linear(DETER_DIM + LATENT_DIM, DETER_DIM), nn.ELU(),
            nn.Linear(DETER_DIM, STOCH_DIM * 2)  # mu + log_std
        )

        # Reward predictor r_t = f(h_t, s_t)
        self.reward_predictor = nn.Sequential(
            nn.Linear(STATE_DIM, REWARD_HIDDEN), nn.ELU(),
            nn.Linear(REWARD_HIDDEN, 1)
        )

        # Observation predictor: enc_t = g(h_t, s_t)
        # Forces the posterior to encode observation-dependent info in s_t.
        # Without this, s_t collapses to prior noise since reward is action-dependent only.
        self.obs_predictor = nn.Sequential(
            nn.Linear(STATE_DIM, DYNAMICS_HIDDEN), nn.ELU(),
            nn.Linear(DYNAMICS_HIDDEN, LATENT_DIM)
        )

    def _reparameterize(self, mu, log_std):
        std = torch.exp(log_std.clamp(-4, 4))
        eps = torch.randn_like(mu)
        return mu + eps * std, mu, std

    def get_prior(self, h):
        """Prior distribution p(s | h). Returns (s_sample, mu, std)."""
        out = self.prior_net(h)
        mu, log_std = out.chunk(2, dim=-1)
        return self._reparameterize(mu, log_std)

    def get_posterior(self, h, enc):
        """Posterior distribution q(s | h, enc). Returns (s_sample, mu, std)."""
        out = self.posterior_net(torch.cat([h, enc], dim=-1))
        mu, log_std = out.chunk(2, dim=-1)
        return self._reparameterize(mu, log_std)

    def forward(self, enc_seq, a_seq):
        """
        Full sequence rollout using posteriors — for world model training.
        enc_seq: [B, T, LATENT_DIM]  — VAE-encoded observations
        a_seq:   [B, T, ACTION_DIM]
        Returns: prior_stats (mu, std), post_stats (mu, std),
                 reward_preds [B, T, 1], obs_preds [B, T, LATENT_DIM]
        """
        batch_size, seq_len, _ = enc_seq.shape
        h = torch.zeros(batch_size, DETER_DIM, device=enc_seq.device)
        s = torch.zeros(batch_size, STOCH_DIM, device=enc_seq.device)

        prior_mus, prior_stds = [], []
        post_mus, post_stds = [], []
        reward_preds = []
        obs_preds = []

        for t in range(seq_len):
            # Recurrent update: uses prev s and prev action
            a_prev = a_seq[:, t - 1] if t > 0 else torch.zeros(batch_size, ACTION_DIM, device=enc_seq.device)
            h = self.h_norm(self.recurrent(torch.cat([s, a_prev], dim=-1), h))

            # Prior (no observation)
            _, p_mu, p_std = self.get_prior(h)

            # Posterior (with observation)
            s, q_mu, q_std = self.get_posterior(h, enc_seq[:, t])

            state = torch.cat([h, s], dim=-1)
            reward_preds.append(self.reward_predictor(state))
            obs_preds.append(self.obs_predictor(state))
            prior_mus.append(p_mu);  prior_stds.append(p_std)
            post_mus.append(q_mu);   post_stds.append(q_std)

        prior_stats = (torch.stack(prior_mus, 1), torch.stack(prior_stds, 1))
        post_stats  = (torch.stack(post_mus, 1),  torch.stack(post_stds, 1))
        return prior_stats, post_stats, torch.stack(reward_preds, 1), torch.stack(obs_preds, 1)

    def forward_step(self, s_t, a_t, h_t):
        """
        Single imagination step using prior — no observation needed.
        Returns (s_next, reward_t, h_next).
        Reward is predicted from the NEXT state (h_next encodes a_t,
        matching training where r_pred[t] corresponds to a_{t-1}).
        """
        h_next = self.h_norm(self.recurrent(torch.cat([s_t, a_t], dim=-1), h_t))
        s_next, _, _ = self.get_prior(h_next)
        r_t = self.reward_predictor(torch.cat([h_next, s_next], dim=-1))
        return s_next, r_t, h_next

    def observe_step(self, s_prev, a_prev, h_prev, enc_t):
        """
        Single posterior step — encodes a real observation into RSSM state.
        Used to seed imagination from real data.
        Returns (s_t, h_t).
        """
        h_t = self.h_norm(self.recurrent(torch.cat([s_prev, a_prev], dim=-1), h_prev))
        s_t, _, _ = self.get_posterior(h_t, enc_t)
        return s_t, h_t


# Keep alias for backward compatibility with any remaining imports
WorldModel = RSSMWorldModel
