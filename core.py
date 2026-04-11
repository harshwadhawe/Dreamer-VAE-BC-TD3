"""
Centralized configuration and model architectures for the Dreamer pipeline.

Requires: Nothing (this is the root dependency)
Downstream: Every other script imports from here.

Contains all shared constants (image dims, latent dim, batch sizes, device selection),
unified image preprocessing (process_sim_image), and all neural network architectures
(DeterministicEncoder, Actor, Critic, RSSM). Hardware-aware config auto-selects
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
#TUB_NAME = f"tub_{_env_name}"
TUB_NAME = f"tub_generated_track"
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
LATENT_DIM = 32
ACTION_DIM = 2
HIDDEN_DIM = 256
STOCH_DIM = 32                            # Stochastic state dimension in RSSM
RSSM_STATE_DIM = HIDDEN_DIM + STOCH_DIM  # Combined state fed to Actor/Critic = 288
ENCODER_CHANNELS = [3, 32, 64, 128, 256]
ACTOR_HIDDEN = 64
CRITIC_HIDDEN = 64
DYNAMICS_HIDDEN = 128
REWARD_HIDDEN = 64

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
            nn.Linear(RSSM_STATE_DIM, ACTOR_HIDDEN), nn.ELU(),
            nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN), nn.ELU(),
            nn.Linear(ACTOR_HIDDEN, ACTION_DIM)
        )

    def forward(self, state):
        raw_action = self.net(state)
        steer = torch.tanh(raw_action[:, 0:1])
        throttle = torch.sigmoid(raw_action[:, 1:2]) * THROTTLE_CAP
        return torch.cat([steer, throttle], dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(RSSM_STATE_DIM, CRITIC_HIDDEN), nn.ELU(),
            nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN), nn.ELU(),
            nn.Linear(CRITIC_HIDDEN, 1)
        )

    def forward(self, state):
        return self.net(state)

class RSSM(nn.Module):
    """
    Recurrent State Space Model (Dreamer-style).

    State = (h_t, z_t):
      h_t — deterministic hidden state from GRU, carries persistent memory
      z_t — stochastic state sampled from prior (imagination) or posterior (real obs)

    Training:  posterior q(z_t | h_t, x_t) updates z_t with VAE observations.
    Imagination / inference: prior p(z_t | h_t) predicts z_t without observations.
    KL(posterior || prior) trains the prior to match the posterior over time.
    """
    def __init__(self):
        super().__init__()
        # Deterministic recurrence: prev stochastic state + prev action → new hidden state
        self.cell = nn.GRUCell(STOCH_DIM + ACTION_DIM, HIDDEN_DIM)

        # Prior p(z_t | h_t) — used during imagination, no observation needed
        self.prior_net = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ELU(),
            nn.Linear(HIDDEN_DIM, STOCH_DIM * 2)
        )

        # Posterior q(z_t | h_t, x_t) — used during training with real VAE latents
        self.posterior_net = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, HIDDEN_DIM), nn.ELU(),
            nn.Linear(HIDDEN_DIM, STOCH_DIM * 2)
        )

        # Predictors operate on full state cat(h_t, z_t)
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(RSSM_STATE_DIM, DYNAMICS_HIDDEN), nn.ReLU(),
            nn.Linear(DYNAMICS_HIDDEN, LATENT_DIM)
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(RSSM_STATE_DIM, REWARD_HIDDEN), nn.ReLU(),
            nn.Linear(REWARD_HIDDEN, 1)
        )

    @staticmethod
    def _split_stats(raw):
        mu, logvar = raw.chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 2)  # numerical stability
        return mu, logvar

    @staticmethod
    def _sample(mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    @staticmethod
    def kl_loss(post_mu, post_logvar, prior_mu, prior_logvar, free_nats=0.5):
        """KL(posterior || prior), summed over stoch dims, averaged over batch."""
        var_post = post_logvar.exp()
        var_prior = prior_logvar.exp()
        kl = 0.5 * (prior_logvar - post_logvar + var_post / var_prior
                    + (post_mu - prior_mu).pow(2) / var_prior - 1)
        kl_per_sample = kl.sum(dim=-1)
        return kl_per_sample.clamp(min=free_nats * STOCH_DIM).mean()

    def forward(self, z_vae_seq, a_seq):
        """
        Training pass — uses posterior at every step.
        z_vae_seq : [B, T, LATENT_DIM]  VAE latents as observations
        a_seq      : [B, T, ACTION_DIM]
        Returns: pred_latents [B, T, LATENT_DIM], pred_rewards [B, T, 1], kl scalar
        """
        B, T, _ = z_vae_seq.shape
        dev = z_vae_seq.device
        h = torch.zeros(B, HIDDEN_DIM, device=dev)
        z = torch.zeros(B, STOCH_DIM, device=dev)

        pred_latents, pred_rewards, kl_losses = [], [], []

        for t in range(T):
            a_prev = a_seq[:, t - 1] if t > 0 else torch.zeros(B, ACTION_DIM, device=dev)
            h = self.cell(torch.cat([z, a_prev], dim=-1), h)

            prior_mu, prior_logvar   = self._split_stats(self.prior_net(h))
            post_mu,  post_logvar    = self._split_stats(
                self.posterior_net(torch.cat([h, z_vae_seq[:, t]], dim=-1))
            )
            z = self._sample(post_mu, post_logvar)
            state = torch.cat([h, z], dim=-1)

            pred_latents.append(self.dynamics_predictor(state))
            pred_rewards.append(self.reward_predictor(state))
            kl_losses.append(self.kl_loss(post_mu, post_logvar, prior_mu, prior_logvar))

        return (torch.stack(pred_latents, dim=1),
                torch.stack(pred_rewards, dim=1),
                torch.stack(kl_losses).mean())

    def imagine_step(self, z_t, a_t, h_t):
        """Single imagination step — prior only, no observation.
        Returns: z_next, reward, h_next, state
        """
        h_next = self.cell(torch.cat([z_t, a_t], dim=-1), h_t)
        prior_mu, prior_logvar = self._split_stats(self.prior_net(h_next))
        z_next = self._sample(prior_mu, prior_logvar)
        state = torch.cat([h_next, z_next], dim=-1)
        return z_next, self.reward_predictor(state), h_next, state

    def observe_step(self, x_t, a_prev, z_prev, h_prev):
        """Single step with real observation — posterior update.
        Used during real driving and actor-critic seed extraction.
        Returns: z_t, h_t, state
        """
        h_t = self.cell(torch.cat([z_prev, a_prev], dim=-1), h_prev)
        post_mu, post_logvar = self._split_stats(
            self.posterior_net(torch.cat([h_t, x_t], dim=-1))
        )
        z_t = self._sample(post_mu, post_logvar)
        state = torch.cat([h_t, z_t], dim=-1)
        return z_t, h_t, state
