import os
import torch
import torch.nn as nn

# --- UNIFIED CONFIGURATION ---
TUB_DIR = './tub_sim'
MODEL_DIR = './models/vae'
VAE_WEIGHTS = os.path.join(MODEL_DIR, 'vae_encoder.pth')
WORLD_MODEL_WEIGHTS = os.path.join(MODEL_DIR, 'world_model.pth')
ACTOR_WEIGHTS = os.path.join(MODEL_DIR, 'dreamer_actor.pth')

LATENT_DIM = 32
ACTION_DIM = 2
HIDDEN_DIM = 256

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- UNIFIED IMAGE PREPROCESSING ---
def process_sim_image(img_tensor):
    """
    Ensures identical cropping across Data Collection, Training, and Simulation.
    Expects PyTorch Tensor (B, C, H, W). Crops top 40px and center 128px width.
    """
    _, _, h, w = img_tensor.shape
    start_x = (w - 128) // 2
    return img_tensor[:, :, 40:120, start_x:start_x+128]

# --- UNIFIED ARCHITECTURE ---
class DeterministicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(10240, LATENT_DIM)

    def forward(self, x):
        return self.fc_mu(self.encoder(x))

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
            nn.Linear(64, ACTION_DIM)
        )
    def forward(self, z):
        raw_action = self.net(z)
        steer = torch.tanh(raw_action[:, 0:1])
        throttle = torch.sigmoid(raw_action[:, 1:2]) * 0.4
        return torch.cat([steer, throttle], dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
            nn.Linear(64, 1)
        )
    def forward(self, z):
        return self.net(z)

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRUCell(input_size=LATENT_DIM + ACTION_DIM, hidden_size=HIDDEN_DIM)
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Linear(128, LATENT_DIM)
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64), nn.ReLU(), nn.Linear(64, 1)
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
