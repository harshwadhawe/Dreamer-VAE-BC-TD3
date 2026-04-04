# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Dreamer-inspired reinforcement learning pipeline for autonomous driving in the DonkeyCar simulator. The system learns to drive by: collecting human driving data, compressing camera images into a latent space (VAE), learning a world model that predicts future states, and training an actor-critic policy entirely "in imagination" using the world model.

## Architecture & Training Pipeline

The pipeline runs sequentially — each script depends on outputs from the previous step:

1. **`collect_data.py`** — Manual driving data collection via pygame keyboard control. Connects to the DonkeyCar simulator (gym-donkeycar on port 9091), saves camera frames as JPGs and steering/throttle to `tub_sim/telemetry.csv`. Only saves frames when throttle > 0.05. Crops 16px from each side of raw 160px-wide sim images.

2. **`train_vae.py`** — Trains a convolutional VAE on collected images. Input: 80x128x3 (top 40px cropped from 120px height, center-cropped to 128px width). Latent dim: 32. Exports a `DeterministicEncoder` (mu-only, no sampling) as both `.pth` and `.onnx` to `models/vae/`.

3. **`train_world_model.py`** — Trains an RSSM-style world model (GRU-based) that predicts next latent state and reward given current state + action. Uses the frozen VAE encoder to convert images to latents. Sequence length: 16, hidden dim: 256.

4. **`train_actor_critic.py`** — Dreamer-style imagination training. Loads frozen VAE + world model, trains Actor and Critic by rolling out imagined trajectories (horizon=15). Seeds dreams with real latent states from driving data. Includes steering penalty for stability.

5. **`train_latent_imitation.py`** — Alternative to step 4: behavioral cloning in latent space. Directly maps VAE latents to human actions via supervised learning. Simpler but less capable than actor-critic.

6. **`drive_sim.py`** — Inference/deployment. Loads VAE encoder + trained Actor, connects to simulator, runs autonomous driving loop.

## Running Scripts

Requires a conda environment (configured via VS Code python-envs). The simulator must be running on `127.0.0.1:9091` for data collection and driving.

```bash
# Collect training data (requires simulator + pygame window focused)
python collect_data.py

# Train in order
python train_vae.py
python train_world_model.py
python train_actor_critic.py   # OR: python train_latent_imitation.py

# Run autonomous driving (requires simulator)
python drive_sim.py
```

## Key Constants Shared Across Scripts

These values must stay consistent across all scripts:
- **Latent dim**: 32
- **Image preprocessing**: crop top 40px, center-crop to 128px width → 80x128x3
- **Action space**: 2 (steering [-1,1], throttle [0, 0.4 via sigmoid cap])
- **VAE encoder architecture**: 4 conv layers (3→32→64→128→256, stride 2), flatten (10240), linear to 32
- **Actor architecture**: 3 linear layers (32→64→64→2) with ELU, tanh on steering, sigmoid*0.4 on throttle
- **World model hidden dim**: 256 (GRU)

## Model Weights

All saved to `models/vae/`:
- `vae_encoder.pth` / `vae_encoder.onnx` — frozen encoder
- `world_model.pth` — trained RSSM
- `dreamer_actor.pth` — trained policy

## Data

- `tub_sim/` — collected frames (`frame_XXXXX.jpg`) and `telemetry.csv` (columns: frame, steering, throttle)

## Dependencies

PyTorch (with MPS/CUDA support), gymnasium, gym-donkeycar, opencv-python, pygame, PIL, matplotlib, numpy
