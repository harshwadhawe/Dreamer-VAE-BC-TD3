"""
Manual driving data collection from the DonkeyCar simulator.

Requires: Running simulator on SIM_HOST:SIM_PORT (configured in core.py)
Downstream: train_vae.py -> train_world_model.py -> train_actor_critic.py

Connects to the sim via gym-donkeycar, captures camera frames as JPGs and logs
steering/throttle/episode_id to tub_sim/telemetry.csv using pygame keyboard input.
Append-safe: detects existing data and continues episode numbering.
Only saves frames when throttle > 0.05. Crops 16px from each side of raw sim images.
"""

import os
import time
import cv2
import pygame
import csv
import glob
import gymnasium as gym
import gym_donkeycar

from core import SIM_HOST, SIM_PORT, SIM_ENV, THROTTLE_CAP

_env_name = SIM_ENV.removeprefix("donkey-").removesuffix("-v0").replace("-", "_")
TUB_DIR = f"./data/tub_{_env_name}"

HEADER = ['frame', 'steering', 'throttle', 'episode_id', 'reward', 'speed', 'cte']

def collect_data():
    os.makedirs(TUB_DIR, exist_ok=True)

    telemetry_path = os.path.join(TUB_DIR, "telemetry.csv")

    # --- Detect existing data ---
    existing_frames = 0
    last_episode_id = 0

    if os.path.exists(telemetry_path):
        with open(telemetry_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_frames += 1
                ep = int(row.get('episode_id', 0))
                if ep > last_episode_id:
                    last_episode_id = ep

    existing_images = len(glob.glob(os.path.join(TUB_DIR, '*.jpg')))

    if existing_frames > 0:
        print(f"Found existing dataset: {existing_frames} rows in CSV, {existing_images} images on disk")
        if existing_frames != existing_images:
            print(f"  WARNING: CSV rows ({existing_frames}) != image count ({existing_images}) — possible data mismatch")
        print(f"  Last episode ID: {last_episode_id}")
        print(f"  Appending new data...")
        csv_file = open(telemetry_path, mode='a', newline='')
        csv_writer = csv.writer(csv_file)
    else:
        print(f"No existing data found in '{TUB_DIR}'. Starting fresh.")
        csv_file = open(telemetry_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(HEADER)

    print(f"Connecting to Simulator ({SIM_ENV} @ {SIM_HOST}:{SIM_PORT})...")
    conf = {
        "exe_path": "remote",
        "host": SIM_HOST,
        "port": SIM_PORT,
        "body_style": "donkey",
        "body_rgb": (255, 0, 0),
        "car_name": "Data_Collector",
        "font_size": 100
    }

    env = gym.make(SIM_ENV, conf=conf)
    obs, info = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((300, 200))
    pygame.display.set_caption("Drive (Click Here First!)")

    frame_count = 0
    episode_id = last_episode_id + 1
    running = True
    steering, throttle = 0.0, 0.0

    print(f"Recording from Episode {episode_id}. Press ESC to stop.")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP]: throttle = min(throttle + 0.05, THROTTLE_CAP)
            elif keys[pygame.K_DOWN]: throttle = max(throttle - 0.1, -THROTTLE_CAP)
            else: throttle = throttle * 0.9

            if keys[pygame.K_LEFT]: steering = max(steering - 0.1, -1.0)
            elif keys[pygame.K_RIGHT]: steering = min(steering + 0.1, 1.0)
            else: steering = steering * 0.8

            obs, reward, terminated, truncated, info = env.step([steering, throttle])

            if throttle > 0.05:
                bgr_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                cropped_img = bgr_img[:, 16:-16, :]

                timestamp = int(time.time() * 1000)
                img_name = f"frame_{timestamp}.jpg"
                cv2.imwrite(os.path.join(TUB_DIR, img_name), cropped_img)

                speed = info.get('speed', 0.0)
                cte = info.get('cte', 0.0)

                csv_writer.writerow([img_name, steering, throttle, episode_id, reward, speed, cte])
                csv_file.flush()

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"  +{frame_count} new frames (total: {existing_frames + frame_count}, episode {episode_id})")

            if terminated or truncated:
                print(f"Crash detected! Ending Episode {episode_id}...")
                env.reset()
                steering, throttle = 0.0, 0.0
                episode_id += 1

            time.sleep(0.05)

    finally:
        csv_file.close()
        env.close()
        pygame.quit()
        total = existing_frames + frame_count
        print(f"\nDone! Added {frame_count} frames. Total dataset: {total} frames across episodes 1-{episode_id} in '{TUB_DIR}'.")

if __name__ == "__main__":
    collect_data()
