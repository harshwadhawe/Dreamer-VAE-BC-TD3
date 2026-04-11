"""
Converts physical DonkeyCar .catalog files to telemetry.csv for the Dreamer pipeline.

Requires: A tub directory with .catalog files and images/ folder (from physical DonkeyCar)
Downstream: auto_run.py (train_vae.py -> train_world_model.py -> train_actor_critic.py)

Parses JSON-line .catalog files, extracts steering/throttle/timestamps, and auto-detects
episode boundaries via >1s time gaps. Outputs a telemetry.csv compatible with the
training pipeline. Speed/CTE/reward are zeroed (physical car has no sim metrics).

Usage:
    python add_telemetry_to_tub.py <tub_dir>
    python add_telemetry_to_tub.py tub_1_26-04-01
"""

import os
import sys
import json
import csv
import glob
from core import TUB_DIR
def convert_donkey_catalog(tub_dir):
    catalog_files = glob.glob(os.path.join(tub_dir, '*.catalog'))
    if not catalog_files:
        print(f"No .catalog files found in {tub_dir}!")
        return

    # Sort files to ensure chronological order (catalog_0.catalog, catalog_1.catalog, etc.)
    catalog_files.sort()

    csv_path = os.path.join(tub_dir, 'telemetry.csv')
    
    # The header exactly as your train_world_model.py expects it
    header = ['frame', 'steering', 'throttle', 'episode_id', 'reward', 'speed', 'cte']

    total_frames = 0
    current_episode = 1
    last_timestamp = None

    with open(csv_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for cat_file in catalog_files:
            print(f"Parsing {os.path.basename(cat_file)}...")
            with open(cat_file, 'r') as f_in:
                for line in f_in:
                    try:
                        record = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue # Skip corrupted lines
                    
                    # 1. Extract physical data
                    img_name = record.get('cam/image_array', '')
                    steering = record.get('user/angle', 0.0)
                    throttle = record.get('user/throttle', 0.0)
                    timestamp = record.get('_timestamp_ms', 0)

                    if not img_name:
                        continue

                    # Physical tubs put images in an 'images' folder. 
                    # We append it so the CSV points to the right place.
                    relative_img_path = os.path.join('images', img_name)

                    # 2. Automatic Episode Slicing (The Time Gap trick)
                    if last_timestamp is not None:
                        time_gap = timestamp - last_timestamp
                        # If there is a >1000ms gap, the user picked up the car. New episode!
                        if time_gap > 1000:
                            current_episode += 1
                            
                    last_timestamp = timestamp

                    # 3. Handle Missing Sim Metrics
                    # Speed and CTE don't exist in reality. We hardcode 0.0.
                    # Your new "Survival Only" architecture ignores these anyway!
                    speed = 0.0 
                    cte = 0.0
                    reward = 0.0 # Handled dynamically in train_world_model.py

                    writer.writerow([relative_img_path, steering, throttle, current_episode, reward, speed, cte])
                    total_frames += 1

    print("\n--- Conversion Complete ---")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Total Unique Episodes Detected: {current_episode}")
    print(f"Saved to: {csv_path}")

if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("Usage: python add_telemetry_to_tub.py <tub_dir>")
    #     print("Example: python add_telemetry_to_tub.py tub_1_26-04-01")
    #     sys.exit(1)
    convert_donkey_catalog(TUB_DIR)