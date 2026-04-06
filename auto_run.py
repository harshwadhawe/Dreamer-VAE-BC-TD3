"""
Automated Training Pipeline
============================
Runs the full VAE -> World Model -> Actor-Critic pipeline.

Usage:
    python auto_run.py                      # Default: tub_sim -> models/vae
    python auto_run.py tub_physical_monaco  # Custom:  tub_physical_monaco -> models/tub_physical_monaco

Drive with trained models:
    MODEL_DIR=./models/tub_physical_monaco python drive_sim.py
"""

import os
import shutil
import csv
import subprocess
import sys

if len(sys.argv) > 1:
    tub_name = sys.argv[1]
    TUB_DIR = f"./{tub_name}"
    MODEL_DIR = f"./models/{tub_name}"
else:
    TUB_DIR = "./tub_sim"
    MODEL_DIR = "./models/vae"

# def bridge_data():
#     print("========================================")
#     print("🌉 STAGE 1: AUTOMATED DATA BRIDGING")
#     print("========================================")
#     os.makedirs(TUB_DIR, exist_ok=True)
#     merged_csv_path = os.path.join(TUB_DIR, "telemetry.csv")

#     current_episode_offset = 0
#     total_frames = 0

#     with open(merged_csv_path, 'w', newline='') as f_out:
#         writer = csv.writer(f_out)
#         writer.writerow(['frame', 'steering', 'throttle', 'episode_id', 'reward', 'speed', 'cte'])

#         for tub in SOURCE_TUBS:
#             csv_path = os.path.join(tub, "telemetry.csv")
#             if not os.path.exists(csv_path):
#                 print(f"⚠️ Skipping {tub} (No telemetry.csv found)")
#                 continue

#             print(f"-> Stitching data from {tub}...")
#             max_ep_in_tub = 0

#             with open(csv_path, 'r') as f_in:
#                 reader = csv.DictReader(f_in)
#                 for row in reader:
#                     # Safely copy image if it doesn't exist in the merged folder
#                     src_img = os.path.join(tub, row['frame'])
#                     dst_img = os.path.join(TUB_DIR, row['frame'])
#                     if os.path.exists(src_img) and not os.path.exists(dst_img):
#                         shutil.copy2(src_img, dst_img)

#                     # Offset episode IDs to prevent physics teleportation glitches
#                     original_ep_id = int(row.get('episode_id', 0))
#                     new_ep_id = original_ep_id + current_episode_offset
#                     max_ep_in_tub = max(max_ep_in_tub, original_ep_id)

#                     writer.writerow([
#                         row['frame'], row['steering'], row['throttle'], 
#                         new_ep_id, row.get('reward', 0), row.get('speed', 0), row.get('cte', 0)
#                     ])
#                     total_frames += 1

#             current_episode_offset += max_ep_in_tub
            
#     print(f"✅ Bridge Complete! {total_frames} total frames ready.\n")

def run_script(script_name, env_vars):
    print("========================================")
    print(f"🚀 STAGE: Executing {script_name}")
    print("========================================")
    
    # Merge the current system environment variables with our custom ones
    custom_env = os.environ.copy()
    custom_env.update(env_vars)
    
    # Run the script and halt the entire pipeline if it crashes
    result = subprocess.run([sys.executable, script_name], env=custom_env)
    if result.returncode != 0:
        print(f"\n❌ FATAL ERROR: {script_name} crashed. Pipeline halted.")
        sys.exit(1)
    print(f"✅ {script_name} completed successfully.\n")

if __name__ == '__main__':
    # Validate tub exists
    if not os.path.exists(os.path.join(TUB_DIR, "telemetry.csv")):
        print(f"No telemetry.csv found in '{TUB_DIR}'")
        print(f"Usage: python auto_run.py <tub_name>")
        print(f"Example: python auto_run.py tub_physical_monaco")
        sys.exit(1)

    print(f"TUB_DIR:   {TUB_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")

    os_env = {
        "TUB_DIR": TUB_DIR,
        "MODEL_DIR": MODEL_DIR
    }

    run_script("train_vae.py", os_env)
    run_script("train_world_model.py", os_env)
    run_script("train_actor_critic.py", os_env)

    print("========================================")
    print("PIPELINE COMPLETE!")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"To drive, run:")
    print(f"  MODEL_DIR={MODEL_DIR} python drive_sim.py")
    print("========================================")