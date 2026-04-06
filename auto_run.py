"""
Automated Training Pipeline - runs VAE -> World Model -> Actor-Critic sequentially.

Requires: A tub directory with telemetry.csv and images (from collect_data.py or
          add_telemetry_to_tub.py)
Downstream: drive_sim.py or export_pth_to_tflite.py (uses trained models)

Passes TUB_DIR and MODEL_DIR as environment variables to each training script.
Halts the entire pipeline if any step fails. Models are saved to models/<tub_name>/.

Usage:
    python auto_run.py                      # Default: tub_sim -> models/vae
    python auto_run.py tub_physical_monaco  # Custom:  tub_physical_monaco -> models/tub_physical_monaco

Drive with trained models:
    MODEL_DIR=./models/tub_physical_monaco python drive_sim.py
"""

import os
import subprocess
import sys

if len(sys.argv) > 1:
    tub_name = sys.argv[1]
    TUB_DIR = f"./{tub_name}"
    MODEL_DIR = f"./models/{tub_name}"
else:
    TUB_DIR = "./tub_sim"
    MODEL_DIR = "./models/vae"

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
    run_script("export_pth_to_tflite.py", os_env)

    print("========================================")
    print("PIPELINE COMPLETE!")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"To drive, run:")
    print(f"  MODEL_DIR={MODEL_DIR} python drive_sim.py")
    print("========================================")