"""
Physical autonomous driving on Raspberry Pi 5.

Replaces drive_physical_tflite.py — uses picamera2 (libcamera stack, required on Pi 5)
and PyTorch CPU inference instead of TFLite. Runs VAE -> RSSM -> Actor in a tight loop.

--- Pi 5 Setup ---

1. Install PyTorch (CPU):
   pip install torch --index-url https://download.pytorch.org/whl/cpu

2. Install camera + hardware libs:
   pip install picamera2
   pip install donkeycar          # for PCA9685 actuator support

3. Copy from training machine:
   scp core.py pi@<pi-ip>:~/dreamer/
   scp -r models/<tub_name>/ pi@<pi-ip>:~/dreamer/models/

4. Run:
   cd ~/dreamer
   MODEL_DIR=./models/tub_generated_track python drive_pi5.py

--- Hardware wiring (same as before) ---
Steering ESC  -> PCA9685 channel 14
Throttle ESC  -> PCA9685 channel 1
PCA9685 SDA/SCL -> Pi GPIO 2/3 (I2C bus 1)
"""

import os
import sys
import time
import warnings
import torch
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: picamera2 not found. Install with: pip install picamera2")
    sys.exit(1)

try:
    from donkeycar.parts.actuator import PCA9685
except ImportError:
    print("ERROR: donkeycar not found. Install with: pip install donkeycar")
    sys.exit(1)

from core import (
    DeterministicEncoder, Actor, RSSM, process_sim_image,
    VAE_WEIGHTS, ACTOR_WEIGHTS, WORLD_MODEL_WEIGHTS, MODEL_DIR,
    device, HIDDEN_DIM, STOCH_DIM, ACTION_DIM,
    IMG_CROP_TOP, IMG_HEIGHT, IMG_WIDTH
)

# --- HARDWARE CONFIG (match your physical setup) ---
STEERING_CHANNEL     = 14
THROTTLE_CHANNEL     = 1
STEERING_LEFT_PWM    = 470
STEERING_RIGHT_PWM   = 320
THROTTLE_FORWARD_PWM = 480
THROTTLE_STOPPED_PWM = 400

# --- TUNING KNOBS ---
THROTTLE_BOOST = 1.0   # raise if car is too slow against real-world friction
STEERING_GAIN  = 1.0   # raise to 1.1-1.2 if turns are too wide

# Camera must capture at least IMG_CROP_TOP + IMG_HEIGHT tall
CAMERA_W = IMG_WIDTH              # 128
CAMERA_H = IMG_CROP_TOP + IMG_HEIGHT  # 120


def _map_pwm(value, in_min, in_max, pwm_min, pwm_max):
    return int((value - in_min) * (pwm_max - pwm_min) / (in_max - in_min) + pwm_min)


class DreamerPi5:
    def __init__(self):
        print(f"Model directory: {MODEL_DIR}")

        # --- Load models (CPU on Pi 5) ---
        print("Loading VAE encoder...")
        self.vae = DeterministicEncoder().to(device)
        self.vae.load_state_dict(torch.load(VAE_WEIGHTS, map_location=device))
        self.vae.eval()

        print("Loading RSSM world model...")
        self.world_model = RSSM().to(device)
        self.world_model.load_state_dict(torch.load(WORLD_MODEL_WEIGHTS, map_location=device))
        self.world_model.eval()

        print("Loading Actor...")
        self.actor = Actor().to(device)
        self.actor.load_state_dict(torch.load(ACTOR_WEIGHTS, map_location=device))
        self.actor.eval()

        # --- Camera (picamera2 / libcamera — required on Pi 5) ---
        print(f"Initialising camera ({CAMERA_W}x{CAMERA_H})...")
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(
            main={"size": (CAMERA_W, CAMERA_H), "format": "RGB888"}
        )
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1)  # let camera warm up

        # --- Actuators ---
        print("Initialising steering and throttle (PCA9685)...")
        self.steering_ctrl = PCA9685(channel=STEERING_CHANNEL, busnum=1)
        self.throttle_ctrl  = PCA9685(channel=THROTTLE_CHANNEL,  busnum=1)
        self.throttle_ctrl.run(THROTTLE_STOPPED_PWM)  # safe init

        self._reset_rssm()
        print("Ready.\n")

    def _reset_rssm(self):
        self.h_t   = torch.zeros(1, HIDDEN_DIM, device=device)
        self.z_t   = torch.zeros(1, STOCH_DIM,  device=device)
        self.a_prev = torch.zeros(1, ACTION_DIM, device=device)

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """picamera2 gives HxWx3 uint8 RGB. Convert to [1,3,H,W] float tensor."""
        t = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        return process_sim_image(t.to(device))  # crops to [1,3,80,128]

    def run(self):
        print("=" * 45)
        print("  AUTONOMOUS MODE ENGAGED — Ctrl+C to stop")
        print("=" * 45 + "\n")

        try:
            while True:
                t0 = time.time()

                frame = self.cam.capture_array()
                img   = self._preprocess(frame)

                with torch.no_grad():
                    x_t = self.vae(img)
                    self.z_t, self.h_t, state = self.world_model.observe_step(
                        x_t, self.a_prev, self.z_t, self.h_t
                    )
                    action = self.actor(state).squeeze(0).cpu().numpy()

                steering_val = float(np.clip(action[0] * STEERING_GAIN, -1.0, 1.0))
                throttle_val = float(np.clip(action[1] * THROTTLE_BOOST,  0.0, 1.0))

                self.a_prev = torch.tensor([[steering_val, throttle_val]], device=device)

                steer_pwm = _map_pwm(steering_val, -1.0, 1.0,
                                     STEERING_LEFT_PWM, STEERING_RIGHT_PWM)
                throt_pwm = _map_pwm(throttle_val,  0.0, 1.0,
                                     THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM)

                self.steering_ctrl.run(steer_pwm)
                self.throttle_ctrl.run(throt_pwm)

                fps = 1.0 / (time.time() - t0)
                print(f"FPS: {fps:5.1f} | Steer: {steering_val:+.3f} ({steer_pwm}) "
                      f"| Throttle: {throttle_val:.3f} ({throt_pwm})", end="\r")

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self._shutdown()

    def _shutdown(self):
        self.throttle_ctrl.run(THROTTLE_STOPPED_PWM)
        self.steering_ctrl.run(
            int((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2)
        )
        self.cam.stop()
        print("Motors off. Camera stopped.")


if __name__ == "__main__":
    car = DreamerPi5()
    car.run()
