"""
Autonomous driving on a physical DonkeyCar using TFLite models.

Requires: vae_encoder.tflite + dreamer_actor.tflite (from export_pth_to_tflite.py),
          PCA9685 I2C PWM board, PiCamera
Downstream: None (end of pipeline - physical deployment)

Pure numpy preprocessing (no PyTorch dependency on Pi). Runs VAE encoder + Actor
inference via tflite_runtime. Maps neural network outputs to PWM signals for
steering servo and throttle ESC. Bulletproof shutdown: motors are killed on any
exit (Ctrl+C, crash, or hardware failure) via finally block with None-checks.

Usage:
    python drive_physical_tflite.py <model_dir>
    python drive_physical_tflite.py models/tub_1_26-04-01
"""

import sys
import time
import warnings
import numpy as np
import tflite_runtime.interpreter as tflite

# Silence the DonkeyCar deprecation terminal spam
warnings.filterwarnings("ignore", category=DeprecationWarning)

from donkeycar.parts.actuator import PCA9685
from donkeycar.parts.camera import PiCamera 

# --- HARDWARE CONFIGURATION ---
STEERING_LEFT_PWM = 470
STEERING_RIGHT_PWM = 320
THROTTLE_FORWARD_PWM = 480
THROTTLE_STOPPED_PWM = 400
THROTTLE_REVERSE_PWM = 320

# --- IMAGE CROP CONFIGURATION ---
IMG_CROP_TOP = 40  

def map_action_to_pwm(action_val, min_action, max_action, min_pwm, max_pwm):
    return int((action_val - min_action) * (max_pwm - min_pwm) / (max_action - min_action) + min_pwm)

class PhysicalDreamerCar:
    def __init__(self, model_dir):
        print("\n" + "="*50)
        print("🚀 BOOTING DREAMER TFLITE ENGINE")
        print("="*50)
        
        # Pre-declare hardware variables so shutdown doesn't crash if init fails
        self.steering_controller = None
        self.throttle_controller = None
        self.camera = None
        
        try:
            # 1. Load VAE
            print("[INFO] Allocating VAE Subsystem...")
            self.vae_interpreter = tflite.Interpreter(model_path=f"{model_dir}/vae_encoder.tflite")
            self.vae_interpreter.allocate_tensors()
            self.vae_input = self.vae_interpreter.get_input_details()[0]['index']
            self.vae_output = self.vae_interpreter.get_output_details()[0]['index']
            print("  -> VAE Loaded Successfully.")

            # 2. Load Actor
            print("[INFO] Allocating Actor Subsystem...")
            self.actor_interpreter = tflite.Interpreter(model_path=f"{model_dir}/dreamer_actor.tflite")
            self.actor_interpreter.allocate_tensors()
            self.actor_input = self.actor_interpreter.get_input_details()[0]['index']
            self.actor_output = self.actor_interpreter.get_output_details()[0]['index']
            print("  -> Actor Loaded Successfully.")

            # 3. Initialize Hardware
            print("[INFO] Initializing I2C PWM Controllers...")
            self.steering_controller = PCA9685(channel=14, busnum=1)
            self.throttle_controller = PCA9685(channel=1, busnum=1)
            print("  -> PWM Controllers Linked.")
            
            print("[INFO] Warming up PiCamera module...")
            self.camera = PiCamera(image_w=128, image_h=120) 
            time.sleep(2) # Give sensor time to adjust white balance
            print("  -> Camera Online.")
            
            print("[OK] All systems go.\n")
            
        except Exception as e:
            print(f"\n[FATAL] Hardware initialization failed: {e}")
            self.shutdown()
            sys.exit(1)

    def run(self):
        print("="*50)
        print("🟢 AUTONOMOUS MODE ENGAGED")
        print("   Press [CTRL+C] for Emergency Stop")
        print("="*50 + "\n")
        
        try:
            while True:
                start_time = time.time()
                
                # 1. Vision (Grab raw frame)
                img_array = self.camera.run()
                
                # 2. PURE NUMPY PREPROCESSING
                img_cropped = img_array[IMG_CROP_TOP:, :, :] 
                img_normalized = img_cropped.astype(np.float32) / 255.0
                img_transposed = np.transpose(img_normalized, (2, 0, 1))
                img_input = np.expand_dims(img_transposed, axis=0)

                # 3. VAE Inference
                self.vae_interpreter.set_tensor(self.vae_input, img_input)
                self.vae_interpreter.invoke()
                latent_state = self.vae_interpreter.get_tensor(self.vae_output)

                # 4. Actor Inference
                self.actor_interpreter.set_tensor(self.actor_input, latent_state)
                self.actor_interpreter.invoke()
                action = self.actor_interpreter.get_tensor(self.actor_output)[0]

                steering_action = float(action[0]) 
                throttle_action = float(action[1]) 

                # 5. Math to Metal 
                steering_pwm = map_action_to_pwm(steering_action, -1.0, 1.0, STEERING_LEFT_PWM, STEERING_RIGHT_PWM)
                throttle_pwm = map_action_to_pwm(throttle_action, 0.0, 1.0, THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM)

                # 6. Fire Hardware
                self.steering_controller.run(steering_pwm)
                self.throttle_controller.run(throttle_pwm)

                # Diagnostic output
                fps = 1.0 / (time.time() - start_time)
                print(f"⚡ FPS: {fps:4.1f} | 🧭 Steer: {steering_action:>5.2f} ({steering_pwm}) | 🚀 Throttle: {throttle_action:>5.2f} ({throttle_pwm})", end='\r')

        except KeyboardInterrupt:
            print("\n\n[WARN] Manual override triggered (CTRL+C).")
        except Exception as e:
            print(f"\n\n[ERROR] CRITICAL CRASH IN MAIN LOOP: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Bulletproof shutdown sequence to ensure motors never run wild."""
        print("\n" + "="*50)
        print("🛑 INITIATING SAFE SHUTDOWN")
        print("="*50)
        
        if self.throttle_controller:
            print("[SHUTDOWN] Cutting throttle power (Neutral PWM: 400)...")
            try:
                self.throttle_controller.run(THROTTLE_STOPPED_PWM)
            except: pass # Ignore errors if hardware is completely disconnected
            
        if self.steering_controller:
            print("[SHUTDOWN] Centering steering servo...")
            try:
                self.steering_controller.run(int((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2))
            except: pass

        if self.camera:
            print("[SHUTDOWN] Powering down camera interface...")
            try:
                self.camera.shutdown()
            except: pass

        print("[SHUTDOWN] Sequence complete. Safe to power off.")
        print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = './models/vae'
    car = PhysicalDreamerCar(model_dir=model_dir)
    car.run()