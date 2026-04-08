import sys
import time
import warnings
import numpy as np
import tflite_runtime.interpreter as tflite

# Silence the DonkeyCar deprecation terminal spam
warnings.filterwarnings("ignore", category=DeprecationWarning)

from donkeycar.parts.actuator import PCA9685
from donkeycar.parts.camera import PiCamera 

# --- HARDWARE CONFIGURATION (Matches your physical setup) ---
STEERING_LEFT_PWM = 470
STEERING_RIGHT_PWM = 320
THROTTLE_FORWARD_PWM = 480
THROTTLE_STOPPED_PWM = 400
THROTTLE_REVERSE_PWM = 320

# --- TUNING KNOBS (Adjust these for performance) ---
# Pulling from your core.py logic
IMG_CROP_TOP = 40  
THROTTLE_BOOST = 1.0  # Multiplier to overcome real-world friction
STEERING_GAIN = 1.0     # Adjust to 1.1 or 1.2 if turns are too wide

def map_action_to_pwm(action_val, min_action, max_action, min_pwm, max_pwm):
    """Linear interpolation from AI space [-1, 1] to Hardware PWM."""
    return int((action_val - min_action) * (max_pwm - min_pwm) / (max_action - min_action) + min_pwm)

class PhysicalDreamerCar:
    def __init__(self, model_dir):
        print("\n" + "="*50)
        print(f"🚀 BOOTING DREAMER TFLITE ENGINE | BOOST: {THROTTLE_BOOST}x")
        print("="*50)
        
        self.steering_controller = None
        self.throttle_controller = None
        self.camera = None
        
        try:
            # 1. Load VAE (Sim-to-Real Bridge)
            print("[INFO] Loading VAE Encoder...")
            self.vae_interpreter = tflite.Interpreter(model_path=f"{model_dir}/vae_encoder.tflite")
            self.vae_interpreter.allocate_tensors()
            self.vae_input = self.vae_interpreter.get_input_details()[0]['index']
            self.vae_output = self.vae_interpreter.get_output_details()[0]['index']

            # 2. Load Actor (The Decision Maker)
            print("[INFO] Loading Dreamer Actor...")
            self.actor_interpreter = tflite.Interpreter(model_path=f"{model_dir}/dreamer_actor.tflite")
            self.actor_interpreter.allocate_tensors()
            self.actor_input = self.actor_interpreter.get_input_details()[0]['index']
            self.actor_output = self.actor_interpreter.get_output_details()[0]['index']

            # 3. Initialize Hardware
            print("[INFO] Binding I2C PWM Bus (40 and 70)...")
            self.steering_controller = PCA9685(channel=14, busnum=1)
            self.throttle_controller = PCA9685(channel=1, busnum=1)
            
            print("[INFO] Initializing PiCamera (128x120)...")
            self.camera = PiCamera(image_w=128, image_h=120) 
            time.sleep(2) 
            
            print("[OK] Systems Nominal. Ready for Autonomous Run.\n")
            
        except Exception as e:
            print(f"\n[FATAL] Hardware init failed: {e}")
            self.shutdown()
            sys.exit(1)

    def run(self):
        print("="*50)
        print("🟢 AUTONOMOUS MODE ENGAGED")
        print("   Emergency Stop: [CTRL+C]")
        print("="*50 + "\n")
        
        try:
            while True:
                start_time = time.time()
                
                # 1. Capture & Pure NumPy Preprocessing
                img_array = self.camera.run()
                # Crop to match core.IMG_HEIGHT (80)
                img_cropped = img_array[IMG_CROP_TOP:IMG_CROP_TOP+80, :, :] 
                # Normalize & Transpose (H, W, C) -> (C, H, W)
                img_input = np.expand_dims(np.transpose(img_cropped.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)

                # 2. VAE Inference
                self.vae_interpreter.set_tensor(self.vae_input, img_input)
                self.vae_interpreter.invoke()
                latent_state = self.vae_interpreter.get_tensor(self.vae_output)

                # 3. Actor Inference
                self.actor_interpreter.set_tensor(self.actor_input, latent_state)
                self.actor_interpreter.invoke()
                action = self.actor_interpreter.get_tensor(self.actor_output)[0]

                # 4. Apply Tuning Variables & Safety Clips
                # action[0] is Steering (tanh), action[1] is Throttle (sigmoid)
                steering_val = np.clip(float(action[0]) * STEERING_GAIN, -1.0, 1.0)
                throttle_val = np.clip(float(action[1]) * THROTTLE_BOOST, 0.0, 1.0) 

                # 5. Convert to Metal (PWM)
                steer_pwm = map_action_to_pwm(steering_val, -1.0, 1.0, STEERING_LEFT_PWM, STEERING_RIGHT_PWM)
                throt_pwm = map_action_to_pwm(throttle_val, 0.0, 1.0, THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM)

                # 6. Command Actuators
                self.steering_controller.run(steer_pwm)
                self.throttle_controller.run(throt_pwm)

                # Telemetry
                fps = 1.0 / (time.time() - start_time)
                print(f"⚡ FPS: {fps:4.1f} | 🧭 Steer: {steering_val:>5.2f} ({steer_pwm}) | 🚀 Throt: {throttle_val:>5.2f} ({throt_pwm})", end='\r')

        except KeyboardInterrupt:
            print("\n\n[WARN] Manual override triggered.")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n" + "="*50)
        print("🛑 SAFE SHUTDOWN INITIATED")
        print("="*50)
        
        if self.throttle_controller:
            try: self.throttle_controller.run(THROTTLE_STOPPED_PWM)
            except: pass
        if self.steering_controller:
            try: self.steering_controller.run(int((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2))
            except: pass
        if self.camera:
            try: self.camera.shutdown()
            except: pass
        print("[DONE] Car secured. Motors off.")

if __name__ == "__main__":
    car = PhysicalDreamerCar(model_dir='./models')
    car.run()