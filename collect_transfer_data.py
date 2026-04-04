import os
import time
import cv2
import pygame
import csv
import gymnasium as gym
import gym_donkeycar

from core import SIM_HOST, SIM_PORT

def collect_transfer_data():
    # NEW: Isolate the new domain data
    tub_dir = "./tub_transfer" 
    os.makedirs(tub_dir, exist_ok=True)
    
    csv_file = open(os.path.join(tub_dir, "telemetry.csv"), mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'steering', 'throttle', 'episode_id', 'reward', 'speed', 'cte'])
    
    print("Connecting to Warehouse Simulator...")
    conf = {
        "exe_path": "remote",
        "host": SIM_HOST,
        "port": SIM_PORT,
        "body_style": "donkey",
        "body_rgb": (0, 255, 0), # Let's make the car green for this one
        "car_name": "Domain_Explorer",
        "font_size": 100
    }
    
    # NEW: Switch to the Warehouse environment
    env = gym.make("donkey-generated-roads-v0", conf=conf)
    obs, info = env.reset()
    
    pygame.init()
    screen = pygame.display.set_mode((300, 200))
    pygame.display.set_caption("Drive Warehouse")
    
    frame_count = 0
    episode_id = 1
    running = True
    steering, throttle = 0.0, 0.0
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]: throttle = min(throttle + 0.05, 0.3) 
            elif keys[pygame.K_DOWN]: throttle = max(throttle - 0.1, -0.3)
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
                cv2.imwrite(os.path.join(tub_dir, img_name), cropped_img)
                
                speed = info.get('speed', 0.0)
                cte = info.get('cte', 0.0)
                
                csv_writer.writerow([img_name, steering, throttle, episode_id, reward, speed, cte])
                csv_file.flush() 

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Warehouse: Saved {frame_count} images...")
            
            if terminated or truncated:
                print(f"Crash! Ending Episode {episode_id}...")
                env.reset()
                steering, throttle = 0.0, 0.0
                episode_id += 1
                
            time.sleep(0.05) 
            
    finally:
        csv_file.close()
        env.close()
        pygame.quit()
        print(f"\nDone! Collected {frame_count} frames in '{tub_dir}'.")

if __name__ == "__main__":
    collect_transfer_data()