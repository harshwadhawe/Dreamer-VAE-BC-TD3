import os
import time
import cv2
import pygame
import csv
import gymnasium as gym
import gym_donkeycar

def collect_data():
    tub_dir = "./tub_sim" 
    os.makedirs(tub_dir, exist_ok=True)
    
    # --- UPGRADED: Rich Telemetry CSV ---
    csv_file = open(os.path.join(tub_dir, "telemetry.csv"), mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'steering', 'throttle', 'episode_id', 'reward', 'speed', 'cte'])
    
    print("Connecting to Simulator...")
    conf = {
        "exe_path": "remote", 
        "host": "127.0.0.1",
        "port": 9091,
        "body_style": "donkey",
        "body_rgb": (255, 0, 0),
        "car_name": "Data_Collector",
        "font_size": 100
    }
    
    env = gym.make("donkey-generated-track-v0", conf=conf)
    obs, info = env.reset()
    
    pygame.init()
    screen = pygame.display.set_mode((300, 200))
    pygame.display.set_caption("Drive (Click Here First!)")
    
    frame_count = 0
    episode_id = 1 # NEW: Track continuous driving episodes
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
                
                # FIX: Timestamp prevents accidental data overwriting
                timestamp = int(time.time() * 1000)
                img_name = f"frame_{timestamp}.jpg"
                cv2.imwrite(os.path.join(tub_dir, img_name), cropped_img)
                
                # NEW: Extract Physics from Unity
                speed = info.get('speed', 0.0)
                cte = info.get('cte', 0.0)
                
                # Save the rich data
                csv_writer.writerow([img_name, steering, throttle, episode_id, reward, speed, cte])
                csv_file.flush() 

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Saved {frame_count} images & telemetry...")
            
            if terminated or truncated:
                print(f"Crash detected! Ending Episode {episode_id}...")
                env.reset()
                steering, throttle = 0.0, 0.0
                episode_id += 1 # NEW: Increment episode tracker on crash
                
            time.sleep(0.05) 
            
    finally:
        csv_file.close()
        env.close()
        pygame.quit()
        print(f"\nDone! Collected {frame_count} frames across {episode_id} episodes in '{tub_dir}'.")

if __name__ == "__main__":
    collect_data()