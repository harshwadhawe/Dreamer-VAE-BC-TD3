import os
import csv
import shutil

SOURCE_TUB = './tub_sim'
GOLDEN_TUB = './tub_golden'

# --- THE GOLDEN FILTERS ---
MAX_STEERING_JERK = 0.2  # Max allowed change in steering between frames
FRAMES_TO_DROP_END = 30  # Safety horizon (drop crash-adjacent frames)

os.makedirs(GOLDEN_TUB, exist_ok=True)

with open(os.path.join(SOURCE_TUB, 'telemetry.csv'), 'r') as f:
    rows = list(csv.DictReader(f))

# Prepare the new Golden CSV
golden_csv = open(os.path.join(GOLDEN_TUB, 'telemetry.csv'), 'w', newline='')
writer = csv.DictWriter(golden_csv, fieldnames=rows[0].keys())
writer.writeheader()

golden_frames = 0
dropped_jerk = 0
dropped_crash = 0

for i in range(1, len(rows)):
    row = rows[i]
    prev_row = rows[i-1]
    
    ep_id = int(row.get('episode_id', 0))
    
    # 1. THE CRASH FILTER (Lookahead)
    is_near_crash = False
    for lookahead in range(FRAMES_TO_DROP_END):
        idx = i + lookahead
        if idx >= len(rows) or int(rows[idx].get('episode_id', 0)) != ep_id:
            is_near_crash = True
            break
            
    if is_near_crash:
        dropped_crash += 1
        continue

    # 2. THE SMOOTHNESS FILTER (Steering Derivative)
    current_steer = float(row['steering'])
    prev_steer = float(prev_row['steering'])
    steer_jerk = abs(current_steer - prev_steer)
    
    if steer_jerk > MAX_STEERING_JERK:
        dropped_jerk += 1
        continue

    # If it passes both filters, it is a Golden Frame
    img_name = row['frame']
    src_img = os.path.join(SOURCE_TUB, img_name)
    dst_img = os.path.join(GOLDEN_TUB, img_name)
    
    if os.path.exists(src_img):
        shutil.copy(src_img, dst_img)
        writer.writerow(row)
        golden_frames += 1

golden_csv.close()

print(f"Extraction Complete!")
print(f"Total Source Frames: {len(rows)}")
print(f"Dropped (Erratic Steering): {dropped_jerk}")
print(f"Dropped (Crash Horizon): {dropped_crash}")
print(f"Golden Frames Retained: {golden_frames}")