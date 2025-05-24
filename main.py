import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Directories
VIDEO_DIR = "videos"
OUTPUT_DIR = "output"
CSV_PATH = os.path.join(OUTPUT_DIR, "pose_data.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize list to hold landmark data
all_data = []

# Function to extract label from filename
def infer_label(filename):
    if "sitting" in filename.lower():
        return "sitting"
    elif "walking" in filename.lower():
        return "walking"
    else:
        return "unknown"

# List video files
video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])

# Process each video
for video_file in tqdm(video_files, desc="Processing videos"):
    label = infer_label(video_file)
    if label == "unknown":
        print(f"⚠️ Skipping unknown label video: {video_file}")
        continue

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_file))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = result.pose_landmarks.landmark
            row = []
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            row.append(label)
            all_data.append(row)

        # Show video with overlay
        cv2.imshow(f"Processing {video_file}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Save to CSV if data exists
if all_data:
    num_coords = len(all_data[0]) - 1
    headers = []
    for i in range(num_coords // 4):
        headers += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
    headers.append("label")

    df = pd.DataFrame(all_data, columns=headers)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n✅ Data saved to {CSV_PATH}")
else:
    print("\n❌ No pose data found.")
