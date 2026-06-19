import cv2
import os
import glob
from tqdm import tqdm

video_dir = r"C:\mnt\data\raw_videos"
output_dir = r"C:\mnt\data\all_frames"
os.makedirs(output_dir, exist_ok=True)

video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
total_extracted = 0

def extract_frames(video_path, output_folder, frame_interval=0.25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open {video_path}")
        return 0

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # fallback if FPS=0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n Processing: {os.path.basename(video_path)} | {total_frames} frames | {fps} fps")

    count, saved = 0, 0
    step = max(1, int(fps * frame_interval))
    success, frame = cap.read()

    with tqdm(total=total_frames, desc=f"Extracting {os.path.basename(video_path)}") as pbar:
        while success:
            if count % step == 0:
                frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{saved:05d}.jpg"
                cv2.imwrite(os.path.join(output_folder, frame_name), frame)
                saved += 1
            success, frame = cap.read()
            count += 1
            pbar.update(1)

    cap.release()
    print(f"✅ Extracted {saved} frames from {os.path.basename(video_path)}")
    return saved


# Loop through all videos
for video_path in video_files:
    total_extracted += extract_frames(video_path, output_dir)

print(f"\n🎞️ Total frames extracted from all videos: {total_extracted}")



