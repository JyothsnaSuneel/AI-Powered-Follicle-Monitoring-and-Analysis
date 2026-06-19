# pixel_calibration_final

import cv2
import numpy as np
import os

# Folder containing calibration frames
folder_path = r"C:\Users\sunee\IVF_Follicle_AI\calibration_frames" 
known_mm = 10  # The real-world measurement shown on ultrasound (e.g. 10 mm)

# Collect calibration frames
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print("⚠️ No calibration frames found. Please check your folder path.")
else:
    print(f"🧩 Found {len(image_files)} calibration frames...")

pixels_per_mm_list = []

for img_name in image_files:
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)

    # Convert to grayscale and threshold bright areas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)  # detect white areas

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours (potential measurement lines)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_lengths = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Filter likely horizontal lines (long and thin)
        if aspect_ratio > 5 and 2 < w < 500 and h < 10:
            line_lengths.append(w)

    if line_lengths:
        # Pick the longest detected line
        best_line = max(line_lengths)
        px_per_mm = best_line / known_mm
        pixels_per_mm_list.append(px_per_mm)
        print(f"{img_name}: Detected line = {best_line:.1f}px  →  {px_per_mm:.2f} px/mm")
    else:
        print(f"⚠️ No clear measurement line detected in {img_name}")

#  Compute final calibration mean
if pixels_per_mm_list:
    mean_px_mm = np.mean(pixels_per_mm_list)
    print("\n✅ Calibration complete!")
    print(f"Mean Pixels per mm: {mean_px_mm:.2f}")

    # Save result
    with open("mean_calibration_result.txt", "w") as f:
        f.write(str(round(mean_px_mm, 2)))
    print("💾 Saved calibration result → mean_calibration_result.txt")

else:
    print("❌ Could not detect any valid measurement lines.")


