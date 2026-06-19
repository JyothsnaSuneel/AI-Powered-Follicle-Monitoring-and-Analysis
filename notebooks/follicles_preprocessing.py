import cv2
import os
from tqdm import tqdm

# Input and output directories
input_folder = r"C:\mnt\data\all_frames"        
output_folder = r"C:/mnt/data/preprocessed_frames"
os.makedirs(output_folder, exist_ok=True)

# Parameters
TARGET_SIZE = (640, 640)  # YOLOv8 standard input size
BLUR_THRESHOLD = 80       # Below this Laplacian variance, frame considered blurry

# CLAHE for contrast enhancement (adaptive histogram equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Function to check blur using Laplacian variance
def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

# Process each frame
saved, skipped = 0, 0
for filename in tqdm(os.listdir(input_folder), desc="Preprocessing frames"):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    path = os.path.join(input_folder, filename)
    img = cv2.imread(path)

    if img is None:
        continue

    # Resize to target size
    img = cv2.resize(img, TARGET_SIZE)

    # Convert to LAB color space for better brightness control
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel (improves brightness & contrast)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Blur filtering
    blurry, variance = is_blurry(img_enhanced)
    if blurry:
        skipped += 1
        continue  # Skip blurry frames

    # Save the cleaned and enhanced frame
    cv2.imwrite(os.path.join(output_folder, filename), img_enhanced)
    saved += 1

print(f"Preprocessing complete!")
print(f"Frames saved (clean): {saved}")
print(f"Frames skipped (blurry): {skipped}")
print(f"Enhanced frames stored in: {output_folder}")



