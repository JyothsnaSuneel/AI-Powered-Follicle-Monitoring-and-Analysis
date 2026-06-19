import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# Path to all extracted frames
frames_dir = r"C:\mnt\data\all_frames"
frame_paths = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")]

brightness_values = []
contrast_values = []
edge_density_values = []

print(f" Analyzing {len(frame_paths)} frames for EDA...")

for path in tqdm(frame_paths[:500]):  # analyze 500 samples to save time
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # Calculate brightness and contrast
    brightness = np.mean(img)
    contrast = np.std(img)

    # Edge density (Canny edge detection)
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    brightness_values.append(brightness)
    contrast_values.append(contrast)
    edge_density_values.append(edge_density)

# Summary 
print("\n EDA Summary:")
print(f"Total frames analyzed: {len(brightness_values)}")
print(f"Brightness range: {min(brightness_values):.2f} - {max(brightness_values):.2f}")
print(f"Contrast range: {min(contrast_values):.2f} - {max(contrast_values):.2f}")
print(f"Average edge density: {np.mean(edge_density_values):.4f}")

# Visualizations 

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
sns.histplot(brightness_values, bins=30, color='gold')
plt.title("Brightness Distribution")

plt.subplot(1, 3, 2)
sns.histplot(contrast_values, bins=30, color='skyblue')
plt.title("Contrast Distribution")

plt.subplot(1, 3, 3)
sns.histplot(edge_density_values, bins=30, color='salmon')
plt.title("Edge Density Distribution")

plt.tight_layout()
plt.show()



