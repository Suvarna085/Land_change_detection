import cv2
import numpy as np

# Check a raw original label file
raw = cv2.imread('./archive/labels_land_cover_2006/2006/D14/1.tif', cv2.IMREAD_UNCHANGED)
print("Raw label dtype:", raw.dtype)
print("Raw unique values:", np.unique(raw))
print("Raw shape:", raw.shape)

# Check a saved patch label
patch = cv2.imread('./patches/D14_1_000_lbl1.png', cv2.IMREAD_GRAYSCALE)
print("\nPatch label dtype:", patch.dtype)
print("Patch unique values:", np.unique(patch))