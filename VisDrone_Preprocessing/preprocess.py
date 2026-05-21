import os
import cv2

# --- 1. SETTINGS & PATHS ---
RAW_ANN_PATH = r"D:\VS Code\OpenCV\data\raw_annotations\uav0000073_00600_v.txt"
IMAGES_DIR = r"D:\VSCode\OpenCV\data\yolo_dataset\images"
LABELS_DIR = r"D:\VSCode\OpenCV\data\yolo_dataset\labels"

# Standard VisDrone Dimensions (Check one of your images properties to confirm)
# i confirmed!
IMG_W = 1920 
IMG_H = 1080 

# Ensure the labels folder exists
os.makedirs(LABELS_DIR, exist_ok=True)

# --- 2. THE MATH ENGINE ---
def convert_to_yolo(left, top, width, height, img_w, img_h):
    """Calculates Center-XY and Normalizes the values"""
    x_center = (left + (width / 2)) / img_w  #so im 
    y_center = (top + (height / 2)) / img_h  #
    w_norm = width / img_w
    h_norm = height / img_h
    return x_center, y_center, w_norm, h_norm

# --- 3. PROCESSING ---
print("Starting conversion...")

# We use a dictionary to group detections by frame
# Key: Frame Number, Value: List of YOLO strings
dataset_map = {}

with open(RAW_ANN_PATH, 'r') as f:  #raw annotation wala data ko single txt file me change karre,with its specificindex
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 8: continue # Skip broken lines
        
        frame_idx = int(parts[0])
        left, top = float(parts[2]), float(parts[3])
        width, height = float(parts[4]), float(parts[5])
        category = int(parts[7])

        # Class FILTERING : Only Class 1 (Pedestrian) and 2 (People)
        if category in [1, 2]:
            class_id = 0 # Combine both into one 'Human' class
            
            # Apply our Center Math
            x, y, w, h = convert_to_yolo(left, top, width, height, IMG_W, IMG_H)
            
            # Format as YOLO string: "class x_center y_center width height"
            yolo_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            
            if frame_idx not in dataset_map:
                dataset_map[frame_idx] = []
            dataset_map[frame_idx].append(yolo_line)



# --- 4. WRITING THE FILES ---
for frame_idx, lines in dataset_map.items():
    # VisDrone images are named like 0000001.jpg, 0000002.jpg...
    # We create matching 0000001.txt, 0000002.txt...
    file_name = f"{frame_idx:07d}.txt" 
    output_path = os.path.join(LABELS_DIR, file_name)
    
    with open(output_path, 'w') as out_f:
        out_f.write("\n".join(lines))

print(f"Success! {len(dataset_map)} label files created in {LABELS_DIR}")