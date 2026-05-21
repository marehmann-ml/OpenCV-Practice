import os
import cv2
import random
import shutil

# --- CONFIGURATION ---
SOURCE_IMG_DIR = r'C:\Users\Abdur Rehman\Downloads\VisDrone2019-VID-test-dev\VisDrone2019-VID-test-dev\sequences' 
SOURCE_LAB_DIR = r'C:\Users\Abdur Rehman\Downloads\VisDrone2019-VID-test-dev\VisDrone2019-VID-test-dev\annotations'
OUTPUT_BASE = r'D:\VSCode\OpenCV\yolo_dataset_v2'

# VisDrone Class 1 = Pedestrian, 2 = People. We map both to YOLO Class 0.
TARGET_CLASSES = [1, 2] 

# --- GEOMETRIC MATH CONVERSION ---
def convert_to_yolo(left, top, width, height, img_w, img_h):
    """Calculates Center-XY and Normalizes the values"""
    x_center = (left + (width / 2)) / img_w 
    y_center = (top + (height / 2)) / img_h     #dimension se divide kardere = 0 & 1 btw (normalized)
    w_norm = width / img_w
    h_norm = height / img_h
    return x_center, y_center, w_norm, h_norm

# 1. Get all sequence folders automatically
# collecting names of video sequence folders
all_folders = [f for f in os.listdir(SOURCE_IMG_DIR) if os.path.isdir(os.path.join(SOURCE_IMG_DIR, f))]
random.seed(42) # Keeps the shuffle consistent every time you run it
random.shuffle(all_folders)

# 2. Split logic: 80% for Training, 20% for Validation
split_idx = int(len(all_folders) * 0.8) #80:20
train_folders = all_folders[:split_idx] #slices 80% from first folderr to cutoff
val_folders = all_folders[split_idx:] #remaining portion
#Splitting at the folder level ensures that the AI is tested on completely unseen backgrounds
#proving it can actually adapt to new environments.

def process_set(folders, set_name):
    img_out = os.path.join(OUTPUT_BASE, 'images', set_name) #creating folder as needed
    lab_out = os.path.join(OUTPUT_BASE, 'labels', set_name)
    os.makedirs(img_out, exist_ok=True) #if folder are already there it shouldnt crash
    os.makedirs(lab_out, exist_ok=True)

    print(f"Processing {set_name} set: {len(folders)} folders...")

    for folder in folders:  #pair of vdo folder wrt annotation file
        img_folder_path = os.path.join(SOURCE_IMG_DIR, folder)
        annotation_file = os.path.join(SOURCE_LAB_DIR, folder + '.txt')

        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation not found for {folder}, skipping.")
            continue

        # Load all annotations for this folder into a dictionary for speed
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        # Group annotations by frame number
        # Reads VisDrone text file and organizes coordinates by frame number.
        anno_dict = {}  
        for line in lines:
            parts = line.strip().split(',')     #['1','3','218',...]
            f_num = int(parts[0])
            if f_num not in anno_dict:
                anno_dict[f_num] = []
            anno_dict[f_num].append(parts) 

        # Process each image in the folder
        for img_name in os.listdir(img_folder_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(img_folder_path, img_name)
            
            # Use the number from '0000001.jpg' as frame index
            try:
                frame_num = int(os.path.splitext(img_name)[0]) #extract no. out of img filename
            except ValueError:
                continue
            
            # Read image to get its resolution 
            img = cv2.imread(img_path)
            if img is None: 
                continue
            h, w, _ = img.shape # h = img height, w = img width

            #filtering out any imgs that doenst have a HUMAN!
            yolo_labels = []    #for every single frame in an empty list it loops thro 'annotation', if it finds class pedestr/person,
            if frame_num in anno_dict:     #it appends to list
                for parts in anno_dict[frame_num]:
                    v_class = int(parts[7])
                    score = int(parts[6]) # 0 means ignore region
                    
                    if v_class in TARGET_CLASSES and score != 0:
                        # Extract bounding box pixel parameters
                        left = float(parts[2])
                        top = float(parts[3])
                        width = float(parts[4])
                        height = float(parts[5])
                        
                        # Process centers and get normalized values
                        yolo_box = convert_to_yolo(left, top, width, height, w, h)
                        
                        # Save tracking details as Class 0 (Human)
                        yolo_labels.append(f"0 {' '.join([f'{coord:.6f}' for coord in yolo_box])}")

            # Only copy img and save label if a human was actually found
            if yolo_labels:
                new_filename = f"{folder}_{img_name}"
                # Copy the img
                shutil.copy(img_path, os.path.join(img_out, new_filename))
                # Write YOLO Label file
                label_filename = new_filename.rsplit('.', 1)[0] + '.txt'
                with open(os.path.join(lab_out, label_filename), 'w') as f:
                    f.write('\n'.join(yolo_labels))


    print(f"Finished {set_name} set.")

# Run the processing sets
process_set(train_folders, 'train')
process_set(val_folders, 'val')

print(f"\nSUCCESS! Your packed dataset is ready at: {OUTPUT_BASE}")