import cv2
import os
from ultralytics import YOLO

# 1. SETUP PATHS
# Using the full path we confirmed earlier
IMAGE_DIR = r"D:\VSCode\OpenCV\data_task1_v1\raw_images"
model = YOLO("yolov8n.pt")

# 2. PREPARE THE IMAGE LIST
# Get all .jpg files and sort them to ensure a smooth video sequence
files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
files.sort()   #ascending order arranged

print(f"Starting playback of {len(files)} frames...")


# 3. THE PROCESSING LOOP
for filename in files:
    path = os.path.join(IMAGE_DIR, filename)
    frame = cv2.imread(path)
    
    if frame is None:
        continue
    
    # Run the AI
    # iou=0.5: reduces the "patchy" overlapping boxes
    # imgsz=1280: helps detection of tiny people                        #PCs Speed
    results = model(frame, imgsz=1280, conf=0.3, classes=[0], iou=0.5, verbose=False)
                    #resolution increase(upscale the img)    #overlapping filter

    # 4. AUTO PLOTTING 
    # line_width=1: makes the box borders very thin
    # font_size=0.5: makes the "person" text as small as possible
    # labels=True: ensures the "person" text stays visible
    annotated_frame = results[0].plot(line_width=2, font_size=1, labels=True)

    # 5. RESIZE & DISPLAY
    # Resize to fit your monitor (standard HD size)
    display_frame = cv2.resize(annotated_frame, (1280, 720))
    
    cv2.imshow("VisDrone - Automatic Detection", display_frame)
    
    # Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Process Finished.")