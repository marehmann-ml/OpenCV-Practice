import cv2
import os
from ultralytics import YOLO

# 1. SETUP
IMAGE_DIR = r"D:\VSCode\OpenCV\data\raw_images"
model = YOLO("yolov8n.pt")

# 2. PREPARE THE FILES
files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
files.sort()

print(f"Processing {len(files)} frames... Press 'q' to stop.")

# 3. THE LOOP
for filename in files:
    path = os.path.join(IMAGE_DIR, filename)
    frame = cv2.imread(path)
    
    if frame is None:
        continue
    
    # Run the AI
                                                                        #PCs Speed
    results = model(frame, imgsz=1280, conf=0.3, classes=[0], iou=0.5, verbose=False)
                    #resolution increase(upscale the img)    #overlapping filter

    # --- MANUAL CLEAN DRAWING (No big blue labels) ---
    for box in results[0].boxes:
        # 1. Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 2. Get data for the text
        conf = float(box.conf[0])
        label = f"person {conf:.2f}" # This looks like: person 0.82(round off 2decimal)

        # 3. Draw the green box (Thickness 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 4. DRAW THE TEXT (The missing part!)
        # We use a small font_scale (0.4) and thickness (1) to keep it clean
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Resize the final view so it fits your monitor perfectly
    display_frame = cv2.resize(frame, (1280, 720))
    
    # Show the result
    cv2.imshow("VisDrone Human Detection", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Done!")