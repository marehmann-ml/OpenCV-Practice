import cv2
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────────
IMAGE_PATH = r"D:\VSCode\OpenCV\CV basics\street.jpg"
MODEL_PATH = "yolov8n.pt" 
# ───────────────────────────────────────────────────────────────

# Load the brain
model = YOLO(MODEL_PATH)

# Load the image
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("[ERROR] Could not find the image. Check the path!")
    exit()

# Run detection
results = model(img) # No filters! Let's see everything.

# We loop through every box found
for box in results[0].boxes:
    # 1. Get Coordinates (x, y coordinates of the center)
    # xywh = center_x, center_y, width, height
    c_x, c_y, w, h = box.xywh[0]
    
    # 2. Get Class ID and Name
    class_id = int(box.cls[0])
    label = model.names[class_id]
    
    # 3. Get Confidence
    conf = float(box.conf[0])

    # 4. PRINT the data (The 'Detective' part)
    print(f"Found {label} | Confidence: {conf:.2f} | Center: ({int(c_x)}, {int(c_y)})")

    # 5. Manual Drawing (Instead of a box, let's draw a dot on the object)
    cv2.circle(img, (int(c_x), int(c_y)), 5, (0, 255, 0), -1)
    cv2.putText(img, label, (int(c_x), int(c_y) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Show our custom result
cv2.imshow("Data Detective", img)
cv2.waitKey(0) # Wait forever until a key is pressed
cv2.destroyAllWindows()