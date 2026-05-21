import cv2
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────────
VIDEO_PATH = r"D:\VSCode\OpenCV\CV basics\sample_vdo.mp4"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.4
# ───────────────────────────────────────────────────────────────

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Cannot open video file.")
    exit()

print("[INFO] Running detection... Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[INFO] Video ended.")
        break

    # Run YOLO detection
    results = model(frame, conf=CONFIDENCE, verbose=False)[0]

    person_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if label == "person":
            person_count += 1

    # Show people count on screen
    cv2.putText(frame, f"People detected: {person_count}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Store Aisle Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()