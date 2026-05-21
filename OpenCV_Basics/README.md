# OpenCV Basics

This section contains my initial hands-on practice with OpenCV during the ML internship.

The focus was on understanding how images and videos are handled internally before moving into advanced Computer Vision and YOLO workflows.

---

# Topics Practiced

- Reading and writing images
- Accessing webcam/video streams
- Understanding pixel matrices
- Learning BGR color channels
- Coordinate systems in images
- Frame-by-frame video handling
- Basic OpenCV display windows

---

# Development Environment

- VS Code
- Anaconda (`internship_env`)
- Python
- OpenCV (`cv2`)

---

# Sample Learning Snippet

```python
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("OpenCV Workspace", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

# Learning Outcome

This stage helped me understand:
- how video frames are processed
- how OpenCV handles image arrays
- real-time frame loops
- coordinate-based image operations

These concepts later helped during YOLO detection and bounding box workflows.
