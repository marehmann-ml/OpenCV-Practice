# VisDrone Preprocessing

This section documents the preprocessing workflow used for preparing the VisDrone dataset for YOLOv8 training.

The work mainly focused on understanding annotation formats, coordinate conversions, and dataset organization.

---

# Tasks Performed

- Downloaded and explored the VisDrone dataset
- Rearranged sequence and annotation folders
- Filtered metadata text files
- Cleaned annotation rows
- Organized dataset structure for training

---

# Coordinate Conversion

Learned how to convert VisDrone annotations into YOLO format.

Conversion involved:
- bounding box center calculations
- width and height normalization
- scaling coordinates between 0 and 1

---

# Dataset Splitting

Implemented:
- Training / Validation split
- Randomized dataset mixing
- Consistent dataset shuffling using:

```python
random.seed()
random.shuffle()
```

---

# Technical Understanding

This phase helped me understand:
- annotation systems
- bounding box mathematics
- dataset preprocessing pipelines
- normalization workflows
- structured AI dataset preparation

These preprocessing steps were later used for custom YOLOv8 training.
