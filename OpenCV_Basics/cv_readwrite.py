import cv2

# 1. Read the image from disk
# The second argument (1) loads it in BGR color mode by default
img = cv2.imread('input_image.jpg', 1)

# Check if image was successfully loaded
if img is None:
    print("Error: Could not read the image.")
else:
    # 2. Display the image in a window
    cv2.imshow('Display Window', img)

    # 3. Wait for a key press (0 means wait indefinitely)
    cv2.waitKey(0)

    # 4. Write (Save) the image to a new file
    # This saves the loaded 'img' data as 'output_image.png'
    cv2.imwrite('output_image.png', img)

    # 5. Close all open windows
    cv2.destroyAllWindows()
