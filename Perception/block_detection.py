import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('frame_750.jpg')

# Convert to HSV for better color detection
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for red, green, blue, yellow
color_ranges = {
    'red': np.array([[0, 100, 100], [10, 255, 255]]),  # Adjust these values
    'green': np.array([[40, 50, 50], [85, 255, 255]]),
    'blue': np.array([[90, 30, 30], [130, 255, 255]]),
    'yellow': np.array([[20, 100, 100], [40, 255, 255]])
}

# Detect blocks (binarize image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# _, thresh = cv2.threshold(gray, 100, 250, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)



# Find contours
yellow_mask = cv2.inRange(hsv_image, color_ranges['red'][0], color_ranges['red'][1])
plt.imshow(yellow_mask, cmap='gray')

contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw contours on the original image
# cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# # Display the image with contours
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Detected Blocks')
# plt.show()

blue_mask = cv2.inRange(hsv_image, color_ranges['red'][0], color_ranges['red'][1])

# plt.imshow(blue_mask, cmap='gray')
# plt.title('Blue Mask')
# plt.show()

# Process each block
blocks = []
for contour in contours:
    # Get minimum area rectangle

    if cv2.contourArea(contour) < 100:
        continue
    rect = cv2.minAreaRect(contour)
    
    # Extract box points (corner points of the rectangle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # Convert to integer
    
    # Draw the oriented bounding box on the image
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    
    # Print details of the bounding box
    print(f"Center: {rect[0]}, Size: {rect[1]}, Angle: {rect[2]}")

# Display the result
plt.imshow(image)
plt.show()


