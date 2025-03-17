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
    'yellow': np.array([[15, 100, 100], [45, 255, 255]])
}

color_rgb = {
    'red': (255, 0, 0),  # Adjust these values
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0)
}

# Detect blocks (binarize image)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# _, thresh = cv2.threshold(gray, 100, 250, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)



# Find contours
# yellow_mask = cv2.inRange(hsv_image, color_ranges['yellow'][0], color_ranges['yellow'][1])
# kernel = np.ones((7, 7), np.uint8)
# yellow_mask = cv2.erode(yellow_mask, kernel, iterations=1)
# # plt.imshow(yellow_mask, cmap='gray')
# # plt.title('Yellow Mask')
# # plt.show()

# contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # Draw contours on the original image
# cv2.drawContours(image, contours, -1, (0, 255, 0), 2)   

# # # # Display the image with contours
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Detected Blocks')
# plt.show()

# blue_mask = cv2.inRange(hsv_image, color_ranges['blue'][0], color_ranges['blue'][1])

# plt.imshow(blue_mask, cmap='gray')
# plt.title('Blue Mask')
# plt.show()

# Process each block
blocks = []
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for color, _ in color_ranges.items():
    mask = cv2.inRange(hsv_image, color_ranges[color][0], color_ranges[color][1])
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=3)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color_rgb[color], 2) 
    # plt.imshow(image)
    # plt.show()
    for contour in contours:
        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]

        if cv2.contourArea(contour) < 100 or cv2.contourArea(contour) > 5000 or abs(width - height) > 30:
            continue
        
        # Extract box points (corner points of the rectangle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # Convert to integer
        
        # Draw the center of the rectangle
        center = (int(rect[0][0]), int(rect[0][1]))
        cv2.circle(image, center, 5, color_rgb[color], -1)
        
        # Draw the angle of the rectangle
        angle = rect[2]
        length = 50  # Length of the line to indicate the angle
        end_point = (int(center[0] + length * np.cos(np.deg2rad(angle))), 
                    int(center[1] + length * np.sin(np.deg2rad(angle))))
        cv2.line(image, center, end_point, color_rgb[color], 2)
        
        # Draw the oriented bounding box on the image
        cv2.drawContours(image, [box], 0, color_rgb[color], 2)

# Display the result
plt.imshow(image)
plt.show()


