import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyrealsense2 as rs

# Load the image
# image = cv2.imread('frame_750.jpg')

# Convert to HSV for better color detection
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for red, green, blue, yellow



class BlockDetection2D:

    def __init__(self):
        self.color_ranges = {
            'red': np.array([[0, 100, 100], [10, 255, 255]]),  # Adjust these values
            'green': np.array([[40, 50, 50], [85, 255, 255]]),
            'blue': np.array([[90, 30, 30], [130, 255, 255]]),
            'yellow': np.array([[15, 100, 100], [45, 255, 255]])
        }

        self.color_rgb = {
            'red': (255, 0, 0),  # Adjust these values
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0)
        }




    def get_blocks2d(self, image):


        # Process each block
        blocks = {
            'red': [],
            'green': [],
            'blue': [],
            'yellow': []
        }
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for color, _ in self.color_ranges.items():
            mask = cv2.inRange(hsv_image, self.color_ranges[color][0], self.color_ranges[color][1])
            # kernel = np.ones((3, 3), np.uint8)
            # mask = cv2.dilate(mask, kernel, iterations=1)
            # plt.imshow(mask, cmap='gray')
            # plt.show()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(image, contours, -1, color_rgb[color], 2) 
            # plt.imshow(image)
            # plt.show()
            for contour in contours:
                # Get minimum area rectangle
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]

                if cv2.contourArea(contour) < 800 or cv2.contourArea(contour) > 16000 or abs(width - height) > 30:
                    continue
                
                # Extract box points (corner points of the rectangle)
                box = cv2.boxPoints(rect)
                box = np.clip(box, [0, 0], [847, 479])  # Clip points to image dimensions (848, 480)
                box = np.int0(box)  # Convert to integer
                blocks[color].append(box)
                
                # Draw the center of the rectangle
                # center = (int(rect[0][0]), int(rect[0][1]))
                # print("Center2d:", center)
                # cv2.circle(image, center, 5, color_rgb[color], -1)
                
                # # Draw the angle of the rectangle
                # angle = rect[2]
                # length = 50  # Length of the line to indicate the angle
                # end_point = (int(center[0] + length * np.cos(np.deg2rad(angle))), 
                #             int(center[1] + length * np.sin(np.deg2rad(angle))))
                # cv2.line(image, center, end_point, color_rgb[color], 2)
                
                # # Draw the oriented bounding box on the image
                # cv2.drawContours(image, [box], 0, color_rgb[color], 2)

        return blocks
    

class BlocksDetection3D:
    def __init__(self):

        # These are the depth intrinsics for the Intel RealSense D435

        # self.fx = 421.9288330078125
        # self.fy = 421.9288330078125
        # self.cx = 419.6107482910156
        # self.cy = 244.43862915039062

        # self.intrinsics = np.array([
        #     [self.fx, 0,  self.cx],
        #     [0,  self.fy, self.cy],
        #     [0,  0,  1]
        # ])

        # self.depth_intrinsics = rs.pyrealsense2.intrinsics()
        # self.depth_intrinsics.height = 480
        # self.depth_intrinsics.width = 848
        # self.depth_intrinsics.ppx = self.fx
        # self.depth_intrinsics.ppy = self.fy
        # self.depth_intrinsics.fx = self.cx
        # self.depth_intrinsics.fy = self.cy
        # self.depth_intrinsics.model = rs.pyrealsense2.distortion.inverse_brown_conrady
        # self.depth_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.depth_scale = 0.001
        self.depth_intrinsics = rs.intrinsics()
        self.depth_intrinsics.width = 848
        self.depth_intrinsics.height = 480
        

    
    def get_depth_at_point(self, depth_image, x, y):
        
        depth_value = depth_image[y, x]

        # Calculate real-world coordinates
        depth_in_meters = depth_value * self.depth_scale
        pixel = [float(x), float(y)]  # Convert pixel coordinates to floats
        point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, pixel, depth_in_meters)

        return point
    
    def get_block_center_3d(self, depth_image, blocks2d):

        block_color_poses = {
            'red': [],
            'green': [],
            'blue': [],
            'yellow': []
        }
        for color, boxes in blocks2d.items():
            for box in boxes:
                corners3d = []
                for corner in box:
                    x, y = corner
                    corner3d = self.get_depth_at_point(depth_image, x, y)
                    corners3d.append(corner3d)
                corners3d = np.array(corners3d)
                center = self.get_pose(corners3d)
                # return center, normal, v1
                block_color_poses[color].append(center)

        return block_color_poses

    def get_pose(self, corners3d):
        # Calculate the pose of the block
        # This function should return the pose of the block in the world frame
        # The pose should be a 4x4 transformation matrix

        # Calculate the center of the block
        center = np.mean(corners3d, axis=0)

        # # Calculate vectors from the first corner to the other corners
        # v1 = np.array(corners3d[1]) - np.array(corners3d[0])
        # v2 = np.array(corners3d[3]) - np.array(corners3d[0])

        # # Normalize the vectors
        # if np.linalg.norm(v1) == 0:
        #     v1 /= 1e-6
        # else:
        #     v1 /= np.linalg.norm(v1)

        # if np.linalg.norm(v2) == 0:
        #     v2 /= 1e-6
        # else:
        #     v2 /= np.linalg.norm(v2)

        # # Calculate the normal vector to the plane of the block
        # normal = np.cross(v1, v2)

        # if np.linalg.norm(normal) == 0:
        #     normal /= 1e-6
        # else:
        #     normal /= np.linalg.norm(normal)

        # # Create the rotation matrix
        # rotation_matrix = np.eye(4)
        # rotation_matrix[:3, 0] = v1
        # rotation_matrix[:3, 1] = v2
        # rotation_matrix[:3, 2] = normal

        # # Create the transformation matrix
        # transformation_matrix = np.eye(4)
        # transformation_matrix[:3, 3] = center
        # transformation_matrix[:3, :3] = rotation_matrix[:3, :3]

        return center
                    


# blocks = get_blocks2d(image)
# for color, block_list in blocks.items():
#     print(f'{color}: {len(block_list)} blocks')
#     for block in block_list:
#         print(block)


# Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

# block_detection_2d = BlockDetection2D()

# try:
#     while True:
#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
#         if not color_frame:
#             continue

#         # Convert images to numpy arrays
#         color_image = np.asanyarray(color_frame.get_data())
       

#         # Get blocks from the color image
#         blocks2d = block_detection_2d.get_blocks2d(color_image)

#         # Print the number of blocks detected for each color
#         for color, boxes in blocks2d.items():
#             for box in boxes:
#                 for corner in box:
#                     cv2.circle(color_image, tuple(corner), 2, block_detection_2d.color_rgb[color], -1)
                    

#         # Display the resulting frame
#         cv2.imshow('Frame', color_image)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     # Stop streaming
#     pipeline.stop()
#     cv2.destroyAllWindows()


# def process_video(input_video_path):
#     cap = cv2.VideoCapture(input_video_path)
#     block_detection_2d = BlockDetection2D()
#     i = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = np.asanyarray(frame)
#         blocks2d = block_detection_2d.get_blocks2d(frame)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         for color, boxes in blocks2d.items():
#             for box in boxes:
#                 for corner in box:
#                     cv2.circle(frame, tuple(corner), 2, block_detection_2d.color_rgb[color], -1)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # cv2.imshow('Frame', frame)
#         i +=1
        
#         if i == 640:
#             # cv2.imwrite(f'frame_{i}.jpg', frame)
#             cv2.imshow('Frame', frame)
#             cv2.waitKey(0)
        
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        

#     # cap.release()

# process_video('output_2.mp4')


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

block_detection_2d = BlockDetection2D()
blocks3d = BlocksDetection3D()
try:

    while True:


        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Get intrinsics for the depth camera
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        print("depth intrinsics: ", depth_intrinsics)   
        print("fx: ", depth_intrinsics.fx)
        print("fy: ", depth_intrinsics.fy)
        print("ppx: ", depth_intrinsics.ppx)
        print("ppy: ", depth_intrinsics.ppy)
        print("depth scale: ", depth_sensor.get_depth_scale())

        # blocks3d.depth_intrinsics = depth_intrinsics


        # blocks2d = block_detection_2d.get_blocks2d(color_image)
        # frame = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        # for color, boxes in blocks2d.items():
        #     for box in boxes:
        #         # print("Box: ", box)
        #         for corner in box:
        #             cv2.circle(frame, tuple(corner), 2, block_detection_2d.color_rgb[color], -1)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # blocks_colors_poses = blocks3d.get_block_center_3d(depth_image, blocks2d)

        # # center_pixel = rs.rs2_project_point_to_pixel(depth_intrinsics, center)
        # # print("center pixel: ", center_pixel)
        # # cv2.circle(frame, tuple(center_pixel), 2, (255,255,255), -1)
        # for color, poses in blocks_colors_poses.items():
        #     print(poses)

        # cv2.imshow('Frame', frame)
        
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break



finally:
    pipeline.stop()



