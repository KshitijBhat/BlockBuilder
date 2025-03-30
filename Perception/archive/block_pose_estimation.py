#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import ColorRGBA, Header
from tf.transformations import quaternion_from_matrix
import cv2
import numpy as np
import pyrealsense2 as rs


class BlocksDetection3D:
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

        # Change these depth intrinsics for the Intel RealSense D435

        self.fx = 593.1236572265625
        self.fy = 593.1236572265625
        self.cx = 421.08984375
        self.cy = 240.96368408203125

        self.depth_intrinsics = rs.pyrealsense2.intrinsics()
        self.depth_intrinsics.height = 480
        self.depth_intrinsics.width = 848
        self.depth_intrinsics.ppx = self.cx
        self.depth_intrinsics.ppy = self.cy
        self.depth_intrinsics.fx = self.fx
        self.depth_intrinsics.fy = self.fy
        self.depth_intrinsics.model = rs.pyrealsense2.distortion.inverse_brown_conrady
        self.depth_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.depth_scale = 0.001


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
        """
        Calculate the pose (position and orientation) of a block from its 3D corners
        Returns: position (xyz) and quaternion (xyzw)
        """
        
        
        # Calculate the center of the block (position)
        center = np.mean(corners3d, axis=0)
        
        # Calculate vectors from the first corner to the other corners
        v1 = np.array(corners3d[1]) - np.array(corners3d[0])
        v2 = np.array(corners3d[3]) - np.array(corners3d[0])
        
        # Normalize the vectors safely
        v1_norm = np.linalg.norm(v1)
        if v1_norm > 1e-6:
            v1 = v1 / v1_norm
        else:
            v1 = np.array([1.0, 0.0, 0.0])  # Default to x-axis if degenerate
        
        v2_norm = np.linalg.norm(v2)
        if v2_norm > 1e-6:
            v2 = v2 / v2_norm
        else:
            v2 = np.array([0.0, 1.0, 0.0])  # Default to y-axis if degenerate
        
        # Calculate the normal vector (perpendicular to the plane)
        normal = np.cross(v1, v2)
        
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-6:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, 0.0, 1.0])  # Default to z-axis if degenerate
        
        # Ensure orthogonality by recomputing v2
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)  # This should be safe now
        
        # Create the rotation matrix
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[:, 0] = v1
        rotation_matrix[:, 1] = v2
        rotation_matrix[:, 2] = normal
        
        # Create the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = center
        transformation_matrix[:3, :3] = rotation_matrix
        
        # Extract position and orientation
        position = center
        quaternion = quaternion_from_matrix(transformation_matrix)
        
        return position, quaternion



class BlockVisualizer:
    def __init__(self):
        rospy.init_node('block_visualizer')
        self.publisher = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
        
        # Define color mappings
        self.color_map = {
            "red": ColorRGBA(1.0, 0.0, 0.0, 1.0),
            "green": ColorRGBA(0.0, 1.0, 0.0, 1.0),
            "blue": ColorRGBA(0.0, 0.0, 1.0, 1.0),
            "yellow": ColorRGBA(1.0, 1.0, 0.0, 1.0)
        }

    def create_marker(self, block_id, pose, color_name):
        """Create a cube marker for a block"""
        marker = Marker()
        marker.header = Header(frame_id="camera_depth_optical_frame")
        marker.header.stamp = rospy.Time.now()
        marker.ns = "blocks"
        marker.id = block_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Set pose and scale
        marker.pose = pose
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # Set color
        marker.color = self.color_map.get(color_name.lower(), self.color_map["red"])
        marker.lifetime = rospy.Duration(1)  # Refresh every 1 second
        
        return marker

    def publish_blocks(self, blocks_data):
        """Publish all blocks as a MarkerArray"""
        marker_array = MarkerArray()
        
        for idx, block in enumerate(blocks_data):
            marker = self.create_marker(
                block_id=idx,
                pose=block["pose"],
                color_name=block["color"]
            )
            marker_array.markers.append(marker)
        
        self.publisher.publish(marker_array)

if __name__ == '__main__':
    visualizer = BlockVisualizer()
    rate = rospy.Rate(10)  # 10 Hz
    
    # Example block data - replace with your actual data
    example_blocks = [
        {
            "color": "red",
            "pose": Pose(position=Point(0.5, 0.2, 0.3))
        },
        {
            "color": "green",
            "pose": Pose(position=Point(-0.3, 0.4, 0.5))
        },
        {
            "color": "blue",
            "pose": Pose(position=Point(0.0, -0.2, 0.4))
        }
    ]
    
    while not rospy.is_shutdown():
        visualizer.publish_blocks(example_blocks)
        rate.sleep()