#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge, CvBridgeError 
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import ColorRGBA, Header
from tf.transformations import quaternion_from_matrix
import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt


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
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for color, _ in self.color_ranges.items():
            mask = cv2.inRange(hsv_image, self.color_ranges[color][0], self.color_ranges[color][1])
            # kernel = np.ones((3, 3), np.uint8)
            # mask = cv2.dilate(mask, kernel, iterations=1)
            # plt.imshow(mask, cmap='gray')
            # plt.show()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(image, contours, -1, (0, 255, 0), 2) 
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
                # cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

                # plt.imshow(image)
                # plt.show()

        return blocks
        

    
    def get_depth_at_point(self, depth_image, x, y):
        
        depth_value = depth_image[y, x]

        # Calculate real-world coordinates
        depth_in_meters = depth_value * self.depth_scale
        pixel = [float(x), float(y)]  # Convert pixel coordinates to floats
        point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, pixel, depth_in_meters)

        return point
    
    def get_block_center_3d(self, color_image, depth_image):

        block_color_poses = {
            'red': [],
            'green': [],
            'blue': [],
            'yellow': []
        }

        blocks2d = self.get_blocks2d(color_image)

        for color, boxes in blocks2d.items():
            for box in boxes:
                # print("Box: ", box)
                for corner in box:
                    cv2.circle(color_image, tuple(corner), 2, (0, 255, 0), -1)
        # plt.imshow(color_image)
        # plt.show()
        for color, boxes in blocks2d.items():
            for box in boxes:
                corners3d = []
                for corner in box:
                    x, y = corner
                    corner3d = self.get_depth_at_point(depth_image, x, y)
                    corners3d.append(corner3d)
                corners3d = np.array(corners3d)
                center = np.mean(corners3d, axis=0)
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



class BlocksPublisher:
    def __init__(self):
        rospy.init_node('blocks_publisher')
        self.publisher = rospy.Publisher('/marker_array', MarkerArray, queue_size=10)

        self.latest_color = None
        self.latest_depth = None

        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)

        self.rate = rospy.Rate(10)

        rospy.loginfo("Block detection node initialized")
        
        # Define color mappings
        self.color_map = {
            "red": ColorRGBA(1.0, 0.0, 0.0, 1.0),
            "green": ColorRGBA(0.0, 1.0, 0.0, 1.0),
            "blue": ColorRGBA(0.0, 0.0, 1.0, 1.0),
            "yellow": ColorRGBA(1.0, 1.0, 0.0, 1.0)
        }

        self.BlockDetector = BlocksDetection3D()

    def color_callback(self, msg):
        
        self.color_bridge = CvBridge()
        try:
            cv_image = self.color_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            color_np = np.array(cv_image)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error {e}")

        self.latest_color = color_np

    def depth_callback(self, msg):
        
        self.depth_bridge = CvBridge()
        try:
            cv_image = self.depth_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_np = np.array(cv_image)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error {e}")

        self.latest_depth = depth_np

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
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        
        # Set color
        marker.color = self.color_map.get(color_name.lower(), self.color_map["red"])
        marker.lifetime = rospy.Duration(1)  # Refresh every 1 second
        
        return marker

    def publish_blocks(self):
        """Publish all blocks as a MarkerArray"""
        if self.latest_color is not None and self.latest_depth is not None:
            block_centers = self.BlockDetector.get_block_center_3d(color_image=self.latest_color, depth_image=self.latest_depth)
            marker_array = MarkerArray()
            
            # for idx, block in enumerate(blocks_data):
            #     marker = self.create_marker(
            #         block_id=idx,
            #         pose=block["pose"],
            #         color_name=block["color"]
            #     )
            #     marker_array.markers.append(marker)

            id_counter=0
            for color, centers in block_centers.items():
                for center in centers:
                    # print(center)
                    marker = self.create_marker(
                        block_id=id_counter,
                        pose=Pose(position=Point(*center)),
                        color_name=color
                    )
                    marker_array.markers.append(marker)
                    id_counter += 1
            
            self.publisher.publish(marker_array)

if __name__ == '__main__':
    visualizer = BlocksPublisher()
    rate = rospy.Rate(10)  # 10 Hz

    
    while not rospy.is_shutdown():
        visualizer.publish_blocks()
        rate.sleep()