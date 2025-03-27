#!/usr/bin/env python
import rospy
from visualization_msgs.msg import MarkerArray
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

class MarkerTransformer:
    def __init__(self):
        # Create TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publisher for transformed markers
        self.marker_pub = rospy.Publisher('/world_marker_array', MarkerArray, queue_size=1)
        
        # Subscribe to your original marker array
        self.marker_sub = rospy.Subscriber('/marker_array', MarkerArray, 
                                           self.marker_callback)
        
        # Wait for TF to be ready
        rospy.sleep(1.0)
        
    def marker_callback(self, marker_array):
        transformed_markers = MarkerArray()
        
        try:
            # Look up transform from camera frame to world frame
            transform = self.tf_buffer.lookup_transform(
                'panda_link0',
                'camera_depth_optical_frame',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            # Transform each marker in the array
            for marker in marker_array.markers:
                # Create PoseStamped from marker's pose
                pose_stamped = PoseStamped()
                pose_stamped.header = marker.header
                pose_stamped.pose = marker.pose
                
                # Transform the pose
                transformed_pose = tf2_geometry_msgs.do_transform_pose(
                    pose_stamped, transform)
                
                # Update the marker
                marker.header.frame_id = 'panda_link0'
                marker.pose = transformed_pose.pose
                
                transformed_markers.markers.append(marker)
            
            # Publish the transformed marker array
            self.marker_pub.publish(transformed_markers)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to transform markers: {e}")

if __name__ == '__main__':
    rospy.init_node('marker_tf_changer')
    transformer = MarkerTransformer()
    rospy.spin()