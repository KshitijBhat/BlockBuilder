import logging
import argparse
import yaml

import numpy as np

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from autolab_core import RigidTransform, YamlConfig
from frankapy import FrankaArm

import sys
sys.path.append("/home/ros_ws")
from Planning.moveit_class import MoveItPlanner
from geometry_msgs.msg import Pose, Point, Quaternion
import scipy.spatial.transform as spt

def subsample(data, rate=0.1):
    idx = np.random.choice(np.arange(len(data)), size=int(rate * len(data)))
    return data[idx]


def make_det_one(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.eye(len(R)) @ Vt


def execute_pose(planner, pose):
    # convert pose goal to the panda_hand frame (the frame that moveit uses)
    pose_goal_moveit = planner.get_moveit_pose_given_frankapy_pose(pose)
    # plan a straight line motion to the goal
    plan = planner.get_straight_plan_given_pose(pose_goal_moveit)
    # execute the plan (uncomment after verifying plan on rviz)
    planner.execute_plan(plan)


def get_closest_grasp_pose(block_pose, world_pose, T_tag_world, T_ee_world, cube_size):
    block_rotation = RigidTransform.rotation_from_quaternion(block_pose.orientation)
    tag_axes = [
        block_rotation[:, 0], -block_rotation[:, 0],
        block_rotation[:, 1], -block_rotation[:, 1]
    ]
    x_axis_ee = RigidTransform.rotation_from_quaternion(world_pose.orientation)
    dots = [axis @ x_axis_ee for axis in tag_axes]
    grasp_x_axis = tag_axes[np.argmax(dots)]
    grasp_z_axis = np.array([0, 0, -1])
    grasp_y_axis = np.cross(grasp_z_axis, grasp_x_axis)

    grasp_rotation = spt.Rotation.from_matrix(make_det_one(np.c_[grasp_x_axis, grasp_y_axis, grasp_z_axis]))
    grasp_quaternion = grasp_rotation.as_quat()
    grasp_translation = block_pose.position
    grasp_translation.z = cube_size/2

    return Pose(position=grasp_translation, orientation=Quaternion(*grasp_quaternion))


def perform_pick(fa, pick_pose, lift_pose, no_gripper):
    fa.goto_gripper(0.05)
    fa.goto_pose(lift_pose)
    fa.goto_pose(pick_pose)

    if not no_gripper:
        fa.close_gripper()

    fa.goto_pose(lift_pose)


def calculate_pose(count):
    place_pose = RigidTransform(
        translation=[0.54875245, 0.11862949 + count * 0.05, 0.01705035],
        rotation=[[-0.02087884, 0.99942336, 0.02641552],
                  [0.99757839, 0.01907633, 0.06674037],
                  [0.06619797, 0.02774502, -0.99742065]],
        from_frame="franka_tool",
        to_frame="world")
    return place_pose


def perform_place(fa, place_pose, lift_pose, no_gripper):
    fa.goto_pose(lift_pose)
    fa.goto_pose(place_pose)

    if not no_gripper:
        fa.goto_gripper(0.05)

    fa.goto_pose(lift_pose)


def color_matches(marker_color, target_color, threshold=0.7):
    return (abs(marker_color.r - target_color[0]) < threshold and
            abs(marker_color.g - target_color[1]) < threshold and
            abs(marker_color.b - target_color[2]) < threshold)


def get_block_by_color(target_color_name):
    # Define color mappings (RGB values)
    COLOR_MAP = {
        'red': (1.0, 0.0, 0.0),
        'green': (0.0, 1.0, 0.0),
        'blue': (0.0, 0.0, 1.0),
        'yellow': (1.0, 1.0, 0.0),
    }

    target_color = COLOR_MAP.get(target_color_name.lower())
    if target_color is None:
        raise ValueError(f"Unknown color: {target_color_name}")

    rospy.loginfo(f"Looking for {target_color_name} block (RGB: {target_color})")

    i = 0
    while not rospy.is_shutdown() and i < 100:
        marker_list = rospy.wait_for_message('/world_marker_array', MarkerArray)

        for block_marker in marker_list.markers:
            print(block_marker)
            if color_matches(block_marker.color, target_color):
                rospy.loginfo(f"Found {target_color_name} block at {block_marker.pose.position}")
                return block_marker.pose
                # pose = block_marker.pose
                #
                # translation = [pose.position.x, pose.position.y, pose.position.z]
                # quaternion = [pose.orientation.x, pose.orientation.y,
                #               pose.orientation.z, pose.orientation.w]
                # rotation = RigidTransform.rotation_from_quaternion(quaternion)
                #
                # block_pose = Pose()
                # block_position_point = Point(translation*)
                # block_pose.position.x = 0.50688894
                # block_pose.position.y = -0.23398067
                # observe_pose.position.z = 0.47516064
                # observe_pose.orientation.x = 0.02366586
                # observe_pose.orientation.y = 0.70650001
                # observe_pose.orientation.z = -0.70463537
                # observe_pose.orientation.w = 0.06145785
                #
                # T_block_camera = RigidTransform(
                #     translation=translation,
                #     rotation=rotation,
                #     from_frame='block',
                #     to_frame='world'
                # )
                # rospy.loginfo(f"Found {target_color_name} block at {translation}")
                # return T_block_camera
            else:
                rospy.logdebug(
                    f"Ignoring block with color: {block_marker.color.r}, {block_marker.color.g}, {block_marker.color.b}")
        i += 1
    return None


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_grasp', '-ng', action='store_true')
    args = parser.parse_args()
    cfg = yaml.load(open('cfg.yaml'))
    # Load the predetermined camera info
    # TODO: Test removal of this
    T_camera_mount_delta = RigidTransform.load(cfg['T_tool_base_path'])
    cube_size = cfg['cube_size']

    count = 0  # identifies which block we're placing in a given row

    # Init the arm
    logging.info('Starting robot')
    fa = FrankaArm()
    fa.set_tool_delta_pose(T_camera_mount_delta)
    fa.reset_joints()
    fa.open_gripper()

    # This script plans a straight line path from the current robot pose to pose_goal
    # This plan can then be executed on the robot using the execute_plan method

    # create a MoveItPlanner object and start the moveit node
    franka_moveit = MoveItPlanner()

    # Get the world frame and create a "ready" position
    T_ready_world = fa.get_pose()
    T_ready_world.translation[0] += 0.25
    T_ready_world.translation[2] = 0.4
    ready_pose = Pose(position=Point(*T_ready_world.translation), rotation=Quaternion(*T_ready_world.quaternion))
    execute_pose(fa, ready_pose)
    # Move to ready position
    # fa.goto_pose(T_ready_world)

    # Pose for observing the picking area
    # T_observe_pick_world = RigidTransform(
    #     translation=[ 0.50688894, -0.23398067,  0.47516064],
    #     rotation=[
    #         [-.000595312285, -.998558656,  .0534888540],
    #         [-.992740906, -.00583873282, -.120051772],
    #         [.120191043, -.0531720416, -.991325635]
    #     ],
    #     from_frame='franka_tool',
    #     to_frame='world'
    # )

    wall_configuration = [
        [
            "red",
            "red",
            "blue",
            "green",
            "yellow"
        ]
    ]

    # Construct the Pose goal in panda_end_effector frame (that you read from fa.get_pose())
    observe_pose = Pose()
    observe_pose.position.x = 0.50688894
    observe_pose.position.y = -0.23398067
    observe_pose.position.z = 0.47516064
    observe_pose.orientation.x = 0.02366586
    observe_pose.orientation.y = 0.70650001
    observe_pose.orientation.z = -0.70463537
    observe_pose.orientation.w = 0.06145785


    while len(wall_configuration) > 0:
        row_configuration = wall_configuration.pop()
        logging.info(f'Row configuration: {row_configuration}')
        while len(row_configuration) > 0:
            color_to_find = row_configuration.pop()

            # fa.goto_pose(T_observe_pick_world)
            execute_pose(franka_moveit, observe_pose)

            block_pose = get_block_by_color(color_to_find)

            if not block_pose:
                logging.error(f"{color_to_find} block not found, exiting.")
                exit(2)

            # Get grasp pose
            T_grasp_world = get_closest_grasp_pose(block_pose, ready_pose, cube_size)
            print(T_grasp_world)

            T_place_world = calculate_pose(count)

            # Pose closer to pick/place poses
            T_lift = RigidTransform(translation=[0, 0, 0.05], from_frame=T_ready_world.to_frame,
                                    to_frame=T_ready_world.to_frame)
            T_lift_pick_world = T_lift * T_grasp_world
            T_lift_place_world = T_lift * T_place_world

            perform_pick(fa, T_grasp_world, T_lift_pick_world, args.no_grasp)
            fa.goto_pose(T_ready_world)
            perform_place(fa, T_place_world, T_lift_place_world, args.no_grasp)
            fa.goto_pose(T_ready_world)

            count += 1

    exit(0)
