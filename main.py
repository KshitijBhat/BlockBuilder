import os
import logging
import argparse
import yaml
import json
from time import sleep

import numpy as np

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from autolab_core import RigidTransform, YamlConfig
from frankapy import FrankaArm


def subsample(data, rate=0.1):
    idx = np.random.choice(np.arange(len(data)), size=int(rate * len(data)))
    return data[idx]


def make_det_one(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.eye(len(R)) @ Vt


def get_closest_grasp_pose(T_tag_world, T_ee_world):
    tag_axes = [
        T_tag_world.rotation[:, 0], -T_tag_world.rotation[:, 0],
        T_tag_world.rotation[:, 1], -T_tag_world.rotation[:, 1]
    ]
    x_axis_ee = T_ee_world.rotation[:, 0]
    dots = [axis @ x_axis_ee for axis in tag_axes]
    grasp_x_axis = tag_axes[np.argmax(dots)]
    grasp_z_axis = np.array([0, 0, -1])
    grasp_y_axis = np.cross(grasp_z_axis, grasp_x_axis)
    grasp_R = make_det_one(np.c_[grasp_x_axis, grasp_y_axis, grasp_z_axis])
    # Adjust cube size to match
    cube_size = .0254
    grasp_translation = (T_tag_world.translation + np.array([0, 0, -cube_size / 2]))
    grasp_translation[-1] = cube_size/2
    return RigidTransform(
        rotation=grasp_R,
        translation=grasp_translation,
        from_frame=T_ee_world.from_frame, to_frame=T_ee_world.to_frame
    )


def perform_pick(fa, pick_pose, lift_pose, use_gripper=True):
    # fa.goto_gripper(0.05)
    fa.open_gripper()
    fa.goto_pose(lift_pose)
    fa.goto_pose(pick_pose)

    if use_gripper:
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


def perform_place(fa, place_pose, lift_pose, use_gripper=True):
    fa.goto_pose(lift_pose)
    fa.goto_pose(place_pose)

    if use_gripper:
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
        # print(marker_list)

        for block_marker in marker_list.markers:
            print(block_marker)
            if color_matches(block_marker.color, target_color):
                pose = block_marker.pose
                translation = [pose.position.x, pose.position.y, pose.position.z]
                quaternion = [pose.orientation.x, pose.orientation.y,
                              pose.orientation.z, pose.orientation.w]
                rotation = RigidTransform.rotation_from_quaternion(quaternion)

                T_block_camera = RigidTransform(
                    translation=translation,
                    rotation=rotation,
                    from_frame='block',
                    to_frame='world'
                )
                rospy.loginfo(f"Found {target_color_name} block at {translation}")
                return T_block_camera
            else:
                rospy.logdebug(
                    f"Ignoring block with color: {block_marker.color.r}, {block_marker.color.g}, {block_marker.color.b}")
        i += 1


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_grasp', '-ng', action='store_true')
    args = parser.parse_args()
    cfg = yaml.load(open('cfg.yaml'))
    # Load the predetermined camera info
    T_camera_ee = RigidTransform.load(cfg['T_rs_tool_path'])
    T_camera_mount_delta = RigidTransform.load(cfg['T_tool_base_path'])
    # T_camera_world = RigidTransform.load(cfg['T_rs_base_path'])

    # Load the wall that we want to build, can disable once we're recognizing blocks
    # blocks = json.load(open('blocks.json'))
    # print(blocks)
    count = 0  # identifies which block we're placing in a given row

    # Init the arm
    logging.info('Starting robot')
    fa = FrankaArm()
    fa.set_tool_delta_pose(T_camera_mount_delta)
    fa.reset_joints()
    fa.open_gripper()

    # Get the world frame and create a "ready" position
    T_ready_world = fa.get_pose()
    T_ready_world.translation[0] += 0.25
    T_ready_world.translation[2] = 0.4
    # Move to ready position
    fa.goto_pose(T_ready_world)

    T_observe_pick_world = RigidTransform(
        translation=[ 0.50688894, -0.23398067,  0.47516064],
        rotation=[
            [-.000595312285, -.998558656,  .0534888540],
            [-.992740906, -.00583873282, -.120051772],
            [.120191043, -.0531720416, -.991325635]
        ],
        from_frame='franka_tool',
        to_frame='world'
    )

    wall_configuration = [
        [
            "red",
            "red",
            "blue",
            "green",
            "yellow"
        ]
    ]

    while len(wall_configuration) > 0:
        row_configuration = wall_configuration.pop()
        logging.info(f'Row configuration: {row_configuration}')
        while len(row_configuration) > 0:
            color_block_to_find = row_configuration.pop()

            fa.goto_pose(T_observe_pick_world)
            T_block_world = get_block_by_color(color_block_to_find)

            # TODO: There's no need to adjust the pose given by camera,
            #  grasp function performs that calculation using the block size

            # Calc translation for block
            # print(T_camera_ee)
            # print(T_ready_world)
            # T_camera_world = T_ready_world * T_camera_ee
            # print(T_camera_world)
            # T_block_world = T_camera_world * T_block_camera
            print(T_block_world)
            # logging.info(f'{color_block_to_find} block has translation {T_block_world}')
            # T_tag_world = T_camera_world * T_tag_camera
            # block_pose = blocks.pop()
            # logging.info('Tag has translation {}'.format(T_tag_world.translation))

            # logging.info('Finding closest orthogonal grasp')
            # Get grasp pose
            T_grasp_world = get_closest_grasp_pose(T_block_world, T_ready_world)
            print(f"Grasp in world frame: {T_grasp_world}")
            # exit(0)
            # T_grasp_world = get_closest_grasp_pose(T_tag_world, T_ready_world)
            # print(T_grasp_world)
            # T_grasp_world = RigidTransform(translation=block_pose["translation"], rotation=block_pose["rotation"],
            #                                from_frame="franka_tool", to_frame="world")
            T_place_world = calculate_pose(count)

            # Pose closer to pick/place poses
            T_lift = RigidTransform(translation=[0, 0, 0.05], from_frame=T_ready_world.to_frame,
                                    to_frame=T_ready_world.to_frame)
            T_lift_pick_world = T_lift * T_grasp_world
            T_lift_place_world = T_lift * T_place_world

            perform_pick(fa, T_grasp_world, T_lift_pick_world, not args.no_grasp)
            fa.goto_pose(T_ready_world)
            perform_place(fa, T_place_world, T_lift_place_world, not args.no_grasp)
            fa.goto_pose(T_ready_world)

            count += 1

    exit(0)
