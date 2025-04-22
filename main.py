import logging
import argparse
import yaml

import numpy as np
import csv
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from autolab_core import RigidTransform, YamlConfig
from frankapy import FrankaArm
from time import sleep
# from extract_color import extract


def make_det_one(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.eye(len(R)) @ Vt


def get_closest_grasp_pose(T_tag_world, T_ee_world, cube_size):
   tag_axes = [
       T_tag_world.rotation[:, 0], -T_tag_world.rotation[:, 0],
       T_tag_world.rotation[:, 1], -T_tag_world.rotation[:, 1]
   ]
   x_axis_ee = T_ee_world.rotation[:, 0]
   dots = [axis.dot(x_axis_ee) for axis in tag_axes]
   grasp_x_axis = tag_axes[np.argmax(dots)]
   grasp_z_axis = np.array([0, 0, -1])
   grasp_y_axis = np.cross(grasp_z_axis, grasp_x_axis)
   grasp_R = make_det_one(np.c_[grasp_x_axis, grasp_y_axis, grasp_z_axis])
    # Adjust cube size to match
   translation_R = T_tag_world.translation.copy()
   translation_R[-1] = cube_size/2
   grasp_translation = (T_tag_world.translation + np.array([0, 0, -cube_size / 2]))
   grasp_translation[-1] = cube_size/2
   return RigidTransform(
       rotation=grasp_R,
       translation=translation_R,
       from_frame=T_ee_world.from_frame, to_frame=T_ee_world.to_frame
   )


def perform_pick(fa, pick_pose, lift_pose, no_gripper):
    fa.open_gripper()
    fa.goto_pose(lift_pose, duration=2)
    fa.goto_pose(pick_pose)

    if not no_gripper:
        fa.close_gripper()

    fa.goto_pose(lift_pose, duration=2)


def calculate_pose(col, row):
    place_pose = RigidTransform(
        translation=[0.54875245, 0.11862949 + col * 0.06, 0.01705035 + row*0.05],
        rotation=[[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        from_frame="franka_tool",
        to_frame="world")
    return place_pose


def perform_place(fa, place_pose, lift_pose, no_gripper):
    fa.goto_pose(lift_pose, duration=2)
    fa.goto_pose(place_pose)

    if not no_gripper:
        fa.open_gripper()

    fa.goto_pose(lift_pose, duration=2)


def get_all_visible_blocks():
    return rospy.wait_for_message('/world_marker_array', MarkerArray).markers


def get_stable_visible_blocks(num_samples=50, delay=0.01, match_threshold=0.02):
    """
    Collect multiple samples of visible blocks and average positions for what appear
    to be the same blocks based on spatial proximity.

    Returns:
        A list of averaged block positions: [[x, y, z], ...]
    """
    block_groups = []

    for _ in range(num_samples):
        for marker in get_all_visible_blocks():
            pos = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
            quat = np.array([marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w])

            matched = False
            for group in block_groups:
                # Compare with the first item in the group (reference)
                if np.linalg.norm(pos - group["translations"][0]) < match_threshold:
                    group["translations"].append(pos)
                    group["quaternions"].append(quat)
                    matched = True
                    break

            if not matched:
                # Start a new group for a new block
                block_groups.append(
                    {
                        "color": [marker.color.r, marker.color.g, marker.color.b],
                        "translations": [pos],
                        "quaternions": [quat],
                    }
                )

        rospy.sleep(delay)

    # Average positions in each group
    return [
        {
            "color": group["color"],
            "translation": np.mean(group["translations"], axis=0),
            "quaternion": np.mean(group["quaternions"], axis=0)
        }
        for group in block_groups
    ]


def find_free_space(cube_size, visible_blocks):
    # Define the threshold distance for free space (i.e., at least cube_size apart in x and y)
    threshold = cube_size * 1.5  # Allowing a little padding around the cube

    # TODO: Adjust to use new format from stable_visible_blocks
    # Iterate over each visible block and search for free space around it
    for block_pos in visible_blocks:
        # Check for a nearby empty space by trying surrounding positions in the XY plane
        for dx in [-threshold, 0, threshold]:
            for dy in [-threshold, 0, threshold]:
                # TODO: We shouldn't search just neighbors, we should sample our picking area randomly
                #  and detect whether there are blocks there.
                # Candidate position in XY plane (keep Z the same)
                print(f"Block_pos: {block_pos[:2]}")
                candidate_pos = np.array(block_pos[:2]) + np.array([dx, dy])  # Only change x and y
                print(f"Candidate pos: {candidate_pos}")

                # Ensure the candidate position is not too close to any other blocks in the XY plane
                is_free = True

                if block_pos[0] > 0.65 or block_pos[1] < -0.34 or block_pos[1] > -0.06:
                    continue

                for other_block in visible_blocks:
                    # TODO: We have to adjust this so that we're using the actual cube and not the center of the cube
                    if np.linalg.norm(candidate_pos - np.array(other_block[:2])) < cube_size:
                        is_free = False
                        break

                # If this position is free, return it (with the fixed z-position)
                if is_free:
                    return np.append(candidate_pos, cube_size / 2)  # Add the fixed z-position

    # If no free space is found, return None
    return None


def color_matches(marker_color, target_color, threshold=0.7):
    return (abs(marker_color[0] - target_color[0]) < threshold and
            abs(marker_color[1] - target_color[1]) < threshold and
            abs(marker_color[2] - target_color[2]) < threshold)


def get_block_position(fa, color_to_find, duration=1.5):
    fa.goto_pose(T_observe_pick_world, duration=duration)
    sleep(1)
    return get_block_by_color(color_to_find)


def get_block_by_color(target_color_name=None):
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
        for block_marker in get_stable_visible_blocks():
            if color_matches(block_marker["color"], target_color):
                translation = block_marker["translation"]
                quaternion = block_marker["quaternion"]
                if translation[0] > 0.65 or translation[1] < -0.34 or translation[1] > -0.06:
                    continue

                rotation = RigidTransform.rotation_from_quaternion(quaternion)

                T_block_camera = RigidTransform(
                    translation=translation,
                    rotation=rotation,
                    from_frame='block',
                    to_frame='world'
                )
                rospy.loginfo(f"Found {target_color_name} block at {translation}")
                rospy.loginfo(f"{target_color_name} block rotation \n{rotation}")
                rospy.loginfo(f"{target_color_name} block quaternion \n{quaternion}")
                return T_block_camera
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

    row = 0  # identifies which row is currently being built

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
    ready_joints = fa.get_joints()

    # Pose for observing the picking area
    T_observe_pick_world = RigidTransform(
        translation=[0.50688894, -0.23398067, 0.47516064],
        rotation=[[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
        from_frame='franka_tool',
        to_frame='world'
    )

    T_lift = RigidTransform(translation=[0, 0, cube_size * 2], from_frame=T_ready_world.to_frame,
                            to_frame=T_ready_world.to_frame)

    # wall_configuration = extract("2D.png")
    # print(wall_configuration)

    wall_configuration = [
        ["red"],
        ["green"],
    ]

    block_placement_positions = []

    # Header row that gets ignored when writing the file
    pick_stats = [
        ["color", "pick_success", "x", "y", "row-col-configuration_type"]
    ]

    configuration_type = "2-1-red-green"

    while len(wall_configuration) > 0:
        col = 0 # identifies which block we're placing in a given row
        row_configuration = wall_configuration.pop(0)
        logging.info(f'Row configuration: {row_configuration}')
        while len(row_configuration) > 0:
            color_to_find = row_configuration.pop(0)
            T_block_world = get_block_position(fa, color_to_find)

            if not T_block_world:
                logging.error(f"{color_to_find} block not found, exiting.")
                pick_stats.append([color_to_find, False, -999, -999,
                                   configuration_type])
                exit(2)

            # Get grasp pose
            T_grasp_world = get_closest_grasp_pose(T_block_world, T_observe_pick_world, cube_size)
            T_grasp_world.translation[0] += cube_size/4
            T_grasp_world.translation[1] += cube_size/6

            grasp_quat = T_grasp_world.quaternion.copy()
            block_quat = T_block_world.quaternion.copy()
            grasp_quat[2] = block_quat[0]
            T_grasp_world.rotation = RigidTransform.rotation_from_quaternion(grasp_quat)
            print(f"calculated grasp pose: \n{T_grasp_world}")

            T_place_world = calculate_pose(col, row)
            block_placement_positions.append(T_place_world)

            # Pose closer to pick/place poses
            T_lift_pick_world = T_lift * T_grasp_world
            T_lift_place_world = T_lift * T_place_world

            num_pick_attempts = 3
            pick_failure = True
            while pick_failure:
                perform_pick(fa, T_grasp_world, T_lift_pick_world, args.no_grasp)
                pick_failure = fa.get_gripper_width() < 0.004
                pick_stats.append([color_to_find, not pick_failure, T_block_world.translation[0], T_block_world.translation[1], configuration_type])

                if num_pick_attempts == 0 or not pick_failure:
                    break

                num_pick_attempts -= 1
                T_block_world = get_block_position(fa, color_to_find, duration=2)
                T_grasp_world = get_closest_grasp_pose(T_block_world, T_ready_world, cube_size)
                T_lift_pick_world = T_lift * T_grasp_world

            fa.goto_joints(ready_joints, duration=1.5)
            perform_place(fa, T_place_world, T_lift_place_world, args.no_grasp)
            fa.goto_joints(ready_joints, duration=1.5)
            col += 1

        row += 1

    with open('stats.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(pick_stats[1:])

    # while block_placement_positions:
    #     block_placement_position = block_placement_positions.pop()
    #     T_lift_pick_world = T_lift * block_placement_position
    #     perform_pick(fa, block_placement_position, T_lift_pick_world, args.no_grasp)
    #     fa.goto_joints(ready_joints, duration=1.5)
    #     fa.goto_pose(T_observe_pick_world, duration=1.5)
    #     blocks = get_stable_visible_blocks(num_samples=10)
    #     i = 0
    #     while not blocks and i < 50:
    #         blocks = get_stable_visible_blocks(num_samples=10)
    #         i += 1
    #
    #     print(blocks)
    #     if (free_space := find_free_space(cube_size, blocks)) is not None:
    #         print(f"Free space: {free_space}")
    #         fa.goto_pose(RigidTransform(
    #             translation=free_space,
    #             rotation=[
    #                 [0, -1, 0], [-1, 0, 0], [0, 0, -1]
    #             ],
    #             from_frame='franka_tool',
    #             to_frame='world'
    #         ))
    #
    #     fa.open_gripper()
    #     fa.goto_joints(ready_joints, duration=1.5)


    exit(0)
