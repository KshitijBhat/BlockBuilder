import logging
import argparse
import yaml

import numpy as np

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from autolab_core import RigidTransform, YamlConfig
from frankapy import FrankaArm
from time import sleep


def subsample(data, rate=0.1):
    idx = np.random.choice(np.arange(len(data)), size=int(rate * len(data)))
    return data[idx]


def make_det_one(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.eye(len(R)) @ Vt


def get_closest_grasp_pose(T_tag_world, T_ee_world, cube_size):
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
    T_tag_world.translation[-1] = cube_size/2
    # grasp_translation = (T_tag_world.translation + np.array([0, 0, -cube_size / 2]))
    # grasp_translation[-1] = cube_size/2
    return RigidTransform(
        rotation=grasp_R,
        translation=T_tag_world.translation,
        from_frame=T_ee_world.from_frame, to_frame=T_ee_world.to_frame
    )


def perform_pick(fa, pick_pose, lift_pose, no_gripper):
#    fa.goto_gripper(0.05)
    fa.open_gripper()
    fa.goto_pose(lift_pose)
    fa.goto_pose(pick_pose)

    if not no_gripper:
        fa.close_gripper()

    fa.goto_pose(lift_pose)


def calculate_pose(col, row):
    place_pose = RigidTransform(
        translation=[0.54875245, 0.11862949 + col * 0.06, 0.01705035 + row*0.05],
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
        fa.open_gripper()

    fa.goto_pose(lift_pose)


def color_matches(marker_color, target_color, threshold=0.7):
    return (abs(marker_color.r - target_color[0]) < threshold and
            abs(marker_color.g - target_color[1]) < threshold and
            abs(marker_color.b - target_color[2]) < threshold)

def get_block_position(fa, color_to_find):
    fa.goto_pose(T_observe_pick_world)
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
        marker_list = rospy.wait_for_message('/world_marker_array', MarkerArray)

        for block_marker in marker_list.markers:
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
                rospy.loginfo(f"{target_color_name} block rotation {rotation}")
                return T_block_camera
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
    print(T_ready_world)
    # Move to ready position
    fa.goto_pose(T_ready_world)

    # Pose for observing the picking area
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
            "blue",
            "red",
            "green",
        ],
        [
            "green",
            "red",
            "blue",
        ]
    ]

    block_placement_positions = []

    while len(wall_configuration) > 0:
        col = 0 # identifies which block we're placing in a given row
        row_configuration = wall_configuration.pop(0)
        logging.info(f'Row configuration: {row_configuration}')
        while len(row_configuration) > 0:
            color_to_find = row_configuration.pop(0)
            T_block_world = get_block_position(fa, color_to_find)

            if not T_block_world:
                logging.error(f"{color_to_find} block not found, exiting.")
                exit(2)

            print(T_block_world)

           # T_block_world.translation[0] += cube_size/2
           # T_block_world.translation[1] -= cube_size/6

            print(T_block_world)
            # Get grasp pose
            T_grasp_world = get_closest_grasp_pose(T_block_world, T_ready_world, cube_size)
            print(T_grasp_world)
            # T_grasp_world.translation[0] += cube_size/2

            T_place_world = calculate_pose(col, row)
            block_placement_positions.append(T_place_world)

            # Pose closer to pick/place poses
            T_lift = RigidTransform(translation=[0, 0, cube_size*2], from_frame=T_ready_world.to_frame,
                                    to_frame=T_ready_world.to_frame)
            T_lift_pick_world = T_lift * T_grasp_world
            T_lift_place_world = T_lift * T_place_world

            num_pick_attempts = 3
            pick_failure = True
            while pick_failure:
                perform_pick(fa, T_grasp_world, T_lift_pick_world, args.no_grasp)
                pick_failure = fa.get_gripper_width() < 0.004
                if num_pick_attempts == 0 or not pick_failure:
                    break

                num_pick_attempts -= 1
                T_block_world = get_block_position(fa, color_to_find)
                T_grasp_world = get_closest_grasp_pose(T_block_world, T_ready_world, cube_size)
                T_lift_pick_world = T_lift * T_grasp_world

            fa.goto_pose(T_ready_world)
            perform_place(fa, T_place_world, T_lift_place_world, args.no_grasp)
            fa.goto_pose(T_ready_world)
            col += 1
        
        row += 1

    while block_placement_positions:
        block_placement_position = block_placement_positions.pop()
        T_grasp_world = get_closest_grasp_pose(block_placement_position, T_ready_world, cube_size)
        T_lift_pick_world = T_lift * block_placement_position
        perform_pick(fa, T_grasp_world, T_lift_pick_world, args.no_grasp)
        fa.goto_pose(T_ready_world)
        fa.open_gripper()


    exit(0)
