import os
import logging
import argparse
import yaml
import json
from time import sleep

import numpy as np

from autolab_core import RigidTransform, YamlConfig

# from perception_utils.apriltags import AprilTagDetector
# from perception_utils.realsense import get_first_realsense_sensor

from frankapy import FrankaArm


def subsample(data, rate=0.1):
    idx = np.random.choice(np.arange(len(data)), size=int(rate * len(data)))
    return data[idx]


def make_det_one(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.eye(len(R)) @ Vt


def get_closest_grasp_pose(T_tag_world, T_ee_world):
    tag_axes = [
        T_tag_world.rotation[:,0], -T_tag_world.rotation[:,0],
        T_tag_world.rotation[:,1], -T_tag_world.rotation[:,1]
    ]
    x_axis_ee = T_ee_world.rotation[:,0]
    dots = [axis @ x_axis_ee for axis in tag_axes]
    grasp_x_axis = tag_axes[np.argmax(dots)]
    grasp_z_axis = np.array([0, 0, -1])
    grasp_y_axis = np.cross(grasp_z_axis, grasp_x_axis)
    grasp_R = make_det_one(np.c_[grasp_x_axis, grasp_y_axis, grasp_z_axis])
    # Adjust cube size to match
    cube_size = .0254
    grasp_translation = T_tag_world.translation + np.array([0, 0, -cube_size / 2])
    return RigidTransform(
        rotation=grasp_R,
        translation=grasp_translation,
        from_frame=T_ee_world.from_frame, to_frame=T_ee_world.to_frame
    )


def perform_pick(fa, pick_pose, lift_pose):
    fa.goto_pose(lift_pose)
    fa.goto_gripper(0.04)
    fa.goto_pose(lift_pose)
    fa.goto_pose(pick_pose)
    fa.close_gripper()
    fa.goto_pose(lift_pose)


def calculate_pose(count):
    place_pose = RigidTransform(
            translation = [0.54875245, 0.11862949 + count*0.04, 0.01705035],
            rotation = [[-0.02087884,  0.99942336,  0.02641552],
            [ 0.99757839,   0.01907633,  0.06674037],
            [ 0.06619797,  0.02774502, -0.99742065]],
            from_frame="franka_tool",
            to_frame="world")
    return place_pose

def perform_place(fa, place_pose, lift_pose):
    fa.goto_pose(place_pose)
    fa.open_gripper()
    fa.goto_pose(lift_pose)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg.yaml')
    parser.add_argument('--no_grasp', '-ng', action='store_true')
    args = parser.parse_args()
    cfg = yaml.load(open(args.cfg))
    T_camera_ee = RigidTransform.load(cfg['T_camera_ee_path'])
    T_camera_mount_delta = RigidTransform.load(cfg['T_camera_mount_path'])
    blocks = json.load(open('blocks.json'))
    count = 0 # identifies which block we're placing in a given row

    # Init the arm
    logging.info('Starting robot')
    fa = FrankaArm()
    fa.set_tool_delta_pose(T_camera_mount_delta)
    fa.reset_joints()
    fa.open_gripper()

    # Get the world frame
    T_ready_world = fa.get_pose()
    T_ready_world.translation[0] += 0.25
    T_ready_world.translation[2] = 0.4

    # Move slightly
    fa.goto_pose(T_ready_world)

    # Init the camera
    # logging.info('Init camera')
    # sensor = get_first_realsense_sensor(cfg['rs'])
    # sensor.start()

    # logging.info('Detecting Color Blocks')
    # Replace this with perception code for "ColorBlockDetector"
    # april = AprilTagDetector(cfg['april_tag'])
    # intr = sensor.color_intrinsics

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
            # Get all blocks
            # T_blocks_camera = color_blocks.detect(sensor, intr, vis=cfg['vis_detect'])
            # {"red": [
            # {
            #   "rotation": [],
            #   "translation": []
            # }
            # ], "blue": []}
            # block_pose = T_blocks_camera[color_block_to_find][0]
            # T_block_camera = RigidTransform(
            #   translation=block_pose["translation"],
            #   rotation=block_pose["rotation"],
            #   from_frame="block",
            #   to_frame="realsense")
            # TODO: There's no need to adjust the pose given by camera,
            #  grasp function performs that calculation using the block size
            # T_tag_camera = april.detect(sensor, intr, vis=cfg['vis_detect'])[0]
            #T_tag_camera = RigidTransform(
            #    translation=[0, 0, 0.0127],
            #    from_frame='tag',
            #    to_frame='realsense'
            #)

            # Calc translation for block
            #T_camera_world = T_ready_world * T_camera_ee
            # T_block_world = T_camera_world * T_block_camera
            # logging.info(f'{color_block_to_find} block has translation {T_block_world}')
            #T_tag_world = T_camera_world * T_tag_camera
            block_pose = blocks.pop()
            # logging.info('Tag has translation {}'.format(T_tag_world.translation))

            # logging.info('Finding closest orthogonal grasp')
            # Get grasp pose
            # T_grasp_world = get_closest_grasp_pose(T_block_world, T_ready_world)
            # T_grasp_world = get_closest_grasp_pose(T_tag_world, T_ready_world)
            # print(T_grasp_world)
            T_grasp_world = RigidTransform(translation=block_pose["translation"], rotation=block_pose["rotation"], from_frame="franka_tool", to_frame="world")
            # Pose closer to grasp pose
            T_lift = RigidTransform(translation=[0, 0, 0.05], from_frame=T_ready_world.to_frame, to_frame=T_ready_world.to_frame)
            T_lift_pick_world = T_lift * T_grasp_world

            place_pose = calculate_pose(count)
            T_lift_place_world = T_lift * place_pose

            logging.info('Visualizing poses')
            #_, depth_im, _ = sensor.frames()
            #points_world = T_camera_world * intr.deproject(depth_im)

            if not args.no_grasp:
                logging.info('Commanding robot')
                perform_pick(fa, T_grasp_world, T_lift_pick_world)
                fa.goto_pose(T_ready_world)
                perform_place(fa, place_pose, T_lift_place_world)
                fa.goto_pose(T_ready_world)
                count += 1

    exit(0)
