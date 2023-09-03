import argparse
import json
import os

import numpy as np
import rospy
from addict import Dict

from ros_uois.camera.realsense_camera import RealsenseRGBDSyncCamera
from ros_uois.scene_processor import RGBDSceneProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Runner for Training Grasp Samplers")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/uois_3d.json",
        help="Path to config file",
    )

    parser.add_argument(
        "-debug",
        action="store_true",
        default=False,
        help="Setting this will disable wandb logger and ... TODO",
    )
    return parser.parse_args()


def load_config(config_fp):
    if not os.path.isfile(config_fp):
        raise ValueError(f"Config file {os.path.abspath(config_fp)} does not exist")

    with open(config_fp, "r") as f:
        config = json.load(f)
    return Dict(config)


def main(args):
    # Config
    config = load_config(args.config)

    # Initialize Camera Subscriber
    rospy.loginfo("Initializing Camera Subscriber")

    # Initialize RGBD Camera Object
    camera = RealsenseRGBDSyncCamera(
        color_image_topic=config.camera.color.image_topic,
        color_info_topic=config.camera.color.info_topic,
        depth_image_topic=config.camera.depth.image_topic,
        depth_info_topic=config.camera.depth.info_topic,
        response_timeout=config.camera.response_timeout,
    )

    # Initialize Scene Processor and underlying models
    scene_processor = RGBDSceneProcessor(
        # pipeline_type="uois_segmentation",
        pipeline_type="sam_query_onnx",
        pipeline_config=config.pipeline,
        device="cuda:0",
        camera_intrinsic_matrix=camera.K,
        num_points_out=2048,
    )

    # Initialize Camera Subscribers
    camera.initialize()
    scene_processor.initialize()

    for _ in range(10):
        color_image, depth_image = camera.get_image()

        intrinsics = camera.K

        # Process Scene
        results = scene_processor.process(
            color_image=color_image,
            depth_image=depth_image,
            color_cam_K=intrinsics,
            depth_cam_K=intrinsics,
        )
        print(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    rospy.spin()
