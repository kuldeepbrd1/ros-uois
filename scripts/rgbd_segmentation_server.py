#!/usr/bin/python

import argparse
import json
import os
import sys

import numpy as np
import rospy
from addict import Dict

from ros_uois.base_action_server import (
    ACTION_STATES,
    ActionStateManager,
    BaseActionServer,
)
from ros_uois.camera.realsense_camera import RealsenseRGBDSyncCamera
from ros_uois.config import Config
from ros_uois.scene_processor import RGBDSceneProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Runner for Training Grasp Samplers")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="/root/workspaces/src/ros_uois/configs/sam_query_rgb_onnx.json",
        help="Path to config file",
    )

    parser.add_argument(
        "-debug",
        action="store_true",
        default=False,
        help="Setting this will disable wandb logger and ... TODO",
    )
    return parser.parse_args(sys.argv[4:])


class RGBDSceneSegmentationServer(BaseActionServer):
    def __init__(self, config, is_debug=False):
        super().__init__()

        self.config = load_config(config)

        # Initialize RGBD Camera Object
        self.camera = RealsenseRGBDSyncCamera(
            color_image_topic=self.config.camera.color.image_topic,
            color_info_topic=self.config.camera.color.info_topic,
            depth_image_topic=self.config.camera.depth.image_topic,
            depth_info_topic=self.config.camera.depth.info_topic,
            response_timeout=self.config.camera.response_timeout,
        )

        # Initialize Scene Processing Pipeline Object
        self.scene_processor = RGBDSceneProcessor(
            # pipeline_type="uois_segmentation",
            pipeline_type="sam_query_onnx",
            pipeline_config=self.config.pipeline,
            device="cuda:0",
            camera_intrinsic_matrix=self.camera.K,
            state_manager=self.state_manager,
            is_debug=is_debug,
        )

    def initialize_pipeline(self) -> None:
        """Initialize the pipeline components

        Args:
            state_manager (ActionStateManager): Action State Manager instance
        """
        self.camera.initialize()
        self.scene_processor.initialize()
        rospy.loginfo(
            "\n====\nInitialized RGBD Scene Segmentation Server \nWaiting for action request ...\n===="
        )

    def process_scene(self) -> dict:
        """Process the scene and return the results

        Returns:
            [type]: [description]
        """

        # Get Images
        self.update_state(ACTION_STATES.IMAGE_ACQUISITION)
        color_image, depth_image = self.camera.get_image()
        # Get Camera Intrinsics
        intrinsics = self.camera.K

        # Process Scene
        self.update_state(ACTION_STATES.PROCESSING)

        results = self.scene_processor.process(
            color_image=color_image,
            depth_image=depth_image,
            color_cam_K=intrinsics,
            depth_cam_K=intrinsics,
        )

        # Update Result
        self.update_state(ACTION_STATES.COMPLETED)

        return results


def load_config(config_fp: Config):
    if not os.path.isfile(config_fp):
        raise ValueError(f"Config file {os.path.abspath(config_fp)} does not exist")

    with open(config_fp, "r") as f:
        config = json.load(f)
    return Dict(config)


def main(args):
    server = RGBDSceneSegmentationServer(args.config)
    server.initialize_pipeline()


if __name__ == "__main__":
    args = parse_args()
    rospy.init_node("rgbd_segmentation_server")
    main(args)
    rospy.spin()
