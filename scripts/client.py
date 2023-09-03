import copy

import actionlib
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2_helpers
from sensor_msgs.msg import Image, PointCloud2

from ros_uois.msg import (
    SceneSegmentationAction,
    SceneSegmentationGoal,
    SegmentationRequest,
)


class SceneSegmentationActionClient:
    def __init__(
        self,
        namespace="segmentation_server",
        action_spec=SceneSegmentationAction,
        timeout=15,
        is_debug=False,
        filter_pointcloud=True,
    ) -> None:
        """Scene segmentation action client

        Args:
            namespace (str, optional): Namespace for the action server. Defaults to "segmentation_server".
            action_spec (actionlib.ActionSpec, optional): Action specification. Defaults to SceneSegmentationAction.
            timeout (int, optional): Response timeout in seconds. Defaults to 30.
            is_debug (bool, optional): Run in debug mode. Enables visualization etc. Defaults to False.
            filter_pointcloud (bool, optional): Filter spillover points from the depth image using DBSCAN. Defaults to True.
        """
        # ROS Action client
        self._is_initialized = False
        self._action_spec = action_spec

        self.namespace = namespace
        self.response_timeout = timeout

        self.client = None

        # Post-process
        self.filter_pointcloud = filter_pointcloud

        # Debug
        self.is_debug = is_debug

    def initialize(self):
        """Initialize action client and wait for server to come up"""
        # Initialize action client
        self.client = actionlib.SimpleActionClient(
            ns=self.namespace, ActionSpec=self.action_spec
        )

        # Wait for the server to come up
        success = self.wait_for_server()

        if not success:
            raise ValueError(
                f"Could not connect to action server {self.namespace} within timeout: {self.response_timeout} seconds"
            )

        self._is_initialized = success
        return

    def wait_for_server(self) -> bool:
        """Wait for server to come up within response timeout

        Returns:
            bool: True if server is up, False otherwise
        """
        return self.client.wait_for_server(
            timeout=rospy.Duration.from_sec(self.response_timeout)
        )

    def decode_image(self, img: Image) -> np.ndarray:
        """Image msg decoding helper

        Args:
            img (Image): Image message

        Returns:
            np.ndarray: Image as numpy array
        """
        # Determine output dtype
        if img.encoding == "rgb8":
            res_dtype = np.uint8
        elif img.encoding == "16UC1":
            res_dtype = np.uint16
        elif img.encoding == "mono8":
            res_dtype = np.uint8
        else:
            raise NotImplementedError(
                f"Image encoding {img.encoding} is not supported. Please add it to the list of supported encodings"
            )

        # Decode image buffer
        image = np.frombuffer(img.data, dtype=res_dtype).reshape(
            img.height, img.width, -1
        )

        return image

    def get_rgbd_segmentation(
        self,
        num_min_points: int = 1024,
        return_mask: bool = True,
        return_rgb: bool = True,
        return_depth: bool = True,
        return_pointcloud: bool = True,
    ):
        """Get segmented depth pointcloud from action server

        NOTE:
            if pointcloud has less than 1024 points, oversamples existing points to provide num_min_points
            Filtering if enabled, uses DBSCAN to filter spillover points from the depth image

        Args:
            num_min_points (int, optional): Minimum number of points to return. Defaults to 1024.
            return_mask (bool, optional): Return binary segmentation mask. Defaults to True.
            return_rgb (bool, optional): Return RGB image. Defaults to True.
            return_depth (bool, optional): Return depth image. Defaults to True.
            return_pointcloud (bool, optional): Return pointcloud. Defaults to True.

        Returns:
            dict: Results dict with keys:
                    ["pointcloud", "depth_image", "rgb_image", "mask"]
        """
        limit_request_fail = 20
        pc_np = None
        counter = 0

        # Send request to action server
        while counter < limit_request_fail:
            results = self.send_request(
                return_pointcloud=return_pointcloud,
                return_depth=return_depth,
                return_mask=return_mask,
                return_rgb=return_rgb,
            )

            if results is not None:
                break

            counter += 1

        # Pointcloud
        pc_np = results["pointcloud"]

        # Filter spillover points from the depth image using DBSCAN
        if self.filter_pointcloud:
            clustering = DBSCAN(eps=0.05, min_samples=50)
            labels = clustering.fit_predict(pc_np)

            pc_np = pc_np[labels == 0]

        # Oversample pointcloud if necessary
        if pc_np.shape[0] < num_min_points:
            rospy.logwarn(
                "Obtained pointcloud has less than 1024 points. Oversampling existing points. You've been warned."
            )
            pc_np = np.repeat(pc_np, num_min_points // pc_np.shape[0], axis=0)[
                :num_min_points
            ]

        results["pointcloud"] = pc_np
        return results

    def send_request(
        self,
        return_pointcloud: bool = True,
        return_rgb: bool = False,
        return_depth: bool = False,
        return_mask: bool = False,
    ):
        # Check request
        for opt in (return_depth, return_mask, return_pointcloud, return_rgb):
            if not isinstance(opt, bool):
                raise ValueError(
                    f"return_depth, return_mask, return_pointcloud, return_rgb should all be bool."
                )

        # Construct goal msg
        goal_request = SegmentationRequest()
        goal_request.return_pointcloud = return_pointcloud
        goal_request.return_rgb = return_rgb
        goal_request.return_depth = return_depth
        goal_request.return_mask = return_mask

        # Send to server and wait
        self.client.send_goal(SceneSegmentationGoal(goal_request=goal_request))
        self.client.wait_for_result(rospy.Duration.from_sec(self.response_timeout))
        results = None

        try:
            # Get result
            result = self.client.get_result()

            ## Post-process result
            # Pointcloud
            if return_pointcloud:
                pc = result.result.pointcloud
                pc_buffer = copy.deepcopy(pc.data)
                pc_np = np.frombuffer(pc_buffer, dtype=np.float32).reshape((-1, 3))
            else:
                pc_np = None

            # Depth image
            depth_image = (
                self.decode_image(result.result.depth_image) if return_depth else None
            )

            # RGB image
            rgb_image = (
                self.decode_image(result.result.rgb_image) if return_rgb else None
            )

            # Binary segmentation mask
            mask = self.decode_image(result.result.mask) if return_mask else None

            # Results dict
            results = dict(
                pointcloud=pc_np,
                depth_image=depth_image,
                rgb_image=rgb_image,
                mask=mask,
            )

        except Exception as e:
            rospy.logerr(e)
            pc_np = None

        if self.is_debug:
            # Display point cloud
            import trimesh

            trimesh.Scene(
                trimesh.points.PointCloud(pc_np, colors=[150, 150, 150, 100]),
            ).show(flags={"axis": True})

        return results


if __name__ == "__main__":
    rospy.init_node("scene_processing_client")
    client = SceneSegmentationActionClient()
    client.initialize()
    results = client.send_request(return_depth=True, return_mask=True)
