from abc import abstractmethod

import numpy as np

from .camera.utils import DepthCameraUtils

# from .config import Config
from .scene_segmentor import (
    SAMQuerySegmentor,
    SAMQuerySegmentorONNX,
    SAMSegmentor,
    UOIS3DSceneSegmentor,
)


class SceneProcessor:
    def __init__(
        self,
        device,
        state_manager,
        use_tensors=False,
        return_pointcloud=False,
        is_debug=False,
    ):
        self.device = device
        self.use_tensors = True if use_tensors or "cuda" in device else False
        self.return_pointcloud = return_pointcloud
        self.is_debug = is_debug
        self.state_manager = state_manager

    @abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def process(self, *args, **kwargs):
        raise NotImplementedError


class RGBDSceneProcessor(SceneProcessor):
    PIPELINES = {
        "uois_segmentation": UOIS3DSceneSegmentor,
        "sam_segmentation": SAMSegmentor,
        "sam_query_segmentation": SAMQuerySegmentor,
        "sam_query_onnx": SAMQuerySegmentorONNX,
    }

    def __init__(
        self,
        state_manager,
        pipeline_type=None,
        pipeline_config=None,
        device="cuda:0",
        use_tensors=True,
        camera_intrinsic_matrix=None,
        return_pointcloud=True,
        num_points_out=4096,
        is_debug=False,
    ):
        super().__init__(
            device=device,
            use_tensors=use_tensors,
            state_manager=state_manager,
            return_pointcloud=return_pointcloud,
            is_debug=is_debug,
        )

        self.pipeline = self.PIPELINES[pipeline_type](
            pipeline_config,
            device=device,
            is_debug=self.is_debug,
            state_manager=self.state_manager,
        )

        self.num_points_out = num_points_out

    def process(
        self, color_image=None, depth_image=None, color_cam_K=None, depth_cam_K=None
    ):
        input_images = {}

        if self.pipeline.requires_rgb:
            if color_image is None:
                raise ValueError(
                    f"Pipeline {type(self.pipeline).__name__} requires RGB image"
                )
            else:
                input_images["color_image"] = color_image

            if color_cam_K is None:
                raise ValueError(
                    f"Pipeline {type(self.pipeline).__name__} requires RGB camera intrinsics"
                )

        if self.pipeline.requires_depth:
            if depth_image is None:
                raise ValueError(
                    f"Pipeline {type(self.pipeline).__name__} requires depth image"
                )
            if depth_cam_K is None:
                raise ValueError(
                    f"Pipeline {type(self.pipeline).__name__} requires depth camera intrinsics"
                )

        # TODO: Pre-proc filters go here, if any

        mask = self.pipeline.segment(
            color_images=[color_image],
            depth_images=[depth_image],
        )

        output = dict(
            mask=mask,
            color_image=color_image,
            depth_image=depth_image,
        )

        if self.is_debug:
            self.debug_show_images(color_image, depth_image, mask)

        if self.return_pointcloud:
            masked_depth = depth_image.squeeze(-1) * mask
            pointcloud, _ = self.depth_to_pointcloud(masked_depth, depth_cam_K)

            # filter [0,0,0] or [-0, -0, -0] zero points
            pointcloud[pointcloud == -0.0] = 0.0
            non_zero_rows = ~np.all(pointcloud == [0.0, 0.0, 0.0], axis=1)
            pointcloud = pointcloud[non_zero_rows]

            # Shuffle and sample {num_points_out} points
            np.random.shuffle(pointcloud)
            pointcloud = pointcloud[: self.num_points_out]
            output["pointcloud"] = pointcloud

            # If debug, show pointcloud here
            if self.is_debug:
                scene_pc, _ = self.depth_to_pointcloud(
                    depth_image.squeeze(-1), depth_cam_K
                )
                # self._debug_show_pointcloud(scene_pc=scene_pc, obj_pc=pointcloud)
        return output

    def initialize(
        self,
    ):
        self.pipeline.initialize_models()
        self.initialization_test()

    def initialization_test(self):
        print("Sanity checking pipeline initialization\n")
        pass

    def depth_to_pointcloud(
        self,
        depth_image,
        depth_cam_K,
        z_near=0.1,
        z_far=2,
    ):
        return DepthCameraUtils.depth_to_pointcloud(
            depth=depth_image,
            camera_intrinsics_matrix=depth_cam_K,
            z_near=z_near,
            z_far=z_far,
        )

    def debug_show_images(self, color_image, depth_image, mask):
        import cv2

        # concatenate images
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

        mask = mask.astype(np.uint8) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        mosaic = np.concatenate((color_image, depth_image, mask), axis=1)

        cv2.imshow("Mosaic", mosaic)
        cv2.waitKey(1)
        return

    def _debug_show_pointcloud(self, obj_pc: np.ndarray, scene_pc: np.ndarray = None):
        import trimesh

        pc = obj_pc.copy()
        r = np.ones_like(pc[..., 0]) * 255
        g = pc[..., 1] * 0 / max(pc[..., 1])
        b = pc[..., 2] * 0 / max(pc[..., 2])
        a = np.ones(pc.shape[0]) * 200
        colors = np.clip(np.array([r, g, b, a]).T, 0, 255).astype(np.uint8)

        # Trimesh
        pc = trimesh.PointCloud(pc, colors=colors)

        if scene_pc is not None:
            scene_pc = trimesh.points.PointCloud(scene_pc, colors=[50, 50, 50, 100])
            trimesh.Scene([pc, scene_pc]).show()
        else:
            trimesh.Scene([pc]).show()
