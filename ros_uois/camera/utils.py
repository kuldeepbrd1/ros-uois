from typing import Union

import numpy as np
import torch

from .camera import BaseCamera


class DepthCameraUtils:
    def depth_to_pointcloud(
        depth: np.ndarray,
        rgb: np.ndarray = None,
        camera_intrinsics_matrix: np.ndarray = None,
        z_near: float = 0.0,
        z_far: float = 10,
    ) -> np.ndarray:
        """Convert depth image to pointcloud given camera intrinsics.
        Args:
                                        depth (np.ndarray): Depth image. ** In meters**
        Returns:
                                        np.ndarray: [nx3] (x, y, z) Point cloud.
        """

        # Params
        K = camera_intrinsics_matrix

        _fx = K[0, 0]
        _fy = K[1, 1]
        _cx = K[0, 2]
        _cy = K[1, 2]

        # Mask z_far > depth > z_near
        mask = np.where(depth > z_near)
        mask = np.where(depth < z_far)
        x, y = mask[1], mask[0]

        # world/camera frame de-projection
        world_x = (x.astype(np.float32) - _cx) * depth[y, x] / _fx
        world_y = (y.astype(np.float32) - _cy) * depth[y, x] / _fy
        world_z = depth[y, x]

        # Return RGB value at each point, if rgb
        if rgb is not None:
            rgb = rgb[y, x, :]

        #
        pc = np.vstack((world_x, world_y, world_z)).T

        if rgb is not None:
            rgb = rgb[y, x, :]

        return pc, rgb

    def depth_to_pointcloud_torch(
        depth: torch.Tensor,
        rgb: torch.Tensor = None,
        camera_intrinsics_matrix: Union[np.ndarray, torch.Tensor] = None,
        z_near: float = 0.0,
        z_far: float = 10,
    ) -> torch.Tensor:
        """Convert depth image to pointcloud given camera intrinsics.
        Args:
                                        depth (torch.Tensor): Depth image.
        Returns:
                                        torch.Tensor: [nx3] (x, y, z) Point cloud.
        """

        # Params
        K = camera_intrinsics_matrix

        _fx = K[0, 0]
        _fy = K[1, 1]
        _cx = K[0, 2]
        _cy = K[1, 2]

        # Mask z_far > depth > z_near
        mask = torch.where(depth > z_near and depth < z_far)
        x, y = mask[1], mask[0]

        # World /cam 3d frame de-projection
        world_x = (x.to(torch.float32) - _cx) * depth[y, x] / _fx
        world_y = (y.to(torch.float32) - _cy) * depth[y, x] / _fy
        world_z = depth[y, x]

        if rgb is not None:
            rgb = rgb[y, x, :]

        pc = torch.vstack((world_x, world_y, world_z)).T

        if rgb is not None:
            rgb = rgb[y, x, :]

        return pc, rgb


# class RGBDHelpers:
#     def initialization_test(rgb_camera: BaseCamera, depth_camera: BaseCamera):
#         if not rgb_camera.is_model_initialized:
#             raise ValueError(
#                 f"RGB Camera ({type(rgb_camera)}) intrinsics are not initialized"
#             )

#         if not depth_camera.is_model_initialized:
#             raise ValueError(
#                 f"Depth Camera ({type(rgb_camera)}) intrinsics are not initialized"
#             )

#         if rgb_camera.image_size != depth_camera.image_size:
#             raise ValueError(
#                 f"RGB image ({rgb_camera.shape}) and Depth image ({rgb_camera.shape}) image sizes do not match. Make sure Realsense node is initialized with `align_depth:=true`"
#             )
