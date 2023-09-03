import copy
import warnings
from abc import abstractmethod
from typing import Any, List, Tuple, Union

import cv2
import einops
import numpy as np
import torch
from addict import Dict
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from segment_anything import sam_model_registry as SAM_MODEL_REGISTRY
from uois.segmentation import UOISNet3D

from uois import data_augmentation

from .base_action_server import ACTION_STATES
from .camera import BaseCamera
from .config import Config
from .config_types import *
from .utils import alpha_blend


class RGBDSceneSegmentor:
    """Abstract Base class for scene segmentation methods"""

    _REQUIRES_RGB = None
    _REQUIRES_DEPTH = None

    def __init__(self, config, device, state_manager, is_debug=False):
        self._validate_config(config)

        self.config = config
        self.device = device
        self.is_debug = is_debug
        self.state_manager = state_manager

    @property
    def requires_rgb(self):
        if self._REQUIRES_RGB is None:
            raise NotImplementedError(
                f"_REQUIRES_RGB not provided by {type(self).__name__}"
            )
        return self._REQUIRES_RGB

    @property
    def requires_depth(self):
        if self._REQUIRES_DEPTH is None:
            raise NotImplementedError(
                f"_REQUIRES_DEPTH not provided by {type(self).__name__}"
            )
        return self._REQUIRES_DEPTH

    @abstractmethod
    def initialize_models(self):
        """Initialize the models"""
        raise NotImplementedError

    @abstractmethod
    def segment(
        self, color_images: List[np.ndarray], depth_images: List[np.ndarray]
    ) -> np.ndarray:
        """Segment the scene into objects

        Args:
            color_image (np.ndarray): Color image
            depth_image (np.ndarray): Depth image

        Returns:
            np.ndarray: Segmentation mask
        """
        raise NotImplementedError

    def _validate_config(self, config: dict):
        """Check if config keys and their types are valid"""
        return Config._check_types(config, self._config_keys)


class UOIS3DSceneSegmentor(RGBDSceneSegmentor):
    _MODEL = UOISNet3D
    _REQUIRES_DEPTH = True
    _REQUIRES_RGB = True

    __rrn_config_keys = {
        "feature_dim": MultipleOf(4),  # 32 would be normal
        "img_H": PositiveInt,
        "img_W": PositiveInt,
        "use_coordconv": Bool,
    }

    __dsn_config_keys = {
        "feature_dim": EvenInt,
        "max_GMS_iters": int,
        "epsilon": float,
        "sigma": float,
        "num_seeds": int,
        "subsample_factor": int,
        "min_pixels_thresh": int,
        "tau": float,
    }

    _config_keys = {
        "padding_percentage": Percentage,
        "use_open_close_morphology": Bool,
        "open_close_morphology_ksize": OddInt,
        "use_largest_connected_component": Bool,
        "final_close_morphology": Bool,
        "dsn_checkpoint_path": FilePath,
        "rrn_checkpoint_path": FilePath,
        "rrn_config": __rrn_config_keys,
        "dsn_config": __dsn_config_keys,
    }

    _optional_keys = {}

    def __init__(
        self,
        state_manager,
        config: dict,
        device: torch.device,
        input_img_height: int = 480,
        input_img_width: int = 640,
        is_debug=False,
    ):
        """UOISNet scene segmentor

        Args:
            config (dict): Config dictionary. Must contain the following keys:
                    padding_percentage (float): Percentage of padding to add to the image
                    use_open_close_morphology (bool): Whether to use open close morphology
                    open_close_morphology_ksize (int): Kernel size for open close morphology
                    use_largest_connected_component (bool): Whether to use largest connected component
                    final_close_morphology (bool): Whether to use final close morphology
                    dsn_checkpoint_path (str): Path to DSN checkpoint
                    rrn_checkpoint_path (str): Path to RRN checkpoint

            device (torch.device): Device to run the model on
            camera_intrinsic_matrix (np.ndarray): Camera intrinsic matrix K (3x3)
        """
        super().__init__(
            config=config, device=device, state_manager=state_manager, is_debug=is_debug
        )

        self.model = self._initalize_model()

        self.input_img_height = input_img_height
        self.input_img_width = input_img_width

        self.rgb_input_shape = (3, input_img_height, input_img_width)
        self.depth_input_shape = (3, input_img_height, input_img_width)

        self.camera_intrinsic_matrix = config.camera_intrinsic_matrix

    def segment(
        self, color_images: List[np.ndarray], depth_images: List[np.ndarray]
    ) -> np.ndarray:
        """Segment the scene into objects

        Args:
            color_image (np.ndarray): Color image
            depth_image (np.ndarray): Depth image

        Returns:
            np.ndarray: Segmentation mask
        """
        mask = None

        # Check if right modality images are provided
        if self.requires_rgb:
            if color_images is None:
                raise ValueError(
                    f"{type(self).__name__} requires RGB image but none provided"
                )
            # Batch list items to array
            color_images = np.array(color_images)

        if self.requires_depth:
            if depth_images is None:
                raise ValueError(
                    f"{type(self).__name__} requires depth image but none provided"
                )
            # Batch list items to array
            depth_images = np.array(depth_images)

        # Preprocess
        rgb_img, depth_img, metas = self.preprocess(color_images, depth_images)

        # Segment
        masks = self.segment_image(rgb_img, depth_img, metas)

        # Postprocess
        results = self.postprocess(masks, metas)

        return results

    def preprocess(self, color_images: np.ndarray, depth_images: np.ndarray) -> Tuple:
        """Preprocess the input images

        Args:
            color_images (List[np.ndarray]): List of color images
                            shape = (N, H, W, 3) or (H, W, 3)
                            dtype = np.uint8
            depth_images (List[np.ndarray]): List of depth images
                            shape = (N, H, W, 1) or (H, W, 1)
                            dtype = np.uint16
                            depth values are in m
        """

        # Check shape and convert to (N,C,H,W)
        color_images, depth_images = self._conform_to_input_shapes(
            color_images, depth_images
        )

        batch_size = color_images.shape[0]

        # Initialize batch tensors
        batch_rgb = torch.zeros(batch_size, *self.rgb_input_shape)
        batched_depth = torch.zeros(batch_size, *self.depth_input_shape)

        # Metas
        metas = {"K": self.camera_intrinsic_matrix, "scale": "mm"}

        for idx, (rgb, depth) in enumerate(zip(color_images, depth_images)):
            # Standardize RGB
            batch_rgb[idx] = torch.from_numpy(
                data_augmentation.standardize_image(rgb)
            ).float()

            # Convert to structured pointcloud. (N,1,W,H) -> (N,3,W,H)
            # Structured pointcloud is when each pixel in the image is a 3D point
            batched_depth[idx] = torch.from_numpy(
                self.depth_to_structured_pointcloud(
                    depth[0], self.camera_intrinsic_matrix
                )
            ).float()

        return batch_rgb, batched_depth, metas

    def segment_image(
        self,
        color_images: torch.Tensor = None,
        depth_images: torch.Tensor = None,
        metas: dict = None,
    ) -> torch.Tensor:
        batch = {
            "rgb": color_images.to(self.device),
            "xyz": depth_images.to(self.device),
        }

        output = self.model.run_on_batch(batch)
        # fg_masks, center_offsets, initial_masks, seg_masks = output
        seg_masks = output[-1]

        return seg_masks

    def postprocess(self, masks: torch.Tensor) -> Any:
        return masks

    def depth_to_structured_pointcloud(
        self, depth: np.ndarray, K: np.ndarray
    ) -> np.ndarray:
        """Compute ordered point cloud from depth image and camera parameters.
        Assumes camera uses left-handed coordinate system, with
            x-axis pointing right
            y-axis pointing up
            z-axis pointing "forward"

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used

        @return: a [3 x H x W] numpy array
        """

        # Compute focal length from camera parameters
        fx = K[0, 0]
        fy = K[1, 1]
        x_offset = K[0, 2]
        y_offset = K[1, 2]

        indices = np.indices(
            (self.input_img_height, self.input_img_width), dtype=np.float32
        ).transpose(1, 2, 0)
        indices[..., 0] = np.flipud(
            indices[..., 0]
        )  # pixel indices start at top-left corner. for these equations, it starts at bottom-left
        z_e = depth
        x_e = (indices[..., 1] - x_offset) * z_e / fx
        y_e = (indices[..., 0] - y_offset) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]

        # Transpose to (3, H, W)
        return xyz_img.transpose(2, 0, 1)

    def _initalize_model(self):
        """Build the model instance"""
        uois_config = copy.deepcopy(self.config)
        dsn_ckpt_filename = uois_config.pop("dsn_checkpoint_path")
        rrn_ckpt_filename = uois_config.pop("rrn_checkpoint_path")

        rrn_config = uois_config.pop("rrn_config")
        dsn_config = uois_config.pop("dsn_config")

        model = self._MODEL(
            config=uois_config,
            rrn_config=rrn_config,
            dsn_config=dsn_config,
            dsn_filename=dsn_ckpt_filename,
            rrn_filename=rrn_ckpt_filename,
        )

        # model.to(self.device)
        # model.eval()

        return model

    def _conform_to_input_shapes(
        self, color_images: np.ndarray, depth_images: np.ndarray
    ):
        """Validate the input shapes

        Args:
            color_images (np.ndarray): Color images
                            shape = (N, H, W, 3) or (H, W, 3)
                            dtype = np.uint8

            depth_images (np.ndarray): Depth images
                            shape = (N, H, W, 1) or (H, W, 1)
                            dtype = np.uint16

        Raises:
            ValueError: If the input shapes are invalid

        Returns:
            Tuple[np.ndarray, np.ndarray]: Color and depth images
                            color images shape = (N, H, W, 3)
                            depth images shape = (N, H, W, 1)
        """
        # Shape check: Color
        if color_images.ndim == 3:
            color_images = color_images[None, ...]
        elif color_images.ndim == 4:
            pass
        else:
            raise ValueError(
                f"Invalid color image shape. Expected (N, H, W, 3) or (H, W, 3). Got {color_images.shape}"
            )

        bc, hc, wc, cc = color_images.shape
        if hc != self.input_img_height or wc != self.input_img_width:
            raise ValueError(
                f"Invalid color image shape. Expected (N, {self.input_img_height}, {self.input_img_width}, 3). Got {color_images.shape}"
            )

        # Shape check Depth
        if depth_images.ndim == 3:
            depth_images = depth_images[None, ...]
        elif depth_images.ndim == 4:
            pass
        else:
            raise ValueError(
                f"Invalid depth image shape. Expected (N, H, W, 1) or (H, W, 1). Got {depth_images.shape}"
            )

        bd, hd, wd, cd = depth_images.shape
        if hd != self.input_img_height or wd != self.input_img_width:
            raise ValueError(
                f"Invalid depth image shape. Expected (N, {self.input_img_height}, {self.input_img_width}, 1). Got {depth_images.shape}"
            )

        # Batch size check
        if bc != bd:
            raise ValueError(
                f"Batch size not matching. Expected color_images.shape[0] == depth_images.shape[0]. Got {color_images.shape[0]} != {depth_images.shape[0]}"
            )

        # Transpose channels
        color_images = einops.rearrange(color_images, "n h w c -> n c h w")
        depth_images = einops.rearrange(depth_images, "n h w c -> n c h w")

        return color_images, depth_images


class SAMSegmentor(RGBDSceneSegmentor):
    _REQUIRES_DEPTH = False
    _REQUIRES_RGB = True

    _MODEL_REGISTRY = SAM_MODEL_REGISTRY

    __optional_config_keys = {
        "points_per_side": int,
        "points_per_batch": int,
        "pred_iou_thresh": float,
        "stability_score_thresh": float,
        "stability_score_offset": float,
        "box_nms_thresh": float,
        "crop_n_layers": int,
        "crop_nms_thresh": float,
        "crop_overlap_ratio": float,
        "crop_n_points_downscale_factor": int,
        "min_mask_region_area": int,
        "output_mode": str,
    }
    _config_keys = {}

    def __init__(self, config, device, is_debug=True):
        super().__init__(config, device, is_debug=is_debug)

        # Need to be initialized using initialize_models()
        self.model = None
        self.mask_generator = None

    def initialize_models(self):
        self.model = self._MODEL_REGISTRY[self.config.model_type](
            checkpoint=self.config.model_path
        )
        self.model.to(self.device)
        self.model.eval()

        self.mask_generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=32,
            pred_iou_thresh=0.99,
            stability_score_thresh=0.98,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            crop_nms_thresh=0.6,
            crop_overlap_ratio=0.8,
        )
        return

    def segment(
        self, color_images: List[np.ndarray], depth_images: List[np.ndarray] = None
    ) -> np.ndarray:
        for color_image in color_images:
            masks = self.mask_generator.generate(color_image)

        if self.is_debug:
            self.debug_show_masks(masks)

        return masks

    def debug_show_masks(self, masks):
        import matplotlib.pyplot as plt

        grid_length = 5
        grid_width = min(5, len(masks) // grid_length)

        # Assuming you have a list of images called 'images'
        fig, axs = plt.subplots(grid_width, grid_length, figsize=(20, 20))
        for i in range(grid_width):
            for j in range(grid_length):
                axs[i, j].imshow(
                    masks[i * grid_length + j]["segmentation"], cmap="gray"
                )
        plt.savefig("a.png")
        return


class SAMQuerySegmentor(RGBDSceneSegmentor):
    _REQUIRES_DEPTH = False
    _REQUIRES_RGB = True

    _MODEL_REGISTRY = SAM_MODEL_REGISTRY

    _selected_points = []
    _config_keys = {}

    def __init__(
        self,
        config,
        device,
        state_manager,
        is_debug=False,
    ):
        super().__init__(
            config=config,
            device=device,
            state_manager=state_manager,
            is_debug=is_debug,
        )

        self.model = None
        self.predictor = None

    def initialize_models(self):
        self.model = self._MODEL_REGISTRY[self.config.model_type](
            checkpoint=self.config.model_path
        )
        self.model.to(self.device)
        self.model.eval()

        self.predictor = SamPredictor(self.model)

        return

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._selected_points.append((x, y))
            print(f"Clicked: ({x}, {y})")

    def ui_select_point(self, image):
        window_title = "Select Points (Exit: 'q')"
        try:
            cv2.namedWindow(window_title)
        except Exception as e:
            print(
                "Could not open a window. If you are running this in Docker, "
                "This may be due to X11 authorization for host."
                "Try `xhost +` in a host terminal and try again."
            )
            raise e

        # Set callback function for mouse events
        cv2.setMouseCallback(window_title, self.mouse_callback)

        while True:
            # Show image
            cv2.imshow(window_title, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Preview selected points to image
            for point in self._selected_points:
                cv2.circle(image, point, 5, (0, 0, 255), -1)

            key = cv2.waitKey(1) & 0xFF

            # Check if 'q' key is pressed to exit the window
            if key == ord("q"):
                break

        cv2.destroyAllWindows()

    def segment(
        self, color_images: List[np.ndarray], depth_images: List[np.ndarray] = None
    ) -> np.ndarray:
        for color_image in color_images:
            # Clear Query points
            self._selected_points = []

            # Set image input
            self.predictor.set_image(color_image)

            # Use the UI to select query points
            self.ui_select_point(color_image.copy())
            input_point = np.array(self._selected_points)

            # Assign foreground label (1) to all points
            input_label = np.array([1] * len(self._selected_points))

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            # Auto-select the mask with highest score
            mask = masks[scores == max(scores)][0]

            if self.is_debug:
                self.debug_show_masks(color_image, mask)
        # TODO: Postprocess
        return mask

    def debug_show_masks(self, img, mask):
        import os
        import time

        from matplotlib import pyplot as plt

        # mask = mask[..., None].repeat(img.shape[2], axis=-1)

        mask_img = mask.astype(np.uint8) * 255
        mask_rgb = np.zeros_like(img).astype(np.uint8)
        mask_rgb[..., 0] = mask_img

        alpha = 0.6
        # Perform the blending using cv2.addWeighted
        blended_image = cv2.addWeighted(mask_rgb, alpha, img, 1 - alpha, 0)

        save_dir = "img_cache"
        savepath = os.path.join(save_dir, f"a{time.time()}.png")
        os.makedirs(save_dir, exist_ok=True)

        plt.imshow(blended_image)
        plt.savefig(savepath)
        return


class SAMQuerySegmentorONNX(SAMQuerySegmentor):
    def __init__(
        self,
        config,
        device,
        state_manager,
        is_debug=False,
    ):
        import onnxruntime

        super().__init__(
            config=config,
            device=device,
            state_manager=state_manager,
            is_debug=is_debug,
        )

        # ORT
        self.ort_session = onnxruntime.InferenceSession(
            config.onnx_model_path, providers=["CUDAExecutionProvider"]
        )

    def segment(
        self, color_images: List[np.ndarray], depth_images: List[np.ndarray] = None
    ) -> np.ndarray:
        for color_image in color_images:
            # Clear Query points
            self._selected_points = []

            # Set image input
            self.predictor.set_image(color_image)

            # Embed image with the encoder
            img_embedding = self.predictor.get_image_embedding().cpu().numpy()

            # Use the UI to select query points
            self.state_manager.update(ACTION_STATES.AWAITING_USER_INPUT)
            self.ui_select_point(color_image.copy())
            self.state_manager.update(ACTION_STATES.PROCESSING)

            # Prepare selected points input. assign foreground label (1) to all points
            input_point = np.array(self._selected_points)
            input_label = np.array([1] * len(self._selected_points))

            # ONNX input formatting
            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[
                None, :, :
            ]
            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[
                None, :
            ].astype(np.float32)
            onnx_coord = self.predictor.transform.apply_coords(
                onnx_coord, color_image.shape[:2]
            ).astype(np.float32)

            # Create empty mask
            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)

            # ORT inputs
            ort_inputs = {
                "image_embeddings": img_embedding,
                "point_coords": onnx_coord,
                "point_labels": onnx_label,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array(color_image.shape[:2], dtype=np.float32),
            }

            # Inference

            # import time

            # start = time.time()
            masks, scores, low_res_logits = self.ort_session.run(None, ort_inputs)
            masks = masks > self.predictor.model.mask_threshold

            # print(f"Inference time: {time.time() - start}")

            # Auto-select the mask with highest score
            mask = masks[scores == max(scores)][0]

            if self.is_debug:
                self.debug_show_masks(color_image, mask)
        return mask
