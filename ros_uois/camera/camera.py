from abc import abstractmethod, abstractproperty
from typing import Tuple

import cv2
import numpy as np


class BaseCamera:
    """Abstract Base Camera class
    Inherit camera models from here. Virtual or real
    """

    def __init__(self) -> None:
        self.K = None
        self.dists = None
        self.distortion_model = None
        self.fov = None
        self.img_height = None
        self.img_width = None

    @abstractproperty
    def is_model_initialized(self) -> bool:
        """Abstract property: Returns True if camera model is initialized"""
        raise NotImplementedError

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """Abstract method: Returns the current image"""
        raise NotImplementedError

    def project_points_to_image(
        self, points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points to image plane

        Args:
            points (np.ndarray): 3D points
            rvec (np.ndarray): rotation vector
            tvec (np.ndarray): translation vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: (projected points, jacobian)
        """
        if not self.is_model_initialized:
            raise ValueError("Camera model is not initialized")

        projected_points, jac = cv2.projectPoints(
            points, rvec, tvec, self.K, self.dists
        )
        return projected_points, jac
