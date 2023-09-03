from collections import deque
from typing import List, Tuple

import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image

from .camera import BaseCamera


class RealsenseCamera(BaseCamera):
    """Realsense camera model to directly subscribe to camera_info topic in Realsense ROS wrapper"""

    def __init__(
        self,
        image_topic="/camera/color/image_raw",
        camera_info_topic="/camera/color/camera_info",
        response_timeout=10,
    ) -> None:
        """Realsense camera helper class

        Args:
            image_topic (str, optional): topic to get raw image as sensor_msgs.msg.Image. Defaults to "/camera/color/image_raw".
            camera_info_topic (str, optional): topic to get camera info as sensor_msgs.msg.CameraInfo . Defaults to "/camera/color/camera_info".
            response_timeout (int, optional): response_timeout for communication. Defaults to 10.
        """

        super().__init__()

        self.info_topic = camera_info_topic
        self.image_topic = image_topic
        self.response_timeout = response_timeout

        # Intrinsics
        self.K = None
        self.dists = None

        # FOV- (HFOV,VFOV,DFOV)
        self.fov = None

        # Image size
        self.img_height = None
        self.img_width = None
        self.img_encoding = None

        # # Near/Far limits in boresight
        # self.z_near = z_near
        # self.z_far = z_far

        self.distortion_model = None

        self.processed_ids = []

        _buffer_size = 100

        self._rolling_image_buffer = deque(maxlen=_buffer_size)

        # rospy.init_node("_rgbd_camera", anonymous=True)

    @property
    def image_size(self) -> Tuple[int, int, int]:
        """Image size

        Returns:
            Tuple[int, int, int]: height, width, channels
        """
        # if self.img_encoding == "rgb8":
        #     channels = 3
        # elif self.img_encoding == "16UC1":
        #     channels = 1
        # else:
        #     raise NotImplementedError(
        #         f"Image encoding {self.img_encoding} is not implemented. Please add it to the list of supported encodings"
        #     )
        return (
            self.img_height,
            self.img_width,
        )

    @property
    def _last_frame_id(self) -> int:
        """Assign current frame id

        Returns:
            int: id
        """
        return self.processed_ids[-1] if self.processed_ids else 0

    @property
    def _last_buffer_entry(self) -> bytes:
        """Get last buffer entry

        Returns:
            bytes: last buffer entry
        """
        return self._rolling_image_buffer[-1] if self._rolling_image_buffer else None

    @property
    def _image(self) -> np.ndarray:
        """Image: process last buffer and convert to np array (image)

        Returns:
            np.ndarray: deserialized image array
        """
        if self._last_buffer_entry is None:
            return None

        img_data = self._last_buffer_entry.data
        img_height = self._last_buffer_entry.height
        img_width = self._last_buffer_entry.width
        img_encoding = self._last_buffer_entry.encoding

        # Sanity check that intrinsics and image are not mismatched
        assert img_height == self.img_height, "Image height does not match"
        assert img_width == self.img_width, "Image width does not match"
        assert img_encoding == self.img_encoding, "Image encoding does not match"

        # Deserialize image
        image = None
        if img_encoding == "rgb8":
            image = np.frombuffer(img_data, dtype=np.uint8).reshape(
                img_height, img_width, -1
            )
        elif img_encoding == "16UC1":
            image = np.frombuffer(img_data, dtype=np.uint16).reshape(
                img_height, img_width, -1
            )
        else:
            raise NotImplementedError(
                f"Image encoding {img_encoding} is not supported. Please add it to the list of supported encodings"
            )

        return image

    @property
    def is_model_initialized(self) -> bool:
        """Check if intrinsics are initialized

        Returns:
            bool: true if initialized, false if not
        """
        return (
            self.K is not None
            and self.dists is not None
            and self.image_size is not None
            and self.distortion_model is not None
        )

    def get_image(self) -> np.ndarray:
        """public method to get current image

        Returns:
            np.ndarray: image
        """
        return self._image

    def initialize(self) -> None:
        """Initialize the subscriber

        Raises:
            e: if subscriber fails

        """
        # Update intrinsics
        self.update_intrinsics()

        # Start a subscriber for getting image buffers
        try:
            rospy.Subscriber(self.image_topic, Image, self.image_buffer_callback)
            return True

        except Exception as e:
            raise e

    def update_intrinsics(self) -> None:
        """Update intrinsics from camera info topic

        User should call it from their code, if/when image size settings are expected to change

        Raises:
            e: if subscriber fails

        """
        msg = None
        try:
            rospy.loginfo(
                f"RealsenseCamera: Waiting ({self.response_timeout}s) to get the intrinsics from topic:/{self.info_topic}"
            )
            msg = rospy.wait_for_message(
                self.info_topic, CameraInfo, self.response_timeout
            )

        except Exception as e:
            rospy.loginfo(
                f"RealsenseCamera: Timeout({self.response_timeout}s) exceeded. Could not get a message from topic:/{self.info_topic}."
                " \n Action client cannot issue commands!! "
            )
            rospy.logerr(e)

        if msg:
            rospy.loginfo(
                f"RealsenseCamera: Intrinsics are loaded from {self.info_topic}."
            )
            self.K = np.array(list(msg.K)).reshape(3, 3)
            self.dists = msg.D
            self.img_height = msg.height
            self.img_width = msg.width
            self.distortion_model = msg.distortion_model

            # Focal Length in px
            self._fx = self.K[0, 0]
            self._fy = self.K[1, 1]

            # Principal centers
            self._cx = self.K[0, 2]
            self._cy = self.K[1, 2]

            self.fov = self.compute_fov()

        return

    def compute_fov(self) -> Tuple[float, float, float]:
        """Get FOV from camera info

        Returns:
            Tuple[float, float, float]: hfov, vfov, dfov  in radians
        """
        hfov = 2 * np.arctan(self.img_width / (2 * self._fx))
        vfov = 2 * np.arctan(self.img_height / (2 * self._fy))
        dfov = np.linalg.norm(np.array([hfov, vfov]))
        return (hfov, vfov, dfov)

    def image_buffer_callback(self, message) -> None:
        """Image call back for subscriber

        Args:
            message (Image): Message with raw image buffer
        """
        try:
            # Store the buffer and only do the expensive np
            # conversion when needed not on every callback
            img_acquisition_id = self._last_frame_id + 1
            self._rolling_image_buffer.append(message)

            self.processed_ids.append(img_acquisition_id)

        except Exception as e:
            raise e


class RealsenseRGBDSyncCamera(BaseCamera):
    def __init__(
        self,
        color_image_topic="/camera/color/image_raw",
        color_info_topic="/camera/color/camera_info",
        depth_image_topic="/camera/depth/image_rect_raw",
        depth_info_topic="/camera/depth/camera_info",
        response_timeout=5,
    ) -> None:
        # Initialize the camera objects
        self.color_camera = RealsenseCamera(
            color_image_topic, color_info_topic, response_timeout
        )
        self.depth_camera = RealsenseCamera(
            depth_image_topic, depth_info_topic, response_timeout
        )

        # Update intrinsics
        self.color_camera.update_intrinsics()
        self.depth_camera.update_intrinsics()

        assert (
            self.color_camera.is_model_initialized
            and self.depth_camera.is_model_initialized
        ), "Camera intrinsics are not initialized."

        # Expose intrinsics
        self.K = self.depth_camera.K
        self.dists = self.depth_camera.dists

        # Sanity check: Both camera images are of same size
        assert (
            self.color_camera.image_size == self.depth_camera.image_size
        ), f"Color image size {self.color_camera.image_size} and Depth image size {self.depth_camera.image_size} do not match. Make sure Realsense node is initialized with `align_depth:=true`"

        # Image buffer queues
        self._buffer_queue_length = 100
        self.color_image_buffer = deque(maxlen=self._buffer_queue_length)
        self.depth_image_buffer = deque(maxlen=self._buffer_queue_length)

        # Frame ids to track data in queues
        self.frame_ids = deque(maxlen=self._buffer_queue_length)

    @property
    def _last_frame_id(self) -> int:
        return self.frame_ids[-1] if self.frame_ids else None

    def get_buffer_idx(self, frame_id) -> int:
        """Get buffer index for a given frame id

        Args:
            frame_id (int): frame id

        Returns:
            int: buffer index
        """
        return self.frame_ids.index(frame_id)

    def get_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the last synchronized color and depth image

        Returns:
            Tuple[np.ndarray, np.ndarray]: color image, depth image
                            color image: np.ndarray (dtype: uint8) of shape (H,W,3)
                            depth image: np.ndarray (dtype: uint16) of shape (H,W,1)
        """

        # Retrieve data from buffer queue
        buffer_idx = self.get_buffer_idx(self._last_frame_id)
        color_image_data = self.color_image_buffer[buffer_idx]
        depth_image_data = self.depth_image_buffer[buffer_idx]

        # Deserialize Color Image
        assert color_image_data.encoding == "rgb8"
        color_img = np.frombuffer(color_image_data.data, dtype=np.uint8).reshape(
            color_image_data.height, color_image_data.width, -1
        )

        assert depth_image_data.encoding == "16UC1"
        depth_img = np.frombuffer(depth_image_data.data, dtype=np.uint16).reshape(
            depth_image_data.height, depth_image_data.width, -1
        )

        depth_img = depth_img.astype(np.float32) / 1000.0

        return color_img, depth_img

    def initialize(self) -> None:
        color_img_sub = message_filters.Subscriber(self.color_camera.image_topic, Image)
        depth_img_sub = message_filters.Subscriber(self.depth_camera.image_topic, Image)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_img_sub, depth_img_sub], 10, 0.1
        )
        self.sync.registerCallback(self.callback)

        print("Callback warmup: Wait for 10 frames to be received")
        while len(self.frame_ids) < 10:
            continue
        return

    def callback(self, color_image_msg, depth_image_msg):
        try:
            # Store the buffer and only do the expensive np
            # conversion when needed not on every callback
            img_acquisition_id = (
                self._last_frame_id + 1 if self._last_frame_id is not None else 0
            )
            self.color_image_buffer.append(color_image_msg)
            self.depth_image_buffer.append(depth_image_msg)
            self.frame_ids.append(img_acquisition_id)

        except Exception as e:
            raise e
