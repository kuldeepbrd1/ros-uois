import datetime
import enum
from abc import abstractmethod
from typing import Tuple

import actionlib
import rospy
import sensor_msgs.point_cloud2 as pc2_helpers
from actionlib_msgs.msg import GoalStatus
from sensor_msgs.msg import Image, PointCloud2

from ros_uois.msg import (
    SceneSegmentationAction,
    SceneSegmentationActionFeedback,
    SceneSegmentationActionGoal,
    SceneSegmentationActionResult,
    SceneSegmentationResult,
    SegmentationResult,
)


class ACTION_STATES(enum.Enum):
    """States for the scene processing pipeline

    OFF: The pipeline is not running
    INITIALIZING: The pipeline is initializing
    READY: The pipeline is ready to process a scene
    IMAGE_ACQUISITION: The pipeline is acquiring images from the camera
    AWAITING_USER_INPUT: The pipeline is waiting for user input
    PROCESSING: The pipeline is processing the scene
    COMPLETED: The pipeline has completed processing the scene
    FAILED: The pipeline has failed to process the scene

    NOTE: Add more states as needed here
    """

    IDLE = enum.auto()
    IMAGE_ACQUISITION = enum.auto()
    AWAITING_USER_INPUT = enum.auto()
    PROCESSING = enum.auto()
    COMPLETED = enum.auto()
    FAILED = enum.auto()

    # TODO: Remove. Not necessary
    # goal_status_map = {
    #     OFF: GoalStatus.LOST,
    #     INITIALIZING: GoalStatus.PENDING,
    #     READY: GoalStatus.PENDING,
    #     IMAGE_ACQUISITION: GoalStatus.ACTIVE,
    #     AWAITING_USER_INPUT: GoalStatus.ACTIVE,
    #     PROCESSING: GoalStatus.ACTIVE,
    #     COMPLETED: GoalStatus.SUCCEEDED,
    #     FAILED: GoalStatus.ABORTED,
    # }

    def state_list(self):
        return [state for state in ACTION_STATES]


class ActionStateManager:
    """Makeshift global tracker of states for action feedback"""

    def __init__(self, server: actionlib.SimpleActionServer) -> None:
        super().__init__()
        self._current_state = None
        self._server = server
        self._state_history = []

        # TODO: Add history logging?

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, state: ACTION_STATES):
        assert isinstance(state, ACTION_STATES)
        self._current_state = state
        self._state_history.append((state, datetime.datetime.now()))
        # Add to history
        # self._server.update_feedback(state)

    def update(self, state: ACTION_STATES):
        self.current_state = state


class BaseActionServer:
    _action_type = SceneSegmentationAction
    _feedback_type = SceneSegmentationActionFeedback
    _result_type = SceneSegmentationActionResult

    def __init__(self):
        self.server = actionlib.SimpleActionServer(
            "segmentation_server",
            self._action_type,
            self._execute_request,
            False,
        )
        self.state_manager = ActionStateManager(self)

        # To be overwritten by child class
        self.camera = None
        self.scene_processor = None

        self._num_received_requests = 0

        self.server.start()

    def _execute_request(self, goal: SceneSegmentationActionGoal):
        self._num_received_requests += 1
        rospy.loginfo(
            f"Received request ({self._num_received_requests}): {goal}. Executing ..."
        )

        try:
            results = self.process_scene()

        except Exception as e:
            rospy.logerr(f"Error processing scene: {e}")
            self.update_state(ACTION_STATES.FAILED)
            self.server.set_aborted()
            return

        action_result = self._get_action_result(goal, results)
        # rospy.loginfo(f"Action Result: {action_result}")
        self.server.set_succeeded(action_result)

    def update_state(self, state: ACTION_STATES):
        # Do not publish feedback if the state is IDLE
        # To prevent sending feedback when there is no goal
        # if not state == ACTION_STATES.IDLE:
        #     self.update_feedback(state)
        self.state_manager.update(state)

        # def update_feedback(self, state: ACTION_STATES):
        #     # Construct the custom feedback message
        #     feedback = ProcessingState()
        #     feedback.state_id = state.value
        #     feedback.state_name = state.name

        #     # Publish the feedback
        #     self.server.publish_feedback(
        #         SceneSegmentationActionFeedback(
        #             feedback=feedback,
        #         )
        #     )
        #     return

        # self.server.set_succeeded(result)

    @abstractmethod
    def initialize_pipeline(self, *args, **kwargs) -> None:
        """Initialize the scene processing pipeline

        This is where you should initialize the models and pipeline properties
        """
        raise NotImplementedError

    @abstractmethod
    def process_scene(self, *args, **kwargs) -> Tuple[dict, bool]:
        """Process the scene and return the results"""
        raise NotImplementedError

    def _get_action_result(self, goal, results_dict):
        action_result = SegmentationResult()

        # Nothing is good :'()
        is_good = False

        # Header
        header = rospy.Header()
        header.stamp = rospy.Time.now()

        # Pointcloud
        if goal.goal_request.return_pointcloud:
            is_good = self._check_results_dict(results_dict, "pointcloud")
            if is_good:
                action_result.pointcloud = pc2_helpers.create_cloud_xyz32(
                    header=header, points=results_dict["pointcloud"]
                )

        # Depth
        if goal.goal_request.return_depth:
            is_good = self._check_results_dict(results_dict, "depth_image")
            depth_img = results_dict["depth_image"]
            if is_good:
                action_result.depth_image = Image(
                    header=header,
                    height=depth_img.shape[0],
                    width=depth_img.shape[1],
                    encoding="16UC1",
                    data=depth_img.tobytes(),
                )

        # RGB
        if goal.goal_request.return_rgb:
            is_good = self._check_results_dict(results_dict, "color_image")
            color_img = results_dict["color_image"]
            if is_good:
                action_result.rgb_image = Image(
                    header=header,
                    height=color_img.shape[0],
                    width=color_img.shape[1],
                    encoding="rgb8",
                    data=color_img.tobytes(),
                )

        # Segmentation Mask
        if goal.goal_request.return_mask:
            is_good = self._check_results_dict(results_dict, "mask")
            if is_good:
                action_result.mask = Image(
                    header=header,
                    height=results_dict["mask"].shape[0],
                    width=results_dict["mask"].shape[1],
                    encoding="mono8",
                    data=results_dict["mask"].tobytes(),
                )

        return SceneSegmentationResult(result=action_result)

    def _check_results_dict(self, res_dict, key):
        if key not in res_dict:
            rospy.logerror(
                f"Current Pipeline ({self.scene_processor}) did not return {key}, but was requested."
                f"Verify that the request for {key} is valid for the chosen pipeline ({self.scene_processor})."
                "Or that the pipeline is implemented correctly"
            )
            return False
        return True

    def _check_numpy_array(self, array, dtype, shape_spec=None):
        # Check dtype
        if array.dtype != dtype:
            rospy.logerror(
                f"Expected {dtype} for {array}, but got {array.dtype} instead."
            )
            return False

        # Check shape
        if shape_spec is not None:
            # Check if num dimensions match
            is_in_good_shape = array.ndim == len(shape_spec)

            # Check if each dimension matches. None in shape_spec means any number of elements is ok
            for idx, num_elems in enumerate(shape_spec):
                if num_elems is None:
                    continue
                else:
                    is_in_good_shape = num_elems == array.shape[idx]

            if not is_in_good_shape:
                _str_spec = [str(s) if s is not None else ":" for s in shape_spec]
                rospy.logerror(
                    f"Expected {_str_spec} for {array}, but got {array.shape} instead."
                )
                return False

        return True
