cmake_minimum_required(VERSION 3.0.2)
project(ros_uois)

# Set the build dependency packages
find_package(catkin REQUIRED COMPONENTS
  actionlib
  rospy
  roscpp
  std_msgs
  actionlib_msgs
  sensor_msgs
  message_generation
)

# Find pkg-config
find_package(PkgConfig)

# Setup python modules of the package: calls setup.py
catkin_python_setup()

## Add message files
add_message_files(
  DIRECTORY msg
  FILES SegmentationRequest.msg SegmentationResult.msg SegmentationState.msg
)

# Add Action files
add_action_files(
  DIRECTORY action
  FILES SceneSegmentation.action
)

# Catkin dependencies
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  actionlib_msgs
)



catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    rospy
    roscpp
    std_msgs
    actionlib_msgs
    message_generation
    message_runtime
)

# Scripts to install
catkin_install_python(PROGRAMS
    scripts/process_rgbd_scene.py
    scripts/rgbd_segmentation_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
