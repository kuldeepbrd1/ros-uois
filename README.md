# ROS-UOIS: ROS Wrapper and Service for UOIS

## Get Started

```
pip install -e ./segment-anything
pip install -e ./uois

catkin build

roslaunch ros_uois rgbd_realsense_segment_server.launch
```

```
roslaunch realsense2_camera rs_camera.launch enable_depth:=false enable_infra1:=false enable_infra2:=false color_height:=1080 color_width:=1920 color_fps:=6
roslaunch realsense2_camera rs_camera.launch enable_pointcloud:=true
```
