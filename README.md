# piper_ws

ROS workspace containing Piper robot control packages and RealSense/YOLO-based
vision experiments.

## Repository layout

- `src/piper_ros`: Piper ROS packages and local manipulation experiments.
- `src/realsense-D455-YOLOV5`: RealSense D455 and YOLOv5 vision code.
- `src/zxgjgh`: Local motion-filter experiment notes.

Generated catkin directories such as `build/`, `devel/`, and `logs/` are kept
locally and excluded from Git.

## Force-controlled grasping

`src/piper_ros/src/bottle_grab` supports an RPPS-200 10x20 tactile sensor on
the inner side of one gripper finger.  The target normal force is set by
`grasp_force_n` (default 8 N) and closed-loop controlled; see
[`FORCE_CONTROL.md`](src/piper_ros/src/bottle_grab/FORCE_CONTROL.md).
The vendor calibration export lives in `calibration/rpps200/`.

## Model files

The local file
`src/realsense-D455-YOLOV5/weights/yolov5x.pt` is not tracked because it is too
large for regular GitHub storage. Restore or download the required model after
cloning the repository.

## Source provenance

The two source trees were consolidated from nested repositories so this
workspace can be cloned as a single repository:

- `src/piper_ros` was based on
  [`agilexrobotics/piper_ros`](https://github.com/agilexrobotics/piper_ros) at
  commit `6cf947c58797775dba148861729a5e87c99564bf`.
- `src/realsense-D455-YOLOV5` was based on
  [`wenyishengkingkong/realsense-D455-YOLOV5`](https://github.com/wenyishengkingkong/realsense-D455-YOLOV5)
  at commit `127fd44d490aae852a6468635f3d5dc693427554`.

The upstream license files remain alongside their respective source trees.
