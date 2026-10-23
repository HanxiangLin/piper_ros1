#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RealSense D435i camera wrapper.

Start/stop the pipeline, align depth to color, build the post-processing
filters and expose the depth scale.  The SDK is imported lazily so importing
this module does not require pyrealsense2.
"""

import rospy

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
FPS = 30


def _import_rs():
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError(
            "当前Python环境缺少RealSense依赖: " + str(error)) from error
    return rs


def build_filters(rs_module=None):
    """Build the standard decimation/spatial/temporal/hole-filling chain."""
    rs = rs_module if rs_module is not None else _import_rs()
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 1)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)

    return [decimation, spatial, temporal, hole_filling]


def apply_filters(depth_frame, filters):
    """Run every filter in ``filters`` over ``depth_frame``."""
    for f in filters:
        depth_frame = f.process(depth_frame)
    return depth_frame


class RealSenseCamera:
    """Owns the RealSense pipeline and the depth post-processing chain."""

    def __init__(self, color_width=COLOR_WIDTH, color_height=COLOR_HEIGHT,
                 depth_width=DEPTH_WIDTH, depth_height=DEPTH_HEIGHT, fps=FPS):
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps

        self._rs = None
        self.pipeline = None
        self.align = None
        self.filters = []
        self.depth_scale = None

    def _import_rs(self):
        if self._rs is None:
            self._rs = _import_rs()
        return self._rs

    def start(self):
        """Start the pipeline and return the depth scale (m/count)."""
        rs = self._import_rs()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.depth, self.depth_width, self.depth_height,
            rs.format.z16, self.fps)
        config.enable_stream(
            rs.stream.color, self.color_width, self.color_height,
            rs.format.bgr8, self.fps)
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as error:
            raise RuntimeError(
                "RealSense启动失败；运行本脚本前停止ROS相机launch以释放设备: {}"
                .format(error))
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.filters = build_filters(rs)
        rospy.loginfo("RealSense初始化完成；depth scale=%.6f", self.depth_scale)
        return self.depth_scale

    def frames(self, timeout_ms=5000):
        """Wait for and return the next aligned frame set."""
        if self.pipeline is None:
            raise RuntimeError("RealSense相机尚未启动")
        return self.pipeline.wait_for_frames(timeout_ms=timeout_ms)

    def stop(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None
