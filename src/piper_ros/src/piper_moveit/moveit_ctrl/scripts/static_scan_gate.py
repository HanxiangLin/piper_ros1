#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward fresh PointCloud2 messages only while an explicit scan is enabled."""

import argparse
import json
import sys

from piper_static_scene.gate import GateState


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cloud", default="/camera/depth/color/points")
    parser.add_argument("--output-cloud", default="/piper/static_scan/points")
    parser.add_argument("--max-age", type=float, default=0.75)
    raw = sys.argv[1:] if argv is None else argv
    return parser.parse_args([argument for argument in raw if ":=" not in argument])


class ScanGateNode:
    def __init__(self, args):
        import rospy
        from sensor_msgs.msg import PointCloud2
        from std_srvs.srv import Trigger

        self.rospy = rospy
        self.state = GateState(args.max_age)
        self.publisher = rospy.Publisher(args.output_cloud, PointCloud2, queue_size=1)
        self.subscriber = rospy.Subscriber(
            args.input_cloud, PointCloud2, self.on_cloud, queue_size=1,
            buff_size=32 * 1024 * 1024)
        self.start_service = rospy.Service("~start", Trigger, self.on_start)
        self.stop_service = rospy.Service("~stop", Trigger, self.on_stop)
        self.status_service = rospy.Service("~status", Trigger, self.on_status)
        rospy.logwarn("静态扫描门已启动但默认关闭；输入=%s，输出=%s", args.input_cloud,
                      args.output_cloud)

    def on_cloud(self, message):
        accepted, reason = self.state.accept(
            message.header.frame_id, message.header.stamp.to_sec(), self.rospy.Time.now().to_sec())
        if accepted:
            # Preserve the sensor timestamp and frame. MoveIt's updater must use
            # the matching historical TF; this node never substitutes "latest".
            self.publisher.publish(message)
        elif self.state.summary()["enabled"]:
            self.rospy.logwarn_throttle(2.0, "静态扫描点云被拒绝：%s", reason)

    def on_start(self, _request):
        from std_srvs.srv import TriggerResponse
        self.state.start()
        return TriggerResponse(success=True, message=(
            "扫描已开启；不会自动清空旧地图，也不会控制机械臂运动"))

    def on_stop(self, _request):
        from std_srvs.srv import TriggerResponse
        self.state.stop()
        return TriggerResponse(success=True, message=(
            "扫描已停止；不再转发点云，当前 MoveIt OctoMap 保持冻结"))

    def on_status(self, _request):
        from std_srvs.srv import TriggerResponse
        return TriggerResponse(success=True, message=json.dumps(
            self.state.summary(), ensure_ascii=False, sort_keys=True))


def main(argv=None):
    args = parse_args(argv)
    import rospy
    rospy.init_node("piper_static_scan_gate")
    try:
        ScanGateNode(args)
        rospy.spin()
        return 0
    except (ValueError, RuntimeError) as error:
        rospy.logerr("静态扫描门终止：%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
