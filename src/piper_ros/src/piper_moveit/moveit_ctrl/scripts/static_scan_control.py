#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicitly clear, start, stop or inspect the gated static scan."""

import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("clear", "start", "stop", "status"))
    parser.add_argument("--gate-node", default="/piper_static_scan_gate")
    parser.add_argument("--clear-service", default="/clear_octomap")
    parser.add_argument("--timeout", type=float, default=5.0)
    raw = sys.argv[1:] if argv is None else argv
    args = parser.parse_args([argument for argument in raw if ":=" not in argument])
    if args.timeout <= 0:
        parser.error("--timeout 必须为正数")
    return args


def main(argv=None):
    args = parse_args(argv)
    import rospy
    from std_srvs.srv import Empty, Trigger

    rospy.init_node("piper_static_scan_control", anonymous=True)
    try:
        if args.command == "clear":
            rospy.wait_for_service(args.clear_service, timeout=args.timeout)
            rospy.ServiceProxy(args.clear_service, Empty)()
            print("MoveIt OctoMap 已清空；此操作不启动扫描。")
            return 0
        service_name = args.gate_node.rstrip("/") + "/" + args.command
        rospy.wait_for_service(service_name, timeout=args.timeout)
        response = rospy.ServiceProxy(service_name, Trigger)()
        print(response.message)
        return 0 if response.success else 2
    except (rospy.ROSException, rospy.ServiceException) as error:
        print("静态扫描控制失败：{}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
