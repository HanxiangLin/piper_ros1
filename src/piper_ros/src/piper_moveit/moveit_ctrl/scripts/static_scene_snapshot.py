#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save or restore MoveIt's internal OctoMap as a ROS bag snapshot."""

import argparse
import copy
import os
from pathlib import Path
import re
import sys
import uuid

from piper_static_scene.octomap import describe_octomap, has_octomap, octomap_frame

SNAPSHOT_TOPIC = "/piper_static_scan/octomap_snapshot"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=8.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    save = subparsers.add_parser("save", help="保存当前非空 MoveIt OctoMap，不覆盖旧文件")
    save.add_argument("--output", required=True)
    save.add_argument("--get-service", default="/get_planning_scene")
    inspect = subparsers.add_parser("inspect", help="只读检查当前 MoveIt OctoMap 是否非空")
    inspect.add_argument("--get-service", default="/get_planning_scene")
    load = subparsers.add_parser("load", help="清空当前 OctoMap 后恢复一个快照")
    load.add_argument("--input", required=True)
    load.add_argument("--apply-service", default="/apply_planning_scene")
    load.add_argument("--clear-service", default="/clear_octomap")
    load.add_argument("--expected-frame", default="dummy_link")
    load.add_argument("--replace", action="store_true",
                      help="确认先清空 MoveIt 当前 OctoMap；恢复时必须显式提供")
    raw = sys.argv[1:] if argv is None else argv
    args = parser.parse_args([argument for argument in raw if ":=" not in argument])
    if args.timeout <= 0:
        parser.error("--timeout 必须为正数")
    if args.command == "load":
        if not args.replace:
            parser.error("恢复地图会清空当前 OctoMap，必须显式添加 --replace")
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_/]*", args.expected_frame)
                or args.expected_frame.startswith("/")):
            parser.error("--expected-frame 不是有效 TF frame")
    return args


def world_only_diff(scene):
    """Return an apply-planning-scene diff containing only the saved OctoMap."""
    from moveit_msgs.msg import PlanningScene
    result = PlanningScene()
    result.name = "piper_static_scene_snapshot"
    result.is_diff = True
    result.robot_state.is_diff = True
    result.world.octomap = copy.deepcopy(scene.world.octomap)
    return result


def query_octomap(service_name, timeout):
    import rospy
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest

    rospy.wait_for_service(service_name, timeout=timeout)
    request = GetPlanningSceneRequest()
    request.components.components = PlanningSceneComponents.OCTOMAP
    return rospy.ServiceProxy(service_name, GetPlanningScene)(request).scene


def output_path(value):
    path = Path(value).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("快照输出已存在，请使用新文件名：{}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_snapshot(path):
    import rosbag
    path = Path(path).expanduser().absolute()
    if not path.is_file():
        raise ValueError("快照文件不存在：{}".format(path))
    messages = []
    with rosbag.Bag(str(path), "r") as bag:
        for _topic, message, _stamp in bag.read_messages(topics=[SNAPSHOT_TOPIC]):
            messages.append(message)
    if len(messages) != 1 or getattr(messages[0], "_type", "") != "moveit_msgs/PlanningScene":
        raise ValueError("快照必须恰好包含一个 moveit_msgs/PlanningScene 消息")
    if not has_octomap(messages[0]):
        raise ValueError("快照中的 OctoMap 为空或无效")
    return messages[0]


def save_command(args):
    import rosbag
    import rospy

    destination = output_path(args.output)
    scene = query_octomap(args.get_service, args.timeout)
    if not has_octomap(scene):
        raise ValueError("MoveIt 当前 OctoMap 为空；先完成扫描并停止扫描门")
    frame = octomap_frame(scene)
    temporary = destination.parent / ("." + destination.name + "." + str(uuid.uuid4()) + ".tmp")
    try:
        with rosbag.Bag(str(temporary), "w") as bag:
            bag.write(SNAPSHOT_TOPIC, scene, t=rospy.Time.now())
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    print("已保存静态 OctoMap：{}；frame={}，resolution={:.3f} m，序列化字节={}".format(
        destination, frame, scene.world.octomap.octomap.resolution,
        len(scene.world.octomap.octomap.data)))
    return 0


def inspect_command(args):
    report = describe_octomap(query_octomap(args.get_service, args.timeout))
    if not report["present"]:
        print("MoveIt OctoMap：空。不要开始避障规划；检查扫描门和点云更新器。")
        return 2
    print("MoveIt OctoMap：非空")
    print("  frame: {}".format(report["frame"]))
    print("  tree: {} ({})".format(report["id"], "binary" if report["binary"] else "full"))
    print("  resolution: {:.3f} m".format(report["resolution_m"]))
    print("  serialized bytes: {}（不是体素数量）".format(report["serialized_bytes"]))
    return 0


def load_command(args):
    import rospy
    from moveit_msgs.srv import ApplyPlanningScene
    from std_srvs.srv import Empty

    scene = read_snapshot(args.input)
    frame = octomap_frame(scene)
    if frame != args.expected_frame:
        raise ValueError("快照 frame={}，但期望 {}；拒绝加载".format(frame, args.expected_frame))
    # Validate the complete snapshot before explicitly replacing current state.
    rospy.wait_for_service(args.clear_service, timeout=args.timeout)
    rospy.wait_for_service(args.apply_service, timeout=args.timeout)
    rospy.ServiceProxy(args.clear_service, Empty)()
    response = rospy.ServiceProxy(args.apply_service, ApplyPlanningScene)(world_only_diff(scene))
    if not response.success:
        raise RuntimeError("MoveIt 拒绝应用 OctoMap 快照；当前地图已经清空")
    print("已清空并恢复 MoveIt 静态 OctoMap：{}；扫描门应保持关闭。".format(args.input))
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        # rosbag imports Cryptodome on Noetic. The workspace's Piper Conda
        # interpreter lacks it, while ROS Noetic's system Python provides it.
        # Fail with an actionable message before initializing a ROS node.
        if args.command in ("save", "load"):
            import rosbag  # noqa: F401
        import rospy
        rospy.init_node("piper_static_scene_snapshot", anonymous=True)
        if args.command == "save":
            return save_command(args)
        if args.command == "inspect":
            return inspect_command(args)
        return load_command(args)
    except ModuleNotFoundError as error:
        if error.name == "Cryptodome":
            print("静态地图快照操作失败：当前 Python 缺少 Cryptodome。请用 /usr/bin/python3 "
                  "直接运行 moveit_ctrl/scripts/static_scene_snapshot.py。", file=sys.stderr)
            return 1
        raise
    except (ValueError, RuntimeError, OSError) as error:
        print("静态地图快照操作失败：{}".format(error), file=sys.stderr)
        return 1
    except Exception as error:
        # Keep ROS transport failures concise without importing rospy in an
        # interpreter where its dependency import already failed.
        if error.__class__.__module__.startswith("rospy"):
            print("静态地图快照操作失败：{}".format(error), file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
