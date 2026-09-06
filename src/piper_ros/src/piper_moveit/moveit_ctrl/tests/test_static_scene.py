"""Offline tests for gated scanning configuration and snapshot safeguards."""

import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import xml.etree.ElementTree as ET

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))
from piper_static_scene.gate import GateState


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GateStateTests(unittest.TestCase):
    def test_disabled_start_forward_stop(self):
        state = GateState(max_age=.75)
        self.assertFalse(state.accept("camera_depth_optical_frame", 10., 10.1)[0])
        state.start()
        self.assertTrue(state.accept("camera_depth_optical_frame", 11., 11.2)[0])
        self.assertEqual(state.summary()["session_forwarded"], 1)
        state.stop()
        self.assertFalse(state.accept("camera_depth_optical_frame", 12., 12.1)[0])
        self.assertEqual(state.summary()["forwarded"], 1)

    def test_rejects_stale_future_and_invalid_frames(self):
        state = GateState(max_age=.5, future_tolerance=.1)
        state.start()
        for frame, stamp, now in (("", 1., 1.), ("/camera", 1., 1.),
                                  ("bad frame", 1., 1.), ("camera", 0., 1.),
                                  ("camera", 1., 2.), ("camera", 2., 1.5)):
            self.assertFalse(state.accept(frame, stamp, now)[0])
        self.assertEqual(state.summary()["dropped_invalid"], 6)

    def test_bad_settings_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                GateState(max_age=value)


class ConfigurationTests(unittest.TestCase):
    def test_moveit_sensor_configuration_uses_gated_topic(self):
        path = PACKAGE.parent / "piper_with_gripper_moveit" / "config" / "sensors_3d.yaml"
        config = yaml.safe_load(path.read_text())
        self.assertEqual(len(config["sensors"]), 1)
        sensor = config["sensors"][0]
        self.assertEqual(sensor["sensor_plugin"],
                         "occupancy_map_monitor/PointCloudOctomapUpdater")
        self.assertEqual(sensor["point_cloud_topic"], "/piper/static_scan/points")
        self.assertGreaterEqual(sensor["padding_offset"], .02)

    def test_mapping_launch_starts_only_disabled_gate(self):
        root = ET.parse(PACKAGE / "launch" / "static_scene_mapping.launch").getroot()
        nodes = root.findall("node")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].attrib["type"], "static_scan_gate.py")
        self.assertNotIn("eye_in_hand_camera", ET.tostring(root, encoding="unicode"))

    def test_ros_remappings_are_filtered_from_cli(self):
        gate = load_script("static_scan_gate")
        args = gate.parse_args(["--input-cloud", "/points", "__name:=renamed"])
        self.assertEqual(args.input_cloud, "/points")
        control = load_script("static_scan_control")
        self.assertEqual(control.parse_args(["status", "__log:=/tmp/log"]).command, "status")


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.module = load_script("static_scene_snapshot")

    @staticmethod
    def fake_scene(outer="base_link", inner="base_link", data=None):
        octomap = SimpleNamespace(id="OcTree", resolution=.02,
                                  data=[1, 2] if data is None else data,
                                  binary=True, header=SimpleNamespace(frame_id=inner))
        wrapped = SimpleNamespace(header=SimpleNamespace(frame_id=outer), octomap=octomap)
        return SimpleNamespace(world=SimpleNamespace(octomap=wrapped))

    def test_octomap_validation_and_frame(self):
        scene = self.fake_scene()
        self.assertTrue(self.module.has_octomap(scene))
        self.assertEqual(self.module.octomap_frame(scene), "base_link")
        report = self.module.describe_octomap(scene)
        self.assertTrue(report["present"])
        self.assertEqual(report["frame"], "base_link")
        self.assertEqual(report["serialized_bytes"], 2)
        self.assertFalse(self.module.has_octomap(self.fake_scene(data=[])))
        self.assertEqual(self.module.describe_octomap(self.fake_scene(data=[])), {"present": False})
        with self.assertRaisesRegex(ValueError, "不一致"):
            self.module.octomap_frame(self.fake_scene(inner="world"))

    def test_load_requires_explicit_replace(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.module.parse_args(["load", "--input", "map.bag"])
        args = self.module.parse_args(["load", "--input", "map.bag", "--replace"])
        self.assertTrue(args.replace)

    def test_inspect_is_read_only_subcommand(self):
        args = self.module.parse_args(["inspect"])
        self.assertEqual(args.command, "inspect")
        self.assertEqual(args.get_service, "/get_planning_scene")

    def test_world_diff_contains_no_robot_snapshot(self):
        from moveit_msgs.msg import PlanningScene
        scene = PlanningScene()
        scene.world.octomap.header.frame_id = "base_link"
        scene.world.octomap.octomap.header.frame_id = "base_link"
        scene.world.octomap.octomap.id = "OcTree"
        scene.world.octomap.octomap.resolution = .02
        scene.world.octomap.octomap.data = [1, 2]
        result = self.module.world_only_diff(scene)
        self.assertTrue(result.is_diff)
        self.assertTrue(result.robot_state.is_diff)
        self.assertEqual(result.world.octomap.octomap.data, [1, 2])
        self.assertEqual(result.world.collision_objects, [])


if __name__ == "__main__":
    unittest.main()
