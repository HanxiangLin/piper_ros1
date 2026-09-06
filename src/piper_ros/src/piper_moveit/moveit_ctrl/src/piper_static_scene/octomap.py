"""ROS-message-compatible OctoMap inspection helpers."""

import hashlib


def has_octomap(scene):
    octomap = scene.world.octomap.octomap
    return bool(octomap.id and octomap.resolution > 0 and octomap.data)


def octomap_frame(scene):
    outer = scene.world.octomap.header.frame_id
    inner = scene.world.octomap.octomap.header.frame_id
    if outer and inner and outer != inner:
        raise ValueError("OctoMap 内外 header frame 不一致")
    frame = outer or inner
    if not frame:
        raise ValueError("OctoMap 缺少 frame_id")
    return frame.lstrip("/")


def describe_octomap(scene):
    if not has_octomap(scene):
        return {"present": False}
    octomap = scene.world.octomap.octomap
    payload = bytes((value & 0xff for value in octomap.data))
    return {
        "present": True,
        "frame": octomap_frame(scene),
        "id": octomap.id,
        "binary": bool(octomap.binary),
        "resolution_m": float(octomap.resolution),
        "serialized_bytes": len(octomap.data),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def octomap_signature(scene):
    report = describe_octomap(scene)
    if not report["present"]:
        raise ValueError("MoveIt 当前 OctoMap 为空")
    return tuple(report[key] for key in (
        "frame", "id", "binary", "resolution_m", "serialized_bytes", "sha256"))
