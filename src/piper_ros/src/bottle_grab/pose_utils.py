#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure pose/matrix helpers.

No ROS hardware or MoveIt dependency.  Only numpy and ``tf.transformations``
are used, so these functions can be imported from any script or test.
"""

import math

import numpy as np
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped


def normalize_vector(v, eps=1e-9):
    """向量归一化，避免除零。"""
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def build_level_grasp_rotation(reference_R=None):
    """
    构造“只约束夹爪水平”的目标旋转矩阵。

    约束条件：
        夹爪/抓取坐标系 X 轴 = 基坐标系 -Z 方向
        即 grasp_x = [0, 0, -1]

    不再强制：
        grasp_z = base_x 或 grasp_y = base_y

    做法：
        1. 固定 grasp_x 为 [0, 0, -1]，保证夹爪 X 轴与基坐标系 Z 轴反向重合；
        2. 从当前末端姿态 reference_R 中取一个水平面内的参考方向，
           尽量保留当前末端在水平面内的朝向；
        3. 用右手系补全 grasp_y、grasp_z。

    返回：
        R_base_grasp，3x3 旋转矩阵。矩阵三列分别是：
        grasp_x、grasp_y、grasp_z 在基坐标系下的方向。
    """
    # 只强制这一条：夹爪 X 轴朝向基坐标系 -Z。
    grasp_x = np.array([0.0, 0.0, -1.0], dtype=float)

    candidate_refs = []
    if reference_R is not None:
        reference_R = np.array(reference_R, dtype=float)
        # 优先沿用当前末端的 Z 轴作为水平面内朝向参考。
        # 如果当前 Z 轴接近竖直，投影后会退化，再尝试当前 Y/X 轴。
        candidate_refs.extend([
            reference_R[0:3, 2],
            reference_R[0:3, 1],
            reference_R[0:3, 0],
        ])

    # 最后的兜底方向。只有在当前末端几个轴都不适合时才会用到。
    candidate_refs.extend([
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    ])

    grasp_z = None
    for ref in candidate_refs:
        # 去掉 ref 在 grasp_x 方向上的分量，只保留水平面内分量。
        z_candidate = ref - np.dot(ref, grasp_x) * grasp_x
        z_candidate = normalize_vector(z_candidate)
        if z_candidate is not None:
            grasp_z = z_candidate
            break

    if grasp_z is None:
        # 理论上不会走到这里，只是为了安全。
        grasp_z = np.array([1.0, 0.0, 0.0], dtype=float)

    # 右手系要求：grasp_x × grasp_y = grasp_z
    # 因此 grasp_y = grasp_z × grasp_x
    grasp_y = np.cross(grasp_z, grasp_x)
    grasp_y = normalize_vector(grasp_y)

    R_base_grasp = np.column_stack((grasp_x, grasp_y, grasp_z))
    return R_base_grasp


def level_grasp_rotation_for_azimuth(azimuth_rad):
    """水平抓取姿态：X 轴朝下，Z 轴（接近方向）指向给定方位角。

    ``azimuth_rad`` 是 base_link 中从原点指向抓取点的水平方位角。与
    :func:`build_level_grasp_rotation` 不同，这里的接近方向由瓶子方位决定，
    不再继承扫描结束时的末端姿态，因此 ``0°`` 就是正对瓶子的径向接近。
    """
    grasp_x = np.array([0.0, 0.0, -1.0])
    grasp_z = np.array([
        math.cos(float(azimuth_rad)), math.sin(float(azimuth_rad)), 0.0])
    grasp_z /= np.linalg.norm(grasp_z)
    grasp_y = np.cross(grasp_z, grasp_x)
    grasp_y /= np.linalg.norm(grasp_y)
    return np.column_stack((grasp_x, grasp_y, grasp_z))


def validate_transform(matrix, name):
    """Raise ValueError unless ``matrix`` is a valid 4x4 rigid transform."""
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-4)):
        raise ValueError(name + "不是有效的刚体变换")


def pose_matrix(pose):
    """Convert a geometry_msgs/Pose (or PoseStamped.pose) to a 4x4 matrix."""
    matrix = tf_trans.quaternion_matrix([
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w])
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def pose_from_matrix(matrix):
    """Convert a 4x4 matrix to a PoseStamped in base_link."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("目标矩阵无效")
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = matrix[:3, 3]
    quaternion = tf_trans.quaternion_from_matrix(matrix)
    (pose.pose.orientation.x, pose.pose.orientation.y,
     pose.pose.orientation.z, pose.pose.orientation.w) = quaternion
    return pose


def quaternion_from_matrix(matrix):
    """Return (x, y, z, w) quaternion from a 4x4 matrix."""
    return tf_trans.quaternion_from_matrix(matrix)


def quaternion_matrix(quaternion):
    """Return a 4x4 matrix from a quaternion (x, y, z, w)."""
    return tf_trans.quaternion_matrix(quaternion)


def euler_matrix(roll, pitch, yaw, axes="sxyz"):
    """Return a 4x4 matrix from Euler angles."""
    return tf_trans.euler_matrix(roll, pitch, yaw, axes=axes)


# Backwards-compatible aliases used by the previous monolithic script.
_validate_transform = validate_transform
_pose_matrix = pose_matrix
