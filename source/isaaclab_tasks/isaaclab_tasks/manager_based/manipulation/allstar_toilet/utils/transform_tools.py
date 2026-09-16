import numpy as np
import scipy.spatial.transform as tf
import torch

from isaaclab.utils.math import quat_from_angle_axis


class Rotation:
    _rotation: tf.Rotation

    def __init__(self):
        self._rotation = tf.Rotation.from_matrix(np.eye(3))

    @property
    def rot_list(self):
        return self.get_quat_list()

    def add_euler(self, roll, pitch, yaw, degrees=False):
        self._rotation = self._rotation * tf.Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees)
        return self

    def add_rotvec(self, rotvec, degrees=False):
        self._rotation = self._rotation * tf.Rotation.from_rotvec(rotvec, degrees=degrees)
        return self

    def get_quat_list(self):
        quat = self._rotation.as_quat()
        return tuple(quat[[3, 0, 1, 2]].tolist())


def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (w, x, y, z)."""
    quat = tf.Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat[[3, 0, 1, 2]].tolist())


def combine_transforms(transform1, transform2):
    r1 = tf.Rotation.from_euler("xyz", transform1[3:], degrees=False)
    r2 = tf.Rotation.from_euler("xyz", transform2[3:], degrees=False)

    T1 = np.eye(4)
    T1[:3, 3] = transform1[:3]
    T1[:3, :3] = r1.as_matrix()

    T2 = np.eye(4)
    T2[:3, 3] = transform2[:3]
    T2[:3, :3] = r2.as_matrix()

    T_combined = T1 @ T2
    translation = T_combined[:3, 3]
    rotation_matrix = T_combined[:3, :3]
    rotation = tf.Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)
    return np.concatenate((translation, rotation))


def pos_quat_from_rotpose(pose: torch.Tensor):
    pos = pose[:, :3]
    rot = pose[:, 3:]
    angle = torch.linalg.vector_norm(rot, dim=1)
    axis = rot / angle.unsqueeze(-1)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=pose.device).repeat(pose.shape[0], 1)
    quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > 1e-6, quat_from_angle_axis(angle, axis), identity_quat)
    return pos, quat
