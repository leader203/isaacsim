import os

import torch

from curobo._src.types.robot import RobotCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose


URDF_PATH = (
    "/home/stardust/haozhao/pro/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
    "manager_based/manipulation/allstar_toilet/data/robots/chassis_arm/"
    "motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_brush_rolling_0512.urdf"
)

BASE_LINK = "XMS5-R800-W4G3B4C_base"
TOOL_FRAME = "tool_tip_link"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    kin_cfg = KinematicsCfg.from_basic_urdf(
        urdf_path=URDF_PATH,
        base_link=BASE_LINK,
        tool_frames=[TOOL_FRAME],
    )
    kin = Kinematics(kin_cfg)

    q_seed = torch.tensor(
        [[0.0, -0.6, 0.8, 0.0, 0.4, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    fk_state = kin.compute_kinematics(
        JointState.from_position(q_seed, joint_names=kin.joint_names)
    )
    target_pose = fk_state.tool_poses.get_link_pose(TOOL_FRAME)

    print("seed_joint_names:", kin.joint_names)
    print("seed_q:", q_seed)
    print("target_position:", target_pose.position)
    print("target_quaternion:", target_pose.quaternion)

    robot_cfg = RobotCfg.from_basic(
        urdf_path=URDF_PATH,
        base_link=BASE_LINK,
        tool_frames=[TOOL_FRAME],
    )

    ik_cfg = InverseKinematicsCfg.create(
        robot=robot_cfg,
        num_seeds=32,
    )
    ik = InverseKinematics(ik_cfg)

    result = ik.solve_pose(
        GoalToolPose.from_poses({TOOL_FRAME: Pose(
            position=target_pose.position,
            quaternion=target_pose.quaternion,
        )}, num_goalset=1)
    )

    print("ik_success:", bool(result.success.item()))
    print("ik_joint_names:", ik.joint_names)
    print("ik_solution:", result.js_solution.position)
    print("ik_position_error_mm:", result.position_error.item() * 1000.0)


if __name__ == "__main__":
    main()
