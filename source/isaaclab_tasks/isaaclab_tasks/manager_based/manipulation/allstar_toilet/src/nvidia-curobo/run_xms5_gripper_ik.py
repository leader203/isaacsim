import argparse

import torch

from curobo._src.types.robot import RobotCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose


URDF_PATH = (
    "/home/stardust/haozhao/pro/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
    "manager_based/manipulation/allstar_toilet/data/robots/chassis_arm/"
    "motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_gripper_joint.urdf"
)

BASE_LINK = "XMS5-R800-W4G3B4C_base"
TOOL_FRAME = "tool_tip_link"
EXPECTED_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def make_joint_state(q, joint_names):
    return JointState.from_position(
        torch.tensor([q], device="cuda", dtype=torch.float32),
        joint_names=joint_names,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test cuRobo inverse kinematics for the XMS5 gripper URDF."
    )
    parser.add_argument("--urdf", default=URDF_PATH, help="Path to the gripper URDF.")
    parser.add_argument("--base-link", default=BASE_LINK, help="cuRobo kinematic base link.")
    parser.add_argument("--tool-frame", default=TOOL_FRAME, help="Tool frame to solve.")
    parser.add_argument("--num-seeds", type=int, default=64, help="Number of IK seeds.")
    parser.add_argument(
        "--sample-q",
        type=float,
        nargs=6,
        default=[0.0, -0.6, 0.8, 0.0, 0.4, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Six arm joint values in radians used to generate a reachable FK target.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cuRobo IK needs a CUDA device.")

    kin_cfg = KinematicsCfg.from_basic_urdf(
        urdf_path=args.urdf,
        base_link=args.base_link,
        tool_frames=[args.tool_frame],
    )
    kin = Kinematics(kin_cfg)

    print("urdf_path:", args.urdf)
    print("base_link:", kin.base_link)
    print("tool_frames:", kin.tool_frames)
    print("joint_names:", kin.joint_names)
    print("dof:", kin.get_dof())

    if kin.joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(
            f"Unexpected active joint chain. Expected {EXPECTED_JOINT_NAMES}, got {kin.joint_names}"
        )

    target_q = make_joint_state(args.sample_q, kin.joint_names)
    target_fk = kin.compute_kinematics(target_q)
    target_pose = target_fk.tool_poses.get_link_pose(args.tool_frame)

    print("target_q:", args.sample_q)
    print("target_position_in_base:", target_pose.position)
    print("target_quaternion_wxyz:", target_pose.quaternion)

    robot_cfg = RobotCfg.from_basic(
        urdf_path=args.urdf,
        base_link=args.base_link,
        tool_frames=[args.tool_frame],
    )
    ik_cfg = InverseKinematicsCfg.create(
        robot=robot_cfg,
        num_seeds=args.num_seeds,
    )
    ik = InverseKinematics(ik_cfg)

    result = ik.solve_pose(
        GoalToolPose.from_poses(
            {
                args.tool_frame: Pose(
                    position=target_pose.position,
                    quaternion=target_pose.quaternion,
                )
            },
            num_goalset=1,
        )
    )

    success = bool(result.success.any().item())
    print("ik_success:", success)
    print("ik_joint_names:", ik.joint_names)
    print("ik_solution:", result.js_solution.position)
    print("ik_position_error_mm:", result.position_error * 1000.0)
    print("ik_rotation_error_rad:", result.rotation_error)

    if not success:
        raise RuntimeError("IK failed to solve the FK-generated target pose.")

    solution_q = result.js_solution.position.view(-1, kin.get_dof())[0:1]
    solution_fk = kin.compute_kinematics(
        JointState.from_position(solution_q, joint_names=kin.joint_names)
    )
    solution_pose = solution_fk.tool_poses.get_link_pose(args.tool_frame)

    fk_position_error = torch.linalg.norm(solution_pose.position - target_pose.position, dim=-1)
    quat_dot = torch.abs(torch.sum(solution_pose.quaternion * target_pose.quaternion, dim=-1))
    quat_dot = torch.clamp(quat_dot, -1.0, 1.0)
    fk_rotation_error = 2.0 * torch.acos(quat_dot)

    print("solution_fk_position:", solution_pose.position)
    print("solution_fk_quaternion_wxyz:", solution_pose.quaternion)
    print("fk_recheck_position_error_mm:", fk_position_error * 1000.0)
    print("fk_recheck_rotation_error_rad:", fk_rotation_error)
    print("OK: IK solved a reachable FK target for the 6-DOF XMS5 gripper chain.")


if __name__ == "__main__":
    main()
