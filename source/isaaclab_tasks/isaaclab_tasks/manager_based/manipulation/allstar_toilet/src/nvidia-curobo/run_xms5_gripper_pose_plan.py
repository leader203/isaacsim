import argparse

import torch

from curobo._src.types.robot import RobotCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


URDF_PATH = (
    "/home/stardust/haozhao/pro/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
    "manager_based/manipulation/allstar_toilet/data/robots/chassis_arm/"
    "motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_gripper_joint.urdf"
)

BASE_LINK = "XMS5-R800-W4G3B4C_base"
TOOL_FRAME = "gripper_tcp_link"
EXPECTED_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def make_joint_state(q, joint_names):
    return JointState.from_position(
        torch.tensor([q], device="cuda", dtype=torch.float32),
        joint_names=joint_names,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test cuRobo pose planning for the XMS5 gripper TCP."
    )
    parser.add_argument("--urdf", default=URDF_PATH, help="Path to the gripper URDF.")
    parser.add_argument("--base-link", default=BASE_LINK, help="cuRobo kinematic base link.")
    parser.add_argument("--tool-frame", default=TOOL_FRAME, help="Tool frame to plan.")
    parser.add_argument("--num-ik-seeds", type=int, default=64)
    parser.add_argument("--num-trajopt-seeds", type=int, default=8)
    parser.add_argument(
        "--start-q",
        type=float,
        nargs=6,
        default=[0.0, -0.6, 0.8, 0.0, 0.4, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Start arm joint values in radians.",
    )
    parser.add_argument(
        "--target-q",
        type=float,
        nargs=6,
        default=[0.25, -0.9, 1.0, 0.2, 0.2, 0.35],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Joint values used only to generate a reachable target TCP pose.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cuRobo pose planning needs a CUDA device.")

    robot_cfg = RobotCfg.from_basic(
        urdf_path=args.urdf,
        base_link=args.base_link,
        tool_frames=[args.tool_frame],
    )
    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        self_collision_check=False,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=args.num_trajopt_seeds,
        use_cuda_graph=False,
    )
    planner = MotionPlanner(planner_cfg)
    planner.warmup(enable_graph=False, num_warmup_iterations=3)

    kin_cfg = KinematicsCfg.from_basic_urdf(
        urdf_path=args.urdf,
        base_link=args.base_link,
        tool_frames=[args.tool_frame],
    )
    kin = Kinematics(kin_cfg)

    print("urdf_path:", args.urdf)
    print("base_link:", args.base_link)
    print("tool_frames:", planner.tool_frames)
    print("joint_names:", planner.joint_names)
    print("dof:", kin.get_dof())

    if planner.joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(
            f"Unexpected active joint chain. Expected {EXPECTED_JOINT_NAMES}, got {planner.joint_names}"
        )

    start = make_joint_state(args.start_q, planner.joint_names)
    target_q = make_joint_state(args.target_q, planner.joint_names)
    target_fk = kin.compute_kinematics(target_q)
    target_pose = target_fk.tool_poses.get_link_pose(args.tool_frame)

    print("start_q:", args.start_q)
    print("target_q_for_fk:", args.target_q)
    print("target_position_in_base:", target_pose.position)
    print("target_quaternion_wxyz:", target_pose.quaternion)

    goal_pose = GoalToolPose.from_poses(
        {
            args.tool_frame: Pose(
                position=target_pose.position,
                quaternion=target_pose.quaternion,
            )
        },
        num_goalset=1,
    )

    result = planner.plan_pose(goal_pose, start)
    success = result is not None and bool(result.success.any().item())
    print("pose_planning_success:", success)

    if not success:
        raise RuntimeError("Pose motion planning failed.")

    plan = result.get_interpolated_plan()
    positions = plan.position.reshape(-1, kin.get_dof())
    print("trajectory_shape:", tuple(positions.shape))
    print("first_joint_waypoint:", positions[0])
    print("last_joint_waypoint:", positions[-1])

    final_q = positions[-1:].contiguous()
    final_fk = kin.compute_kinematics(
        JointState.from_position(final_q, joint_names=planner.joint_names)
    )
    final_pose = final_fk.tool_poses.get_link_pose(args.tool_frame)
    pos_err = torch.linalg.norm(final_pose.position - target_pose.position, dim=-1)
    quat_dot = torch.abs(torch.sum(final_pose.quaternion * target_pose.quaternion, dim=-1))
    quat_dot = torch.clamp(quat_dot, -1.0, 1.0)
    rot_err = 2.0 * torch.acos(quat_dot)
    print("final_position_in_base:", final_pose.position)
    print("final_quaternion_wxyz:", final_pose.quaternion)
    print("final_position_error_mm:", pos_err * 1000.0)
    print("final_rotation_error_rad:", rot_err)
    print("OK: pose planning reached the gripper TCP target.")


if __name__ == "__main__":
    main()
