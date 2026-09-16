import argparse

import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState


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


def print_tool_pose(label, kin, q, base_link, tool_frame):
    state = kin.compute_kinematics(make_joint_state(q, kin.joint_names))
    pose = state.tool_poses.get_link_pose(tool_frame)
    print(f"{label}_q:", q)
    print(f"{label}_position_in_{base_link}:", pose.position)
    print(f"{label}_quaternion_wxyz:", pose.quaternion)


def main():
    parser = argparse.ArgumentParser(
        description="Test cuRobo forward kinematics for the XMS5 gripper URDF."
    )
    parser.add_argument("--urdf", default=URDF_PATH, help="Path to the gripper URDF.")
    parser.add_argument("--base-link", default=BASE_LINK, help="cuRobo kinematic base link.")
    parser.add_argument("--tool-frame", default=TOOL_FRAME, help="Tool frame to report.")
    parser.add_argument(
        "--sample-q",
        type=float,
        nargs=6,
        default=[0.0, -0.6, 0.8, 0.0, 0.4, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Six arm joint values in radians.",
    )
    args = parser.parse_args()

    base_link = args.base_link
    tool_frame = args.tool_frame

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cuRobo FK needs a CUDA device.")

    cfg = KinematicsCfg.from_basic_urdf(
        urdf_path=args.urdf,
        base_link=base_link,
        tool_frames=[tool_frame],
    )
    kin = Kinematics(cfg)

    print("urdf_path:", args.urdf)
    print("base_link:", kin.base_link)
    print("tool_frames:", kin.tool_frames)
    print("joint_names:", kin.joint_names)
    print("dof:", kin.get_dof())

    if kin.joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(
            f"Unexpected active joint chain. Expected {EXPECTED_JOINT_NAMES}, got {kin.joint_names}"
        )

    print_tool_pose("zero", kin, [0.0] * kin.get_dof(), base_link, tool_frame)
    print_tool_pose("sample", kin, args.sample_q, base_link, tool_frame)

    q_batch = torch.tensor(
        [[0.0] * kin.get_dof(), args.sample_q],
        device="cuda",
        dtype=torch.float32,
    )
    batch_state = kin.compute_kinematics(
        JointState.from_position(q_batch, joint_names=kin.joint_names)
    )
    batch_pose = batch_state.tool_poses.get_link_pose(tool_frame)
    print("batch_positions_shape:", tuple(batch_pose.position.shape))
    print("batch_quaternions_shape:", tuple(batch_pose.quaternion.shape))
    print("OK: cuRobo FK sees only the 6-DOF XMS5 arm chain with the gripper tool frame.")


if __name__ == "__main__":
    main()
