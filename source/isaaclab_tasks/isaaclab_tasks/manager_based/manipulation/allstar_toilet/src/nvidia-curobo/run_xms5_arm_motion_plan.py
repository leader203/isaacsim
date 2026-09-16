import torch

from curobo._src.types.robot import RobotCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState


URDF_PATH = (
    "/home/stardust/haozhao/pro/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
    "manager_based/manipulation/allstar_toilet/data/robots/chassis_arm/"
    "motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_brush_rolling_0512.urdf"
)

BASE_LINK = "XMS5-R800-W4G3B4C_base"
TOOL_FRAME = "tool_tip_link"


def make_joint_state(q, joint_names):
    return JointState.from_position(
        torch.tensor([q], device="cuda", dtype=torch.float32),
        joint_names=joint_names,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    robot_cfg = RobotCfg.from_basic(
        urdf_path=URDF_PATH,
        base_link=BASE_LINK,
        tool_frames=[TOOL_FRAME],
    )

    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        self_collision_check=False,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        use_cuda_graph=False,
    )
    planner = MotionPlanner(planner_cfg)
    planner.warmup(enable_graph=False, num_warmup_iterations=3)

    print("base_link:", BASE_LINK)
    print("tool_frames:", planner.tool_frames)
    print("joint_names:", planner.joint_names)

    start = make_joint_state(
        [0.0, -0.6, 0.8, 0.0, 0.4, 0.0],
        planner.joint_names,
    )
    goal = make_joint_state(
        [0.25, -0.9, 1.0, 0.2, 0.2, 0.35],
        planner.joint_names,
    )

    result = planner.plan_cspace(goal, start)
    success = result is not None and bool(result.success.any().item())
    print("planning_success:", success)

    if not success:
        raise RuntimeError("Motion planning failed")

    plan = result.get_interpolated_plan()
    positions = plan.position.squeeze(0)
    print("num_waypoints:", positions.shape[0])
    print("start_waypoint:", positions[0])
    print("end_waypoint:", positions[-1])


if __name__ == "__main__":
    main()
