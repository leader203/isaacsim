from curobo.kinematics import Kinematics, KinematicsCfg


URDF_PATH = (
    "/home/stardust/haozhao/pro/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
    "manager_based/manipulation/allstar_toilet/data/robots/chassis_arm/"
    "motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_brush_rolling_0512.urdf"
)

BASE_LINK = "XMS5-R800-W4G3B4C_base"
TOOL_FRAME = "tool_tip_link"


def main():
    cfg = KinematicsCfg.from_basic_urdf(
        urdf_path=URDF_PATH,
        base_link=BASE_LINK,
        tool_frames=[TOOL_FRAME],
    )

    kin = Kinematics(cfg)

    print("base_link:", kin.base_link)
    print("tool_frames:", kin.tool_frames)
    print("joint_names:", kin.joint_names)
    print("dof:", kin.get_dof())

    expected = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    if kin.joint_names != expected:
        raise RuntimeError(
            f"Unexpected joint chain. Expected {expected}, got {kin.joint_names}"
        )

    print("OK: cuRobo sees only the 6-DOF XMS5 arm chain.")


if __name__ == "__main__":
    main()
