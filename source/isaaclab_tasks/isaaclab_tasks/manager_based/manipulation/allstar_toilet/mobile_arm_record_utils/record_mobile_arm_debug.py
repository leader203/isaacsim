"""Debug-print helpers for mobile-arm recording."""

import contextlib

import numpy as np
import torch


def _debug_values(value, joint_ids=None, decimals: int = 6):
    """Convert tensors/scalars to compact printable values without affecting sim state."""
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if joint_ids is not None:
                ids = [int(i) for i in joint_ids]
                tensor = tensor[ids] if tensor.dim() == 1 else tensor[0, ids]
            elif tensor.dim() > 1:
                tensor = tensor[0]
            values = tensor.flatten().cpu().tolist()
            return [round(float(v), decimals) for v in values]
        if isinstance(value, np.ndarray):
            values = value.reshape(-1).tolist()
            return [round(float(v), decimals) for v in values]
        if isinstance(value, (list, tuple)):
            return [round(float(v), decimals) if isinstance(v, (int, float)) else v for v in value]
        if isinstance(value, (int, float)):
            return round(float(value), decimals)
    except Exception as exc:
        return f"<debug-error: {exc}>"
    return value


def _debug_joint_ids(value):
    if value is None:
        return None
    if isinstance(value, slice):
        return str(value)
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.detach().cpu().flatten().tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return value


def _print_joint_prim_attrs(prefix: str, prim) -> None:
    attrs = {
        "stiff": "drive:angular:physics:stiffness",
        "damp": "drive:angular:physics:damping",
        "max_force": "drive:angular:physics:maxForce",
        "target_pos": "drive:angular:physics:targetPosition",
        "target_vel": "drive:angular:physics:targetVelocity",
        "drive_type": "drive:angular:physics:type",
        "lower": "physics:lowerLimit",
        "upper": "physics:upperLimit",
        "max_vel": "physxJoint:maxJointVelocity",
    }
    values = {}
    for label, attr_name in attrs.items():
        attr = prim.GetAttribute(attr_name)
        values[label] = attr.Get() if attr and attr.IsValid() else "<missing>"
    print(f"{prefix} {prim.GetPath()} {prim.GetTypeName()} {values}", flush=True)


def _dump_usd_file_joint_debug(usd_path: str | None, arm_joint_names: list[str]) -> None:
    if not usd_path:
        print("[ArmDebug][USD file] no usd_path on robot spawn cfg", flush=True)
        return
    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(usd_path)
        if stage is None:
            print(f"[ArmDebug][USD file] failed to open: {usd_path}", flush=True)
            return
        print(f"[ArmDebug][USD file] path={usd_path}", flush=True)
        print(f"[ArmDebug][USD file] defaultPrim={stage.GetDefaultPrim().GetPath()}", flush=True)
        arm_names = set(arm_joint_names)
        for prim in stage.Traverse():
            if prim.GetName() in arm_names:
                _print_joint_prim_attrs("[ArmDebug][USD file joint]", prim)
        for prim in stage.Traverse():
            name = prim.GetName()
            if (
                prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and any(token in name for token in ("XMS5", "link1", "link2", "link3", "link4", "link5", "link6", "tool"))
            ):
                grav_attr = prim.GetAttribute("physxRigidBody:disableGravity")
                mass_attr = prim.GetAttribute("physics:mass")
                gravity = grav_attr.Get() if grav_attr and grav_attr.IsValid() else "<missing>"
                mass = mass_attr.Get() if mass_attr and mass_attr.IsValid() else "<missing>"
                print(f"[ArmDebug][USD file body] {prim.GetPath()} mass={mass} disableGravity={gravity}", flush=True)
    except Exception as exc:
        print(f"[ArmDebug][USD file] skipped: {exc}", flush=True)


def _dump_spawned_stage_joint_debug(base_env, arm_joint_names: list[str]) -> None:
    try:
        from pxr import UsdPhysics

        stage = base_env.sim.stage
        arm_names = set(arm_joint_names)
        print("[ArmDebug][Spawned stage] robot joint/body authored attrs under /robot", flush=True)
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/robot/" in path and prim.GetName() in arm_names:
                _print_joint_prim_attrs("[ArmDebug][Spawned stage joint]", prim)
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            name = prim.GetName()
            if (
                "/robot/" in path
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and any(token in name for token in ("XMS5", "link1", "link2", "link3", "link4", "link5", "link6", "tool"))
            ):
                grav_attr = prim.GetAttribute("physxRigidBody:disableGravity")
                mass_attr = prim.GetAttribute("physics:mass")
                gravity = grav_attr.Get() if grav_attr and grav_attr.IsValid() else "<missing>"
                mass = mass_attr.Get() if mass_attr and mass_attr.IsValid() else "<missing>"
                print(f"[ArmDebug][Spawned stage body] {prim.GetPath()} mass={mass} disableGravity={gravity}", flush=True)
    except Exception as exc:
        print(f"[ArmDebug][Spawned stage] skipped: {exc}", flush=True)


def _dump_mobile_arm_runtime_debug(
    tag: str,
    base_env,
    robot,
    arm_joint_ids,
    arm_joint_names,
    arm_action_term,
    env_cfg=None,
    task_id: str | None = None,
    action: torch.Tensor | None = None,
    include_usd: bool = False,
) -> None:
    ids = [int(i) for i in arm_joint_ids]
    print(f"\n[ArmDebug][{tag}] task={task_id}", flush=True)
    print(f"[ArmDebug][{tag}] robot.num_joints={robot.num_joints}", flush=True)
    print(f"[ArmDebug][{tag}] robot.joint_names={robot.data.joint_names}", flush=True)
    print(f"[ArmDebug][{tag}] robot.body_names={robot.data.body_names}", flush=True)
    print(f"[ArmDebug][{tag}] arm_joint_ids={ids}", flush=True)
    print(f"[ArmDebug][{tag}] arm_joint_names={arm_joint_names}", flush=True)

    spawn_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    spawn_cfg = getattr(spawn_cfg, "spawn", None)
    usd_path = getattr(spawn_cfg, "usd_path", None)
    rigid_props = getattr(spawn_cfg, "rigid_props", None)
    articulation_props = getattr(spawn_cfg, "articulation_props", None)
    print(f"[ArmDebug][{tag}] cfg.usd_path={usd_path}", flush=True)
    print(
        f"[ArmDebug][{tag}] cfg.rigid.disable_gravity={getattr(rigid_props, 'disable_gravity', '<missing>')}",
        flush=True,
    )
    print(
        f"[ArmDebug][{tag}] cfg.articulation.self_collision="
        f"{getattr(articulation_props, 'enabled_self_collisions', '<missing>')}",
        flush=True,
    )

    if hasattr(robot, "actuators"):
        for name, actuator in robot.actuators.items():
            print(
                f"[ArmDebug][{tag}] actuator[{name}] class={type(actuator).__name__} "
                f"joint_names={getattr(actuator, 'joint_names', None)} "
                f"joint_indices={_debug_joint_ids(getattr(actuator, 'joint_indices', None))}",
                flush=True,
            )
            print(
                f"[ArmDebug][{tag}] actuator[{name}] stiffness={_debug_values(getattr(actuator, 'stiffness', None))} "
                f"damping={_debug_values(getattr(actuator, 'damping', None))} "
                f"effort_limit_sim={_debug_values(getattr(actuator, 'effort_limit_sim', None))} "
                f"velocity_limit_sim={_debug_values(getattr(actuator, 'velocity_limit_sim', None))}",
                flush=True,
            )

    data = robot.data
    for label, attr in (
        ("joint_pos", "joint_pos"),
        ("joint_vel", "joint_vel"),
        ("default_joint_pos", "default_joint_pos"),
        ("default_joint_vel", "default_joint_vel"),
        ("joint_pos_target", "joint_pos_target"),
        ("joint_vel_target", "joint_vel_target"),
        ("joint_stiffness", "joint_stiffness"),
        ("joint_damping", "joint_damping"),
        ("default_joint_stiffness", "default_joint_stiffness"),
        ("default_joint_damping", "default_joint_damping"),
        ("joint_effort_limits", "joint_effort_limits"),
        ("joint_vel_limits", "joint_vel_limits"),
        ("soft_joint_vel_limits", "soft_joint_vel_limits"),
    ):
        print(f"[ArmDebug][{tag}] {label}={_debug_values(getattr(data, attr, None), ids)}", flush=True)

    with contextlib.suppress(Exception):
        limits = data.default_joint_pos_limits[0, ids].detach().cpu().tolist()
        limits = [[round(float(v), 6) for v in pair] for pair in limits]
        print(f"[ArmDebug][{tag}] default_joint_pos_limits={limits}", flush=True)

    if action is not None:
        print(f"[ArmDebug][{tag}] raw_action={_debug_values(action)}", flush=True)
        with contextlib.suppress(Exception):
            print(f"[ArmDebug][{tag}] raw_arm_action={_debug_values(action[:, 4:10])}", flush=True)

    if arm_action_term is None:
        print(f"[ArmDebug][{tag}] arm_action_term=None", flush=True)
    else:
        print(
            f"[ArmDebug][{tag}] arm_action_term={type(arm_action_term).__name__} "
            f"joint_ids={_debug_joint_ids(getattr(arm_action_term, '_joint_ids', None))} "
            f"joint_names={getattr(arm_action_term, '_joint_names', None)}",
            flush=True,
        )
        print(
            f"[ArmDebug][{tag}] arm_action_term.offset={_debug_values(getattr(arm_action_term, '_offset', None))} "
            f"scale={_debug_values(getattr(arm_action_term, '_scale', None))}",
            flush=True,
        )
        print(
            f"[ArmDebug][{tag}] arm_action_term.raw_actions={_debug_values(getattr(arm_action_term, '_raw_actions', None))} "
            f"processed_actions={_debug_values(getattr(arm_action_term, '_processed_actions', None))}",
            flush=True,
        )

    with contextlib.suppress(Exception):
        root = data.root_link_pose_w[0].detach().cpu().tolist()
        print(f"[ArmDebug][{tag}] root_link_pose_w={[round(float(v), 6) for v in root]}", flush=True)
    for body_name in (
        "base_link",
        "XMS5_R800_W4G3B4C_base",
        "XMS5_R800_W4G3B4C_link1",
        "XMS5_R800_W4G3B4C_link2",
        "XMS5_R800_W4G3B4C_link3",
        "XMS5_R800_W4G3B4C_link4",
        "XMS5_R800_W4G3B4C_link5",
        "link6",
        "tool_link",
        "tool_tip_link",
    ):
        with contextlib.suppress(Exception):
            body_idx = data.body_names.index(body_name)
            pos = data.body_pos_w[0, body_idx].detach().cpu().tolist()
            print(f"[ArmDebug][{tag}] body_pos_w[{body_name}]={[round(float(v), 6) for v in pos]}", flush=True)
        with contextlib.suppress(Exception):
            body_idx = data.body_names.index(body_name)
            quat = data.body_quat_w[0, body_idx].detach().cpu().tolist()
            print(f"[ArmDebug][{tag}] body_quat_w[{body_name}]={[round(float(v), 6) for v in quat]}", flush=True)
        with contextlib.suppress(Exception):
            body_idx = data.body_names.index(body_name)
            vel = data.body_vel_w[0, body_idx].detach().cpu().tolist()
            print(f"[ArmDebug][{tag}] body_vel_w[{body_name}]={[round(float(v), 6) for v in vel]}", flush=True)

    if include_usd:
        _dump_usd_file_joint_debug(usd_path, arm_joint_names)
        _dump_spawned_stage_joint_debug(base_env, arm_joint_names)


def _print_arm_joint_step_debug(step: int, robot, arm_joint_ids, arm_joint_names, arm_action_term=None) -> None:
    ids = [int(i) for i in arm_joint_ids]
    data = robot.data

    def _values(attr_name: str):
        return _debug_values(getattr(data, attr_name, None), ids)

    q = data.joint_pos[:, ids]
    q_target = getattr(data, "joint_pos_target", None)
    q_target_arm = q_target[:, ids] if q_target is not None else None
    q_err = q_target_arm - q if q_target_arm is not None else None
    q_target_deg = torch.rad2deg(q_target_arm) if q_target_arm is not None else None
    q_err_deg = torch.rad2deg(q_err) if q_err is not None else None

    print(f"\n[ArmJointDebug][step={step}] names={arm_joint_names}", flush=True)
    print(
        f"[ArmJointDebug][step={step}] "
        f"q_deg={_debug_values(torch.rad2deg(q))} "
        f"qd={_values('joint_vel')} "
        f"target_deg={_debug_values(q_target_deg)} "
        f"err_deg={_debug_values(q_err_deg)}",
        flush=True,
    )
    print(
        f"[ArmJointDebug][step={step}] "
        f"stiffness={_values('joint_stiffness')} "
        f"damping={_values('joint_damping')} "
        f"effort_limit={_values('joint_effort_limits')} "
        f"vel_limit={_values('joint_vel_limits')}",
        flush=True,
    )
    if arm_action_term is not None:
        print(
            f"[ArmJointDebug][step={step}] "
            f"raw_action={_debug_values(getattr(arm_action_term, '_raw_actions', None))} "
            f"processed_action={_debug_values(getattr(arm_action_term, '_processed_actions', None))} "
            f"offset={_debug_values(getattr(arm_action_term, '_offset', None))} "
            f"scale={_debug_values(getattr(arm_action_term, '_scale', None))}",
            flush=True,
        )


def print_mobile_arm_pose_line(
    *,
    base_pos,
    base_rpy_deg: tuple[float, float, float],
    tool_tip_pos,
    tool_tip_rpy_deg: tuple[float, float, float],
    tool_tip_quat,
    yellow_frame_origin,
    link6_to_ik_ee_local,
    arm_joint_pos,
    quiet_runtime_debug: bool,
    print_link6_to_ik_ee_local: bool,
) -> None:
    pose_line = (
        "\r[Pose] "
        f"base_link_pos_w=[{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]  "
        f"base_link_rpy_deg_w=[{base_rpy_deg[0]:.1f}, {base_rpy_deg[1]:.1f}, {base_rpy_deg[2]:.1f}]  "
        f"tool_tip_link_pos_w=[{tool_tip_pos[0]:.3f}, {tool_tip_pos[1]:.3f}, {tool_tip_pos[2]:.3f}]  "
        f"tool_tip_link_rpy_deg_w=[{tool_tip_rpy_deg[0]:.1f}, {tool_tip_rpy_deg[1]:.1f}, {tool_tip_rpy_deg[2]:.1f}]  "
        f"tool_tip_link_quat_wxyz=[{tool_tip_quat[0]:.3f}, {tool_tip_quat[1]:.3f}, "
        f"{tool_tip_quat[2]:.3f}, {tool_tip_quat[3]:.3f}]"
    )
    if not quiet_runtime_debug:
        pose_line += (
            f"  yellow_frame_origin_w=[{yellow_frame_origin[0]:.3f}, "
            f"{yellow_frame_origin[1]:.3f}, {yellow_frame_origin[2]:.3f}]"
        )
        if print_link6_to_ik_ee_local:
            pose_line += (
                f"  link6_to_ik_ee_local=[{link6_to_ik_ee_local[0]:.3f}, "
                f"{link6_to_ik_ee_local[1]:.3f}, {link6_to_ik_ee_local[2]:.3f}]"
            )
        pose_line += (
            f"  arm_q_deg=[{np.degrees(arm_joint_pos[0]):.1f}, {np.degrees(arm_joint_pos[1]):.1f}, "
            f"{np.degrees(arm_joint_pos[2]):.1f}, {np.degrees(arm_joint_pos[3]):.1f}, "
            f"{np.degrees(arm_joint_pos[4]):.1f}, {np.degrees(arm_joint_pos[5]):.1f}]"
        )
    print(pose_line, end="", flush=True)


def print_mobile_arm_step_pose_line(
    *,
    step: int,
    base_pos_w,
    tool_pos_w,
    yellow_frame_origin_w,
    tool_tip_pos_w,
    link6_to_ik_ee_local_w,
    print_link6_to_ik_ee_local: bool,
    show_camera_pose_debug: bool,
    sensor_io_enabled: bool,
    cam,
    cam_pos_w,
) -> None:
    base_pos = base_pos_w[0].cpu().tolist()
    tool_pos = tool_pos_w[0].cpu().tolist()
    yellow_origin = yellow_frame_origin_w[0].cpu().tolist()
    tool_tip_pos = tool_tip_pos_w[0].cpu().tolist()
    link6_to_ik_ee = link6_to_ik_ee_local_w[0].cpu().tolist()
    step_line = (
        f"\n[Step {step}] base_link_w={[round(v, 3) for v in base_pos]}"
        f"  tool_link_w={[round(v, 3) for v in tool_pos]}"
        f"  yellow_frame_origin_w={[round(v, 3) for v in yellow_origin]}"
        f"  tool_tip_link_w={[round(v, 3) for v in tool_tip_pos]}"
    )
    if print_link6_to_ik_ee_local:
        step_line += f"  link6_to_ik_ee_local={[round(v, 3) for v in link6_to_ik_ee]}"
    if show_camera_pose_debug and sensor_io_enabled and cam is not None and cam_pos_w is not None:
        cam_target = cam_pos_w[0].cpu().tolist()
        cam_data = cam.data.pos_w[0].cpu().tolist()
        step_line += (
            f"  cam_target={[round(v, 3) for v in cam_target]}"
            f"  cam.data={[round(v, 3) for v in cam_data]}"
        )
    print(step_line)


def print_mobile_arm_control_status(
    *,
    step: int,
    is_chassis_mode: bool,
    wheel_vel,
    delta,
    arm_offsets,
    strict_mode_split: bool,
    use_prismatic_gripper_keyboard: bool,
    gripper_joint1_pos: float,
    teleop_device: str,
    teleop,
) -> None:
    mode_label = "底盘" if is_chassis_mode else "机械臂"
    if is_chassis_mode:
        info = f"wheel={wheel_vel[0].tolist()}"
    else:
        if strict_mode_split:
            info = f"arm_delta={[round(v, 3) for v in delta[0].tolist()]}"
        else:
            info = f"arm_offset={[round(v, 3) for v in arm_offsets[0].tolist()]}"
        if use_prismatic_gripper_keyboard:
            info += f" gripper={gripper_joint1_pos:.4f}m"
    if teleop_device == "keyboard":
        held = [key for key in "WASDZXQETGCVIO MRP".replace(" ", "") if teleop.is_held(key)]
    else:
        held = []
    print(
        f"\r[Step {step:5d}] 模式={mode_label}  "
        f"按键={held if held else '无'}  {info}   ",
        end="", flush=True,
    )
