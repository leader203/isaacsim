项目路径：

`/home/stardust/haozhao/pro/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet`

当前保留的 cuRobo 末端位姿规划测试入口：

```bash
cd /home/stardust/haozhao/pro/IsaacLab
TERM=xterm ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_curobo_pose_plan_test.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0 \
  --initial_robot_id brush_rolling_0512
```

旧的 `plan_arm_to_stain*` / `plan_arm_path_sweep` 规划链路已清理。
