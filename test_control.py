"""
MuJoCo 控制示例（GUI 滑块 + 控制器叠加）
=========================================
使用 launch_passive()，GUI 滑块值自动写入 data.ctrl，
控制器在此基础上做小幅度修改，两者互不冲突。
"""

import mujoco
import mujoco.viewer
import time
import numpy as np

# ======================== XML 模型 =========================
xml = """
<mujoco model="example">
  <default>
    <geom rgba=".8 .6 .4 1"/>
    <joint damping="0.5"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>

    <body pos="0 0 1">
      <joint name="shoulder" type="ball"/>
      <geom type="capsule" size="0.06" fromto="0 0 0  0 0 -.4"/>
      <body pos="0 0 -0.4">
        <joint name="elbow_y" axis="0 1 0"/>
        <joint name="elbow_x" axis="1 0 0"/>
        <geom type="capsule" size="0.04" fromto="0 0 0  .3 0 0"/>
        <body pos=".3 0 0">
          <joint name="wrist_y" axis="0 1 0"/>
          <joint name="wrist_z" axis="0 0 1"/>
          <geom pos=".1 0 0" size="0.1 0.08 0.02" type="ellipsoid"/>
          <site name="end1" pos="0.2 0 0" size="0.01"/>
        </body>
      </body>
    </body>

    <body pos="0.3 0 0.1">
      <joint name="free_obj" type="free"/>
      <geom size="0.07 0.1" type="cylinder" rgba="0 0.6 1 1"/>
      <site name="end2" pos="0 0 0.1" size="0.01"/>
    </body>
  </worldbody>

  <tendon>
    <spatial limited="true" range="0 0.6" width="0.005">
      <site site="end1"/><site site="end2"/>
    </spatial>
  </tendon>

  <actuator>
    <motor name="shoulder_0" joint="shoulder" ctrlrange="-5 5"/>
    <motor name="shoulder_1" joint="shoulder" ctrlrange="-5 5"/>
    <motor name="shoulder_2" joint="shoulder" ctrlrange="-5 5"/>
    <motor name="elbow_y"    joint="elbow_y"    ctrlrange="-3 3"/>
    <motor name="elbow_x"    joint="elbow_x"    ctrlrange="-3 3"/>
    <motor name="wrist_y"    joint="wrist_y"    ctrlrange="-2 2"/>
    <motor name="wrist_z"    joint="wrist_z"    ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

# ======================== 初始化 =========================
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
dt = model.opt.timestep

print(f"dt = {dt*1000:.0f}ms")
for i in range(model.nu):
    print(f"  [{i}] {model.actuator(i).name}")


# ======================== 仿真主循环 =========================
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.perf_counter()

        # GUI 滑块值已自动写入 data.ctrl，这里不需要额外控制

        mujoco.mj_step(model, data)
        viewer.sync()

        # 忙等待对齐物理时间
        while time.perf_counter() - step_start < dt:
            pass