"""
步兵机器人 — Python 仿真主程序
==================================

整体架构:
  Python (ctrl_runner.py)  ←→  C++ (infantry_controller.dll)
    │                                   │
    ├─ 加载 XML 模型                    ├─ 接收传感器数据
    ├─ 管理 mujoco.viewer 渲染          ├─ 接收键盘状态
    ├─ 读传感器 → 传 C++               ├─ 控制算法计算
    ├─ 收控制量 → 写 data.ctrl          └─ 输出控制量
    ├─ 调用 mj_step() 物理仿真
    └─ viewer.sync() 画面同步

键盘方案: pynput 全局监听（独立线程, 不依赖 MuJoCo Viewer 窗口句柄）

使用方法:
  ① 编译 C++ 控制器:
     g++.exe -shared -o infantry_controller.dll infantry_controller.cpp -O2

     参数说明:
       -shared  生成动态链接库 (.dll), 不是 .exe
       -o ...   指定输出文件名
       -O2      编译优化等级（中等优化, 体积与速度平衡）

  ② 运行仿真:
     python ctrl_runner.py

依赖:
  - mujoco (pip install mujoco)
  - numpy
  - pynput (conda install -c conda-forge pynput)
"""

import os
import sys
import time
import ctypes
import numpy as np
from pynput import keyboard

import mujoco
import mujoco.viewer


# ================================================================
#  KeyboardController — 键盘控制器（参照 wheel_leg_mujoco 模式）
# ================================================================
# pynput.keyboard.Listener 在独立线程中用操作系统级 hook 监听全局按键,
# 按下置 1.0, 松开归零, 主循环每帧调用 get_keys() 取当前状态。
# 完全不需要 MuJoCo Viewer 的窗口句柄或 GLFW 回调。
# ================================================================
class KeyboardController:
    def __init__(self):
        # _raw: C++ 控制器期望的 4 键布局 [W, A, S, D]
        self._raw = np.zeros(4, dtype=np.float64)
        # pynput 键盘监听器, 守护线程形式运行
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        """按键按下: 对应键位置 1.0
           pynput 回调接口, 在独立线程中执行"""
        try:
            ch = key.char
            if ch == 'w':      self._raw[0] = 1.0   # 前进
            elif ch == 'a':    self._raw[1] = 1.0   # 左转
            elif ch == 's':    self._raw[2] = 1.0   # 后退
            elif ch == 'd':    self._raw[3] = 1.0   # 右转
        except AttributeError:
            pass  # 特殊键（Shift/Ctrl 等）忽略

    def _on_release(self, key):
        """按键松开: 对应键位归零
           pynput 回调接口, 在独立线程中执行"""
        try:
            ch = key.char
            if ch == 'w':      self._raw[0] = 0.0
            elif ch == 'a':    self._raw[1] = 0.0
            elif ch == 's':    self._raw[2] = 0.0
            elif ch == 'd':    self._raw[3] = 0.0
        except AttributeError:
            pass

    def get_keys(self) -> np.ndarray:
        """返回当前按键状态的副本
           调用方（主循环）每帧调用一次, 线程安全（赋值是原子的）"""
        return self._raw.copy()

    def stop(self):
        """停止键盘监听, 程序退出时调用"""
        if self._listener.is_alive():
            self._listener.stop()


def main():
    """仿真主函数 — 完整流程见下方分段注释"""

    # ============================================================
    #  0. 路径解析 — 确定模型文件和 DLL 的绝对路径
    #     无论从哪个目录执行 python ctrl_runner.py, 都能找到文件
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path   = os.path.join(script_dir, "fudan_infantry_V4",
                              "meshes", "mjmodel_lqr.xml")
    dll_path   = os.path.join(script_dir, "infantry_controller.dll")

    # 文件存在性检查: 少任何一个都无法运行, 提前报错退出
    if not os.path.exists(xml_path):
        print(f"[错误] 找不到模型文件: {xml_path}")
        sys.exit(1)
    if not os.path.exists(dll_path):
        print(f"[错误] 找不到控制器 DLL: {dll_path}")
        print("请先编译 C++ 控制器再运行:")
        print('  g++.exe -shared -o infantry_controller.dll '
              'infantry_controller.cpp -O2')
        sys.exit(1)

    # ============================================================
    #  1. 加载 MuJoCo 模型
    #     from_xml_path() 解析 XML → 构建 mjModel (静态蓝图)
    #     MjData(model)   分配 mjData  (动态快照, 存放位置/速度/力)
    # ============================================================
    print(f"[加载] 模型: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    model.opt.timestep = 0.001  # 仿真步长 (秒), 1ms（决定物理精度）
    dt    = model.opt.timestep
    N_SUBSTEPS = 5              # 每帧物理推进子步数（渲染频率 = 物理频率/N）

    # ============================================================
    #  2. 打印模型接口信息
    #     帮助确认 C++ 代码中 sensor[] 和 ctrl[] 的索引对应关系
    # ============================================================
    print(f"\n===== 模型信息 =====")
    print(f"仿真步长 dt = {dt * 1000:.1f} ms  "
          f"(频率 {1.0 / dt:.0f} Hz)")
    print(f"执行器数量 (nu)        = {model.nu}")
    print(f"传感器总维数 (nsensordata) = {model.nsensordata}")

    print(f"\n执行器列表 (C++ 中 data->ctrl[i] 的 i):")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {name}  ctrlrange={model.actuator_ctrlrange[i]}")

    print(f"\n传感器列表 (C++ 中 sensor[adr] 的 adr):")
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        dim  = model.sensor_dim[i]
        adr  = model.sensor_adr[i]
        print(f"  [{i}] {name}  dim={dim}  adr={adr}")

    # ============================================================
    #  3. 加载 C++ 控制器 DLL
    #     ctypes.CDLL() 把 .dll 映射为 Python 可调用的对象
    #     argtypes 声明 C 函数签名, 确保参数正确传递（类型/数量）
    # ============================================================
    print(f"\n[加载] 控制器 DLL: {dll_path}")
    lib = ctypes.CDLL(dll_path)

    # 声明 C++ 函数:
    #   void compute(const double* sensor, double* ctrl,
    #                int n_sensor, int n_act, const double* keys)
    #
    # ctypes 类型映射:
    #   ctypes.POINTER(c_double) → C 的 double*
    #   ctypes.c_int             → C 的 int
    lib.compute.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # sensor  输入
        ctypes.POINTER(ctypes.c_double),  # ctrl    输出（C++ 写入）
        ctypes.c_int,                      # n_sensor 传感器维度
        ctypes.c_int,                      # n_act    执行器数量
        ctypes.POINTER(ctypes.c_double),  # keys    输入
    ]
    lib.compute.restype = None  # 返回值 void

    # ============================================================
    #  4. 预分配 C++ 交互用的 numpy 数组
    #     预分配 + 复用, 避免每帧创建/销毁数组的开销
    #     ctypes.data_as() 把 numpy 数组指针转成 ctypes 可传的类型
    # ============================================================
    sensor_arr = np.zeros(model.nsensordata, dtype=np.float64)
    ctrl_arr   = np.zeros(model.nu,          dtype=np.float64)

    # ============================================================
    #  5. 启动 pynput 键盘监听 (参照 wheel_leg_mujoco 模式)
    #     pynput 在独立守护线程中监听全局按键
    #     主循环每帧调用 kb.get_keys() 获取当前 WASD 状态
    # ============================================================
    kb = KeyboardController()
    print("[键盘] pynput 全局监听已启动")
    print("        控制: W/S 前进/后退   A/D 左转/右转   "
          "关闭窗口退出")
    print("-" * 50)

    # ============================================================
    #  6. 启动 MuJoCo Viewer 并进入仿真主循环
    #     launch_passive() 返回 Handle → while viewer.is_running() 循环
    #     每步: 读传感器 → C++ 算控制 → 写执行器 → mj_step → 渲染
    # ============================================================
    print("\n[启动] MuJoCo Viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        # 每 200 帧 (≈ 1 秒, dt=0.001, N_SUBSTEPS=5) 打印一次传感器数据
        print_interval = 200

        while viewer.is_running():
            step_start = time.perf_counter()

            # ── 6a. 读传感器: data.sensordata → numpy 数组 ──
            #         sensor_arr 会作为 C++ compute() 的输入
            sensor_arr[:] = data.sensordata

            # ── 6b. 调用 C++ 控制器 ──
            #         Python 传四个参数给 C++:
            #           sensor_arr → 传感器数据
            #           ctrl_arr   → C++ 在此写入控制量
            #           model.nsensordata / model.nu → 数组长度
            #           keys       → WASD 键盘状态
            ctrl_arr.fill(0.0)
            lib.compute(
                sensor_arr.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)),
                ctrl_arr.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)),
                model.nsensordata,
                model.nu,
                kb.get_keys().ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)),
            )

            # ── 6c. 写入 data.ctrl: MuJoCo 执行器接收控制量 ──
            data.ctrl[:] = ctrl_arr

            # ════════════════════════════════════════════════════
            #  6d. 物理推进 × N_SUBSTEPS 次
            #     每个子步都: 读传感器 → C++ 控制 → 写控制量 → mj_step
            #     这样控制频率 = 1/dt = 1000Hz, 渲染频率降低到 1/(dt*N)
            # ════════════════════════════════════════════════════
            for _ in range(N_SUBSTEPS):
                # 读传感器
                sensor_arr[:] = data.sensordata
                # C++ 控制器 (每个子步刷新)
                ctrl_arr.fill(0.0)
                lib.compute(
                    sensor_arr.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)),
                    ctrl_arr.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)),
                    model.nsensordata,
                    model.nu,
                    kb.get_keys().ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)),
                )
                # 写入控制量
                data.ctrl[:] = ctrl_arr
                # 物理推进 (步长 = model.opt.timestep = 0.001s)
                mujoco.mj_step(model, data)

            # ── 6e. 渲染同步 ──
            #         每帧只 sync 一次, 降低渲染开销
            viewer.sync()

            # ── 6f. 调试打印 ──
            #         每 print_interval 步输出关键传感器数据和键盘状态
            step_count += 1
            if step_count % print_interval == 0:
                quat = data.sensordata[0:4]   # 姿态四元数 (w,x,y,z)
                gyro = data.sensordata[4:7]   # 角速度 (gx,gy,gz) rad/s
                lw_v = data.sensordata[18]    # 左轮速度 rad/s
                rw_v = data.sensordata[21]    # 右轮速度 rad/s
                ks   = kb.get_keys()
                print(f"[{step_count:6d}] "
                      f"姿态(wxyz)={quat[0]:6.3f}{quat[1]:7.3f}"
                      f"{quat[2]:7.3f}{quat[3]:7.3f}  "
                      f"角速度={gyro[0]:6.2f}{gyro[1]:6.2f}"
                      f"{gyro[2]:6.2f}  "
                      f"轮速(L/R)={lw_v:7.2f}{rw_v:7.2f}  "
                      f"keys={ks[0]:.0f}{ks[1]:.0f}"
                      f"{ks[2]:.0f}{ks[3]:.0f}")

            # ── 6g. 时间同步 ──
            #         用 perf_counter() 忙等待, 确保每步耗时精确等于 dt
            #         不这样做的话: 仿真跑得比真实时间快/慢, 运动不自然
            while time.perf_counter() - step_start < dt * N_SUBSTEPS:
                pass

    # ============================================================
    #  7. 清理
    #     Viewer 窗口关闭后, 停止键盘监听, 程序退出
    # ============================================================
    print("\n仿真结束。")
    kb.stop()


if __name__ == "__main__":
    main()