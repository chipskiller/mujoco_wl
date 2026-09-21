/**
 * 步兵机器人控制器 — C++ 动态库 (.dll)
 * =========================================
 *
 * 编译命令（在 mujoco_env 环境中执行）：
 *
 *   x86_64-w64-mingw32-g++ -shared -o infantry_controller.dll infantry_controller.cpp -O2
 *
 * 参数说明：
 *   -shared         生成动态链接库 (.dll) 而非可执行文件 (.exe)
 *   -o ...dll       指定输出文件名
 *   -O2             编译优化（发布性能）
 *
 * 这个 .dll 只做一件事：根据传感器数据算出控制量。
 * 不依赖 MuJoCo、不碰渲染、不创建窗口——可以原样移植到实车代码。
 */

#include <cmath>
#include <cstdio>
#include <cstring>

// ================================================================
//  传感器数据布局（与 mjmodel_lqr.xml 中 <sensor> 定义顺序一致）
//  索引 0-21，共 22 个 double
// ================================================================
//
//  [ 0.. 3] orientation     — IMU 姿态四元数 (w, x, y, z)
//  [ 4.. 6] base_ang_vel   — IMU 角速度 (rad/s)  (gx, gy, gz)
//  [ 7.. 9] base_lin_acc   — IMU 线加速度 (m/s²) (ax, ay, az)
//  [10]     lf0_Joint_p    — 左前悬挂上关节 位置 (rad)
//  [11]     lf1_Joint_p    — 左前悬挂下关节 位置 (rad)
//  [12]     l_wheel_Joint_p— 左轮 位置 (rad)
//  [13]     rf0_Joint_p    — 右前悬挂上关节 位置 (rad)
//  [14]     rf1_Joint_p    — 右前悬挂下关节 位置 (rad)
//  [15]     r_wheel_Joint_p— 右轮 位置 (rad)
//  [16]     lf0_Joint_v    — 左前悬挂上关节 速度 (rad/s)
//  [17]     lf1_Joint_v    — 左前悬挂下关节 速度 (rad/s)
//  [18]     l_wheel_Joint_v— 左轮 速度 (rad/s)
//  [19]     rf0_Joint_v    — 右前悬挂上关节 速度 (rad/s)
//  [20]     rf1_Joint_v    — 右前悬挂下关节 速度 (rad/s)
//  [21]     r_wheel_Joint_v— 右轮 速度 (rad/s)

// ================================================================
//  控制量布局（与 mjmodel_lqr.xml 中 <actuator> 定义顺序一致）
//  索引 0-9，共 10 个 double（单位：Nm 力矩）
// ================================================================
//
//  [0] act_lf0         — 左前悬挂上电机   ctrlrange -100..100
//  [1] act_lf1         — 左前悬挂下电机   ctrlrange -100..100
//  [2] act_lw          — 左轮驱动电机     ctrlrange -500..500
//  [3] act_rf0         — 右前悬挂上电机   ctrlrange -100..100
//  [4] act_rf1         — 右前悬挂下电机   ctrlrange -100..100
//  [5] act_rw          — 右轮驱动电机     ctrlrange -500..500
//  [6] act_lf20        — 左后悬挂电机     ctrlrange -100..100
//  [7] act_rf20        — 右后悬挂电机     ctrlrange -100..100
//  [8] Left_loop1_motor — 左侧弹簧/减震器  ctrlrange 0..350
//  [9] Right_loop1_motor— 右侧弹簧/减震器  ctrlrange 0..350

// ================================================================
//  可调参数（修改后重新编译 .dll 即可生效）
// ================================================================
static const double MAX_WHEEL_TORQUE = 200.0;   // 轮子最大力矩 (Nm)
static const double SPRING_FORCE     = 150.0;   // 减震器预载力 (N)

// 悬挂角度参考值（使车身保持水平）
// 这些值需要根据实际仿真/实车标定
static const double SUSPENSION_REF_LF0 = 0.0;   // 左前悬挂上参考角度
static const double SUSPENSION_REF_LF1 = 0.0;   // 左前悬挂下参考角度
static const double SUSPENSION_REF_RF0 = 0.0;   // 右前悬挂上参考角度
static const double SUSPENSION_REF_RF1 = 0.0;   // 右前悬挂下参考角度

// PD 控制增益
static const double KP_SUSPENSION = 30.0;       // 悬挂位置环比例增益
static const double KD_SUSPENSION = 5.0;        // 悬挂位置环微分增益

// ================================================================
//  clamp: 将值限制在 [lo, hi] 范围内
// ================================================================
static inline double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// ================================================================
//  compute — 主控制函数
//
//  参数:
//    sensor   输入数组，长度 n_sensor（当前 = 22）
//    ctrl     输出数组，长度 n_act（当前 = 10），调用前已清零
//    n_sensor 传感器维度数
//    n_act    执行器数量
//    keys     键盘状态数组（由 Python 传入），长度 4
//             keys[0]=W, keys[1]=A, keys[2]=S, keys[3]=D
//             值为 1.0 表示按下，0.0 表示松开
//
//  Python 每个仿真步调用此函数一次，大致每 2ms 调用一次。
// ================================================================
extern "C" void compute(const double* sensor, double* ctrl,
                        int n_sensor, int n_act,
                        const double* keys) {
    // ============================================================
    //  0. 清空控制量（防御性编程）
    //     确保任何未显式赋值的执行器都输出 0 力矩，
    //     避免上一帧的残值泄漏到本帧造成不可预期的运动。
    // ============================================================
    for (int i = 0; i < n_act; i++) {
        ctrl[i] = 0.0;
    }

    // ============================================================
    //  1. 读取键盘状态
    //     keys 数组由 Python 端 GLFW 键盘回调更新，每帧传入。
    //     值为 1.0 = 按下, 0.0 = 松开, 中间值理论上可用于"力度"。
    //     这样做的好处：键盘逻辑与物理仿真彻底解耦。
    //     改键盘映射只需改 Python 端，控制器只管"给什么命令"。
    // ============================================================
    double key_w = keys[0];   // W → 前进
    double key_a = keys[1];   // A → 左转
    double key_s = keys[2];   // S → 后退
    double key_d = keys[3];   // D → 右转

    // ============================================================
    //  2. 差速驱动（Differential Drive）
    //
    //     原理:
    //       四轮机器人有两个驱动轮（左轮 ctrl[2], 右轮 ctrl[5]）。
    //       前悬挂的四个小轮是被动轮或通过连杆联动，不直接驱动。
    //
    //       前进: 左右轮同速同向 → 直行
    //       后退: 左右轮同速反向 → 后退
    //       左转: 左轮慢/右轮快 → 车身左旋
    //       右转: 右轮慢/左轮快 → 车身右旋
    //
    //     数学:
    //       forward ∈ [-1, 1]  直行分量（正=前进）
    //       turn    ∈ [-1, 1]  旋转分量（正=逆时针/左转）
    //       left  = forward + turn    (左转时左轮减速)
    //       right = forward - turn    (左转时右轮加速)
    //
    //     同时按下 W+D: forward=1, turn=-1 → left=0, right=2 → 原地右转
    //     同时按下 W+A: forward=1, turn=1  → left=2, right=0 → 原地左转
    // ============================================================
    double forward = key_w - key_s;         // 范围 [-1, 1]: W 减 S
    double turn    = key_a - key_d;         // 范围 [-1, 1]: A 减 D（A=左转）

    double left_wheel  = (forward + turn) * MAX_WHEEL_TORQUE;
    double right_wheel = (forward - turn) * MAX_WHEEL_TORQUE;

    // clamp 限幅防止控制量超出电机的 ctrlrange（-500 ~ 500）
    ctrl[2] = clamp(left_wheel,  -MAX_WHEEL_TORQUE, MAX_WHEEL_TORQUE);
    ctrl[5] = clamp(right_wheel, -MAX_WHEEL_TORQUE, MAX_WHEEL_TORQUE);

    // ============================================================
    //  3. 前悬挂 PD 位置控制
    //
    //     PD 控制器公式:
    //       u = KP * (ref - pos) - KD * vel
    //
    //     P 项（比例）: 角度偏差越大 → 回正的力矩越大
    //       - 像一根弹簧，把关节"拉"向参考角度
    //       - KP 越大回正越快，但过大导致震荡
    //
    //     D 项（微分）: 速度越快 → 反向阻尼越大
    //       - 像一个阻尼器，抑制震荡
    //       - KD 越大越"肉"，过小则惯性过冲
    //
    //     参考角度 SUSPENSION_REF_* 目前都设为 0，
    //     表示期望悬挂保持初始装配角度（车身水平）。
    //
    //     调参建议:
    //       先单独调 KP，从 10 开始往上加，直到有轻微震荡;
    //       再加 KD，从 KP/10 开始，抑制震荡的同时保持响应速度。
    // ============================================================

    // 左前悬挂 — 上连杆 (lf0_Joint)
    double lf0_pos = sensor[10];   // 当前角度 (rad)
    double lf0_vel = sensor[16];   // 当前角速度 (rad/s)
    ctrl[0] = clamp(KP_SUSPENSION * (SUSPENSION_REF_LF0 - lf0_pos)
                    - KD_SUSPENSION * lf0_vel,
                    -100.0, 100.0);

    // 左前悬挂 — 下连杆 (lf1_Joint)
    double lf1_pos = sensor[11];
    double lf1_vel = sensor[17];
    ctrl[1] = clamp(KP_SUSPENSION * (SUSPENSION_REF_LF1 - lf1_pos)
                    - KD_SUSPENSION * lf1_vel,
                    -100.0, 100.0);

    // 右前悬挂 — 上连杆 (rf0_Joint)
    double rf0_pos = sensor[13];
    double rf0_vel = sensor[19];
    ctrl[3] = clamp(KP_SUSPENSION * (SUSPENSION_REF_RF0 - rf0_pos)
                    - KD_SUSPENSION * rf0_vel,
                    -100.0, 100.0);

    // 右前悬挂 — 下连杆 (rf1_Joint)
    double rf1_pos = sensor[14];
    double rf1_vel = sensor[20];
    ctrl[4] = clamp(KP_SUSPENSION * (SUSPENSION_REF_RF1 - rf1_pos)
                    - KD_SUSPENSION * rf1_vel,
                    -100.0, 100.0);

    // ============================================================
    //  4. 后悬挂（四连杆被动结构）
    //     后悬挂通过 equality connect 约束与前悬挂形成闭链，
    //     前后联动，无需独立控制。先置零。
    //     如需主动调节后悬挂姿态，可在此添加控制逻辑。
    // ============================================================
    ctrl[6] = 0.0;   // act_lf20 — 左后悬挂
    ctrl[7] = 0.0;   // act_rf20 — 右后悬挂

    // ============================================================
    //  5. 弹簧/减震器预载
    //     tendon 型执行器通过改变空间肌腱长度来施加拉力。
    //     SPRING_FORCE = 150 N 给减震器一个初始预紧力，
    //     让悬挂有支撑刚度，不至于一启动就"趴下"。
    //     这个值调大 = 悬挂更硬，调小 = 更软。
    // ============================================================
    ctrl[8]  = SPRING_FORCE;   // 左侧减震器 (Left_loop1_motor)
    ctrl[9]  = SPRING_FORCE;   // 右侧减震器 (Right_loop1_motor)
}