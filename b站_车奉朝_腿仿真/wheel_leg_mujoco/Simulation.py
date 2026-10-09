import mujoco
import mujoco.viewer
import numpy as np
import time
from environment import *
from VMC import *
from keyboard import *
import math
from Controller import *
def main():
    
    TORQUE = 1  #为1时给力矩，为0是无力矩
    SYNC_TIME = True # True=对齐真实世界时间, False=能跑多快跑多快
    GBC486 = LegWheelRobot('MJCF/env.xml')
    dt = GBC486.model.opt.timestep      # 从模型读取真实步长 (0.001s)
    i = 0
    t1 = 1
    t2 = 4
    t3 = 20
    vmc_r = leg_VMC()
    vmc_l = leg_VMC()
    keyboard = KeyboardController()

    # 累计时间对齐: 跟踪总仿真时间 vs 总真实时间
    # 每步累加 dt (仿真时间), 忙等待让真实时间追上仿真时间
    sim_time = 0.0                        # 仿真世界已过的总时间 (s)
    real_start = time.perf_counter()       # 程序启动的真实时间戳

    while True:
        i = i + 1
        
        # 执行仿真步
        GBC486.step()  # 仿真的timestep是1ms，意味着每执行一次step仿真世界时间过去1ms
        #传感器数据获取
        if i % t1 == 0: 
            GBC486.sensor_read_data()
        # ═══════════════════════════════════════════════════════════
        #  控制计算 (每 t2=4 步执行一次, 即 4ms 周期)
        # ═══════════════════════════════════════════════════════════
        if i % t2 == 0:

            # ── 阶段 1: 正向运动学 → 算脚位置和状态 ──────────────────
            # vmc_calc_pos 输入: 两个电机角 + IMU(pitch, gyro)
            #   输出: L0, φ0, theta, d_theta, d_L0 等状态变量
            #   同时更新雅可比矩阵的四个元素 (j11,j12,j21,j22)

            # 右腿: phi1=右前电机角+π, phi4=右后电机角, pitch/gyro 原样传入
            #   joint_pos[0] = Right_front_joint_pos (jAB),  +π 修正零位
            #   joint_pos[1] = Right_rear_joint_pos  (jAG)
            vmc_r.vmc_calc_pos(phi1=GBC486.joint_pos[0]+math.pi,
                               phi4=GBC486.joint_pos[1],
                               pitch= GBC486.euler[1],
                               gyro= GBC486.gyro[1])

            # 左腿: phi1=左后电机角+π, phi4=左前电机角, pitch/gyro 取负(镜像)
            #   joint_pos[3] = Left_rear_joint_pos   (jIO), +π 修正零位
            #   joint_pos[2] = Left_front_joint_pos  (jIJ)
            vmc_l.vmc_calc_pos(phi1=GBC486.joint_pos[3]+math.pi,
                               phi4=GBC486.joint_pos[2],
                               pitch=-GBC486.euler[1],     # 左腿镜像: 俯仰反向
                               gyro=-GBC486.gyro[1])       # 左腿镜像: 角速度反向

            # ── 阶段 2: 设定虚拟力 ──────────────────────────────────
            # F0 = 径向力 (N), 沿 L0 方向: 正=收缩(抬脚), 负=伸长(踩地)
            # Tp = 切向力矩 (Nm), 绕脚 C:  正=逆时针(脚前摆), 负=顺时针(脚后摆)
            # 当前全为零 = 腿不发力, 纯靠重力下垂 ← 这里是要你写控制器的地方!
            vmc_r.F0 = 0
            vmc_l.F0 = 0
            vmc_r.Tp = 0
            vmc_l.Tp = 0

            # ── 阶段 3: VMC 力矩解算 → 虚拟力转关节力矩 ────────────
            #   公式: [τ_front, τ_rear]^T = J^T · [F0, Tp]^T
            #   结果写入 torque_set[1](前电机) 和 torque_set[0](后电机)
            vmc_l.vmc_calc_torque()
            vmc_r.vmc_calc_torque()

            # ── 阶段 4: 轮子力矩 ────────────────────────────────────
            # w_r = 右轮力矩 (Nm), w_l = 左轮力矩 (Nm)
            # 当前全为零 → 轮子不转, 机器人原地不动
            w_r = 0
            w_l = 0
            GBC486.wheel_torque = [w_r, w_l]

            # ── 阶段 5: 组装力矩数组并写入 MuJoCo ───────────────────
            # ctrl 对应关系 (见 env.xml 中 actuator 定义):
            #   ctrl[0]=jAB(右前), ctrl[1]=jAG(右后),
            #   ctrl[2]=jIJ(左前), ctrl[3]=jIO(左后),
            #   ctrl[4]=右轮, ctrl[5]=左轮
            #
            # joint_torque 排列: [右前力矩, 右后力矩, 左前力矩, 左后力矩]
            #   = [vmc_r.torque_set[1], vmc_r.torque_set[0],
            #      vmc_l.torque_set[0], vmc_l.torque_set[1]]
            #   (注意: 右腿 [前,后], 左腿 [前,后] — 下标和左右腿的映射不同!)
            GBC486.joint_torque = [vmc_r.torque_set[1],   # 右前 (φ1 电机)
                                  vmc_r.torque_set[0],    # 右后 (φ4 电机)
                                  vmc_l.torque_set[0],    # 左前 (φ4 电机)
                                  vmc_l.torque_set[1]]    # 左后 (φ1 电机)
            GBC486.actuator_set_torque()

        #键盘控制指令输入,以及打印数据;运行频率低以降低仿真延迟
        if i % t3 == 0:
            cmd = keyboard.get_command()
            # print(vmc_r.L0,vmc_l.L0)

        # 时间同步（可开关）—— 累计对齐模式
        #   每步 sim_time 累加 dt, 然后用忙等待让真实时间追上来
        #   好处: 某步慢（如 viewer.sync 多花了 3ms）不影响整体,
        #         后续步自动多等一点补回来, 长期平均 = 1.0x 实时
        sim_time += dt
        if SYNC_TIME:
            while time.perf_counter() - real_start < sim_time:
                pass



if __name__ == '__main__':
    main()