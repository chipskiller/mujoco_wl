import mujoco
import mujoco.viewer
import numpy as np
import time
from environment import *
from VMC import *
from keyboard import *
import math
from Controller import *
from LQR import LQRController


def set_vertical_pose(robot):
    """把初始位形设为「两腿竖直、L0=0.21 m」，即 LQR 的线性化平衡点。

    背景: qpos=0 时腿是接近全伸、前倾 0.65 rad 的位形，与 LQR 平衡点
    (theta=0, L0=0.21) 相差很远，闭环一上电轮力矩就会顶到 ±4 Nm 上限。

    四个电机角由五连杆网格扫描得到 (VMC 正解: L0=0.2087 m, theta=0):
        jAB=-0.2888 (右前), jAG=-1.0382 (右后),
        jIJ=+0.2570 (左前), jIO=+1.0380 (左后)

    被动关节 (jGH/jBE/jEC/jCF/jJM/jMK/jKN/jOP) 不能随手给 0，否则 4 个
    <connect> 闭链约束初值被破坏，起步瞬间会被约束力弹飞。这里用高斯-牛顿
    迭代把约束残差解到 0。
    """
    m, d = robot.model, robot.data
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    # 机体: 姿态无旋转, 速度清零 (高度后面再算)
    d.qpos[0:3] = [0, 0, 0.3]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[:] = 0

    # 四个电机关节角
    for name, ang in (('jAB', -0.2888), ('jAG', -1.0382),
                      ('jIJ',  0.2570), ('jIO',  1.0380)):
        d.qpos[m.jnt_qposadr[jid(name)]] = ang

    # ── 被动关节: 迭代求解闭链约束 ──────────────────────────
    passive = ['jGH', 'jBE', 'jEC', 'jCF', 'jJM', 'jMK', 'jKN', 'jOP']
    padr = np.array([m.jnt_qposadr[jid(n)] for n in passive])
    eqs = [i for i in range(m.neq) if m.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT]

    def residual():
        """返回全部 connect 约束的位置残差 (site1 - site2 - 偏移)."""
        mujoco.mj_kinematics(m, d)
        r = []
        for i in eqs:
            r.extend(d.site_xpos[m.eq_obj1id[i]] - d.site_xpos[m.eq_obj2id[i]]
                     - m.eq_data[i][0:3])
        return np.array(r)

    for _ in range(50):
        r0 = residual()
        if np.linalg.norm(r0) < 1e-9:
            break
        # 数值雅可比 (被动关节角 -> 约束残差)
        J = np.zeros((r0.size, padr.size))
        for k, a in enumerate(padr):
            d.qpos[a] += 1e-6
            J[:, k] = (residual() - r0) / 1e-6
            d.qpos[a] -= 1e-6
        d.qpos[padr] += np.linalg.lstsq(J, -r0, rcond=None)[0]
        # 尊重关节限位
        for k, j in enumerate(passive):
            lo, hi = m.jnt_range[jid(j)]
            if hi > lo:
                d.qpos[padr[k]] = min(max(d.qpos[padr[k]], lo), hi)

    # ── 机体高度: 让机器人最低点恰好贴地 (z=0) ──────────────
    # 不能用 geom_xpos - geom_rbound 估最低点: geom_rbound 是包围球半径,
    #   对细长连杆/机体网格会高估下沿, 实测会把整机抬离地面 14.5 cm.
    # 改用 mj_geomDistance 求每个几何到地面(plane)的真实最短距离.
    mujoco.mj_forward(m, d)
    floor = next(g for g in range(m.ngeom)
                 if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
    gap = min(mujoco.mj_geomDistance(m, d, g, floor, 10.0, None)
              for g in range(m.ngeom) if m.geom_bodyid[g] != 0)
    d.qpos[2] -= gap

    mujoco.mj_forward(m, d)
    print('初始姿态: 双腿竖直, 轮子贴地, 机体高度 qpos[2] = %.4f m' % d.qpos[2])


def main():
    
    TORQUE = 1  #为1时给力矩，为0是无力矩
    SYNC_TIME = True # True=对齐真实世界时间, False=能跑多快跑多快
    robot = LegWheelRobot('MJCF/env.xml')
    dt = robot.model.opt.timestep      # 从模型读取真实步长 (0.0005s)
    set_vertical_pose(robot)           # 初始位形设为腿竖直 (LQR 平衡点)
    i = 0
    t1 = 1
    t2 = 1
    t3 = 20
    vmc_r = leg_VMC()
    vmc_l = leg_VMC()
    lqr = LQRController()              # LQR 闭环控制器 (K 由腿长多项式给出)

    # ── VMC 预热: 用初始直立姿态先跑一次正解 ────────────────
    # 数值微分需要"上一帧"值; 预热后 last_L0/last_phi0 才是真实值,
    #   否则首帧 d_L0 = L0/dt (几百 m/s), F0 会瞬间上万牛.
    #   同时拿到初始腿长 L0, 作为腿长 PD 的目标值.
    robot.sensor_read_data()
    vmc_r.vmc_calc_pos(dt=dt, phi1=robot.joint_pos[0] + math.pi,
                       phi4=robot.joint_pos[1],
                       pitch=robot.euler[1], gyro=robot.gyro[1])
    vmc_l.vmc_calc_pos(dt=dt, phi1=robot.joint_pos[3] + math.pi,
                       phi4=robot.joint_pos[2],
                       pitch=-robot.euler[1], gyro=-robot.gyro[1])

    # ── 腿长 PD 参数 (F0 虚拟径向力) ──────────────────────────
    # 符号标定(固定机体实测): F0 > 0 → 腿伸长(把机体顶起); F0 < 0 → 腿收缩
    KP_L  = 800.0                        # 腿长刚度 [N/m]
    KD_L  = 40.0                         # 腿长阻尼 [N/(m/s)]
    L_REF = 0.5 * (vmc_l.L0 + vmc_r.L0)   # 目标腿长 = 初始直立腿长 [m]
    F0_FF = 0.7 * 0.5 * 14.546 * 9.81    # 重力前馈: 理论值 71.3 N/腿 的 70% ≈ 50 N
    keyboard = KeyboardController()

    # 累计时间对齐: 跟踪总仿真时间 vs 总真实时间
    # 每步累加 dt (仿真时间), 忙等待让真实时间追上仿真时间
    sim_time = 0.0                        # 仿真世界已过的总时间 (s)
    real_start = time.perf_counter()       # 程序启动的真实时间戳

    while True:
        i = i + 1
        
        # 执行仿真步
        robot.step()  # 仿真的timestep是0.5ms，意味着每执行一次step仿真世界时间过去0.5ms
        #传感器数据获取
        if i % t1 == 0: 
            robot.sensor_read_data()
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
            vmc_r.vmc_calc_pos(dt=dt*t2,          # 控制周期, 供数值微分用
                               phi1=robot.joint_pos[0]+math.pi,
                               phi4=robot.joint_pos[1],
                               pitch= robot.euler[1],
                               gyro= robot.gyro[1])

            # 左腿: phi1=左后电机角+π, phi4=左前电机角, pitch/gyro 取负(镜像)
            #   joint_pos[3] = Left_rear_joint_pos   (jIO), +π 修正零位
            #   joint_pos[2] = Left_front_joint_pos  (jIJ)
            vmc_l.vmc_calc_pos(dt=dt*t2,          # 控制周期, 供数值微分用
                               phi1=robot.joint_pos[3]+math.pi,
                               phi4=robot.joint_pos[2],
                               pitch=-robot.euler[1],     # 左腿镜像: 俯仰反向
                               gyro=-robot.gyro[1])       # 左腿镜像: 角速度反向

            # ── LQR 状态组装 (目前只打印, 尚未参与控制) ─────────────
            # 状态顺序与 LQR.py / MATLAB 完全一致:
            #   [s, dot_s, phi, dot_phi,
            #    theta_l_l, dot_theta_l_l, theta_l_r, dot_theta_l_r,
            #    theta_b, dot_theta_b]
            # 符号约定 (与上交文档 WBR_modeling 一致):
            #   由文档 (2.5) s_b = R_w*theta_w + R_l*phi + l*sin(theta_l) 可知
            #   theta_l > 0 = 髋部在轮子前方(+s); euler[1] > 0 = 机顶朝 +x(+s).
            #   实测(两电机同时 +0.03): 右腿 theta_vmc 减小 0.03 时轮子 x +0.0064
            #   (髋后退), 故 theta_l_r = +theta_vmc_r; 左腿 theta_vmc 减小 0.03 时
            #   轮子 x -0.0062 (髋前进), 故 theta_l_l = -theta_vmc_l.
            state = np.array([
                robot.data.qpos[0],     # s             水平位移 [m]
                robot.data.qvel[0],     # dot_s         水平速度 [m/s]
                robot.euler[2],         # phi           偏航角 [rad]
                robot.gyro[2],          # dot_phi       偏航角速度 [rad/s]
                -vmc_l.theta,           # theta_l_l     左腿倾角 [rad]
                -vmc_l.d_theta,         # dot_theta_l_l 左腿角速度 [rad/s]
                vmc_r.theta,            # theta_l_r     右腿倾角 [rad]
                vmc_r.d_theta,          # dot_theta_l_r 右腿角速度 [rad/s]
                robot.euler[1],         # theta_b       机体俯仰角 [rad]
                robot.gyro[1],          # dot_theta_b   机体俯仰角速度 [rad/s]
            ])

            # LQR 闭环: 按当前左右腿长查 K, 输出 [T_w_l, T_w_r, T_b_l, T_b_r]
            u = lqr.update(state, vmc_l.L0, vmc_r.L0)

            # 每 0.5 s 打印一次 (t2 周期内满足 i % 500 == 0), 首次在 t=0.5 s
            if i % 500 == 0:
                print('[t=%5.2fs] s=%+.4f ds=%+.4f phi=%+.4f dphi=%+.4f | '
                      'thL=%+.4f dthL=%+.4f | thR=%+.4f dthR=%+.4f | '
                      'thB=%+.4f dthB=%+.4f | L0L=%.4f L0R=%.4f | '
                      'u=[%+.2f %+.2f %+.2f %+.2f]'
                      % (i * dt, state[0], state[1], state[2], state[3],
                         state[4], state[5], state[6], state[7],
                         state[8], state[9], vmc_l.L0, vmc_r.L0,
                         u[0], u[1], u[2], u[3]))

            # ── 阶段 2: 设定虚拟力 ──────────────────────────────────
            # F0 = 径向力 (N), 实测标定: 正=伸长(顶起机体), 负=收缩(抬脚)
            # 腿长 PD + 重力前馈: 腿短则伸长, 伸长中则加阻尼回拉
            vmc_r.F0 = F0_FF + KP_L * (L_REF - vmc_r.L0) - KD_L * vmc_r.d_L0
            vmc_l.F0 = F0_FF + KP_L * (L_REF - vmc_l.L0) - KD_L * vmc_l.d_L0
            # Tp = 切向力矩 (Nm), 由 LQR 的腿力矩 T_b 映射
            # 实测标定(机体锁死, 单独给 Tp): 左右腿 Tp>0 都使 theta_vmc 减小.
            # 上交文档 T_b 为「腿部转矩(机体-腿)」, 由 equ1 整理得
            #   (m_l*l_w*l_b - I_l)*theta_ddot = (m_l*l_w+0.5*m_b*l)*g*theta_l + T_b - T_w*(1+l/R),
            #   该系数在 l=0.08~0.30 恒为负 (l=0.21 时 -0.124), 故 T_b>0 使文档意义的
            #   theta_l 增大 = 髋前移.
            # 对应到 Tp: 左腿 theta_vmc 减小 = 髋前移 -> Tp_l = +u[2];
            #            右腿 theta_vmc 减小 = 髋后移 -> Tp_r = -u[3].
            vmc_l.Tp =  u[2]     # T_b_l
            vmc_r.Tp = -u[3]     # T_b_r

            # ── 阶段 3: VMC 力矩解算 → 虚拟力转关节力矩 ────────────
            #   公式: [τ_front, τ_rear]^T = J^T · [F0, Tp]^T
            #   结果写入 torque_set[1](前电机) 和 torque_set[0](后电机)
            vmc_l.vmc_calc_torque()
            vmc_r.vmc_calc_torque()

            # ── 阶段 4: 轮子力矩 ────────────────────────────────────
            # 实测: ctrl[4]=ctrl[5]=+0.5 时 qvel_x>0, 即正力矩=前进
            # LQR 的 u 顺序是 [T_w_l, T_w_r], 而 wheel_torque 是 [右, 左], 需交换
            w_r = u[1]           # T_w_r
            w_l = u[0]           # T_w_l
            robot.wheel_torque = [w_r, w_l]

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
            robot.joint_torque = [vmc_r.torque_set[1],   # 右前 (φ1 电机)
                                  vmc_r.torque_set[0],    # 右后 (φ4 电机)
                                  vmc_l.torque_set[0],    # 左前 (φ4 电机)
                                  vmc_l.torque_set[1]]    # 左后 (φ1 电机)
            robot.actuator_set_torque()

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