"""LQR 闭环控制器.

由 MATLAB (sjtu-wl-lqr-matlab/LQR_K.m, 上海交大 WBR_control 文档 (3.5)) 拟合得到
K 矩阵关于左右腿长的二元二次多项式:

    K(l_l, l_r) = K_K1 + K_K2*l_l + K_K3*l_l^2
                + K_K4*l_r + K_K5*l_r^2 + K_K6*l_l*l_r

用法:
    from LQR import LQRController
    lqr = LQRController()
    u = lqr.update(state, l_l, l_r, x_ref)

状态量 state (10 维, 顺序必须与 MATLAB 完全一致):
    0  s              水平位移        [m]
    1  dot_s          水平速度        [m/s]
    2  phi            偏航角          [rad]
    3  dot_phi        偏航角速度      [rad/s]
    4  theta_l_l      左腿倾角        [rad]
    5  dot_theta_l_l  左腿角速度      [rad/s]
    6  theta_l_r      右腿倾角        [rad]
    7  dot_theta_l_r  右腿角速度      [rad/s]
    8  theta_b        机体俯仰角      [rad]
    9  dot_theta_b    机体俯仰角速度  [rad/s]

控制量 u (4 维, update 的返回值):
    0  T_w_l  左轮力矩  [Nm]
    1  T_w_r  右轮力矩  [Nm]
    2  T_b_l  左腿力矩  [Nm]
    3  T_b_r  右腿力矩  [Nm]

注意: Simulation.py 里 wheel_torque 的顺序是 [右, 左], 与本文件的 [左, 右] 相反,
      接线时需要交换; 腿力矩对应 VMC 的切向虚拟力矩 Tp。
"""

import numpy as np

# ---------------------------------------------------------------------------
# K 多项式系数由 MATLAB 拟合后自动生成, 见 sjtu-wl-lqr-matlab/export_K.m。
# 形状 (6, 4, 10): 第 0 维是"第几项", 每个 "页" 是一个 4×10 矩阵.
# 页顺序 (与上交文档公式下标一致):
#   0: 常数项    1: l_l       2: l_l^2
#   3: l_r       4: l_r^2     5: l_l*l_r
# 重新拟合后只需重跑 export_K 覆盖 LQR_K_coeff.py, 不要手改该文件。
# ---------------------------------------------------------------------------
from LQR_K_coeff import K_COEFF

STATE_SIZE = 10                  # 状态量维数
CONTROL_SIZE = 4                # 控制量维数


class LQRController:
    """LQR 状态反馈控制器: 输入状态/参考/腿长, 输出 4 个控制力矩."""

    def __init__(self, wheel_limit=4.0, leg_limit=20.0, l_min=0.08, l_max=0.30):
        # 力矩限幅 [Nm], 对应 env.xml 中电机 ctrlrange
        self.wheel_limit = wheel_limit
        self.leg_limit = leg_limit
        # 腿长钳位范围 [m], 与 MATLAB 拟合区间一致, 防止多项式外推发散
        self.l_min = l_min
        self.l_max = l_max

    def gains(self, l_l, l_r):
        """按当前腿长还原 4×10 的 K 矩阵.

        L 为 6 项基函数, 顺序与 K_COEFF 的页顺序一一对应:
            L = [1, l_l, l_l^2, l_r, l_r^2, l_l*l_r]
        np.tensordot(K_COEFF, L, axes=([0], [0])) 即对"项"做加权求和:
            K[r,c] = sum_i K_COEFF[i,r,c] * L[i]
        """
        l_l = float(np.clip(l_l, self.l_min, self.l_max))
        l_r = float(np.clip(l_r, self.l_min, self.l_max))
        L = np.array([1.0, l_l, l_l * l_l, l_r, l_r * l_r, l_l * l_r])
        return np.tensordot(K_COEFF, L, axes=([0], [0]))    # (4, 10)

    def update(self, state, l_l, l_r, x_ref=None):
        """一步闭环计算.

        参数:
            state: 10 维状态向量 (顺序见模块说明)
            l_l:   左腿长 [m]
            l_r:   右腿长 [m]
            x_ref: 10 维参考状态, None 表示全零 (直立静止)
        返回:
            4 维控制量表 [T_w_l, T_w_r, T_b_l, T_b_r]
        """
        x = np.asarray(state, dtype=float).reshape(-1)
        if x.size != STATE_SIZE:
            raise ValueError('state 必须是 %d 维, 当前 %d 维' % (STATE_SIZE, x.size))

        if x_ref is None:
            target = np.zeros(STATE_SIZE)
        else:
            target = np.asarray(x_ref, dtype=float).reshape(-1)
            if target.size != STATE_SIZE:
                raise ValueError('x_ref 必须是 %d 维, 当前 %d 维' % (STATE_SIZE, target.size))

        # 状态反馈: u = -K * (x - x_ref), 使状态误差回到零
        u = -self.gains(l_l, l_r) @ (x - target)

        # 力矩限幅, 防止指令超过电机能力
        u[0:2] = np.clip(u[0:2], -self.wheel_limit, self.wheel_limit)
        u[2:4] = np.clip(u[2:4], -self.leg_limit, self.leg_limit)
        return u


if __name__ == '__main__':
    # 自检: 直接运行本文件即可看到不同腿长下的 K 与一组示例控制量
    ctrl = LQRController()

    for leg in (0.15, 0.20, 0.25):
        print('l_l = l_r = %.2f m, K =\n%s\n' % (leg, np.array2string(ctrl.gains(leg, leg), precision=4)))

    # 零状态、零参考 -> 输出应为全零 (直立静止时不需要力矩)
    print('零状态输出:', ctrl.update(np.zeros(10), 0.20, 0.20))

    # 示例: 目标前进速度 0.2 m/s, 其余状态为零
    x_ref = np.zeros(10)
    x_ref[1] = 0.2
    print('目标速度 0.2 m/s 时的输出:', ctrl.update(np.zeros(10), 0.20, 0.20, x_ref))
