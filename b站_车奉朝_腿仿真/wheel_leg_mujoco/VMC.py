import math
from math import sin,cos

class leg_VMC:
    """
    虚拟机械腿参数类
    """
    def __init__(self):
        # 长度参数，单位为m
        self.l5 = 0.0  # AE距离
        self.l1 = 0.215
        self.l4 = 0.215
        self.l2 = 0.258
        self.l3 = 0.258
        
        # 坐标点
        self.XB = 0.0
        self.YB = 0.0
        self.XD = 0.0
        self.YD = 0.0
        
        # C点坐标
        self.XC = 0.0
        self.YC = 0.0
        self.L0 = 0.0      # C点的极径
        self.phi0 = 0.0    # C点的极角
        self.alpha = 0.0
        self.d_alpha = 0.0
        
        # 距离参数
        self.lBD = 0.0     # BD之间的距离
        
        # 角度变化率
        self.d_phi0 = 0.0
        self.last_phi0 = 0.0
        
        # 中间计算参数
        self.A0 = 0.0
        self.B0 = 0.0
        self.C0 = 0.0
        self.phi2 = 0.0
        self.phi3 = 0.0
        self.phi1 = 0.0
        self.phi4 = 0.0
        
        # 雅可比矩阵系数
        self.j11 = 0.0
        self.j12 = 0.0
        self.j21 = 0.0
        self.j22 = 0.0
        
        # 扭矩设置值（2个元素的列表）
        self.torque_set = [0.0, 0.0]
        
        # 力和扭矩参数
        self.F0 = 0.0
        self.Tp = 0.0
        self.F02 = 0.0
        
        # theta相关参数
        self.theta = 0.0
        self.d_theta = 0.0
        self.last_d_theta = 0.0
        self.dd_theta = 0.0
        
        # L0相关参数
        self.d_L0 = 0.0
        self.dd_L0 = 0.0
        self.last_L0 = 0.0
        self.last_d_L0 = 0.0

        # imu相关参数
        self.pitch = 0.0
        self.gyro = 0.0
        
        # 支撑力
        self.FN = 0.0
        
        # 标志位
        self.first_flag = 0
    
    def vmc_calc_pos(self, dt=0.004, phi1=None, phi4=None, pitch=None, gyro=None):
        """
        VMC 正向运动学 — 由两个电机角度算出脚(轮子)位置

        输入 (全部可选, 传了就用新值, 不传就用上次的):
          dt     — 控制周期 (s), 用于数值微分
          phi1   — 前电机角度 (rad), 即 jAB (右腿) 或 jIJ (左腿)
          phi4   — 后电机角度 (rad), 即 jAG (右腿) 或 jIO (左腿)
          pitch  — 车身俯仰角 (rad), IMU 读值
          gyro   — 车身俯仰角速度 (rad/s), IMU 陀螺仪 Y 轴

        输出 (更新 self 的以下属性):
          XC, YC   — 脚 C 的直角坐标 (以底盘 AE 中点为原点, X 指前)
          L0, phi0 — 脚 C 的极坐标 (L0=腿长, phi0=腿相对底盘的角度)
          phi2     — 等价连杆 BC 的角度 (不是物理关节!)
          phi3     — 等价连杆 DC 的角度 (不是物理关节!)
          theta    — 腿的绝对倾角 (相对世界竖直方向, 用于 LQR)
          d_phi0, d_theta, d_L0, dd_theta, dd_L0 — 各阶导数

        几何模型 (等价五杆):
               A ─── l5 ─── E          ← 底盘
              /              \
           l1(前)          l4(后)       ← 主动摇臂 (有电机)
            φ1              φ4
           /                \
          B                  D
           \                /
        l2(等价BC)      l3(等价DC)       ← 虚拟等价杆 (物理上是多根杆)
             \          /
              ─── C ───                 ← 脚/轮子位置 (虚拟点)
        """

        # ── 参数更新: 传入新值就覆盖, 没传就沿用上次 ──────────
        if pitch is not None:
            self.pitch = pitch
        if gyro is not None:
            self.gyro = gyro
        if phi1 is not None:
            self.phi1 = phi1
        if phi4 is not None:
            self.phi4 = phi4
        
        # ── pitch/gyro 符号修正 ──────────────────────────────
        # 实车 C 代码中右腿需要取负号 (坐标系朝向差异)
        # 调用方 (Simulation.py) 已经对左右腿分别处理了符号,
        # 这里再做一次统一取负, 确保与世界坐标系一致
        PitchR = -self.pitch      # 修正后的车身俯仰角
        GyroR  = -self.gyro       # 修正后的车身俯仰角速度

        # ========================================================
        #  阶段一: 几何解算 — 由 φ1, φ4 求脚 C 的坐标
        # ========================================================

        # ── 摇臂端点 B 和 D 的坐标 ────────────────────────────
        # B = 前摇臂(l1)的末端, 角度 φ1
        self.XB = self.l1 * math.cos(self.phi1)       # B 点 X = l1·cos(φ1)
        self.YB = self.l1 * math.sin(self.phi1)       # B 点 Y = l1·sin(φ1)
        # D = 后摇臂(l4)的末端, 角度 φ4, X 方向偏移 l5 (AE 间距)
        self.XD = self.l5 + self.l4 * math.cos(self.phi4)  # D 点 X = l5 + l4·cos(φ4)
        self.YD = self.l4 * math.sin(self.phi4)            # D 点 Y = l4·sin(φ4)

        # ── BD 距离 ──────────────────────────────────────────
        # BD 是两根摇臂末端之间的距离, 也是三角形 BDC 的一条边
        self.lBD = math.sqrt((self.XD - self.XB)**2 + (self.YD - self.YB)**2)

        # ── 三角形 BDC 求解: 已知三边 lBD, l2, l3, 求 φ2 ────
        # 方法: 余弦定理 → 2·l2·(XD-XB)·cos(φ2) + 2·l2·(YD-YB)·sin(φ2)
        #        = l2² + lBD² - l3²
        # 设 A0·cos(φ2) + B0·sin(φ2) = C0
        self.A0 = 2 * self.l2 * (self.XD - self.XB)       # 余弦项系数
        self.B0 = 2 * self.l2 * (self.YD - self.YB)       # 正弦项系数
        self.C0 = self.l2**2 + self.lBD**2 - self.l3**2   # 常数项 (余弦定理结果)

        # 解方程 A0·cos(φ2) + B0·sin(φ2) = C0
        # 通解: φ2 = 2·atan2( (B0 ± √(A0²+B0²-C0²)), (A0+C0) )
        # 这里取 "+" 号对应腿向后弯的构型
        discriminant = self.A0**2 + self.B0**2 - self.C0**2
        if discriminant < 0:
            discriminant = max(discriminant, 0)            # 浮点误差保护

        numerator   = self.B0 + math.sqrt(discriminant)    # 分子: B0 + √Δ
        denominator = self.A0 + self.C0                     # 分母: A0 + C0
        if abs(denominator) < 1e-12:
            self.phi2 = 0.0                                # 奇异位形保护
        else:
            self.phi2 = 2 * math.atan2(numerator, denominator)

        # ── φ3: 由 φ2 反推 DC 杆的角度 ──────────────────────
        # C = B + l2·(cosφ2, sinφ2), 同时 C = D + l3·(cosφ3, sinφ3)
        # 所以 DC 向量 = C - D = (XB-XD+l2·cosφ2, YB-YD+l2·sinφ2)
        dy = self.YB - self.YD + self.l2 * math.sin(self.phi2)
        dx = self.XB - self.XD + self.l2 * math.cos(self.phi2)
        self.phi3 = math.atan2(dy, dx)                    # DC 杆的角度

        # ── C 点坐标 (脚/轮子位置) ──────────────────────────
        # C = B + l2·(cosφ2, sinφ2)
        self.XC = self.XB + self.l2 * math.cos(self.phi2)  # C 点 X
        self.YC = self.YB + self.l2 * math.sin(self.phi2)  # C 点 Y

        # ── C 点极坐标 (以 AE 中点 (l5/2, 0) 为极点) ──────
        # L0   = 脚到车身中点的径向距离 (腿的有效长度)
        # phi0 = 脚相对车身的方位角 (0°=正前方, 正=向上)
        self.L0   = math.sqrt((self.XC - self.l5/2.0)**2 + self.YC**2)
        self.phi0 = math.atan2(self.YC, (self.XC - self.l5/2.0))

        # ── α: 腿的俯仰补偿角 (pi/2 - φ0, 竖直方向为零) ─────
        # 用于后续把世界系角度映射到腿系
        self.alpha = math.pi/2.0 - self.phi0

        # ========================================================
        #  阶段二: 状态变量求导 — 数值微分 + 状态合成
        # ========================================================

        # ── φ0 角速度 (数值微分) ─────────────────────────────
        # 注意: 第一次调用时没有上一帧值, 设 first_flag 跳过微分
        if self.first_flag == 0:
            self.last_phi0 = self.phi0
            self.first_flag = 1

        self.d_phi0  = (self.phi0 - self.last_phi0) / dt     # φ0 的一阶导数 (rad/s)
        self.d_alpha = -self.d_phi0                           # α 的导数 = -φ0

        # ── θ 和 d_θ: 腿的绝对倾角 (相对世界竖直方向) ──────
        #   θ      = π/2 - pitch - φ0
        #   = (世界竖直) - (车身倾角) - (腿相对车身角度)
        #   = 腿在世界系中的绝对倾角, 用于 LQR 平衡控制
        #   d_θ    = -gyro - d_φ0
        #   = -(车身角速度) - (腿相对车身角速度)
        #   = 腿在世界系中的绝对角速度
        self.theta   = math.pi/2.0 - PitchR - self.phi0
        self.d_theta = -GyroR - self.d_phi0

        self.last_phi0 = self.phi0                             # 保存本帧 φ0, 供下帧微分

        # ── L0 及其导数 (腿长变化率) ─────────────────────────
        # d_L0   = 腿的伸缩速度 (m/s), 正=伸长
        # dd_L0  = 腿的伸缩加速度 (m/s²)
        self.d_L0  = (self.L0 - self.last_L0) / dt            # 一阶微分
        self.dd_L0 = (self.d_L0 - self.last_d_L0) / dt        # 二阶微分

        self.last_d_L0 = self.d_L0                             # 保存, 供下帧二阶微分
        self.last_L0   = self.L0                               # 保存, 供下帧一阶微分

        # ── θ 的二阶导数 (腿的绝对角加速度) ──────────────────
        self.dd_theta = (self.d_theta - self.last_d_theta) / dt
        self.last_d_theta = self.d_theta                       # 保存, 供下帧微分

    def vmc_calc_torque(self):
        """
        VMC 力矩解算 — 将脚端的虚拟力(F0, Tp)转为两个电机力矩

        原理: 虚功原理  τ = J^T · F_virtual

          雅可比矩阵 J 定义 (微分层面的映射):
            ⎡ dL0 ⎤   ⎡ j11  j12 ⎤ ⎡ dφ1 ⎤
            ⎢     ⎥ = ⎢          ⎥·⎢     ⎥     (φ1→L0/φ0 的灵敏度)
            ⎣ dφ0 ⎦   ⎣ j21  j22 ⎦ ⎣ dφ4 ⎦

          力矩输出 (J 的转置把虚拟力 "拉回" 到关节空间):
            ⎡ τ_front(φ1电机) ⎤   ⎡ j11  j21 ⎤ ⎡ F0  ⎤
            ⎢                 ⎥ = ⎢          ⎥·⎢     ⎥
            ⎣ τ_rear (φ4电机) ⎦   ⎣ j12  j22 ⎦ ⎣ Tp  ⎦

        虚拟力定义:
          F0  — 径向虚拟力 (N), 沿 L0 方向 (腿伸缩方向)
                正=F0把脚往底盘拉(收缩), 负=把脚往外推(伸长)
          Tp  — 切向虚拟力矩 (Nm), 绕 C 点旋转方向
                正=逆时针(脚相对车身往前摆), 负=顺时针(往后摆)

        输出:
          torque_set[1] = j11·F0 + j12·Tp  →  φ1 电机力矩 (Nm)
          torque_set[0] = j21·F0 + j22·Tp  →  φ4 电机力矩 (Nm)
        """

        # ── 预计算: sin(φ3-φ2), 四个雅可比元素都要除它 ────
        # φ3-φ2 趋近 0 或 ±π 时 sin→0 → 雅可比奇异 (两根等价杆共线)
        sin_phi3_phi2 = math.sin(self.phi3 - self.phi2)

        # ========================================================
        #  雅可比矩阵的四个元素
        #
        #  命名规则: j[行][列]
        #    行 1 = d(L0)/d(·)   (腿长对电机角的偏导)
        #    行 2 = d(φ0)/d(·)   (腿角度对电机角的偏导)
        #    列 1 = d(·)/d(φ1)   (对前电机 φ1 的偏导)
        #    列 2 = d(·)/d(φ4)   (对后电机 φ4 的偏导)
        #
        #  结论: j11/j21 由前摇臂(φ1,l1)产生
        #        j12/j22 由后摇臂(φ4,l4)产生
        # ========================================================

        # j11 = ∂L0/∂φ1: 前电机 φ1 转 1rad 引起腿长 L0 变化多少米
        #   = l1 · sin(φ0-φ3) · sin(φ1-φ2) / sin(φ3-φ2)
        #   分子: l1 是前摇臂长, sin(φ0-φ3) 把 DC 杆方向投影到 L0 径向
        #         sin(φ1-φ2) 把 φ1 的转动投影到 BC 杆的弯曲
        self.j11 = (self.l1 * math.sin(self.phi0 - self.phi3) *
                   math.sin(self.phi1 - self.phi2)) / sin_phi3_phi2

        # j12 = ∂φ0/∂φ1: 前电机 φ1 转 1rad 引起腿角度 φ0 变化多少 rad
        #   = l1 · cos(φ0-φ3) · sin(φ1-φ2) / (L0 · sin(φ3-φ2))
        #   cos(φ0-φ3) 把 DC 杆方向投影到 φ0 切向 (垂直 L0 方向)
        #   除以 L0 是因为角位移 = 线位移 / 半径
        self.j12 = (self.l1 * math.cos(self.phi0 - self.phi3) *
                   math.sin(self.phi1 - self.phi2)) / (self.L0 * sin_phi3_phi2)

        # j21 = ∂L0/∂φ4: 后电机 φ4 转 1rad 引起腿长 L0 变化多少米
        #   = l4 · sin(φ0-φ2) · sin(φ3-φ4) / sin(φ3-φ2)
        #   对称于 j11, 但用的是后摇臂 l4 和 φ4
        self.j21 = (self.l4 * math.sin(self.phi0 - self.phi2) *
                   math.sin(self.phi3 - self.phi4)) / sin_phi3_phi2

        # j22 = ∂φ0/∂φ4: 后电机 φ4 转 1rad 引起腿角度 φ0 变化多少 rad
        #   = l4 · cos(φ0-φ2) · sin(φ3-φ4) / (L0 · sin(φ3-φ2))
        #   对称于 j12, 但用的是后摇臂 l4 和 φ4
        self.j22 = (self.l4 * math.cos(self.phi0 - self.phi2) *
                   math.sin(self.phi3 - self.phi4)) / (self.L0 * sin_phi3_phi2)

        # ========================================================
        #  虚功原理: τ = J^T · F_virtual
        #
        #    脚端虚拟力 F_virtual = [F0, Tp]^T  (2×1)
        #    雅可比转置    J^T    = [[j11, j21],  (2×2)^T
        #                            [j12, j22]]
        #    关节力矩        τ     = [τ_front, τ_rear]^T  (2×1)
        #
        #  torque_set[1] = τ_front (φ1 电机, 前摇臂)
        #  torque_set[0] = τ_rear  (φ4 电机, 后摇臂)
        # ========================================================
        self.torque_set[1] = self.j11 * self.F0 + self.j12 * self.Tp   # 前电机力矩
        self.torque_set[0] = self.j21 * self.F0 + self.j22 * self.Tp   # 后电机力矩