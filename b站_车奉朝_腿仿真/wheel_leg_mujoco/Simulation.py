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
        #vmc计算、观测器计算、LQR计算
        if i % t2 == 0:
            #正向运动学计算，获取状态
            vmc_r.vmc_calc_pos(phi1=GBC486.joint_pos[0]+math.pi,phi4=GBC486.joint_pos[1],pitch= GBC486.euler[1],gyro=GBC486.gyro[1])
            vmc_l.vmc_calc_pos(phi1=GBC486.joint_pos[3]+math.pi,phi4=GBC486.joint_pos[2],pitch=-GBC486.euler[1],gyro=-GBC486.gyro[1])
            vmc_r.F0 = 0
            vmc_l.F0 = 0
            vmc_r.Tp = 0
            vmc_l.Tp = 0
            
            vmc_l.vmc_calc_torque()
            vmc_r.vmc_calc_torque()
            # vmc.vmc_calc()
            w_r = 0
            w_l = 0
            GBC486.wheel_torque = [w_r,w_l]
            GBC486.joint_torque = [vmc_r.torque_set[1],vmc_r.torque_set[0],vmc_l.torque_set[0],vmc_l.torque_set[1]]
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