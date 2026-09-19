"""纯计时测试，无 viewer 干扰"""
import time
import mujoco

# 用 test2 的实际模型
model = mujoco.MjModel.from_xml_path("test2.xml")
data = mujoco.MjData(model)

dt = model.opt.timestep
print(f"dt = {dt}s = {dt*1000:.1f}ms")

# 测试1: 不加控制，纯 mj_step 1000 步
start = time.perf_counter()
for i in range(1000):
    mujoco.mj_step(model, data)
elapsed = time.perf_counter() - start
print(f"测试1 — 纯 mj_step x1000: real={elapsed*1000:.1f}ms, sim={data.time:.3f}s")
print(f"        期望时间: {1000*dt*1000:.0f}ms, 实际/期望={elapsed/(1000*dt):.2f}x")

# 测试2: 带忙等待的时间同步
data.time = 0
start = time.perf_counter()
step_start = time.perf_counter()
for i in range(100):
    step_start = time.perf_counter()
    mujoco.mj_step(model, data)
    # 忙等待
    while time.perf_counter() - step_start < dt:
        pass
elapsed = time.perf_counter() - start
print(f"测试2 — 忙等待 x100:   real={elapsed*1000:.1f}ms, sim={data.time:.3f}s")
print(f"        期望时间: {100*dt*1000:.0f}ms, 实际/期望={elapsed/(100*dt):.2f}x")