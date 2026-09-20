/**
 * 步兵机器人 C++ 控制器示例
 *
 * 功能：
 *   - 加载 mjmodel_lqr.xml
 *   - 键盘 WASD 控制前后左右
 *   - 读取 IMU（姿态 + 角速度 + 加速度）
 *   - 读取关节传感器（位置 + 速度）
 *
 * 编译：
 *   g++ infantry_ctrl.cpp -o infantry_ctrl ^
 *       -I/path/to/mujoco/include -lmujoco ^
 *       -L/path/to/mujoco/bin -lglfw3
 *
 * 或参考同目录 CMakeLists.txt
 */

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <cstring>

// ==================== 键盘状态 ====================
static bool key_w = false, key_a = false, key_s = false, key_d = false;
static bool key_up = false, key_down = false, key_left = false, key_right = false;

static void keyboard_callback(GLFWwindow* window, int key, int scancode, int act, int mods) {
    bool pressed = (act == GLFW_PRESS);
    bool released = (act == GLFW_RELEASE);
    if (!pressed && !released) return;

    switch (key) {
        case GLFW_KEY_W:      key_w = pressed; break;
        case GLFW_KEY_A:      key_a = pressed; break;
        case GLFW_KEY_S:      key_s = pressed; break;
        case GLFW_KEY_D:      key_d = pressed; break;
        case GLFW_KEY_UP:     key_up = pressed; break;
        case GLFW_KEY_DOWN:   key_down = pressed; break;
        case GLFW_KEY_LEFT:   key_left = pressed; break;
        case GLFW_KEY_RIGHT:  key_right = pressed; break;
        case GLFW_KEY_ESCAPE:
            if (pressed) glfwSetWindowShouldClose(window, 1);
            break;
        default: break;
    }
}

// ==================== 读取传感器数据 ====================
static void print_sensor_data(const mjModel* model, const mjData* data) {
    // --- IMU 数据 ---
    // framequat: 4维四元数
    int sens_quat = mj_name2id(model, mjOBJ_SENSOR, "orientation");
    int adr_quat = model->sensor_adr[sens_quat];
    printf("姿态四元数: [%.3f %.3f %.3f %.3f]\n",
           data->sensordata[adr_quat + 0],
           data->sensordata[adr_quat + 1],
           data->sensordata[adr_quat + 2],
           data->sensordata[adr_quat + 3]);

    // gyro: 3维角速度 (rad/s)
    int sens_gyro = mj_name2id(model, mjOBJ_SENSOR, "base_ang_vel");
    int adr_gyro = model->sensor_adr[sens_gyro];
    printf("角速度: [%.3f %.3f %.3f] rad/s\n",
           data->sensordata[adr_gyro + 0],
           data->sensordata[adr_gyro + 1],
           data->sensordata[adr_gyro + 2]);

    // accelerometer: 3维线加速度 (m/s²)
    int sens_acc = mj_name2id(model, mjOBJ_SENSOR, "base_lin_acc");
    int adr_acc = model->sensor_adr[sens_acc];
    printf("加速度: [%.2f %.2f %.2f] m/s²\n",
           data->sensordata[adr_acc + 0],
           data->sensordata[adr_acc + 1],
           data->sensordata[adr_acc + 2]);

    // --- 关节传感器 ---
    // 读取关节位置 (rad) 和速度 (rad/s)
    const char* joint_names[] = {
        "lf0_Joint_p", "lf1_Joint_p", "l_wheel_Joint_p",
        "rf0_Joint_p", "rf1_Joint_p", "r_wheel_Joint_p"
    };
    for (int i = 0; i < 6; i++) {
        int sid = mj_name2id(model, mjOBJ_SENSOR, joint_names[i]);
        int adr = model->sensor_adr[sid];
        printf("  %s: %.3f\n", joint_names[i], data->sensordata[adr]);
    }
}

// ==================== 键盘 → 控制量映射 ====================
static void compute_control(const mjModel* model, mjData* data) {
    // 获取 actuator 的 id
    int act_rw = mj_name2id(model, mjOBJ_ACTUATOR, "act_rw");   // 右轮
    int act_lw = mj_name2id(model, mjOBJ_ACTUATOR, "act_lw");   // 左轮

    // --- WASD 差速控制 ---
    double forward = 0.0, turn = 0.0;
    if (key_w) forward += 1.0;
    if (key_s) forward -= 1.0;
    if (key_a) turn    += 1.0;   // 左转
    if (key_d) turn    -= 1.0;   // 右转

    // 差速模型: 左右轮独立速度
    double max_ctrl = 200.0;  // 最大控制量 (N·m)
    double left  = (forward + turn) * max_ctrl;
    double right = (forward - turn) * max_ctrl;

    data->ctrl[act_lw] = left;
    data->ctrl[act_rw] = right;
}

// ==================== 主函数 ====================
int main(int argc, char** argv) {
    // 1. 加载模型
    const char* xml_path = argc > 1 ? argv[1] : "mjmodel_lqr.xml";
    char error[1000] = {0};
    mjModel* model = mj_loadXML(xml_path, nullptr, error, 1000);
    if (!model) {
        fprintf(stderr, "加载模型失败: %s\n", error);
        return 1;
    }
    mjData* data = mj_makeData(model);

    printf("=== 机器人模型信息 ===\n");
    printf("执行器数量: %d\n", model->nu);
    for (int i = 0; i < model->nu; i++) {
        printf("  actuator[%d] = %s\n", i, mj_id2name(model, mjOBJ_ACTUATOR, i));
    }
    printf("传感器数量: %d\n", model->nsensor);
    for (int i = 0; i < model->nsensor; i++) {
        printf("  sensor[%d] = %s (adr=%d, dim=%d)\n",
               i, mj_id2name(model, mjOBJ_SENSOR, i),
               model->sensor_adr[i], model->sensor_dim[i]);
    }
    printf("时间步长: %.4f s\n", model->opt.timestep);

    // 2. 初始化 GLFW
    if (!glfwInit()) {
        fprintf(stderr, "GLFW 初始化失败\n");
        return 1;
    }

    GLFWwindow* window = glfwCreateWindow(1920, 1080, "步兵机器人控制器", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "窗口创建失败\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyboard_callback);

    // 3. 初始化 MuJoCo 渲染
    mjvScene scn;
    mjv_defaultScene(&scn);
    mjrContext con;
    mjr_defaultContext(&con);
    mjv_makeScene(model, &scn, 2000);
    mjr_makeContext(model, &con, mjFONTSCALE_150);

    mjvCamera cam;
    mjv_defaultCamera(&cam);
    cam.distance = 3.0;
    cam.azimuth = 45;
    cam.elevation = -20;

    mjvOption opt;
    mjv_defaultOption(&opt);

    // 4. 主循环
    double dt = model->opt.timestep;
    int frame = 0;

    while (!glfwWindowShouldClose(window)) {
        double t_start = glfwGetTime();

        // --- 控制器 ---
        compute_control(model, data);

        // --- 物理步进 ---
        mj_step(model, data);

        // --- 每 100 步打印传感器（避免刷屏） ---
        if (frame % 100 == 0) {
            printf("\n=== 时间 t=%.3f ===\n", data->time);
            print_sensor_data(model, data);
        }
        frame++;

        // --- 渲染 ---
        mjv_updateScene(model, data, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(mjr_rect(0, 0, 1920, 1080), &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();

        // --- 时间同步 ---
        double elapsed = glfwGetTime() - t_start;
        if (elapsed < dt) {
            int sleep_us = (int)((dt - elapsed) * 1e6);
            if (sleep_us > 0) {
#ifdef _WIN32
                // Windows 上用 Sleep
                Sleep((DWORD)(sleep_us / 1000));
#else
                usleep(sleep_us);
#endif
            }
            while (glfwGetTime() - t_start < dt) {
                // 忙等待微调
            }
        }
    }

    // 5. 清理
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(data);
    mj_deleteModel(model);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}