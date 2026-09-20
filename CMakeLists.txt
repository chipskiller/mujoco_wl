cmake_minimum_required(VERSION 3.15)
project(infantry_ctrl)

# 查找 MuJoCo（需要设置 MuJoCo_DIR 或通过 pip 安装的路径）
find_package(mujoco REQUIRED)

# 查找 GLFW
find_package(glfw3 REQUIRED)

add_executable(infantry_ctrl infantry_ctrl.cpp)
target_link_libraries(infantry_ctrl PRIVATE mujoco::mujoco glfw)