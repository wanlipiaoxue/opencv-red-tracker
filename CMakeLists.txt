set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_minimum_required(VERSION 3.10)
project(rob_vision) #项目名字，随便改

set(CMAKE_CXX_STANDARD 14) # C++标准，OpenCV用C++14足够

# 查找OpenCV
find_package(OpenCV REQUIRED)
# REQUIRED：意思【必须找到OpenCV，找不到cmake直接报错终止】

add_executable(main main.cpp)
# add_executable(生成的程序名 你的源码文件.cpp)

# 把OpenCV库链接到main程序
target_link_libraries(main ${OpenCV_LIBS})
