cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mask-rcnn)

find_package(Torch REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_compile_options( -std=c++17
                     -fopenmp
                     -pthread
                     -msse3
                     -Wall
                     -Wextra
                     -Wno-int-in-bool-context
                     -Wno-unused-parameter
                     -Wno-unknown-pragmas
                     -Wno-sign-compare)

add_executable(${CMAKE_PROJECT_NAME} main.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} ${TORCH_LIBRARIES})
