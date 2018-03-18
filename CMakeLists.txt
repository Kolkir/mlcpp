cmake_minimum_required(VERSION 2.8)
project(ml-cpp-examples)

set(CMAKE_VERBOSE_MAKEFILE ON)

list(APPEND CMAKE_CXX_FLAGS "-std=c++1z -lstdc++ -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas")

add_definitions(-DMSHADOW_USE_CBLAS=1
                -DMSHADOW_USE_CUDA=0
                -DMSHADOW_USE_MKL=0
                -DMSHADOW_DIST_PS=0
                -DMSHADOW_USE_PASCAL=0)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${CMAKE_SOURCE_DIR}/third_party/mshadow)
include_directories(${CMAKE_SOURCE_DIR}/third_party/fast-cpp-csv-parser)
include_directories(${CMAKE_SOURCE_DIR}/third_party/plotcpp)
include_directories(${CMAKE_SOURCE_DIR}/third_party/xtl/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/xtensor/include)

add_subdirectory(linear_regression)
