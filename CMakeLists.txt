cmake_minimum_required(VERSION 2.8)
project(ml-cpp-examples)

set(CMAKE_VERBOSE_MAKEFILE ON)

list(APPEND CMAKE_CXX_FLAGS "-std=c++1z -Wall -Wextra")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

add_subdirectory(linear_regression)
