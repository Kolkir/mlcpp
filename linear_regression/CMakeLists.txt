cmake_minimum_required(VERSION 2.8)
project(linear-regression)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(SOURCES "linear_reg.cpp"
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

