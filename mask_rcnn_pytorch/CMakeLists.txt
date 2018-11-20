cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mask-rcnn)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

add_compile_options( -fopenmp
                     -pthread
                     -msse3
                     -Wall
                     -Wextra
                     -Wshadow)


set(RESOURCE_FILES
                    README.md)

set(SOURCE_FILES
                    demo.cpp
                    config.h
                    config.cpp
                    maskrcnn.h
                    maskrcnn.cpp
                    visualize.h
                    visualize.cpp
                    imageutils.h
                    imageutils.cpp)

set(REQUIRED_LIBS "stdc++fs")

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES} ${RESOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${TORCH_LIBRARIES}  ${OpenCV_LIBS} ${REQUIRED_LIBS})

