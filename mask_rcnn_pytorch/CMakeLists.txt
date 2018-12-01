cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mask-rcnn)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/../third_party/json/include)
include_directories(${TORCH_INCLUDE_DIRS})

add_compile_options( -fopenmp
                     -pthread
                     -msse3
                     -Wall
                     -Wextra
                     -Wno-unused-parameter)


set(RESOURCE_FILES
                    README.md)

set(SOURCE_FILES
                    config.h
                    config.cpp
                    maskrcnn.h
                    maskrcnn.cpp
                    visualize.h
                    visualize.cpp
                    imageutils.h
                    imageutils.cpp
                    stateloader.h
                    stateloader.cpp
                    resnet.h
                    resnet.cpp
                    nnutils.h
                    nnutils.cpp
                    fpn.h
                    fpn.cpp
                    anchors.h
                    anchors.cpp
                    rpn.h
                    rpn.cpp)

set(REQUIRED_LIBS "stdc++fs")
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})
list(APPEND REQUIRED_LIBS ${OpenCV_LIBS})
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})


add_library("${CMAKE_PROJECT_NAME}_lib" STATIC ${SOURCE_FILES})
target_link_libraries("${CMAKE_PROJECT_NAME}_lib" ${REQUIRED_LIBS})

add_executable(${CMAKE_PROJECT_NAME} ${RESOURCE_FILES} demo.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} "${CMAKE_PROJECT_NAME}_lib" ${REQUIRED_LIBS})


set(TEST_FILES
    tests/catch.hpp
    tests/tests_main.cpp
    tests/nnutils_test.cpp
    tests/anchor_test.cpp
    )

add_executable("${CMAKE_PROJECT_NAME}_test" ${TEST_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME}_test "${CMAKE_PROJECT_NAME}_lib" ${REQUIRED_LIBS})

