cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mask-rcnn)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

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


set(CUDA_NVCC_FLAGS "-std=c++14;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wall;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wextra;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-unused-parameter;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-unknown-pragmas;")
set(CUDA_NVCC_FLAGS_RELEASE "-O3;-DNDEBUG;")
set(CUDA_NVCC_FLAGS_DEBUG "-g;-O0;")
set(CUDA_PROPAGATE_HOST_FLAGS ON)


set(RESOURCE_FILES
                    README.md)

set(SOURCE_FILES
                    debug.h
                    debug.cpp
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
                    rpn.cpp
                    classifier.h
                    classifier.cpp
                    roialign.h
                    roialign.cpp
                    mask.h
                    mask.cpp
                    roialign/cuda/crop_and_resize_kernel.cu
                    roialign/cuda/crop_and_resize_kernel.h
                    roialign/crop_and_resize.h
                    roialign/crop_and_resize.cpp
                    roialign/crop_and_resize_gpu.h
                    roialign/crop_and_resize_gpu.cpp
                    nms/cuda/nms_kernel.cu
                    nms/cuda/nms_kernel.h
                    nms/nms.h
                    nms/nms.cpp
                    nms/nms_cuda.h
                    nms/nms_cuda.cpp
                    proposallayer.h
                    proposallayer.cpp
                    detectionlayer.h
                    detectionlayer.cpp
                    detectiontargetlayer.h
                    detectiontargetlayer.cpp
                    nms.h
                    nms.cpp
                    cocoloader.h
                    cocoloader.cpp
                    cocodataset.h
                    cocodataset.cpp
                    boxutils.h
                    boxutils.cpp)

set(REQUIRED_LIBS "stdc++fs")
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})
list(APPEND REQUIRED_LIBS ${OpenCV_LIBS})
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})


cuda_add_library("${CMAKE_PROJECT_NAME}_lib" STATIC ${SOURCE_FILES})
target_link_libraries("${CMAKE_PROJECT_NAME}_lib" ${REQUIRED_LIBS})

find_library(GOMP_LIBRARY gomp)

add_executable("${CMAKE_PROJECT_NAME}_demo" ${RESOURCE_FILES} demo.cpp)
target_link_libraries("${CMAKE_PROJECT_NAME}_demo" "${CMAKE_PROJECT_NAME}_lib" ${REQUIRED_LIBS} ${GOMP_LIBRARY})


add_executable("${CMAKE_PROJECT_NAME}_train" train.cpp)
target_link_libraries("${CMAKE_PROJECT_NAME}_train" "${CMAKE_PROJECT_NAME}_lib" ${REQUIRED_LIBS} ${GOMP_LIBRARY})



set(TEST_FILES
    tests/catch.hpp
    tests/tests_main.cpp
    tests/nnutils_test.cpp
    tests/anchor_test.cpp
    )

add_executable("${CMAKE_PROJECT_NAME}_test" ${TEST_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME}_test "${CMAKE_PROJECT_NAME}_lib" ${REQUIRED_LIBS})

