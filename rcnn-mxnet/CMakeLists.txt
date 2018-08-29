cmake_minimum_required(VERSION 3.10)
project(rcnn)

set(LIBRARIES_DIR "" CACHE PATH "path to mxnet install dir")

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-int-in-bool-context -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

message(STATUS "mxnet install path: ${LIBRARIES_DIR}")

include_directories(${CMAKE_SOURCE_DIR}/../third_party/eigen)
include_directories(${CMAKE_SOURCE_DIR}/../third_party/json/include)

include_directories(${LIBRARIES_DIR}/include)
include_directories(${LIBRARIES_DIR}/include/mxnet)
include_directories(${LIBRARIES_DIR}/include/dcml)

include_directories(${LIBRARIES_DIR}/sources/incubator-mxnet/3rdparty/dlpack/include/)
include_directories(${LIBRARIES_DIR}/sources/incubator-mxnet/src/operator/)
include_directories(${LIBRARIES_DIR}/sources/incubator-mxnet/mshadow)
include_directories(${LIBRARIES_DIR}/sources/incubator-mxnet/.build/mklml/mklml_lnx_2018.0.3.20180406/include/)

link_directories(${LIBRARIES_DIR}/lib)
link_directories(${LIBRARIES_DIR}/lib64)

find_package(OpenCV REQUIRED)
find_package(BLAS REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs "mxnet")
list(APPEND requiredlibs "mkldnn")

set(SOURCES
    train.cpp
    rcnn.h
    rcnn.cpp
    resnet.h
    resnet.cpp
    coco.h
    coco.cpp
    anchorgenerator.h
    anchorgenerator.cpp
    anchorsampler.h
    anchorsampler.cpp
    trainiter.h
    trainiter.cpp
    bbox.h
    bbox.cpp
    imagedb.h
    imagedb.cpp
    params.h
    params.cpp
    proposaltarget.h
    proposaltarget.cpp
    README.md
    get_model.sh
    )

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${requiredlibs})
target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES} ${OpenCV_LIBS})

