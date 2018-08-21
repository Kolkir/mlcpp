cmake_minimum_required(VERSION 3.10)
project(rcnn)

set(LIBRARIES_DIR "" CACHE PATH "path to mxnet install dir")

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-int-in-bool-context")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

message(STATUS "mxnet install path: ${LIBRARIES_DIR}")

include_directories(${CMAKE_SOURCE_DIR}/../third_party/eigen)
include_directories(${CMAKE_SOURCE_DIR}/../third_party/json/include)

include_directories(${LIBRARIES_DIR}/include)
include_directories(${LIBRARIES_DIR}/include/mxnet)
include_directories(${LIBRARIES_DIR}/include/dcml)

link_directories(${LIBRARIES_DIR}/lib)
link_directories(${LIBRARIES_DIR}/lib64)

find_package(OpenCV REQUIRED)
find_package(BLAS REQUIRED)

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs "mxnet")
list(APPEND requiredlibs "mkldnn")

set(SOURCES
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
    README.md
    get_model.sh
    )

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${requiredlibs})
target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES} ${OpenCV_LIBS})

