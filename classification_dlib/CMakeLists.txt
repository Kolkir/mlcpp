cmake_minimum_required(VERSION 3.10)
project(classify)

set(LIBRARIES_DIR "" CACHE PATH "path to dlib install dir")

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

message(STATUS "dlib install path: ${LIBRARIES_DIR}")

include_directories(${LIBRARIES_DIR}/include)
link_directories(${LIBRARIES_DIR}/lib)
link_directories(${LIBRARIES_DIR}/lib64)

include_directories(${CMAKE_SOURCE_DIR}/../third_party/plotcpp)

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs "cblas")
list(APPEND requiredlibs "lapack")

find_package(CURL REQUIRED)
list(APPEND requiredlibs ${CURL_LIBRARIES})

find_package(CUDA REQUIRED)
list(APPEND requiredlibs ${CUDA_LIBRARIES})
list(APPEND requiredlibs ${CUDA_CUBLAS_LIBRARIES})
list(APPEND requiredlibs ${CUDA_cusolver_LIBRARY})
list(APPEND requiredlibs ${CUDA_curand_LIBRARY})

get_filename_component(cudnn_hint_path "${CUDA_CUBLAS_LIBRARIES}" PATH)
find_library(cudnn cudnn
    HINTS ${cudnn_hint_path} ENV CUDNN_LIBRARY_DIR  ENV CUDNN_HOME
    PATHS /usr/local /usr/local/cuda ENV LD_LIBRARY_PATH
    PATH_SUFFIXES lib64 lib x64
    )
message("CUDNN found ${cudnn}")
list(APPEND requiredlibs ${cudnn})

set(COMMON_SOURCES "../ioutils.h"
                   "../utils.h"
                   "../utils.cpp")

add_executable(${PROJECT_NAME} ${COMMON_SOURCES} "classify_dlib.cpp")
target_link_libraries(${PROJECT_NAME} optimized dlib debug dlibd)
target_link_libraries(${PROJECT_NAME} ${requiredlibs})


