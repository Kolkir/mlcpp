cmake_minimum_required(VERSION 3.10)
project(polynomial-regression-gpu)

set(CMAKE_C_COMPILER gcc-6)
set(CMAKE_CXX_COMPILER g++-6)

find_package(CUDA REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CUDA_NVCC_FLAGS "-std=c++14;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wall;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wextra;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-unused-parameter;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-unknown-pragmas;")

set(CUDA_NVCC_FLAGS_RELEASE "-O3;-DNDEBUG;")
set(CUDA_NVCC_FLAGS_DEBUG "-g;-O0;")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set(CMAKE_CXX_FLAGS "-std=c++14 -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${CMAKE_SOURCE_DIR}/../third_party/mshadow)
include_directories(${CMAKE_SOURCE_DIR}/../third_party/fast-cpp-csv-parser)
include_directories(${CMAKE_SOURCE_DIR}/../third_party/plotcpp)

SET(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")

FIND_PACKAGE(CURL)
IF(CURL_FOUND)
  INCLUDE_DIRECTORIES(${CURL_INCLUDE_DIR})
  SET(requiredlibs ${requiredlibs} ${CURL_LIBRARIES} )
ELSE(CURL_FOUND)
  MESSAGE(FATAL_ERROR "Could not find the CURL library and development files.")
ENDIF(CURL_FOUND)

set(COMMON_SOURCES "../utils.h"
                   "../utils.cpp"
                   "../ioutils.h"
)

SET(requiredlibs ${requiredlibs} cblas)

CUDA_ADD_EXECUTABLE(polynomial-regression-gpu ${COMMON_SOURCES}
                                              poly_reg_gpu.cu
                                              standardizer.h
                                              polynomial.h
                                              common.h
                                              optimizer.h)


target_link_libraries(polynomial-regression-gpu ${requiredlibs} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
target_compile_definitions(polynomial-regression-gpu PRIVATE
                                                 MSHADOW_USE_CBLAS=1
                                                 MSHADOW_USE_CUDA=1
                                                 MSHADOW_USE_CUDNN=0
                                                 MSHADOW_USE_CUBLAS=1
                                                 MSHADOW_USE_MKL=0
                                                 MSHADOW_DIST_PS=0
                                                 MSHADOW_USE_PASCAL=0)
target_include_directories(polynomial-regression-gpu PRIVATE
                                                     ${CUDA_INCLUDE_DIRS})
