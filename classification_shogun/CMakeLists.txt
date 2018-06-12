cmake_minimum_required(VERSION 3.10)
project(classify)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++1z -msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")


include_directories(${LIBRARIES_DIR})
include_directories(${CMAKE_SOURCE_DIR}/../third_party/plotcpp)

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")


set(COMMON_SOURCES "../utils.h"
                   "../utils.cpp"
                   "../ioutils.h"
)

add_executable(${PROJECT_NAME} ${COMMON_SOURCES} "classify_shogun.cpp")
target_link_libraries(${PROJECT_NAME} ${requiredlibs})

