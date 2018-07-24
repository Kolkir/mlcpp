cmake_minimum_required(VERSION 3.10)
project(classify)

set(LIBRARIES_DIR "" CACHE PATH "path to dlib install dir")

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

message(STATUS "shark install path: ${LIBRARIES_DIR}")

include_directories(${LIBRARIES_DIR}/include)
link_directories(${LIBRARIES_DIR}/lib)
link_directories(${LIBRARIES_DIR}/lib64)

include_directories(${CMAKE_SOURCE_DIR}/../third_party/plotcpp)

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs "cblas")
list(APPEND requiredlibs "lapack")

find_package(CURL REQUIRED)
if(CURL_FOUND)
  set(requiredlibs ${requiredlibs} ${CURL_LIBRARIES} )
endif()

set(COMMON_SOURCES "../ioutils.h"
                   "../utils.h"
                   "../utils.cpp")

add_executable(${PROJECT_NAME} ${COMMON_SOURCES} "classify_dlib.cpp")
target_link_libraries(${PROJECT_NAME} optimized dlib debug dlibd)
target_link_libraries(${PROJECT_NAME} ${requiredlibs})


