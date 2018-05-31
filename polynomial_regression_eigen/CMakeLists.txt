cmake_minimum_required(VERSION 3.10)
project(polynomial-regression-eigen)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++1z -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas -Wno-parentheses -Wno-unused-local-typedefs -Wno-int-in-bool-context")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")


include_directories(${CMAKE_SOURCE_DIR}/../third_party/fast-cpp-csv-parser)
include_directories(${CMAKE_SOURCE_DIR}/../third_party/plotcpp)
include_directories(${CMAKE_SOURCE_DIR}/../third_party/eigen)

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs "lapack")

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

add_executable(${PROJECT_NAME} ${COMMON_SOURCES} "poly_reg_eigen.cpp")
target_link_libraries(${PROJECT_NAME} ${requiredlibs})

