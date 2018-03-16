cmake_minimum_required(VERSION 2.8)
project(linear-regression)

set(CMAKE_VERBOSE_MAKEFILE ON)

SET(requiredlibs "stdc++fs")

FIND_PACKAGE(CURL)
IF(CURL_FOUND)
  INCLUDE_DIRECTORIES(${CURL_INCLUDE_DIR})
  SET(requiredlibs ${requiredlibs} ${CURL_LIBRARIES} )
ELSE(CURL_FOUND)
  MESSAGE(FATAL_ERROR "Could not find the CURL library and development files.")
ENDIF(CURL_FOUND)

set(SOURCES "linear_reg.cpp"
            "../utils.h"
            "../utils.cpp"
            "../iotensor.h"
)

add_executable(linear-regression ${SOURCES})
target_link_libraries(linear-regression ${requiredlibs})

