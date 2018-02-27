cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_cc_inference)

## Options
add_compile_options(-std=c++11 -Wall -O3)
include_directories(
  include
)

## Declare a library
add_library(${PROJECT_NAME} SHARED
  lib/Inference.cc
  include/${PROJECT_NAME}/Inference.h
)

## Link library against dependencies
target_link_libraries(${PROJECT_NAME}
  tensorflow
)


## Examples
add_executable(multiply_with_two example/multiply_with_two.cc)
target_link_libraries( multiply_with_two
  ${PROJECT_NAME}
)

## Test
include(gtest.cmake)
find_package(Threads)

add_executable(testrunner test/test_tensorflow_cc_inference.cc)
target_link_libraries(testrunner
  gtest
  ${CMAKE_THREAD_LIBS_INIT}
  ${PROJECT_NAME}
)
