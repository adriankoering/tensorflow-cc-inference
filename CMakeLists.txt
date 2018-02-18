cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_inference)

## Options
add_compile_options(-std=c++11 -Wall -O3)
include_directories(
  include
)

## Declare a library
add_library(${PROJECT_NAME}
  lib/${PROJECT_NAME}.cc
  include/${PROJECT_NAME}/${PROJECT_NAME}.h
)

## Link library against dependencies
target_link_libraries(${PROJECT_NAME}
  tensorflow
)

## Testing

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_tensorflow-cc-inference.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()
