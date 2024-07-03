cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(
    CMAKE_BUILD_TYPE "Release"
    CACHE
    STRING
    "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage."
    FORCE)
endif()

cmake_policy(SET CMP0063 NEW)
cmake_policy(SET CMP0074 NEW)

project(my_ai_training C CXX)

option(MY_AI_TRAINING_BUILD_TESTS "Build my_ai_training C++ Tests" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

set(MY_AI_TRAINING_ROOT ${PROJECT_SOURCE_DIR})

if(MY_AI_TRAINING_BUILD_TESTS)
  include(CTest)
  enable_testing()
  find_package(GTest)
  if(NOT GTest_FOUND)
    list(APPEND CMAKE_MODULE_PATH ${ONNX_ROOT}/cmake/external)
    include(googletest)
  endif()
  set(googletest_STATIC_LIBRARIES GTest::gtest)
endif()

include_directories(src)

file(GLOB SRCS src/ir/*.cc)

add_library(my_ai_training_lib ${SRCS})

if (MY_AI_TRAINING_BUILD_TESTS)
  add_subdirectory(unittests #[[EXCLUDE_FROM_ALL]])
endif()