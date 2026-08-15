# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of the Spazio IT Video-to-Knowledge project.

cmake_minimum_required(VERSION 3.16)

project(VideoToKnowledge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VTK_BUILD_TESTS "Build the pipeline_core unit test suite" ON)

include(FetchContent)

# nlohmann/json is needed by pipeline_core (API response parsing) as well as
# by the main pipeline executable, which also sees openai-cpp's own vendored
# copy of the same header (via OPENAI_CPP_INCLUDE_DIR below). Deliberately
# always fetch our own copy rather than find_package()-ing one off the host:
# nlohmann/json's ABI inline namespace is derived from its version, so a
# system package whose reported version doesn't precisely match what's
# vendored by openai-cpp silently breaks linking between translation units
# that each resolved <nlohmann/json.hpp> differently. Pinning to the same
# release openai-cpp vendors guarantees both are the byte-identical upstream
# header, so it no longer matters which physical copy either target sees.
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui imgcodecs)
find_package(CURL REQUIRED)

# Dependency-light logic (INI/CLI parsing, string/date helpers, API response
# parsing) shared by the executables and covered by unit tests. Deliberately
# has no OpenCV/CURL/openai-cpp dependency. Split by functional area under
# pipeline_core/; pipeline_core.h aggregates it for callers that want the
# full surface.
add_library(pipeline_core STATIC
  pipeline_core.h
  pipeline_core/base64.cpp
  pipeline_core/base64.h
  pipeline_core/cli.cpp
  pipeline_core/cli.h
  pipeline_core/datetime.cpp
  pipeline_core/datetime.h
  pipeline_core/ini_config.cpp
  pipeline_core/ini_config.h
  pipeline_core/json_response.cpp
  pipeline_core/json_response.h
  pipeline_core/numeric.cpp
  pipeline_core/numeric.h
  pipeline_core/strings.cpp
  pipeline_core/strings.h
)
target_include_directories(pipeline_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(pipeline_core PUBLIC nlohmann_json::nlohmann_json)

add_executable(list_cams.exe list_cams.cpp)
target_link_libraries(list_cams.exe PRIVATE
  opencv_core
  opencv_imgproc
  opencv_videoio
  opencv_highgui
)

add_executable(realtime_video_pipeline.exe realtime_video_pipeline.cpp)

# openai-cpp bundles its own copy of nlohmann/json.hpp alongside openai.hpp
# in the very same include directory (see below), so it will always be on
# this target's include path too. A target's own target_include_directories
# entries are searched before any inherited via target_link_libraries, so
# without this explicit (and earlier) entry, openai-cpp's bundled copy would
# shadow the pipeline_core one below and <nlohmann/json.hpp> would resolve
# inconsistently between translation units.
target_include_directories(realtime_video_pipeline.exe PRIVATE "${nlohmann_json_SOURCE_DIR}/include")

# Keep compatibility with the previous Makefile include path for openai-cpp headers.
set(OPENAI_CPP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../openai-cpp/include/openai")
if(EXISTS "${OPENAI_CPP_INCLUDE_DIR}")
  target_include_directories(realtime_video_pipeline.exe PRIVATE "${OPENAI_CPP_INCLUDE_DIR}")
endif()

target_link_libraries(realtime_video_pipeline.exe PRIVATE
  pipeline_core
  opencv_core
  opencv_imgproc
  opencv_videoio
  opencv_highgui
  opencv_imgcodecs
  CURL::libcurl
)

if(VTK_BUILD_TESTS)
  enable_testing()

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(pipeline_core_tests
    tests/test_utils.h
    tests/test_base64.cpp
    tests/test_cli.cpp
    tests/test_datetime.cpp
    tests/test_ini_config.cpp
    tests/test_json_response.cpp
    tests/test_numeric.cpp
    tests/test_strings.cpp
  )
  target_link_libraries(pipeline_core_tests PRIVATE pipeline_core Catch2::Catch2WithMain)

  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  include(Catch)
  catch_discover_tests(pipeline_core_tests)
endif()
