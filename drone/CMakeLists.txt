# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of the Spazio IT Video-to-Knowledge project.
#
# Self-contained build for the drone video-to-knowledge subsystem.
# Deliberately its own CMake project (not part of the root
# CMakeLists.txt): this targets a Jetson/aarch64 companion computer and
# is meant to be independently buildable/deployable. Build with:
#   cmake -S drone -B drone/build && cmake --build drone/build
#   ctest --test-dir drone/build

cmake_minimum_required(VERSION 3.16)

project(DroneVideoToKnowledge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DRONE_BUILD_TESTS "Build the video_core unit test suite" ON)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio objdetect)

# Dependency-light-ish logic (periodicity estimation, optical-flow and
# thermal respiration signals, rPPG, thermal-gradient fallback,
# casualty/face detection) covered by unit tests using synthetic
# in-memory frames -- no camera, gimbal, or video files needed to run
# them. Depends on OpenCV (unavoidable for image/video processing) but
# not on the network/hardware-facing io/ layer below.
add_library(video_core STATIC
  video_core/biquad.cpp
  video_core/biquad.h
  video_core/casualty_detector.cpp
  video_core/casualty_detector.h
  video_core/optical_flow_respiration.cpp
  video_core/optical_flow_respiration.h
  video_core/periodicity.cpp
  video_core/periodicity.h
  video_core/photoplethysmography.cpp
  video_core/photoplethysmography.h
  video_core/respiration_fusion.cpp
  video_core/respiration_fusion.h
  video_core/thermal_gradient.cpp
  video_core/thermal_gradient.h
  video_core/thermal_respiration.cpp
  video_core/thermal_respiration.h
  video_core/vital_signs.h
)
target_include_directories(video_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(video_core PUBLIC
  opencv_core
  opencv_imgproc
  opencv_video
  opencv_objdetect
)

# The gimbal capture interface and its video-file-replay backend:
# hardware-adjacent but not SDK/network-dependent, so still testable
# with real (if synthetic) video file fixtures.
add_library(drone_io STATIC
  io/gimbal_capture.h
  io/video_replay_capture.cpp
  io/video_replay_capture.h
)
target_link_libraries(drone_io PUBLIC video_core opencv_videoio)

add_executable(drone_video_to_knowledge
  main_video_to_knowledge.cpp
)
target_link_libraries(drone_video_to_knowledge PRIVATE
  drone_io
  video_core
  nlohmann_json::nlohmann_json
)

if(DRONE_BUILD_TESTS)
  enable_testing()

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(video_core_tests
    tests/test_casualty_detector.cpp
    tests/test_optical_flow_respiration.cpp
    tests/test_periodicity.cpp
    tests/test_photoplethysmography.cpp
    tests/test_respiration_fusion.cpp
    tests/test_thermal_gradient.cpp
    tests/test_thermal_respiration.cpp
    tests/test_video_replay_capture.cpp
  )
  target_link_libraries(video_core_tests PRIVATE
    drone_io
    video_core
    Catch2::Catch2WithMain
  )

  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  include(Catch)
  catch_discover_tests(video_core_tests)
endif()
