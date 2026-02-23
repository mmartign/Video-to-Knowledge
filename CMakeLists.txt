cmake_minimum_required(VERSION 3.16)

project(VideoToKnowledge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui imgcodecs)
find_package(CURL REQUIRED)

add_executable(list_cams.exe list_cams.cpp)
target_link_libraries(list_cams.exe PRIVATE
  opencv_core
  opencv_imgproc
  opencv_videoio
  opencv_highgui
)

add_executable(realtime_video_pipeline.exe realtime_video_pipeline.cpp)

# Keep compatibility with the previous Makefile include path for openai-cpp headers.
set(OPENAI_CPP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../openai-cpp/include/openai")
if(EXISTS "${OPENAI_CPP_INCLUDE_DIR}")
  target_include_directories(realtime_video_pipeline.exe PRIVATE "${OPENAI_CPP_INCLUDE_DIR}")
endif()

target_link_libraries(realtime_video_pipeline.exe PRIVATE
  opencv_core
  opencv_imgproc
  opencv_videoio
  opencv_highgui
  opencv_imgcodecs
  CURL::libcurl
)
