// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for io/video_replay_capture.h. Fixture videos are written
// on the fly with cv::VideoWriter (MJPG/AVI, reliably available without
// extra codec packages) rather than checked in as binary test data.
#include "../io/video_replay_capture.h"

#include <catch2/catch_test_macros.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>

using namespace drone::io;

namespace {

class TempVideoFile {
public:
    TempVideoFile(int numFrames, int width, int height, double fps, cv::Scalar color)
        : path_(std::filesystem::temp_directory_path() /
                ("drone_video_test_" + std::to_string(counter_++) + ".avi"))
    {
        cv::VideoWriter writer(
            path_.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
            cv::Size(width, height), /*isColor=*/true);
        for (int i = 0; i < numFrames; ++i) {
            cv::Mat frame(height, width, CV_8UC3, color);
            writer.write(frame);
        }
    }

    ~TempVideoFile() { std::filesystem::remove(path_); }

    TempVideoFile(const TempVideoFile&) = delete;
    TempVideoFile& operator=(const TempVideoFile&) = delete;

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
    static inline int counter_ = 0;
};

}  // namespace

TEST_CASE("VideoReplayCapture opens three matching video files and reports the frame rate", "[video_replay_capture]")
{
    TempVideoFile wide(20, 64, 48, 15.0, cv::Scalar(200, 0, 0));
    TempVideoFile zoom(20, 64, 48, 15.0, cv::Scalar(0, 200, 0));
    TempVideoFile thermal(20, 64, 48, 15.0, cv::Scalar(0, 0, 200));

    VideoReplayCapture capture(wide.path(), zoom.path(), thermal.path());
    REQUIRE(capture.ok());
    REQUIRE(capture.frameRateHz() > 0.0);
}

TEST_CASE("VideoReplayCapture reads synchronized frames and converts thermal to grayscale", "[video_replay_capture]")
{
    TempVideoFile wide(10, 64, 48, 15.0, cv::Scalar(200, 0, 0));
    TempVideoFile zoom(10, 64, 48, 15.0, cv::Scalar(0, 200, 0));
    TempVideoFile thermal(10, 64, 48, 15.0, cv::Scalar(0, 0, 200));

    VideoReplayCapture capture(wide.path(), zoom.path(), thermal.path());
    REQUIRE(capture.ok());

    GimbalFrameSet frame;
    REQUIRE(capture.readNext(frame));
    REQUIRE_FALSE(frame.wideBgr.empty());
    REQUIRE_FALSE(frame.zoomBgr.empty());
    REQUIRE_FALSE(frame.thermal.empty());
    REQUIRE(frame.thermal.channels() == 1);
}

TEST_CASE("VideoReplayCapture stops once the shortest stream is exhausted", "[video_replay_capture]")
{
    TempVideoFile wide(5, 64, 48, 15.0, cv::Scalar(200, 0, 0));
    TempVideoFile zoom(10, 64, 48, 15.0, cv::Scalar(0, 200, 0));
    TempVideoFile thermal(10, 64, 48, 15.0, cv::Scalar(0, 0, 200));

    VideoReplayCapture capture(wide.path(), zoom.path(), thermal.path());
    REQUIRE(capture.ok());

    GimbalFrameSet frame;
    int count = 0;
    while (capture.readNext(frame)) {
        ++count;
    }
    REQUIRE(count == 5);
}

TEST_CASE("VideoReplayCapture reports not ok() when a file is missing", "[video_replay_capture]")
{
    TempVideoFile wide(5, 64, 48, 15.0, cv::Scalar(200, 0, 0));
    VideoReplayCapture capture(wide.path(), "/nonexistent/zoom.avi", "/nonexistent/thermal.avi");
    REQUIRE_FALSE(capture.ok());

    GimbalFrameSet frame;
    REQUIRE_FALSE(capture.readNext(frame));
}
