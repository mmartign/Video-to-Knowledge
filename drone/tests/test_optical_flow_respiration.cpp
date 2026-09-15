// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for video_core/optical_flow_respiration.h.
#include "../video_core/optical_flow_respiration.h"

#include <catch2/catch_test_macros.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

using namespace drone::video_core;

namespace {

// A bright rectangle on a dark background, oscillating vertically --
// standing in for chest rise/fall -- at `freqHz`, sampled at
// `frameRateHz` for `numFrames` frames.
std::vector<cv::Mat> makeBreathingFrames(double freqHz, double frameRateHz, int numFrames)
{
    std::vector<cv::Mat> frames;
    frames.reserve(static_cast<size_t>(numFrames));

    constexpr int width = 160, height = 120;
    constexpr int rectWidth = 100, rectHeight = 40;
    constexpr double amplitudePx = 6.0;
    constexpr int centerY = 60;

    for (int i = 0; i < numFrames; ++i) {
        cv::Mat frame(height, width, CV_8UC1, cv::Scalar(20));
        const double t = static_cast<double>(i) / frameRateHz;
        const int yOffset = static_cast<int>(std::round(amplitudePx * std::sin(2.0 * M_PI * freqHz * t)));
        const int y = centerY + yOffset - rectHeight / 2;
        cv::rectangle(
            frame, cv::Rect((width - rectWidth) / 2, y, rectWidth, rectHeight),
            cv::Scalar(220), cv::FILLED);
        frames.push_back(frame);
    }
    return frames;
}

}  // namespace

TEST_CASE("estimateRespirationFromOpticalFlow recovers a known breathing rate", "[optical_flow_respiration]")
{
    constexpr double freqHz = 0.3;  // 18 breaths/min
    constexpr double frameRateHz = 15.0;
    const auto frames = makeBreathingFrames(freqHz, frameRateHz, 300);

    const cv::Rect torsoRoi(10, 20, 140, 80);  // generously covers the moving rectangle
    const auto result = estimateRespirationFromOpticalFlow(frames, torsoRoi, frameRateHz);

    REQUIRE(result.valid);
    REQUIRE(std::fabs(result.ratePerMinute - freqHz * 60.0) < 3.0);
}

TEST_CASE("estimateRespirationFromOpticalFlow is invalid for a static scene", "[optical_flow_respiration]")
{
    std::vector<cv::Mat> frames;
    for (int i = 0; i < 100; ++i) {
        cv::Mat frame(120, 160, CV_8UC1, cv::Scalar(20));
        cv::rectangle(frame, cv::Rect(30, 20, 100, 40), cv::Scalar(220), cv::FILLED);
        frames.push_back(frame);
    }

    const cv::Rect torsoRoi(10, 10, 140, 80);
    const auto result = estimateRespirationFromOpticalFlow(frames, torsoRoi, 15.0);
    REQUIRE_FALSE(result.valid);
}

TEST_CASE("extractThoraxDisplacementSignal returns empty for fewer than 2 frames", "[optical_flow_respiration]")
{
    std::vector<cv::Mat> frames = {cv::Mat(120, 160, CV_8UC1, cv::Scalar(20))};
    const auto signal = extractThoraxDisplacementSignal(frames, cv::Rect(0, 0, 100, 80));
    REQUIRE(signal.empty());
}

TEST_CASE("extractThoraxDisplacementSignal returns empty for an out-of-bounds ROI", "[optical_flow_respiration]")
{
    std::vector<cv::Mat> frames = {
        cv::Mat(120, 160, CV_8UC1, cv::Scalar(20)),
        cv::Mat(120, 160, CV_8UC1, cv::Scalar(20)),
    };
    const auto signal = extractThoraxDisplacementSignal(frames, cv::Rect(500, 500, 10, 10));
    REQUIRE(signal.empty());
}
