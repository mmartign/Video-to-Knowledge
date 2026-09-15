// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for video_core/photoplethysmography.h.
#include "../video_core/photoplethysmography.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace drone::video_core;

namespace {

std::vector<cv::Mat> makePulsatileFaceFrames(
    double freqHz, double frameRateHz, int numFrames, double dc, double amplitude)
{
    std::vector<cv::Mat> frames;
    frames.reserve(static_cast<size_t>(numFrames));
    for (int i = 0; i < numFrames; ++i) {
        const double t = static_cast<double>(i) / frameRateHz;
        const double green = dc + amplitude * std::sin(2.0 * M_PI * freqHz * t);
        // BGR: blue/red held constant, only green pulses -- rPPG's
        // premise is that green carries the plethysmographic signal.
        frames.emplace_back(30, 30, CV_8UC3, cv::Scalar(100, green, 100));
    }
    return frames;
}

}  // namespace

TEST_CASE("estimateHeartRateAndPerfusion recovers a known heart rate and a plausible perfusion index", "[rppg]")
{
    constexpr double freqHz = 1.2;  // 72 bpm
    constexpr double frameRateHz = 30.0;
    const auto frames = makePulsatileFaceFrames(freqHz, frameRateHz, 300, /*dc=*/150.0, /*amplitude=*/6.0);

    const cv::Rect faceRoi(5, 5, 20, 20);
    const auto result = estimateHeartRateAndPerfusion(frames, faceRoi, frameRateHz);

    REQUIRE(result.periodicity.valid);
    REQUIRE(std::fabs(result.periodicity.ratePerMinute - freqHz * 60.0) < 5.0);

    REQUIRE(result.perfusionIndexValid);
    REQUIRE(result.perfusionIndex > 0.0);
    REQUIRE(result.perfusionIndex < 1.0);  // sane range for a small pulsatile fraction of DC
}

TEST_CASE("estimateHeartRateAndPerfusion is invalid for a non-pulsatile face", "[rppg]")
{
    std::vector<cv::Mat> frames;
    for (int i = 0; i < 150; ++i) {
        frames.emplace_back(30, 30, CV_8UC3, cv::Scalar(100, 150, 100));  // constant, no pulse
    }

    const cv::Rect faceRoi(5, 5, 20, 20);
    const auto result = estimateHeartRateAndPerfusion(frames, faceRoi, 30.0);
    REQUIRE_FALSE(result.periodicity.valid);
}

TEST_CASE("extractFaceGreenChannelSignal returns empty for an out-of-bounds ROI", "[rppg]")
{
    std::vector<cv::Mat> frames = {cv::Mat(30, 30, CV_8UC3, cv::Scalar(100, 150, 100))};
    const auto signal = extractFaceGreenChannelSignal(frames, cv::Rect(500, 500, 10, 10));
    REQUIRE(signal.empty());
}
