// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for video_core/thermal_respiration.h.
#include "../video_core/thermal_respiration.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace drone::video_core;

namespace {

// Uniform frames whose intensity oscillates sinusoidally over time,
// standing in for exhaled-breath thermal modulation at the nares.
std::vector<cv::Mat> makeThermalBreathingFrames(double freqHz, double frameRateHz, int numFrames)
{
    std::vector<cv::Mat> frames;
    frames.reserve(static_cast<size_t>(numFrames));
    for (int i = 0; i < numFrames; ++i) {
        const double t = static_cast<double>(i) / frameRateHz;
        const double value = 128.0 + 20.0 * std::sin(2.0 * M_PI * freqHz * t);
        frames.emplace_back(40, 40, CV_8UC1, cv::Scalar(value));
    }
    return frames;
}

}  // namespace

TEST_CASE("estimateRespirationFromThermal recovers a known breathing rate", "[thermal_respiration]")
{
    constexpr double freqHz = 0.25;  // 15 breaths/min
    constexpr double frameRateHz = 15.0;
    const auto frames = makeThermalBreathingFrames(freqHz, frameRateHz, 300);

    const cv::Rect naresRoi(10, 10, 20, 20);
    const auto result = estimateRespirationFromThermal(frames, naresRoi, frameRateHz);

    REQUIRE(result.valid);
    REQUIRE(std::fabs(result.ratePerMinute - freqHz * 60.0) < 2.0);
}

TEST_CASE("extractNaresThermalSignal returns empty for an out-of-bounds ROI", "[thermal_respiration]")
{
    std::vector<cv::Mat> frames = {cv::Mat(40, 40, CV_8UC1, cv::Scalar(128))};
    const auto signal = extractNaresThermalSignal(frames, cv::Rect(500, 500, 10, 10));
    REQUIRE(signal.empty());
}

TEST_CASE("extractNaresThermalSignal returns empty for no frames", "[thermal_respiration]")
{
    const auto signal = extractNaresThermalSignal({}, cv::Rect(0, 0, 10, 10));
    REQUIRE(signal.empty());
}
