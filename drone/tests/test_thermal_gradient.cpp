// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for video_core/thermal_gradient.h.
#include "../video_core/thermal_gradient.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace drone::video_core;

TEST_CASE("computeCoreToPeripheryGradient measures the intensity difference between two ROIs", "[thermal_gradient]")
{
    cv::Mat frame(100, 100, CV_8UC1, cv::Scalar(0));
    frame(cv::Rect(10, 10, 20, 20)).setTo(cv::Scalar(200));  // "core", warm
    frame(cv::Rect(60, 60, 20, 20)).setTo(cv::Scalar(150));  // "periphery", cooler

    const double gradient = computeCoreToPeripheryGradient(
        frame, cv::Rect(10, 10, 20, 20), cv::Rect(60, 60, 20, 20));

    REQUIRE(gradient == Catch::Approx(50.0));
}

TEST_CASE("computeCoreToPeripheryGradient can be negative when periphery reads warmer", "[thermal_gradient]")
{
    cv::Mat frame(100, 100, CV_8UC1, cv::Scalar(0));
    frame(cv::Rect(10, 10, 20, 20)).setTo(cv::Scalar(100));
    frame(cv::Rect(60, 60, 20, 20)).setTo(cv::Scalar(180));

    const double gradient = computeCoreToPeripheryGradient(
        frame, cv::Rect(10, 10, 20, 20), cv::Rect(60, 60, 20, 20));

    REQUIRE(gradient == Catch::Approx(-80.0));
}

TEST_CASE("computeCoreToPeripheryGradient returns 0 when either ROI is out of bounds", "[thermal_gradient]")
{
    cv::Mat frame(100, 100, CV_8UC1, cv::Scalar(128));

    REQUIRE(computeCoreToPeripheryGradient(
                frame, cv::Rect(500, 500, 10, 10), cv::Rect(60, 60, 20, 20)) == 0.0);
    REQUIRE(computeCoreToPeripheryGradient(
                frame, cv::Rect(10, 10, 20, 20), cv::Rect(500, 500, 10, 10)) == 0.0);
}
