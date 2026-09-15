// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Copyright (c) 2026 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
#include "thermal_respiration.h"

#include "optical_flow_respiration.h"  // kMin/MaxRespirationRateBpm

namespace drone::video_core {

std::vector<double> extractNaresThermalSignal(
    const std::vector<cv::Mat>& thermalFrames,
    const cv::Rect& naresRoi)
{
    std::vector<double> signal;
    if (thermalFrames.empty()) {
        return signal;
    }
    signal.reserve(thermalFrames.size());

    const cv::Rect frameBounds(0, 0, thermalFrames.front().cols, thermalFrames.front().rows);
    const cv::Rect roi = naresRoi & frameBounds;
    if (roi.area() <= 0) {
        return signal;
    }

    for (const auto& frame : thermalFrames) {
        // cv::mean() handles any single-channel depth (8U/16U/32F),
        // which is why it's used here rather than manual per-pixel
        // access -- radiometric thermal frames commonly aren't 8-bit.
        signal.push_back(cv::mean(frame(roi))[0]);
    }

    return signal;
}

PeriodicityEstimate estimateRespirationFromThermal(
    const std::vector<cv::Mat>& thermalFrames,
    const cv::Rect& naresRoi,
    double frameRateHz)
{
    const auto signal = extractNaresThermalSignal(thermalFrames, naresRoi);
    return estimatePeriodicity(
        signal, frameRateHz,
        kMinRespirationRateBpm / 60.0, kMaxRespirationRateBpm / 60.0);
}

}  // namespace drone::video_core
