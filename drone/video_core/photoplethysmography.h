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
// "heart rate and a perfusion index are obtained from facial remote
// photoplethysmography on the zoom camera": the green channel carries
// the strongest plethysmographic signal (hemoglobin absorbs green
// light more than red/blue), so the mean green-channel intensity in
// the face ROI oscillates with each heartbeat. Band-pass filtering to
// the plausible heart-rate range isolates that pulsatile (AC)
// component from the slowly-varying baseline (DC) brightness;
// periodicity.h recovers the rate from the AC signal, and the AC/DC
// ratio gives a perfusion index.
#pragma once

#include "periodicity.h"

#include <opencv2/core.hpp>

#include <vector>

namespace drone::video_core {

// Clinically plausible heart-rate range this estimator searches:
// 42-210 bpm covers everything from marked bradycardia to extreme
// tachycardia.
constexpr double kMinHeartRateBpm = 42.0;
constexpr double kMaxHeartRateBpm = 210.0;

struct RppgResult {
    PeriodicityEstimate periodicity;  // .ratePerMinute is heart rate when .valid

    // AC (pulsatile amplitude, RMS of the band-passed signal) / DC
    // (mean raw intensity) ratio. Populated whenever the face ROI is
    // non-empty, independent of whether a heart rate was recoverable,
    // since it's a useful signal-quality indicator either way.
    double perfusionIndex = 0.0;
    bool perfusionIndexValid = false;
};

// Raw (unfiltered) mean green-channel signal in `faceRoi` across
// `colorFrames` (BGR), one value per frame. `faceRoi` is clamped to the
// frame bounds.
std::vector<double> extractFaceGreenChannelSignal(
    const std::vector<cv::Mat>& colorFrames,
    const cv::Rect& faceRoi);

RppgResult estimateHeartRateAndPerfusion(
    const std::vector<cv::Mat>& colorFrames,
    const cv::Rect& faceRoi,
    double frameRateHz);

}  // namespace drone::video_core
