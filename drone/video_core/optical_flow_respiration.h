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
// Respiratory rate from "optical-flow displacement of the thorax"
// (wide/visible camera): dense optical flow between consecutive frames,
// reduced to the mean vertical displacement within the torso ROI --
// chest rise/fall is a roughly periodic vertical motion -- then handed
// to periodicity.h to recover a rate.
#pragma once

#include "periodicity.h"

#include <opencv2/core.hpp>

#include <vector>

namespace drone::video_core {

// One value per consecutive frame pair (so `grayFrames.size() - 1`
// values): the mean vertical (y-axis) dense-optical-flow displacement
// within `torsoRoi`, in pixels between frames. Frames must be
// single-channel (grayscale) and the same size; `torsoRoi` is clamped
// to the frame bounds.
std::vector<double> extractThoraxDisplacementSignal(
    const std::vector<cv::Mat>& grayFrames,
    const cv::Rect& torsoRoi);

// Clinically plausible respiratory rate range this estimator searches:
// 4-60 breaths/min covers everything from severe bradypnea to marked
// tachypnea.
constexpr double kMinRespirationRateBpm = 4.0;
constexpr double kMaxRespirationRateBpm = 60.0;

// Extracts the thorax displacement signal and estimates its
// periodicity within the respiratory rate range above.
PeriodicityEstimate estimateRespirationFromOpticalFlow(
    const std::vector<cv::Mat>& grayFrames,
    const cv::Rect& torsoRoi,
    double frameRateHz);

}  // namespace drone::video_core
