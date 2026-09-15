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
// Respiratory rate from "thermal intensity change at the nares":
// exhaled breath is warmer (or cooler, depending on ambient
// temperature) than the surrounding skin, so the mean thermal
// intensity in a nares ROI oscillates roughly periodically with
// breathing -- reduced to a signal and handed to periodicity.h, the
// same way as the optical-flow estimate in optical_flow_respiration.h.
#pragma once

#include "periodicity.h"

#include <opencv2/core.hpp>

#include <vector>

namespace drone::video_core {

// One value per frame: the mean intensity within `naresRoi` of each
// single-channel thermal frame (radiometric or intensity-mapped --
// units are abstracted away, only relative change over time matters
// here). `naresRoi` is clamped to the frame bounds.
std::vector<double> extractNaresThermalSignal(
    const std::vector<cv::Mat>& thermalFrames,
    const cv::Rect& naresRoi);

// Extracts the nares thermal signal and estimates its periodicity
// within the same respiratory rate range as the optical-flow estimator
// (see optical_flow_respiration.h's kMin/MaxRespirationRateBpm).
PeriodicityEstimate estimateRespirationFromThermal(
    const std::vector<cv::Mat>& thermalFrames,
    const cv::Rect& naresRoi,
    double frameRateHz);

}  // namespace drone::video_core
