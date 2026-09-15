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
#include "thermal_gradient.h"

namespace drone::video_core {

double computeCoreToPeripheryGradient(
    const cv::Mat& thermalFrame,
    const cv::Rect& coreRoi,
    const cv::Rect& peripheryRoi)
{
    const cv::Rect frameBounds(0, 0, thermalFrame.cols, thermalFrame.rows);
    const cv::Rect core = coreRoi & frameBounds;
    const cv::Rect periphery = peripheryRoi & frameBounds;
    if (core.area() <= 0 || periphery.area() <= 0) {
        return 0.0;
    }

    const double coreMean = cv::mean(thermalFrame(core))[0];
    const double peripheryMean = cv::mean(thermalFrame(periphery))[0];
    return coreMean - peripheryMean;
}

}  // namespace drone::video_core
