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
// "When optical signal quality is degraded by dust or vasoconstriction,
// the pipeline falls back to the core-to-periphery thermal gradient"
// for the perfusion index. A poorly perfused casualty (shock, severe
// vasoconstriction) loses the normal warm-core/cool-periphery
// temperature step, so a shallower gradient stands in for a lower
// perfusion index when facial rPPG can't be trusted.
#pragma once

#include <opencv2/core.hpp>

namespace drone::video_core {

// Mean thermal intensity of `coreRoi` minus `peripheryRoi` in a single-
// channel thermal frame (radiometric or intensity-mapped; units are
// abstracted away -- see thermal_respiration.h). Both ROIs are clamped
// to the frame bounds independently; returns 0.0 if either ends up
// empty. This is a relative, uncalibrated proxy, not a clinical
// perfusion index in the same units as the rPPG-derived AC/DC ratio --
// see the header comment in vital_signs.h.
double computeCoreToPeripheryGradient(
    const cv::Mat& thermalFrame,
    const cv::Rect& coreRoi,
    const cv::Rect& peripheryRoi);

}  // namespace drone::video_core
