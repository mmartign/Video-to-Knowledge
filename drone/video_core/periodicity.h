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
// A dependency-light (no OpenCV) periodicity estimator shared by every
// vital sign that reduces to "how fast does this 1D signal repeat":
// chest-displacement respiration, nares thermal respiration, and facial
// rPPG heart rate. Normalized autocorrelation over a plausible rate
// range -- simple, robust to noise and to non-sinusoidal waveforms
// (unlike picking the single dominant FFT bin), and its peak strength
// doubles as a signal-quality estimate, which the pipeline needs anyway
// to decide when to fall back to the thermal gradient.
#pragma once

#include <vector>

namespace drone::video_core {

struct PeriodicityEstimate {
    double rateHz = 0.0;
    double ratePerMinute = 0.0;

    // Normalized autocorrelation at the detected lag: 1.0 is perfectly
    // periodic, near 0 is indistinguishable from noise. Also usable as
    // a general signal-quality proxy.
    double quality = 0.0;

    bool valid = false;
};

// Estimates the dominant periodicity of `samples` (uniformly sampled at
// `sampleRateHz`) within [minRateHz, maxRateHz] Hz. `valid` is false if
// the signal is too short to search the requested range, is DC/flat, or
// its best autocorrelation peak doesn't reach `minQuality`.
PeriodicityEstimate estimatePeriodicity(
    const std::vector<double>& samples,
    double sampleRateHz,
    double minRateHz,
    double maxRateHz,
    double minQuality = 0.3);

}  // namespace drone::video_core
