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
#include "photoplethysmography.h"

#include "biquad.h"

#include <algorithm>
#include <cmath>

namespace drone::video_core {

std::vector<double> extractFaceGreenChannelSignal(
    const std::vector<cv::Mat>& colorFrames,
    const cv::Rect& faceRoi)
{
    std::vector<double> signal;
    if (colorFrames.empty()) {
        return signal;
    }
    signal.reserve(colorFrames.size());

    const cv::Rect frameBounds(0, 0, colorFrames.front().cols, colorFrames.front().rows);
    const cv::Rect roi = faceRoi & frameBounds;
    if (roi.area() <= 0) {
        return signal;
    }

    for (const auto& frame : colorFrames) {
        // BGR order: index 1 is green.
        signal.push_back(cv::mean(frame(roi))[1]);
    }
    return signal;
}

RppgResult estimateHeartRateAndPerfusion(
    const std::vector<cv::Mat>& colorFrames,
    const cv::Rect& faceRoi,
    double frameRateHz)
{
    RppgResult result;

    const auto raw = extractFaceGreenChannelSignal(colorFrames, faceRoi);
    if (raw.size() < 2 || frameRateHz <= 0.0) {
        return result;
    }

    double dc = 0.0;
    for (double v : raw) {
        dc += v;
    }
    dc /= static_cast<double>(raw.size());

    // Band-pass to the plausible heart-rate range to isolate the
    // pulsatile (AC) component from the slowly-varying baseline (DC)
    // brightness.
    Biquad highpass = Biquad::makeHighpass(kMinHeartRateBpm / 60.0, frameRateHz, 0.7071);
    Biquad lowpass = Biquad::makeLowpass(kMaxHeartRateBpm / 60.0, frameRateHz, 0.7071);

    std::vector<double> acFull(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        acFull[i] = lowpass.process(highpass.process(raw[i]));
    }

    // Drop the filter's initial transient (settling in from zero state
    // to the real signal produces a decaying, non-periodic response
    // that both skews the RMS and can itself look spuriously
    // "periodic" to the autocorrelation-based estimator) before either
    // measuring AC amplitude or estimating periodicity -- both need the
    // same steady-state-only signal. Empirically, a quarter of the
    // buffer wasn't enough for this filter's cutoffs to settle (a
    // meaningfully-sized transient, not just floating-point residue,
    // was still leaking past that point); half is.
    const size_t skip = std::min(acFull.size(), acFull.size() / 2);
    const std::vector<double> ac(acFull.begin() + static_cast<long>(skip), acFull.end());

    double sumSq = 0.0;
    for (double v : ac) {
        sumSq += v * v;
    }
    const double acRms = !ac.empty() ? std::sqrt(sumSq / static_cast<double>(ac.size())) : 0.0;

    if (dc > 1e-6) {
        result.perfusionIndex = acRms / dc;
        result.perfusionIndexValid = true;
    }

    // estimatePeriodicity()'s normalized correlation deliberately
    // ignores absolute signal magnitude, so on a genuinely non-
    // pulsatile face it can't tell a real cardiac signal from residual
    // filter transient left over from a perfectly flat input -- even
    // past the skip above, that residual can still be tens of a percent
    // of a real pulsatile amplitude, and it decays in a way that looks
    // "periodic" to an autocorrelation search once amplitude is
    // normalized away. That check belongs here, not in the generic
    // periodicity estimator: what counts as negligible is specific to
    // this caller's pixel-intensity units. An AC RMS well under one
    // 8-bit intensity level isn't a measurable pulsatile signal.
    constexpr double kMinMeasurableAcRms = 0.05;
    if (acRms >= kMinMeasurableAcRms) {
        result.periodicity = estimatePeriodicity(
            ac, frameRateHz, kMinHeartRateBpm / 60.0, kMaxHeartRateBpm / 60.0);
    }

    return result;
}

}  // namespace drone::video_core
