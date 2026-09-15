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
#include "respiration_fusion.h"

#include <cmath>

namespace drone::video_core {

namespace {

// Beyond this much disagreement, the two estimates are more likely
// tracking different things (e.g. motion artifact vs. an actual
// breath) than sampling noise around the same breath -- averaging them
// would produce a number that matches neither, so the higher-quality
// one is trusted instead.
constexpr double kMaxAgreementBpm = 6.0;

}  // namespace

PeriodicityEstimate fuseRespirationEstimates(
    const PeriodicityEstimate& fromOpticalFlow,
    const PeriodicityEstimate& fromThermal)
{
    if (!fromOpticalFlow.valid && !fromThermal.valid) {
        return PeriodicityEstimate{};
    }
    if (fromOpticalFlow.valid && !fromThermal.valid) {
        return fromOpticalFlow;
    }
    if (!fromOpticalFlow.valid && fromThermal.valid) {
        return fromThermal;
    }

    const double disagreement =
        std::fabs(fromOpticalFlow.ratePerMinute - fromThermal.ratePerMinute);
    if (disagreement > kMaxAgreementBpm) {
        return fromOpticalFlow.quality >= fromThermal.quality ? fromOpticalFlow : fromThermal;
    }

    const double qSum = fromOpticalFlow.quality + fromThermal.quality;
    PeriodicityEstimate fused;
    fused.valid = true;
    fused.quality = 0.5 * (fromOpticalFlow.quality + fromThermal.quality);
    if (qSum > 0.0) {
        fused.ratePerMinute =
            (fromOpticalFlow.quality * fromOpticalFlow.ratePerMinute +
             fromThermal.quality * fromThermal.ratePerMinute) / qSum;
    } else {
        fused.ratePerMinute = 0.5 * (fromOpticalFlow.ratePerMinute + fromThermal.ratePerMinute);
    }
    fused.rateHz = fused.ratePerMinute / 60.0;
    return fused;
}

}  // namespace drone::video_core
