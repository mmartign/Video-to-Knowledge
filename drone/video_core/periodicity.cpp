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
#include "periodicity.h"

#include <algorithm>
#include <cmath>

namespace drone::video_core {

PeriodicityEstimate estimatePeriodicity(
    const std::vector<double>& samples,
    double sampleRateHz,
    double minRateHz,
    double maxRateHz,
    double minQuality)
{
    PeriodicityEstimate result;
    if (samples.size() < 2 || sampleRateHz <= 0.0 || minRateHz <= 0.0 || maxRateHz <= minRateHz) {
        return result;
    }

    double mean = 0.0;
    for (double s : samples) {
        mean += s;
    }
    mean /= static_cast<double>(samples.size());

    std::vector<double> centered(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        centered[i] = samples[i] - mean;
    }

    const int n = static_cast<int>(centered.size());
    int minLag = static_cast<int>(std::round(sampleRateHz / maxRateHz));
    int maxLag = static_cast<int>(std::round(sampleRateHz / minRateHz));
    minLag = std::max(minLag, 1);
    // Cap at n/2: below this, the overlap [0, n-lag) has fewer samples
    // than the lag itself, i.e. less than one extra cycle of data
    // beyond the one being matched. Below that point a handful of
    // samples can trivially reach a high normalized correlation by
    // chance, not because the signal is genuinely periodic at that lag
    // -- reliably detecting a period needs at least ~2 cycles in the
    // window.
    maxLag = std::min(maxLag, n / 2);
    if (minLag > maxLag) {
        return result;  // signal too short for this rate range
    }

    // Normalized cross-correlation between the signal and its own
    // lagged copy, both restricted to the overlapping range [0, n-lag).
    // Normalizing by that overlap's own energy (rather than the full
    // signal's) keeps `quality` a proper, lag-independent correlation
    // coefficient in [-1, 1] -- normalizing by full-signal energy
    // instead would systematically under-report quality at larger lags
    // (lower rates) purely because less of the signal overlaps with
    // itself there, not because the signal is actually less periodic.
    int bestLag = -1;
    double bestCorr = -1.0;
    for (int lag = minLag; lag <= maxLag; ++lag) {
        double sum = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i + lag < n; ++i) {
            const double a = centered[static_cast<size_t>(i)];
            const double b = centered[static_cast<size_t>(i + lag)];
            sum += a * b;
            normA += a * a;
            normB += b * b;
        }
        if (normA <= 0.0 || normB <= 0.0) {
            continue;  // flat over this overlap window
        }
        const double normalized = sum / std::sqrt(normA * normB);
        if (normalized > bestCorr) {
            bestCorr = normalized;
            bestLag = lag;
        }
    }

    if (bestLag < 0 || bestCorr < minQuality) {
        return result;
    }

    result.rateHz = sampleRateHz / static_cast<double>(bestLag);
    result.ratePerMinute = result.rateHz * 60.0;
    result.quality = bestCorr;
    result.valid = true;
    return result;
}

}  // namespace drone::video_core
