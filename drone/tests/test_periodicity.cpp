// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for video_core/periodicity.h.
#include "../video_core/periodicity.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>

using namespace drone::video_core;

namespace {

std::vector<double> makeSine(double freqHz, double sampleRateHz, size_t n)
{
    std::vector<double> s(n);
    for (size_t i = 0; i < n; ++i) {
        s[i] = std::sin(2.0 * M_PI * freqHz * static_cast<double>(i) / sampleRateHz);
    }
    return s;
}

}  // namespace

TEST_CASE("estimatePeriodicity recovers the frequency of a clean sine wave", "[periodicity]")
{
    const double sampleRate = 30.0;
    const double freq = 0.3;  // 18 cycles/min, a plausible respiratory rate
    const auto signal = makeSine(freq, sampleRate, 300);

    const auto result = estimatePeriodicity(signal, sampleRate, 0.1, 1.0);
    REQUIRE(result.valid);
    REQUIRE(std::fabs(result.rateHz - freq) < 0.02);
    REQUIRE(result.quality > 0.8);
}

TEST_CASE("estimatePeriodicity is invalid for a flat (DC) signal", "[periodicity]")
{
    std::vector<double> flat(300, 0.5);
    const auto result = estimatePeriodicity(flat, 30.0, 0.1, 1.0);
    REQUIRE_FALSE(result.valid);
}

TEST_CASE("estimatePeriodicity is invalid when the signal is too short for the rate range", "[periodicity]")
{
    // At 30 Hz, a 0.1 Hz cycle needs a lag of 300 samples -- longer than
    // this 50-sample signal.
    const auto signal = makeSine(0.3, 30.0, 50);
    const auto result = estimatePeriodicity(signal, 30.0, 0.1, 1.0);
    REQUIRE_FALSE(result.valid);
}

TEST_CASE("estimatePeriodicity rejects a peak below the quality threshold", "[periodicity]")
{
    // Genuine white noise (via <random>, not a hand-rolled hash --
    // linear-congruential-style formulas like (i*k) mod m increment by
    // a *constant* step between consecutive i, which is a disguised
    // ramp/sawtooth, not noise, and ramps have spuriously high
    // short-lag autocorrelation precisely because they're smooth) has
    // no periodicity to find at any lag, so it shouldn't clear a
    // demanding quality bar.
    std::minstd_rand rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> signal(300);
    for (double& s : signal) {
        s = dist(rng);
    }

    const auto result = estimatePeriodicity(signal, 30.0, 0.1, 1.0, /*minQuality=*/0.9);
    REQUIRE_FALSE(result.valid);
}

TEST_CASE("estimatePeriodicity is invalid for empty input", "[periodicity]")
{
    const auto result = estimatePeriodicity({}, 30.0, 0.1, 1.0);
    REQUIRE_FALSE(result.valid);
}
