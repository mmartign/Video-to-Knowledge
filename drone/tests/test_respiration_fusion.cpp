// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for video_core/respiration_fusion.h.
#include "../video_core/respiration_fusion.h"

#include <catch2/catch_test_macros.hpp>

using namespace drone::video_core;

namespace {

PeriodicityEstimate makeEstimate(double ratePerMinute, double quality)
{
    PeriodicityEstimate e;
    e.valid = true;
    e.ratePerMinute = ratePerMinute;
    e.rateHz = ratePerMinute / 60.0;
    e.quality = quality;
    return e;
}

}  // namespace

TEST_CASE("fuseRespirationEstimates is invalid when neither input is valid", "[respiration_fusion]")
{
    const auto result = fuseRespirationEstimates(PeriodicityEstimate{}, PeriodicityEstimate{});
    REQUIRE_FALSE(result.valid);
}

TEST_CASE("fuseRespirationEstimates passes through the only valid estimate", "[respiration_fusion]")
{
    const auto opticalOnly = makeEstimate(16.0, 0.7);
    const auto r1 = fuseRespirationEstimates(opticalOnly, PeriodicityEstimate{});
    REQUIRE(r1.valid);
    REQUIRE(r1.ratePerMinute == 16.0);

    const auto thermalOnly = makeEstimate(18.0, 0.6);
    const auto r2 = fuseRespirationEstimates(PeriodicityEstimate{}, thermalOnly);
    REQUIRE(r2.valid);
    REQUIRE(r2.ratePerMinute == 18.0);
}

TEST_CASE("fuseRespirationEstimates quality-weights agreeing estimates", "[respiration_fusion]")
{
    // Optical flow: higher quality, should pull the fused rate closer
    // to its own value.
    const auto optical = makeEstimate(16.0, 0.9);
    const auto thermal = makeEstimate(14.0, 0.3);

    const auto result = fuseRespirationEstimates(optical, thermal);
    REQUIRE(result.valid);
    REQUIRE(result.ratePerMinute > 15.0);  // closer to 16 than to 14
    REQUIRE(result.ratePerMinute < 16.0);
}

TEST_CASE("fuseRespirationEstimates trusts the higher-quality estimate on strong disagreement", "[respiration_fusion]")
{
    const auto optical = makeEstimate(30.0, 0.5);   // implausibly different from thermal
    const auto thermal = makeEstimate(14.0, 0.8);

    const auto result = fuseRespirationEstimates(optical, thermal);
    REQUIRE(result.valid);
    REQUIRE(result.ratePerMinute == 14.0);  // thermal has higher quality
}
