// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for pipeline_core/numeric.h.
#include "pipeline_core/numeric.h"

#include <catch2/catch_test_macros.hpp>

using namespace pipeline_core;

TEST_CASE("parseDoubleStrict requires the whole string to be consumed", "[numeric]")
{
    double out = 0.0;
    REQUIRE(parseDoubleStrict("3.14", out));
    REQUIRE(out == 3.14);

    REQUIRE_FALSE(parseDoubleStrict("3.14abc", out));
    REQUIRE_FALSE(parseDoubleStrict("not-a-number", out));
    REQUIRE_FALSE(parseDoubleStrict("", out));
}

TEST_CASE("parseIntStrict requires the whole string to be consumed", "[numeric]")
{
    int out = 0;
    REQUIRE(parseIntStrict("42", out));
    REQUIRE(out == 42);

    REQUIRE_FALSE(parseIntStrict("42.5", out));
    REQUIRE_FALSE(parseIntStrict("42abc", out));
    REQUIRE_FALSE(parseIntStrict("", out));
}

TEST_CASE("isUnsignedIntegerString / isCameraIndexSource", "[numeric]")
{
    REQUIRE(isUnsignedIntegerString("0"));
    REQUIRE(isUnsignedIntegerString("10"));
    REQUIRE_FALSE(isUnsignedIntegerString(""));
    REQUIRE_FALSE(isUnsignedIntegerString("-1"));
    REQUIRE_FALSE(isUnsignedIntegerString("1.5"));
    REQUIRE_FALSE(isUnsignedIntegerString("rtsp://host/stream"));

    REQUIRE(isCameraIndexSource("10"));
    REQUIRE_FALSE(isCameraIndexSource("video.mp4"));
}
