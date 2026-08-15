// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for pipeline_core/base64.h.
#include "pipeline_core/base64.h"

#include <catch2/catch_test_macros.hpp>

using namespace pipeline_core;

TEST_CASE("base64Encode matches RFC 4648 test vectors", "[base64]")
{
    auto bytes = [](const std::string& s) {
        return std::vector<unsigned char>(s.begin(), s.end());
    };

    REQUIRE(base64Encode(bytes("")) == "");
    REQUIRE(base64Encode(bytes("f")) == "Zg==");
    REQUIRE(base64Encode(bytes("fo")) == "Zm8=");
    REQUIRE(base64Encode(bytes("foo")) == "Zm9v");
    REQUIRE(base64Encode(bytes("foob")) == "Zm9vYg==");
    REQUIRE(base64Encode(bytes("fooba")) == "Zm9vYmE=");
    REQUIRE(base64Encode(bytes("foobar")) == "Zm9vYmFy");
}
