// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for pipeline_core/strings.h.
#include "pipeline_core/strings.h"

#include <catch2/catch_test_macros.hpp>

using namespace pipeline_core;

TEST_CASE("trimInPlace trims whitespace and reports emptiness", "[strings]")
{
    std::string s = "  hello world  \t\r\n";
    REQUIRE(trimInPlace(s));
    REQUIRE(s == "hello world");

    std::string onlyWhitespace = " \t\r\n ";
    REQUIRE_FALSE(trimInPlace(onlyWhitespace));
    REQUIRE(onlyWhitespace.empty());

    std::string empty;
    REQUIRE_FALSE(trimInPlace(empty));

    std::string noTrim = "already-trimmed";
    REQUIRE(trimInPlace(noTrim));
    REQUIRE(noTrim == "already-trimmed");
}

TEST_CASE("ensureTrailingSlash", "[strings]")
{
    REQUIRE(ensureTrailingSlash("http://host/api") == "http://host/api/");
    REQUIRE(ensureTrailingSlash("http://host/api/") == "http://host/api/");
    REQUIRE(ensureTrailingSlash("") == "");
}

TEST_CASE("endsWith", "[strings]")
{
    REQUIRE(endsWith("http://host/api/", "/api/"));
    REQUIRE_FALSE(endsWith("http://host/api/", "/v1/"));
    REQUIRE_FALSE(endsWith("short", "longer-than-short"));
    REQUIRE(endsWith("exact", "exact"));
}

TEST_CASE("toLowerAscii", "[strings]")
{
    REQUIRE(toLowerAscii("HTTP://HOST/API") == "http://host/api");
    REQUIRE(toLowerAscii("MiXeD-123") == "mixed-123");
}

TEST_CASE("usesOpenWebUIChatEndpoint detects /api/ and /api/v1/ bases", "[strings]")
{
    REQUIRE(usesOpenWebUIChatEndpoint("http://localhost:3000/api"));
    REQUIRE(usesOpenWebUIChatEndpoint("http://localhost:3000/api/"));
    REQUIRE(usesOpenWebUIChatEndpoint("http://localhost:3000/API/V1"));
    REQUIRE_FALSE(usesOpenWebUIChatEndpoint("http://localhost:11434/v1/"));
    REQUIRE_FALSE(usesOpenWebUIChatEndpoint("http://localhost:11434/v1"));
}
