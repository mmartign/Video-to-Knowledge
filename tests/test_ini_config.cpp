// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for pipeline_core/ini_config.h.
#include "pipeline_core/ini_config.h"

#include "test_utils.h"

#include <catch2/catch_test_macros.hpp>

using namespace pipeline_core;
using pipeline_core_test::TempFile;

TEST_CASE("parseIni parses sections, comments, and flattens keys", "[ini]")
{
    TempFile file(
        "; leading comment\n"
        "# another comment\n"
        "[openai]\n"
        "base_url = http://localhost:11434/v1/ \n"
        "api_key=secret-key\n"
        "vmodel_name = medgemma-15:4b\n"
        "\n"
        "[unused]\n"
        "some_key = some_value  # not stripped, no inline comments\n");

    const auto config = parseIni(file.path());
    REQUIRE(config.at("openai.base_url") == "http://localhost:11434/v1/");
    REQUIRE(config.at("openai.api_key") == "secret-key");
    REQUIRE(config.at("openai.vmodel_name") == "medgemma-15:4b");
    // '#' is only a comment marker at line start; this repo deliberately
    // does not support inline comment stripping.
    REQUIRE(config.at("unused.some_key") == "some_value  # not stripped, no inline comments");
}

TEST_CASE("parseIni returns an empty map for a missing file", "[ini]")
{
    const auto config = parseIni("/nonexistent/path/does-not-exist.ini");
    REQUIRE(config.empty());
}

TEST_CASE("parseIni ignores lines without '=' and keys are trimmed", "[ini]")
{
    TempFile file(
        "[openai]\n"
        "not-a-key-value-line\n"
        "  base_url  =  http://host/ \n");

    const auto config = parseIni(file.path());
    REQUIRE(config.size() == 1);
    REQUIRE(config.at("openai.base_url") == "http://host/");
}

TEST_CASE("loadOpenAIConfig succeeds with all required keys", "[ini]")
{
    TempFile file(
        "[openai]\n"
        "base_url = http://localhost:11434/v1\n"
        "api_key = key123\n"
        "vmodel_name = medgemma-15:4b\n");

    OpenAIConfig cfg;
    REQUIRE(loadOpenAIConfig(file.path(), cfg));
    REQUIRE(cfg.baseUrl == "http://localhost:11434/v1/");  // trailing slash added
    REQUIRE(cfg.apiKey == "key123");
    REQUIRE(cfg.vmodelName == "medgemma-15:4b");
}

TEST_CASE("loadOpenAIConfig fails when required keys are missing", "[ini]")
{
    TempFile file(
        "[openai]\n"
        "base_url = http://localhost:11434/v1\n");

    OpenAIConfig cfg;
    REQUIRE_FALSE(loadOpenAIConfig(file.path(), cfg));
}
