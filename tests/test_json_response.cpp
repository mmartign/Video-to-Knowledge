// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for pipeline_core/json_response.h.
#include "pipeline_core/json_response.h"

#include <catch2/catch_test_macros.hpp>

using namespace pipeline_core;

TEST_CASE("extractMessageText handles chat-completions string content", "[json]")
{
    const json response = {
        {"choices", json::array({
            {{"message", {{"role", "assistant"}, {"content", "hello there"}}}}
        })}
    };
    REQUIRE(extractMessageText(response) == "hello there");
}

TEST_CASE("extractMessageText handles chat-completions array content", "[json]")
{
    const json response = {
        {"choices", json::array({
            {{"message", {{"content", json::array({
                {{"type", "text"}, {"text", "part one "}},
                {{"type", "text"}, {"text", "part two"}}
            })}}}}
        })}
    };
    REQUIRE(extractMessageText(response) == "part one part two");
}

TEST_CASE("extractMessageText falls back to choices[0].text", "[json]")
{
    const json response = {
        {"choices", json::array({{{"text", "fallback text"}}})}
    };
    REQUIRE(extractMessageText(response) == "fallback text");
}

TEST_CASE("extractMessageText falls back to output_text", "[json]")
{
    const json response = {{"output_text", "top level text"}};
    REQUIRE(extractMessageText(response) == "top level text");
}

TEST_CASE("extractMessageText falls back to output[].content[].text", "[json]")
{
    const json response = {
        {"output", json::array({
            {{"content", json::array({
                {{"type", "text"}, {"text", "assembled "}},
                {{"type", "text"}, {"text", "response"}}
            })}}
        })}
    };
    REQUIRE(extractMessageText(response) == "assembled response");
}

TEST_CASE("extractMessageText returns empty string for unrecognized shapes", "[json]")
{
    const json response = {{"unexpected", "shape"}};
    REQUIRE(extractMessageText(response).empty());

    const json emptyChoices = {{"choices", json::array()}};
    REQUIRE(extractMessageText(emptyChoices).empty());
}
