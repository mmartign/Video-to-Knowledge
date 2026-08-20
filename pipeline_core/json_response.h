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
#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace pipeline_core {

using json = nlohmann::json;

// Extract human-readable text from an OpenAI-compatible chat API response.
//
// We support a few plausible shapes because deployments using "OpenAI-like"
// compatibility layers may not always return identical JSON structures.
std::string extractMessageText(const json& response);

// Returns true if `response`'s echoed "model" field is empty or doesn't
// match `requestedModel`.
//
// Some OpenAI-compatible gateways (observed with an Open WebUI backend
// mid-refactor of its multi-round/tool-orchestration handling) can accept
// a chat completion request over HTTP 200 without ever actually
// dispatching it to a model: no text is generated, usage/eval counters are
// all zero, and the "model" field -- normally echoed back from whichever
// model call actually ran -- is left empty because no call ran. An empty
// or mismatched "model" field is the most reliable signal of that failure
// mode, as opposed to a model that ran and legitimately produced no text.
bool looksUndispatched(const json& response, const std::string& requestedModel);

}  // namespace pipeline_core
