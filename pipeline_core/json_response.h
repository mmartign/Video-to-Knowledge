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

}  // namespace pipeline_core
