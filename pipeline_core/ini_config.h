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
// INI file parsing and the OpenAI connection config loaded from it.
#pragma once

#include <map>
#include <string>

namespace pipeline_core {

// OpenAI-related configuration loaded from the INI file.
struct OpenAIConfig {
    std::string baseUrl;
    std::string apiKey;
    std::string vmodelName;
};

// Small INI parser.
//
// Supported features:
// - [section] headers
// - key=value pairs
// - full-line comments starting with ';' or '#'
//
// Deliberately not supported:
// - quoted escaping rules
// - inline comments
// - multi-line values
//
// This keeps the parser predictable and avoids corrupting values containing
// '#' or ';', which the previous version could truncate accidentally.
std::map<std::string, std::string> parseIni(const std::string& filename);

// Load and validate the required OpenAI config values.
bool loadOpenAIConfig(const std::string& path, OpenAIConfig& cfg);

}  // namespace pipeline_core
