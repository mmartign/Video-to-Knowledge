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
// Strict numeric parsing and the camera-index-vs-URI source heuristic.
#pragma once

#include <string>

namespace pipeline_core {

// Parse a double and require that the whole string is consumed.
bool parseDoubleStrict(const std::string& s, double& out);

// Parse an int and require that the whole string is consumed.
bool parseIntStrict(const std::string& s, int& out);

// Return true if the entire string is composed of decimal digits.
bool isUnsignedIntegerString(const std::string& s);

// Treat any non-empty unsigned integer as a camera index.
//
// This is more useful than a single-digit check because device indexes such
// as "10" should still work.
bool isCameraIndexSource(const std::string& s);

}  // namespace pipeline_core
