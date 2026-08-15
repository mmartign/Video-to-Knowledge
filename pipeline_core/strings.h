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
// General-purpose string/URL helpers with no dependency beyond the STL.
#pragma once

#include <string>

namespace pipeline_core {

// Trim leading and trailing whitespace in place.
// Returns false if the resulting string is empty.
bool trimInPlace(std::string& s);

// Ensure base URLs end with '/' so downstream URL construction is consistent.
std::string ensureTrailingSlash(const std::string& url);

bool endsWith(const std::string& value, const std::string& suffix);

std::string toLowerAscii(std::string value);

bool usesOpenWebUIChatEndpoint(const std::string& baseUrl);

}  // namespace pipeline_core
