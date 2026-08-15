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
#include "strings.h"

#include <algorithm>
#include <cctype>

namespace pipeline_core {

bool trimInPlace(std::string& s)
{
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        s.clear();
        return false;
    }

    const auto end = s.find_last_not_of(" \t\r\n");
    s = s.substr(start, end - start + 1);
    return true;
}

std::string ensureTrailingSlash(const std::string& url)
{
    if (url.empty() || url.back() == '/') {
        return url;
    }
    return url + "/";
}

bool endsWith(const std::string& value, const std::string& suffix)
{
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string toLowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool usesOpenWebUIChatEndpoint(const std::string& baseUrl)
{
    const std::string normalized = ensureTrailingSlash(toLowerAscii(baseUrl));
    return endsWith(normalized, "/api/") || endsWith(normalized, "/api/v1/");
}

}  // namespace pipeline_core
