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
#include "ini_config.h"

#include "strings.h"

#include <fstream>
#include <iostream>
#include <vector>

namespace pipeline_core {

std::map<std::string, std::string> parseIni(const std::string& filename)
{
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    if (!file.is_open()) {
        return config;
    }

    std::string line;
    std::string section;

    while (std::getline(file, line)) {
        if (!trimInPlace(line)) {
            continue;
        }

        if (line.empty()) {
            continue;
        }

        // Only treat comment markers as comments when they begin the line.
        // This is simpler and safer than trying to strip inline comments.
        if (line[0] == ';' || line[0] == '#') {
            continue;
        }

        // Section header: [openai]
        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            trimInPlace(section);
            continue;
        }

        const size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, eqPos);
        std::string value = line.substr(eqPos + 1);

        if (!trimInPlace(key)) {
            continue;
        }
        trimInPlace(value);

        // Flatten sections into "section.key".
        if (!section.empty()) {
            key = section + "." + key;
        }

        config[key] = value;
    }

    return config;
}

bool loadOpenAIConfig(const std::string& path, OpenAIConfig& cfg)
{
    const auto config = parseIni(path);

    auto getValue = [&](const std::string& key, std::string& out) {
        const auto it = config.find(key);
        if (it != config.end()) {
            out = it->second;
        }
    };

    getValue("openai.base_url", cfg.baseUrl);
    getValue("openai.api_key", cfg.apiKey);
    getValue("openai.vmodel_name", cfg.vmodelName);

    std::vector<std::string> missing;
    if (cfg.baseUrl.empty()) {
        missing.push_back("openai.base_url");
    }
    if (cfg.apiKey.empty()) {
        missing.push_back("openai.api_key");
    }
    if (cfg.vmodelName.empty()) {
        missing.push_back("openai.vmodel_name");
    }

    if (!missing.empty()) {
        std::cerr << "[ERROR] Missing config values in " << path << ":";
        for (const auto& key : missing) {
            std::cerr << ' ' << key;
        }
        std::cerr << "\n";
        return false;
    }

    cfg.baseUrl = ensureTrailingSlash(cfg.baseUrl);
    return true;
}

}  // namespace pipeline_core
