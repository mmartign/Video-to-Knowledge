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
#include "json_response.h"

namespace pipeline_core {

std::string extractMessageText(const json& response)
{
    const auto choicesIt = response.find("choices");
    if (choicesIt != response.end() && choicesIt->is_array() && !choicesIt->empty()) {
        const auto& first = (*choicesIt)[0];

        // Typical chat-completions shape:
        // choices[0].message.content
        const auto messageIt = first.find("message");
        if (messageIt != first.end()) {
            const auto contentIt = messageIt->find("content");
            if (contentIt != messageIt->end()) {
                if (contentIt->is_string()) {
                    return contentIt->get<std::string>();
                }
                if (contentIt->is_array()) {
                    std::string combined;
                    for (const auto& part : *contentIt) {
                        const auto textIt = part.find("text");
                        if (textIt != part.end() && textIt->is_string()) {
                            combined += textIt->get<std::string>();
                        }
                    }
                    if (!combined.empty()) {
                        return combined;
                    }
                }
            }
        }

        // Fallback shape sometimes seen in wrappers.
        const auto textIt = first.find("text");
        if (textIt != first.end() && textIt->is_string()) {
            return textIt->get<std::string>();
        }
    }

    // Additional defensive fallbacks.
    const auto outputTextIt = response.find("output_text");
    if (outputTextIt != response.end() && outputTextIt->is_string()) {
        return outputTextIt->get<std::string>();
    }

    const auto outputIt = response.find("output");
    if (outputIt != response.end() && outputIt->is_array()) {
        std::string combined;
        for (const auto& item : *outputIt) {
            const auto contentIt = item.find("content");
            if (contentIt == item.end() || !contentIt->is_array()) {
                continue;
            }
            for (const auto& part : *contentIt) {
                const auto textIt = part.find("text");
                if (textIt != part.end() && textIt->is_string()) {
                    combined += textIt->get<std::string>();
                }
            }
        }
        if (!combined.empty()) {
            return combined;
        }
    }

    return {};
}

}  // namespace pipeline_core
