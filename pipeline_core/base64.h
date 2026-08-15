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
#include <vector>

namespace pipeline_core {

// Base64-encode a binary buffer.
//
// Used to embed a JPEG frame as a data URL in the API request body.
std::string base64Encode(const std::vector<unsigned char>& data);

}  // namespace pipeline_core
