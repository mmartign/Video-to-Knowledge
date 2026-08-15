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
// Umbrella header for pipeline_core: dependency-light, unit-testable core
// logic shared by the pipeline executables (INI/CLI parsing, string/date
// helpers, and API response parsing), deliberately free of
// OpenCV/CURL/openai-cpp so it can be built and exercised in isolation by
// the unit test suite.
//
// The logic itself is split by functional area under pipeline_core/ (one
// header/source pair each); this header just aggregates the full surface
// for callers, like realtime_video_pipeline.cpp, that use all of it. Code
// that only needs one area can include the matching pipeline_core/*.h
// directly instead.
#pragma once

#include "pipeline_core/base64.h"
#include "pipeline_core/cli.h"
#include "pipeline_core/datetime.h"
#include "pipeline_core/ini_config.h"
#include "pipeline_core/json_response.h"
#include "pipeline_core/numeric.h"
#include "pipeline_core/strings.h"
