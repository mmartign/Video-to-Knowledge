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
// Command-line option parsing for the pipeline executable.
#pragma once

#include <chrono>
#include <string>

namespace pipeline_core {

// Command-line options with defaults chosen to match the original behavior.
struct ProgramOptions {
    std::string src;
    std::string configPath = "config.ini";

    double intervalSec = 10.0;
    int maxDim = 1024;
    int jpegQuality = 85;
    std::string prompt = "Analyze this frame.";
    bool guiEnabled = true;
    int reconnectSec = 5;

    // Optional explicit base datetime for media files.
    // If absent, we try media metadata creation_time, then application start.
    bool hasPredefinedStartTime = false;
    std::chrono::system_clock::time_point predefinedStartTime{};
};

// Print CLI help.
void printUsage(const char* argv0);

// Parse command-line arguments into ProgramOptions.
//
// Design goals:
// - preserve the original CLI shape
// - validate values with clear error messages
// - avoid uncaught exceptions from stod/stoi
//
// Note: --help/-h prints usage and calls std::exit(0), matching the
// original CLI behavior.
bool parseCommandLine(int argc, char** argv, ProgramOptions& opt);

}  // namespace pipeline_core
