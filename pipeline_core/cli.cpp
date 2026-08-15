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
#include "cli.h"

#include "datetime.h"
#include "numeric.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>

namespace pipeline_core {

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " <video_or_stream_uri> [config.ini] [options]\n"
        << "Options:\n"
        << "  --interval <sec>        Prompt repetition interval in seconds (default 10)\n"
        << "  --max-dim <px>          Resize frames so max(width,height)<=px (default 1024)\n"
        << "  --jpeg-quality <1-100>  JPEG quality (default 85)\n"
        << "  --prompt <text>         Prompt prefix (default: \"Analyze this frame.\")\n"
        << "  --no-gui                Disable OpenCV imshow/waitKey\n"
        << "  --reconnect-sec <sec>   Reconnect window for live streams (default 5)\n"
        << "  --predefined_start_time \"YYYY-mm-dd HH:MM:SS\"\n"
        << "                          Override base datetime for media files\n";
}

bool parseCommandLine(int argc, char** argv, ProgramOptions& opt)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return false;
    }

    opt.src = argv[1];

    // Optional positional config path:
    //   program <src> config.ini --interval 10
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
        opt.configPath = argv[2];
    }

    int argi = 2;
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
        argi = 3;
    }

    auto needValue = [&](const char* name) -> std::optional<std::string> {
        if (argi + 1 >= argc) {
            std::cerr << "[ERROR] Missing value for " << name << "\n";
            return std::nullopt;
        }
        return std::string(argv[++argi]);
    };

    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];

        if (a == "--interval") {
            auto v = needValue("--interval");
            if (!v) return false;

            double parsed = 0.0;
            if (!parseDoubleStrict(*v, parsed) || !std::isfinite(parsed) || parsed <= 0.0) {
                std::cerr << "[ERROR] --interval must be a positive number\n";
                return false;
            }
            opt.intervalSec = std::max(0.1, parsed);
        } else if (a == "--max-dim") {
            auto v = needValue("--max-dim");
            if (!v) return false;

            int parsed = 0;
            if (!parseIntStrict(*v, parsed) || parsed < 0) {
                std::cerr << "[ERROR] --max-dim must be an integer >= 0\n";
                return false;
            }
            opt.maxDim = parsed;
        } else if (a == "--jpeg-quality") {
            auto v = needValue("--jpeg-quality");
            if (!v) return false;

            int parsed = 0;
            if (!parseIntStrict(*v, parsed) || parsed < 1 || parsed > 100) {
                std::cerr << "[ERROR] --jpeg-quality must be 1..100\n";
                return false;
            }
            opt.jpegQuality = parsed;
        } else if (a == "--predefined_start_time") {
            auto v = needValue("--predefined_start_time");
            if (!v) return false;

            if (!parseDateTimeNoConversion(*v, opt.predefinedStartTime)) {
                std::cerr << "[ERROR] --predefined_start_time expects \"YYYY-mm-dd HH:MM:SS\"\n";
                return false;
            }
            opt.hasPredefinedStartTime = true;
        } else if (a == "--prompt") {
            auto v = needValue("--prompt");
            if (!v) return false;
            opt.prompt = *v;
        } else if (a == "--no-gui") {
            opt.guiEnabled = false;
        } else if (a == "--reconnect-sec") {
            auto v = needValue("--reconnect-sec");
            if (!v) return false;

            int parsed = 0;
            if (!parseIntStrict(*v, parsed) || parsed < 0) {
                std::cerr << "[ERROR] --reconnect-sec must be an integer >= 0\n";
                return false;
            }
            opt.reconnectSec = parsed;
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "[ERROR] Unknown option: " << a << "\n";
            printUsage(argv[0]);
            return false;
        }
    }

    return true;
}

}  // namespace pipeline_core
