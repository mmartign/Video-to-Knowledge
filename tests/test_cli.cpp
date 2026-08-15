// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for pipeline_core/cli.h.
#include "pipeline_core/cli.h"

#include "test_utils.h"

#include <catch2/catch_test_macros.hpp>

using namespace pipeline_core;
using pipeline_core_test::Argv;

TEST_CASE("parseCommandLine applies defaults with only a source", "[cli]")
{
    Argv args({"prog", "0"});
    ProgramOptions opt;
    REQUIRE(parseCommandLine(args.argc(), args.argv(), opt));

    REQUIRE(opt.src == "0");
    REQUIRE(opt.configPath == "config.ini");
    REQUIRE(opt.intervalSec == 10.0);
    REQUIRE(opt.maxDim == 1024);
    REQUIRE(opt.jpegQuality == 85);
    REQUIRE(opt.guiEnabled);
    REQUIRE(opt.reconnectSec == 5);
    REQUIRE_FALSE(opt.hasPredefinedStartTime);
}

TEST_CASE("parseCommandLine accepts a positional config path", "[cli]")
{
    Argv args({"prog", "stream.mp4", "myconfig.ini"});
    ProgramOptions opt;
    REQUIRE(parseCommandLine(args.argc(), args.argv(), opt));
    REQUIRE(opt.src == "stream.mp4");
    REQUIRE(opt.configPath == "myconfig.ini");
}

TEST_CASE("parseCommandLine parses all documented options", "[cli]")
{
    Argv args({
        "prog", "0",
        "--interval", "5.5",
        "--max-dim", "512",
        "--jpeg-quality", "90",
        "--prompt", "Describe the scene",
        "--no-gui",
        "--reconnect-sec", "0",
        "--predefined_start_time", "2026-01-01 00:00:00",
    });
    ProgramOptions opt;
    REQUIRE(parseCommandLine(args.argc(), args.argv(), opt));

    REQUIRE(opt.intervalSec == 5.5);
    REQUIRE(opt.maxDim == 512);
    REQUIRE(opt.jpegQuality == 90);
    REQUIRE(opt.prompt == "Describe the scene");
    REQUIRE_FALSE(opt.guiEnabled);
    REQUIRE(opt.reconnectSec == 0);
    REQUIRE(opt.hasPredefinedStartTime);
}

TEST_CASE("parseCommandLine rejects invalid option values", "[cli]")
{
    ProgramOptions opt;

    {
        Argv args({"prog", "0", "--interval", "not-a-number"});
        REQUIRE_FALSE(parseCommandLine(args.argc(), args.argv(), opt));
    }
    {
        Argv args({"prog", "0", "--interval", "0"});  // must be > 0
        REQUIRE_FALSE(parseCommandLine(args.argc(), args.argv(), opt));
    }
    {
        Argv args({"prog", "0", "--jpeg-quality", "101"});  // out of 1..100
        REQUIRE_FALSE(parseCommandLine(args.argc(), args.argv(), opt));
    }
    {
        Argv args({"prog", "0", "--max-dim", "-1"});
        REQUIRE_FALSE(parseCommandLine(args.argc(), args.argv(), opt));
    }
    {
        Argv args({"prog", "0", "--predefined_start_time", "not-a-date"});
        REQUIRE_FALSE(parseCommandLine(args.argc(), args.argv(), opt));
    }
    {
        Argv args({"prog", "0", "--unknown-flag"});
        REQUIRE_FALSE(parseCommandLine(args.argc(), args.argv(), opt));
    }
    {
        Argv args({"prog", "0", "--interval"});  // missing value
        REQUIRE_FALSE(parseCommandLine(args.argc(), args.argv(), opt));
    }
}

TEST_CASE("parseCommandLine fails when no source is given", "[cli]")
{
    Argv args({"prog"});
    ProgramOptions opt;
    REQUIRE_FALSE(parseCommandLine(args.argc(), args.argv(), opt));
}
