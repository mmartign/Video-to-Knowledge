// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for pipeline_core, the dependency-light logic shared by the
// pipeline executables (INI/CLI parsing, string/date helpers, API response
// parsing). These tests intentionally avoid OpenCV/CURL/openai-cpp.
#include "pipeline_core.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace pipeline_core;

namespace {

// Writes `contents` to a uniquely-named temp file and returns its path.
// The file is removed when the returned guard goes out of scope.
class TempFile {
public:
    explicit TempFile(const std::string& contents)
        : path_(std::filesystem::temp_directory_path() /
                ("pipeline_core_test_" + std::to_string(counter_++) + ".ini"))
    {
        std::ofstream out(path_);
        out << contents;
    }

    ~TempFile() { std::filesystem::remove(path_); }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
    static inline int counter_ = 0;
};

// Builds an argv-style char** from a list of strings for parseCommandLine tests.
class Argv {
public:
    explicit Argv(std::vector<std::string> args) : args_(std::move(args))
    {
        for (auto& a : args_) {
            ptrs_.push_back(a.data());
        }
    }

    int argc() const { return static_cast<int>(ptrs_.size()); }
    char** argv() { return ptrs_.data(); }

private:
    std::vector<std::string> args_;
    std::vector<char*> ptrs_;
};

}  // namespace

//------------------------------------------------------------------------------
// String helpers
//------------------------------------------------------------------------------

TEST_CASE("trimInPlace trims whitespace and reports emptiness", "[strings]")
{
    std::string s = "  hello world  \t\r\n";
    REQUIRE(trimInPlace(s));
    REQUIRE(s == "hello world");

    std::string onlyWhitespace = " \t\r\n ";
    REQUIRE_FALSE(trimInPlace(onlyWhitespace));
    REQUIRE(onlyWhitespace.empty());

    std::string empty;
    REQUIRE_FALSE(trimInPlace(empty));

    std::string noTrim = "already-trimmed";
    REQUIRE(trimInPlace(noTrim));
    REQUIRE(noTrim == "already-trimmed");
}

TEST_CASE("ensureTrailingSlash", "[strings]")
{
    REQUIRE(ensureTrailingSlash("http://host/api") == "http://host/api/");
    REQUIRE(ensureTrailingSlash("http://host/api/") == "http://host/api/");
    REQUIRE(ensureTrailingSlash("") == "");
}

TEST_CASE("endsWith", "[strings]")
{
    REQUIRE(endsWith("http://host/api/", "/api/"));
    REQUIRE_FALSE(endsWith("http://host/api/", "/v1/"));
    REQUIRE_FALSE(endsWith("short", "longer-than-short"));
    REQUIRE(endsWith("exact", "exact"));
}

TEST_CASE("toLowerAscii", "[strings]")
{
    REQUIRE(toLowerAscii("HTTP://HOST/API") == "http://host/api");
    REQUIRE(toLowerAscii("MiXeD-123") == "mixed-123");
}

TEST_CASE("usesOpenWebUIChatEndpoint detects /api/ and /api/v1/ bases", "[strings]")
{
    REQUIRE(usesOpenWebUIChatEndpoint("http://localhost:3000/api"));
    REQUIRE(usesOpenWebUIChatEndpoint("http://localhost:3000/api/"));
    REQUIRE(usesOpenWebUIChatEndpoint("http://localhost:3000/API/V1"));
    REQUIRE_FALSE(usesOpenWebUIChatEndpoint("http://localhost:11434/v1/"));
    REQUIRE_FALSE(usesOpenWebUIChatEndpoint("http://localhost:11434/v1"));
}

//------------------------------------------------------------------------------
// INI/config parsing
//------------------------------------------------------------------------------

TEST_CASE("parseIni parses sections, comments, and flattens keys", "[ini]")
{
    TempFile file(
        "; leading comment\n"
        "# another comment\n"
        "[openai]\n"
        "base_url = http://localhost:11434/v1/ \n"
        "api_key=secret-key\n"
        "vmodel_name = medgemma-15:4b\n"
        "\n"
        "[unused]\n"
        "some_key = some_value  # not stripped, no inline comments\n");

    const auto config = parseIni(file.path());
    REQUIRE(config.at("openai.base_url") == "http://localhost:11434/v1/");
    REQUIRE(config.at("openai.api_key") == "secret-key");
    REQUIRE(config.at("openai.vmodel_name") == "medgemma-15:4b");
    // '#' is only a comment marker at line start; this repo deliberately
    // does not support inline comment stripping.
    REQUIRE(config.at("unused.some_key") == "some_value  # not stripped, no inline comments");
}

TEST_CASE("parseIni returns an empty map for a missing file", "[ini]")
{
    const auto config = parseIni("/nonexistent/path/does-not-exist.ini");
    REQUIRE(config.empty());
}

TEST_CASE("parseIni ignores lines without '=' and keys are trimmed", "[ini]")
{
    TempFile file(
        "[openai]\n"
        "not-a-key-value-line\n"
        "  base_url  =  http://host/ \n");

    const auto config = parseIni(file.path());
    REQUIRE(config.size() == 1);
    REQUIRE(config.at("openai.base_url") == "http://host/");
}

TEST_CASE("loadOpenAIConfig succeeds with all required keys", "[ini]")
{
    TempFile file(
        "[openai]\n"
        "base_url = http://localhost:11434/v1\n"
        "api_key = key123\n"
        "vmodel_name = medgemma-15:4b\n");

    OpenAIConfig cfg;
    REQUIRE(loadOpenAIConfig(file.path(), cfg));
    REQUIRE(cfg.baseUrl == "http://localhost:11434/v1/");  // trailing slash added
    REQUIRE(cfg.apiKey == "key123");
    REQUIRE(cfg.vmodelName == "medgemma-15:4b");
}

TEST_CASE("loadOpenAIConfig fails when required keys are missing", "[ini]")
{
    TempFile file(
        "[openai]\n"
        "base_url = http://localhost:11434/v1\n");

    OpenAIConfig cfg;
    REQUIRE_FALSE(loadOpenAIConfig(file.path(), cfg));
}

//------------------------------------------------------------------------------
// Encoding and API response parsing
//------------------------------------------------------------------------------

TEST_CASE("base64Encode matches RFC 4648 test vectors", "[base64]")
{
    auto bytes = [](const std::string& s) {
        return std::vector<unsigned char>(s.begin(), s.end());
    };

    REQUIRE(base64Encode(bytes("")) == "");
    REQUIRE(base64Encode(bytes("f")) == "Zg==");
    REQUIRE(base64Encode(bytes("fo")) == "Zm8=");
    REQUIRE(base64Encode(bytes("foo")) == "Zm9v");
    REQUIRE(base64Encode(bytes("foob")) == "Zm9vYg==");
    REQUIRE(base64Encode(bytes("fooba")) == "Zm9vYmE=");
    REQUIRE(base64Encode(bytes("foobar")) == "Zm9vYmFy");
}

TEST_CASE("extractMessageText handles chat-completions string content", "[json]")
{
    const json response = {
        {"choices", json::array({
            {{"message", {{"role", "assistant"}, {"content", "hello there"}}}}
        })}
    };
    REQUIRE(extractMessageText(response) == "hello there");
}

TEST_CASE("extractMessageText handles chat-completions array content", "[json]")
{
    const json response = {
        {"choices", json::array({
            {{"message", {{"content", json::array({
                {{"type", "text"}, {"text", "part one "}},
                {{"type", "text"}, {"text", "part two"}}
            })}}}}
        })}
    };
    REQUIRE(extractMessageText(response) == "part one part two");
}

TEST_CASE("extractMessageText falls back to choices[0].text", "[json]")
{
    const json response = {
        {"choices", json::array({{{"text", "fallback text"}}})}
    };
    REQUIRE(extractMessageText(response) == "fallback text");
}

TEST_CASE("extractMessageText falls back to output_text", "[json]")
{
    const json response = {{"output_text", "top level text"}};
    REQUIRE(extractMessageText(response) == "top level text");
}

TEST_CASE("extractMessageText falls back to output[].content[].text", "[json]")
{
    const json response = {
        {"output", json::array({
            {{"content", json::array({
                {{"type", "text"}, {"text", "assembled "}},
                {{"type", "text"}, {"text", "response"}}
            })}}
        })}
    };
    REQUIRE(extractMessageText(response) == "assembled response");
}

TEST_CASE("extractMessageText returns empty string for unrecognized shapes", "[json]")
{
    const json response = {{"unexpected", "shape"}};
    REQUIRE(extractMessageText(response).empty());

    const json emptyChoices = {{"choices", json::array()}};
    REQUIRE(extractMessageText(emptyChoices).empty());
}

//------------------------------------------------------------------------------
// Time helpers
//------------------------------------------------------------------------------

TEST_CASE("parseDateTimeNoConversion / formatDateTimeNoConversion roundtrip", "[time]")
{
    std::chrono::system_clock::time_point tp;
    REQUIRE(parseDateTimeNoConversion("2026-04-27 10:15:30", tp));
    REQUIRE(formatDateTimeNoConversion(tp) == "[2026-04-27 10:15:30]");
}

TEST_CASE("parseDateTimeNoConversion rejects malformed and out-of-range dates", "[time]")
{
    std::chrono::system_clock::time_point tp;
    REQUIRE_FALSE(parseDateTimeNoConversion("not-a-date", tp));
    REQUIRE_FALSE(parseDateTimeNoConversion("2026-13-01 00:00:00", tp));  // invalid month
    REQUIRE_FALSE(parseDateTimeNoConversion("2026-02-30 00:00:00", tp));  // invalid day
    REQUIRE_FALSE(parseDateTimeNoConversion("2026-04-27 24:00:00", tp));  // invalid hour
    REQUIRE_FALSE(parseDateTimeNoConversion("2026-04-27T10:15:30", tp));  // wrong separator
}

TEST_CASE("parseDateTimeNoConversion accepts Feb 29 on leap years only", "[time]")
{
    std::chrono::system_clock::time_point tp;
    REQUIRE(parseDateTimeNoConversion("2024-02-29 00:00:00", tp));   // 2024 is a leap year
    REQUIRE_FALSE(parseDateTimeNoConversion("2023-02-29 00:00:00", tp));  // 2023 is not
}

TEST_CASE("isLeapYear follows Gregorian rules", "[time]")
{
    REQUIRE(isLeapYear(2024));
    REQUIRE_FALSE(isLeapYear(2023));
    REQUIRE_FALSE(isLeapYear(1900));  // divisible by 100, not 400
    REQUIRE(isLeapYear(2000));        // divisible by 400
}

TEST_CASE("daysInMonth accounts for leap February", "[time]")
{
    REQUIRE(daysInMonth(2024, 2) == 29);
    REQUIRE(daysInMonth(2023, 2) == 28);
    REQUIRE(daysInMonth(2026, 4) == 30);
    REQUIRE(daysInMonth(2026, 13) == 0);
}

TEST_CASE("daysFromCivil matches known epoch offsets", "[time]")
{
    REQUIRE(daysFromCivil(1970, 1, 1) == 0);
    REQUIRE(daysFromCivil(1969, 12, 31) == -1);
    REQUIRE(daysFromCivil(2000, 1, 1) == 10957);
}

TEST_CASE("parseNDigits requires exact digit count and rejects non-digits", "[time]")
{
    int out = 0;
    REQUIRE(parseNDigits("2026", 0, 4, out));
    REQUIRE(out == 2026);

    REQUIRE_FALSE(parseNDigits("20a6", 0, 4, out));
    REQUIRE_FALSE(parseNDigits("202", 0, 4, out));  // not enough characters
}

TEST_CASE("parseMetadataDateTime normalizes T/Z/fractional/timezone suffixes", "[time]")
{
    std::chrono::system_clock::time_point tp;
    std::chrono::system_clock::time_point tpRef;
    REQUIRE(parseDateTime("2026-04-27 10:15:30", tpRef));

    REQUIRE(parseMetadataDateTime("2026-04-27T10:15:30Z", tp));
    REQUIRE(tp == tpRef);

    REQUIRE(parseMetadataDateTime("2026-04-27T10:15:30.123456Z", tp));
    REQUIRE(tp == tpRef);

    REQUIRE(parseMetadataDateTime("2026-04-27T10:15:30+02:00", tp));
    REQUIRE(tp == tpRef);
}

TEST_CASE("parseMetadataDateTime rejects empty or too-short values", "[time]")
{
    std::chrono::system_clock::time_point tp;
    REQUIRE_FALSE(parseMetadataDateTime("", tp));
    REQUIRE_FALSE(parseMetadataDateTime("   ", tp));
    REQUIRE_FALSE(parseMetadataDateTime("2026-04-27", tp));
}

TEST_CASE("addSecondsToTimePoint advances by fractional seconds", "[time]")
{
    std::chrono::system_clock::time_point base;
    REQUIRE(parseDateTimeNoConversion("2026-04-27 10:15:30", base));

    const auto later = addSecondsToTimePoint(base, 90.0);
    REQUIRE(formatDateTimeNoConversion(later) == "[2026-04-27 10:17:00]");
}

//------------------------------------------------------------------------------
// Strict numeric parsing helpers
//------------------------------------------------------------------------------

TEST_CASE("parseDoubleStrict requires the whole string to be consumed", "[numeric]")
{
    double out = 0.0;
    REQUIRE(parseDoubleStrict("3.14", out));
    REQUIRE(out == 3.14);

    REQUIRE_FALSE(parseDoubleStrict("3.14abc", out));
    REQUIRE_FALSE(parseDoubleStrict("not-a-number", out));
    REQUIRE_FALSE(parseDoubleStrict("", out));
}

TEST_CASE("parseIntStrict requires the whole string to be consumed", "[numeric]")
{
    int out = 0;
    REQUIRE(parseIntStrict("42", out));
    REQUIRE(out == 42);

    REQUIRE_FALSE(parseIntStrict("42.5", out));
    REQUIRE_FALSE(parseIntStrict("42abc", out));
    REQUIRE_FALSE(parseIntStrict("", out));
}

TEST_CASE("isUnsignedIntegerString / isCameraIndexSource", "[numeric]")
{
    REQUIRE(isUnsignedIntegerString("0"));
    REQUIRE(isUnsignedIntegerString("10"));
    REQUIRE_FALSE(isUnsignedIntegerString(""));
    REQUIRE_FALSE(isUnsignedIntegerString("-1"));
    REQUIRE_FALSE(isUnsignedIntegerString("1.5"));
    REQUIRE_FALSE(isUnsignedIntegerString("rtsp://host/stream"));

    REQUIRE(isCameraIndexSource("10"));
    REQUIRE_FALSE(isCameraIndexSource("video.mp4"));
}

//------------------------------------------------------------------------------
// CLI parsing
//------------------------------------------------------------------------------

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
