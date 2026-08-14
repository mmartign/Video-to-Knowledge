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
// Dependency-light, unit-testable core logic shared by the pipeline
// executables: INI/CLI parsing, string/date helpers, and API response
// parsing. Deliberately free of OpenCV/CURL/openai-cpp so it can be built
// and exercised in isolation by the unit test suite.
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace pipeline_core {

using json = nlohmann::json;

//------------------------------------------------------------------------------
// Configuration and runtime option structures
//------------------------------------------------------------------------------

// OpenAI-related configuration loaded from the INI file.
struct OpenAIConfig {
    std::string baseUrl;
    std::string apiKey;
    std::string vmodelName;
};

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

struct DateTimeParts {
    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
};

//------------------------------------------------------------------------------
// CLI help
//------------------------------------------------------------------------------

void printUsage(const char* argv0);

//------------------------------------------------------------------------------
// String helpers
//------------------------------------------------------------------------------

// Trim leading and trailing whitespace in place.
// Returns false if the resulting string is empty.
bool trimInPlace(std::string& s);

// Ensure base URLs end with '/' so downstream URL construction is consistent.
std::string ensureTrailingSlash(const std::string& url);

bool endsWith(const std::string& value, const std::string& suffix);

std::string toLowerAscii(std::string value);

bool usesOpenWebUIChatEndpoint(const std::string& baseUrl);

//------------------------------------------------------------------------------
// INI/config parsing
//------------------------------------------------------------------------------

// Small INI parser.
//
// Supported features:
// - [section] headers
// - key=value pairs
// - full-line comments starting with ';' or '#'
//
// Deliberately not supported:
// - quoted escaping rules
// - inline comments
// - multi-line values
//
// This keeps the parser predictable and avoids corrupting values containing
// '#' or ';', which the previous version could truncate accidentally.
std::map<std::string, std::string> parseIni(const std::string& filename);

// Load and validate the required OpenAI config values.
bool loadOpenAIConfig(const std::string& path, OpenAIConfig& cfg);

//------------------------------------------------------------------------------
// Encoding and API response parsing
//------------------------------------------------------------------------------

// Base64-encode a binary buffer.
//
// Used to embed a JPEG frame as a data URL in the API request body.
std::string base64Encode(const std::vector<unsigned char>& data);

// Extract human-readable text from an API response.
//
// We support a few plausible shapes because deployments using "OpenAI-like"
// compatibility layers may not always return identical JSON structures.
std::string extractMessageText(const json& response);

//------------------------------------------------------------------------------
// Time helpers
//------------------------------------------------------------------------------

// Thread-safe localtime wrapper.
bool safeLocalTime(std::time_t t, std::tm& out);

// Thread-safe gmtime wrapper used for explicit wall-clock timestamps.
bool safeUtcTime(std::time_t t, std::tm& out);

// Format a system_clock time point as "[YYYY-mm-dd HH:MM:SS]".
std::string formatDateTime(const std::chrono::system_clock::time_point& tp);

// Format timestamps derived from --predefined_start_time without applying
// local timezone or DST conversion.
std::string formatDateTimeNoConversion(
    const std::chrono::system_clock::time_point& tp);

// Parse a local datetime string of the form "YYYY-mm-dd HH:MM:SS".
bool parseDateTime(const std::string& s, std::chrono::system_clock::time_point& out);

bool parseNDigits(const std::string& s, size_t pos, size_t count, int& out);

bool isLeapYear(int year);

int daysInMonth(int year, int month);

bool parseDateTimeParts(const std::string& s, DateTimeParts& out);

// Days since 1970-01-01 for a Gregorian civil date. This is intentionally
// timezone-free and does not consult operating-system local timezone rules.
std::int64_t daysFromCivil(int year, unsigned month, unsigned day);

// Parse an explicit datetime without local timezone or DST conversion. The
// resulting time_point must be formatted with formatDateTimeNoConversion().
bool parseDateTimeNoConversion(
    const std::string& s,
    std::chrono::system_clock::time_point& out);

// Normalize metadata datetime variants (T/Z/fractional/tz suffixes) then parse.
bool parseMetadataDateTime(
    std::string value,
    std::chrono::system_clock::time_point& out);

// Add fractional seconds to a time point.
std::chrono::system_clock::time_point addSecondsToTimePoint(
    const std::chrono::system_clock::time_point& base,
    double sec);

//------------------------------------------------------------------------------
// Strict numeric parsing helpers
//------------------------------------------------------------------------------

// Parse a double and require that the whole string is consumed.
bool parseDoubleStrict(const std::string& s, double& out);

// Parse an int and require that the whole string is consumed.
bool parseIntStrict(const std::string& s, int& out);

// Return true if the entire string is composed of decimal digits.
bool isUnsignedIntegerString(const std::string& s);

// Treat any non-empty unsigned integer as a camera index.
//
// This is more useful than a single-digit check because device indexes such
// as "10" should still work.
bool isCameraIndexSource(const std::string& s);

//------------------------------------------------------------------------------
// CLI parsing
//------------------------------------------------------------------------------

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
