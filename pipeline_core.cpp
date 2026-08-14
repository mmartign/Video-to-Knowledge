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
#include "pipeline_core.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>

namespace pipeline_core {

// Print CLI help.
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

//------------------------------------------------------------------------------
// INI/config parsing
//------------------------------------------------------------------------------

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

// Ensure base URLs end with '/' so downstream URL construction is consistent.
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

// Load and validate the required OpenAI config values.
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

//------------------------------------------------------------------------------
// Encoding and API response parsing
//------------------------------------------------------------------------------

std::string base64Encode(const std::vector<unsigned char>& data)
{
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i + 2 < data.size()) {
        const uint32_t triple =
            (static_cast<uint32_t>(data[i]) << 16) |
            (static_cast<uint32_t>(data[i + 1]) << 8) |
            static_cast<uint32_t>(data[i + 2]);

        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(table[(triple >> 6) & 0x3F]);
        encoded.push_back(table[triple & 0x3F]);
        i += 3;
    }

    const size_t rem = data.size() - i;
    if (rem == 1) {
        const uint32_t triple = (static_cast<uint32_t>(data[i]) << 16);
        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back('=');
        encoded.push_back('=');
    } else if (rem == 2) {
        const uint32_t triple =
            (static_cast<uint32_t>(data[i]) << 16) |
            (static_cast<uint32_t>(data[i + 1]) << 8);
        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(table[(triple >> 6) & 0x3F]);
        encoded.push_back('=');
    }

    return encoded;
}

// Extract human-readable text from an API response.
//
// We support a few plausible shapes because deployments using "OpenAI-like"
// compatibility layers may not always return identical JSON structures.
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

//------------------------------------------------------------------------------
// Time helpers
//------------------------------------------------------------------------------

// std::localtime() is not thread-safe, so we use platform-specific safe forms.
bool safeLocalTime(std::time_t t, std::tm& out)
{
#ifdef _WIN32
    return localtime_s(&out, &t) == 0;
#else
    return localtime_r(&t, &out) != nullptr;
#endif
}

// Used for explicit wall-clock timestamps. The timestamp is stored on the
// Unix timeline only for duration arithmetic; gmtime keeps the printed civil
// time independent from the host local timezone.
bool safeUtcTime(std::time_t t, std::tm& out)
{
#ifdef _WIN32
    return gmtime_s(&out, &t) == 0;
#else
    return gmtime_r(&t, &out) != nullptr;
#endif
}

std::string formatDateTime(const std::chrono::system_clock::time_point& tp)
{
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
    if (!safeLocalTime(t, tm)) {
        return "[invalid-local-time]";
    }

    std::ostringstream oss;
    oss << '[' << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << ']';
    return oss.str();
}

std::string formatDateTimeNoConversion(
    const std::chrono::system_clock::time_point& tp)
{
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
    if (!safeUtcTime(t, tm)) {
        return "[invalid-time]";
    }

    std::ostringstream oss;
    oss << '[' << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << ']';
    return oss.str();
}

bool parseDateTime(const std::string& s, std::chrono::system_clock::time_point& out)
{
    std::tm tm{};
    std::istringstream iss(s);
    iss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (iss.fail()) {
        return false;
    }

    const std::time_t tt = std::mktime(&tm);
    if (tt == static_cast<std::time_t>(-1)) {
        return false;
    }

    out = std::chrono::system_clock::from_time_t(tt);
    return true;
}

bool parseNDigits(const std::string& s, size_t pos, size_t count, int& out)
{
    if (pos + count > s.size()) {
        return false;
    }

    int value = 0;
    for (size_t i = 0; i < count; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[pos + i]);
        if (!std::isdigit(c)) {
            return false;
        }
        value = value * 10 + static_cast<int>(c - '0');
    }
    out = value;
    return true;
}

bool isLeapYear(int year)
{
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

int daysInMonth(int year, int month)
{
    static constexpr int kDaysByMonth[] = {
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && isLeapYear(year)) {
        return 29;
    }
    if (month < 1 || month > 12) {
        return 0;
    }
    return kDaysByMonth[month - 1];
}

bool parseDateTimeParts(const std::string& s, DateTimeParts& out)
{
    if (s.size() != 19 ||
        s[4] != '-' ||
        s[7] != '-' ||
        s[10] != ' ' ||
        s[13] != ':' ||
        s[16] != ':') {
        return false;
    }

    DateTimeParts parsed{};
    if (!parseNDigits(s, 0, 4, parsed.year) ||
        !parseNDigits(s, 5, 2, parsed.month) ||
        !parseNDigits(s, 8, 2, parsed.day) ||
        !parseNDigits(s, 11, 2, parsed.hour) ||
        !parseNDigits(s, 14, 2, parsed.minute) ||
        !parseNDigits(s, 17, 2, parsed.second)) {
        return false;
    }

    const int monthDays = daysInMonth(parsed.year, parsed.month);
    if (parsed.year < 1 ||
        monthDays == 0 ||
        parsed.day < 1 ||
        parsed.day > monthDays ||
        parsed.hour > 23 ||
        parsed.minute > 59 ||
        parsed.second > 59) {
        return false;
    }

    out = parsed;
    return true;
}

// Days since 1970-01-01 for a Gregorian civil date. This is intentionally
// timezone-free and does not consult operating-system local timezone rules.
std::int64_t daysFromCivil(int year, unsigned month, unsigned day)
{
    year -= month <= 2;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(year - era * 400);
    const unsigned doy =
        (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<std::int64_t>(era) * 146097 +
           static_cast<std::int64_t>(doe) -
           719468;
}

bool parseDateTimeNoConversion(
    const std::string& s,
    std::chrono::system_clock::time_point& out)
{
    DateTimeParts parts{};
    if (!parseDateTimeParts(s, parts)) {
        return false;
    }

    const std::int64_t days = daysFromCivil(
        parts.year,
        static_cast<unsigned>(parts.month),
        static_cast<unsigned>(parts.day));
    const std::int64_t totalSeconds =
        days * 86400 +
        static_cast<std::int64_t>(parts.hour) * 3600 +
        static_cast<std::int64_t>(parts.minute) * 60 +
        static_cast<std::int64_t>(parts.second);

    out = std::chrono::system_clock::time_point{
        std::chrono::seconds(totalSeconds)};
    return true;
}

// Normalize metadata datetime variants (T/Z/fractional/tz suffixes) then parse.
bool parseMetadataDateTime(
    std::string value,
    std::chrono::system_clock::time_point& out)
{
    if (!trimInPlace(value)) {
        return false;
    }

    std::replace(value.begin(), value.end(), 'T', ' ');
    if (!value.empty() && (value.back() == 'Z' || value.back() == 'z')) {
        value.pop_back();
    }

    const size_t fracPos = value.find('.');
    if (fracPos != std::string::npos) {
        value.erase(fracPos);
    }

    const size_t tzPos = value.find_first_of("+-", 19);
    if (tzPos != std::string::npos) {
        value.erase(tzPos);
    }

    if (!trimInPlace(value)) {
        return false;
    }
    if (value.size() < 19) {
        return false;
    }

    value = value.substr(0, 19);
    return parseDateTime(value, out);
}

std::chrono::system_clock::time_point addSecondsToTimePoint(
    const std::chrono::system_clock::time_point& base,
    double sec)
{
    return base + std::chrono::duration_cast<std::chrono::system_clock::duration>(
                      std::chrono::duration<double>(sec));
}

//------------------------------------------------------------------------------
// Strict numeric parsing helpers
//------------------------------------------------------------------------------

bool parseDoubleStrict(const std::string& s, double& out)
{
    try {
        size_t idx = 0;
        out = std::stod(s, &idx);
        return idx == s.size();
    } catch (...) {
        return false;
    }
}

bool parseIntStrict(const std::string& s, int& out)
{
    try {
        size_t idx = 0;
        out = std::stoi(s, &idx);
        return idx == s.size();
    } catch (...) {
        return false;
    }
}

bool isUnsignedIntegerString(const std::string& s)
{
    if (s.empty()) {
        return false;
    }
    for (unsigned char ch : s) {
        if (!std::isdigit(ch)) {
            return false;
        }
    }
    return true;
}

bool isCameraIndexSource(const std::string& s)
{
    return isUnsignedIntegerString(s);
}

//------------------------------------------------------------------------------
// CLI parsing
//------------------------------------------------------------------------------

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
