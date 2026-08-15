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
#include "datetime.h"

#include "strings.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace pipeline_core {

bool safeLocalTime(std::time_t t, std::tm& out)
{
#ifdef _WIN32
    return localtime_s(&out, &t) == 0;
#else
    return localtime_r(&t, &out) != nullptr;
#endif
}

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

}  // namespace pipeline_core
