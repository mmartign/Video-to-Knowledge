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
// Timestamp parsing/formatting for wall-clock and encoded-media-timeline
// datetimes, including the timezone-free civil-calendar arithmetic used for
// --predefined_start_time and ffprobe metadata.
#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <string>

namespace pipeline_core {

struct DateTimeParts {
    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
};

// Thread-safe localtime wrapper.
//
// std::localtime() is not thread-safe, so we use platform-specific safe forms.
bool safeLocalTime(std::time_t t, std::tm& out);

// Thread-safe gmtime wrapper used for explicit wall-clock timestamps. The
// timestamp is stored on the Unix timeline only for duration arithmetic; gmtime
// keeps the printed civil time independent from the host local timezone.
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

}  // namespace pipeline_core
