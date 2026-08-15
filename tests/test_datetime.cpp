// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for pipeline_core/datetime.h.
#include "pipeline_core/datetime.h"

#include <catch2/catch_test_macros.hpp>

using namespace pipeline_core;

TEST_CASE("parseDateTimeNoConversion / formatDateTimeNoConversion roundtrip", "[datetime]")
{
    std::chrono::system_clock::time_point tp;
    REQUIRE(parseDateTimeNoConversion("2026-04-27 10:15:30", tp));
    REQUIRE(formatDateTimeNoConversion(tp) == "[2026-04-27 10:15:30]");
}

TEST_CASE("parseDateTimeNoConversion rejects malformed and out-of-range dates", "[datetime]")
{
    std::chrono::system_clock::time_point tp;
    REQUIRE_FALSE(parseDateTimeNoConversion("not-a-date", tp));
    REQUIRE_FALSE(parseDateTimeNoConversion("2026-13-01 00:00:00", tp));  // invalid month
    REQUIRE_FALSE(parseDateTimeNoConversion("2026-02-30 00:00:00", tp));  // invalid day
    REQUIRE_FALSE(parseDateTimeNoConversion("2026-04-27 24:00:00", tp));  // invalid hour
    REQUIRE_FALSE(parseDateTimeNoConversion("2026-04-27T10:15:30", tp));  // wrong separator
}

TEST_CASE("parseDateTimeNoConversion accepts Feb 29 on leap years only", "[datetime]")
{
    std::chrono::system_clock::time_point tp;
    REQUIRE(parseDateTimeNoConversion("2024-02-29 00:00:00", tp));   // 2024 is a leap year
    REQUIRE_FALSE(parseDateTimeNoConversion("2023-02-29 00:00:00", tp));  // 2023 is not
}

TEST_CASE("isLeapYear follows Gregorian rules", "[datetime]")
{
    REQUIRE(isLeapYear(2024));
    REQUIRE_FALSE(isLeapYear(2023));
    REQUIRE_FALSE(isLeapYear(1900));  // divisible by 100, not 400
    REQUIRE(isLeapYear(2000));        // divisible by 400
}

TEST_CASE("daysInMonth accounts for leap February", "[datetime]")
{
    REQUIRE(daysInMonth(2024, 2) == 29);
    REQUIRE(daysInMonth(2023, 2) == 28);
    REQUIRE(daysInMonth(2026, 4) == 30);
    REQUIRE(daysInMonth(2026, 13) == 0);
}

TEST_CASE("daysFromCivil matches known epoch offsets", "[datetime]")
{
    REQUIRE(daysFromCivil(1970, 1, 1) == 0);
    REQUIRE(daysFromCivil(1969, 12, 31) == -1);
    REQUIRE(daysFromCivil(2000, 1, 1) == 10957);
}

TEST_CASE("parseNDigits requires exact digit count and rejects non-digits", "[datetime]")
{
    int out = 0;
    REQUIRE(parseNDigits("2026", 0, 4, out));
    REQUIRE(out == 2026);

    REQUIRE_FALSE(parseNDigits("20a6", 0, 4, out));
    REQUIRE_FALSE(parseNDigits("202", 0, 4, out));  // not enough characters
}

TEST_CASE("parseMetadataDateTime normalizes T/Z/fractional/timezone suffixes", "[datetime]")
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

TEST_CASE("parseMetadataDateTime rejects empty or too-short values", "[datetime]")
{
    std::chrono::system_clock::time_point tp;
    REQUIRE_FALSE(parseMetadataDateTime("", tp));
    REQUIRE_FALSE(parseMetadataDateTime("   ", tp));
    REQUIRE_FALSE(parseMetadataDateTime("2026-04-27", tp));
}

TEST_CASE("addSecondsToTimePoint advances by fractional seconds", "[datetime]")
{
    std::chrono::system_clock::time_point base;
    REQUIRE(parseDateTimeNoConversion("2026-04-27 10:15:30", base));

    const auto later = addSecondsToTimePoint(base, 90.0);
    REQUIRE(formatDateTimeNoConversion(later) == "[2026-04-27 10:17:00]");
}
