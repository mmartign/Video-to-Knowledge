// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for video_core/casualty_detector.h.
#include "../video_core/casualty_detector.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <fstream>

using namespace drone::video_core;

TEST_CASE("torsoRegionFromPerson takes the upper fraction of the person box", "[casualty_detector]")
{
    const cv::Rect person(10, 20, 100, 200);
    const cv::Rect torso = torsoRegionFromPerson(person);

    REQUIRE(torso.x == person.x);
    REQUIRE(torso.y == person.y);
    REQUIRE(torso.width == person.width);
    REQUIRE(torso.height < person.height);
    REQUIRE(torso.height > person.height / 2);  // more than half, per kTorsoFractionOfBody = 0.55
}

TEST_CASE("approximateNaresRegion falls within the upper-center of the person box", "[casualty_detector]")
{
    const cv::Rect person(0, 0, 200, 400);
    const cv::Rect nares = approximateNaresRegion(person);

    // Fully contained within the person box.
    REQUIRE((nares & person) == nares);

    // Roughly horizontally centered.
    const int naresCenterX = nares.x + nares.width / 2;
    const int personCenterX = person.x + person.width / 2;
    REQUIRE(std::abs(naresCenterX - personCenterX) < person.width / 10);

    // In the upper portion of the box (a head/nares region, not the torso).
    REQUIRE(nares.y + nares.height < person.y + person.height / 2);
}

TEST_CASE("CasualtyDetector reports not ok() for a nonexistent cascade file", "[casualty_detector]")
{
    CasualtyDetector detector("/nonexistent/cascade.xml");
    REQUIRE_FALSE(detector.ok());

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect face;
    REQUIRE_FALSE(detector.detectFace(frame, face));
}

TEST_CASE("CasualtyDetector::detectPerson returns false on a blank frame", "[casualty_detector]")
{
    CasualtyDetector detector("/nonexistent/cascade.xml");
    cv::Mat blank(200, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Rect person;
    REQUIRE_FALSE(detector.detectPerson(blank, person));
}

TEST_CASE("CasualtyDetector loads a real Haar cascade when one is available on this host", "[casualty_detector]")
{
    // Best-effort: check a few common install locations rather than
    // depending on one; skip (not fail) if none are found, so this test
    // stays portable across hosts/CI images that don't have OpenCV's
    // bundled cascade data installed at a predictable path.
    static const std::array<const char*, 4> candidates = {
        "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    };

    std::string found;
    for (const char* candidate : candidates) {
        if (std::ifstream(candidate).good()) {
            found = candidate;
            break;
        }
    }

    if (found.empty()) {
        SKIP("No OpenCV Haar cascade data found at a known path on this host");
    }

    CasualtyDetector detector(found);
    REQUIRE(detector.ok());

    // A blank frame has no face in it; the real cascade should
    // correctly find nothing rather than a false positive.
    cv::Mat blank(200, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Rect face;
    REQUIRE_FALSE(detector.detectFace(blank, face));
}
