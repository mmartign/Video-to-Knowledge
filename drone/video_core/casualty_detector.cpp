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
#include "casualty_detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace drone::video_core {

namespace {

// The default people detector is trained on upright, mostly-visible
// pedestrians and returns a full-body box; approximating "upper torso"
// as the top fraction of that box (head + shoulders + chest, excluding
// legs) is a coarse stand-in given there's no torso-specific model here.
constexpr double kTorsoFractionOfBody = 0.55;

// Rough head fraction of a full-body box (for approximateNaresRegion),
// and where within that head region the nares would fall for an
// upright, camera-facing subject: horizontally centered, in the lower
// half (a nose/nares sits below eye level, not at the top of the head).
constexpr double kHeadFractionOfBody = 0.18;
constexpr double kNaresWidthFractionOfBody = 0.30;
constexpr double kNaresHeightFractionOfHead = 0.35;
constexpr double kNaresVerticalOffsetFractionOfHead = 0.50;

}  // namespace

CasualtyDetector::CasualtyDetector(const std::string& faceCascadePath)
{
    hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    faceCascadeLoaded_ = faceCascade_.load(faceCascadePath);
}

bool CasualtyDetector::detectPerson(const cv::Mat& wideFrameBgr, cv::Rect& outPersonRoi) const
{
    if (wideFrameBgr.empty()) {
        return false;
    }

    std::vector<cv::Rect> detections;
    std::vector<double> weights;
    hog_.detectMultiScale(
        wideFrameBgr, detections, weights,
        /*hitThreshold=*/0.0, /*winStride=*/cv::Size(8, 8),
        /*padding=*/cv::Size(4, 4), /*scale=*/1.05);

    if (detections.empty()) {
        return false;
    }

    size_t bestIdx = 0;
    for (size_t i = 1; i < detections.size(); ++i) {
        if (weights[i] > weights[bestIdx]) {
            bestIdx = i;
        }
    }

    outPersonRoi = detections[bestIdx];
    return true;
}

bool CasualtyDetector::detectTorso(const cv::Mat& wideFrameBgr, cv::Rect& outTorsoRoi) const
{
    cv::Rect person;
    if (!detectPerson(wideFrameBgr, person)) {
        return false;
    }
    outTorsoRoi = torsoRegionFromPerson(person);
    return true;
}

bool CasualtyDetector::detectFace(const cv::Mat& zoomFrameBgr, cv::Rect& outFaceRoi) const
{
    if (!faceCascadeLoaded_ || zoomFrameBgr.empty()) {
        return false;
    }

    cv::Mat gray;
    cv::cvtColor(zoomFrameBgr, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    faceCascade_.detectMultiScale(
        gray, faces, /*scaleFactor=*/1.1, /*minNeighbors=*/4,
        /*flags=*/0, /*minSize=*/cv::Size(30, 30));

    if (faces.empty()) {
        return false;
    }

    outFaceRoi = *std::max_element(
        faces.begin(), faces.end(),
        [](const cv::Rect& a, const cv::Rect& b) { return a.area() < b.area(); });
    return true;
}

cv::Rect torsoRegionFromPerson(const cv::Rect& personRoi)
{
    return cv::Rect(
        personRoi.x, personRoi.y,
        personRoi.width,
        std::max(1, static_cast<int>(personRoi.height * kTorsoFractionOfBody)));
}

cv::Rect approximateNaresRegion(const cv::Rect& personRoi)
{
    const int headHeight = std::max(1, static_cast<int>(personRoi.height * kHeadFractionOfBody));
    const int naresWidth = std::max(1, static_cast<int>(personRoi.width * kNaresWidthFractionOfBody));
    const int naresHeight = std::max(1, static_cast<int>(headHeight * kNaresHeightFractionOfHead));

    const int naresX = personRoi.x + (personRoi.width - naresWidth) / 2;
    const int naresY = personRoi.y +
        static_cast<int>(headHeight * kNaresVerticalOffsetFractionOfHead);

    return cv::Rect(naresX, naresY, naresWidth, naresHeight);
}

}  // namespace drone::video_core
