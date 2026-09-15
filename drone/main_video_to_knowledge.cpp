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
// Entry point for the drone video-to-knowledge pipeline: three
// synchronized video files (wide, zoom, thermal) in -> casualty
// detection on the wide stream -> per-window vital signs (respiratory
// rate fused from optical flow + thermal, heart rate and perfusion from
// facial rPPG with a thermal-gradient fallback) -> structured JSON out.
//
// Like the drone speech-to-knowledge pipeline, this runs in
// offline/replay mode over video files rather than a live gimbal feed:
// there is no live H20T/Payload SDK backend yet (see
// io/gimbal_capture.h), so a real-time loop would have nothing genuine
// to drive it. This is otherwise the same pipeline a live backend would
// feed.
#include "video_core/casualty_detector.h"
#include "video_core/optical_flow_respiration.h"
#include "video_core/thermal_respiration.h"
#include "video_core/respiration_fusion.h"
#include "video_core/photoplethysmography.h"
#include "video_core/thermal_gradient.h"
#include "video_core/vital_signs.h"

#include "io/video_replay_capture.h"

#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace drone::video_core;

namespace {

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " <wide.mp4> <zoom.mp4> <thermal.mp4> --face-cascade <path> [options]\n"
        << "Options:\n"
        << "  --face-cascade <path>   Haar cascade XML for face detection (required for heart rate/perfusion)\n"
        << "  --window-seconds <n>    Analysis window length in seconds (default: 10)\n";
}

// A thin strip near the edge of the person's box, standing in for an
// exposed limb/extremity -- there's no separate extremity detector
// here; see drone/README.md.
cv::Rect approximatePeripheryRegion(const cv::Rect& personRoi)
{
    const int stripWidth = std::max(1, static_cast<int>(personRoi.width * 0.15));
    return cv::Rect(
        personRoi.x + personRoi.width - stripWidth,
        personRoi.y + personRoi.height / 2,
        stripWidth,
        std::max(1, personRoi.height / 2));
}

std::string vitalSignsToJson(const VitalSigns& v, double windowStartSec)
{
    json j = {
        {"window_start_sec", windowStartSec},
        {"respiratory_rate_bpm", v.respiratoryRateValid ? json(v.respiratoryRateBpm) : json(nullptr)},
        {"heart_rate_bpm", v.heartRateValid ? json(v.heartRateBpm) : json(nullptr)},
        {"perfusion_index", v.perfusionIndexValid ? json(v.perfusionIndex) : json(nullptr)},
        {"perfusion_from_thermal_fallback", v.perfusionFromThermalFallback},
    };
    return j.dump();
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string widePath = argv[1];
    const std::string zoomPath = argv[2];
    const std::string thermalPath = argv[3];
    std::string faceCascadePath;
    double windowSeconds = 10.0;

    for (int argi = 4; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (a == "--face-cascade" && argi + 1 < argc) {
            faceCascadePath = argv[++argi];
        } else if (a == "--window-seconds" && argi + 1 < argc) {
            windowSeconds = std::stod(argv[++argi]);
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "[ERROR] Unknown option: " << a << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    CasualtyDetector detector(faceCascadePath);
    if (!detector.ok()) {
        std::cerr << "[WARN] Could not load face cascade from \"" << faceCascadePath
                  << "\" -- heart rate/perfusion via rPPG will be skipped "
                     "(thermal-gradient fallback will still run).\n";
    }

    drone::io::VideoReplayCapture capture(widePath, zoomPath, thermalPath);
    if (!capture.ok()) {
        std::cerr << "[ERROR] Could not open one or more of the wide/zoom/thermal video files\n";
        return 1;
    }

    const double frameRateHz = capture.frameRateHz();
    if (frameRateHz <= 0.0) {
        std::cerr << "[ERROR] Could not determine frame rate from the wide video stream\n";
        return 1;
    }
    std::cerr << "[INFO] Frame rate: " << frameRateHz << " Hz, window: " << windowSeconds << " s\n";

    const size_t windowFrameCount =
        std::max<size_t>(2, static_cast<size_t>(windowSeconds * frameRateHz));

    std::vector<cv::Mat> wideFrames, zoomFrames, thermalFrames;
    double windowStartSec = 0.0;
    int windowIdx = 0;

    drone::io::GimbalFrameSet frame;
    while (capture.readNext(frame)) {
        wideFrames.push_back(frame.wideBgr);
        zoomFrames.push_back(frame.zoomBgr);
        thermalFrames.push_back(frame.thermal);

        if (wideFrames.size() < windowFrameCount) {
            continue;
        }

        cv::Rect person;
        if (!detector.detectPerson(wideFrames.front(), person)) {
            std::cerr << "[WARN] Window " << windowIdx << ": no casualty detected; skipping\n";
        } else {
            const cv::Rect torsoRoi = torsoRegionFromPerson(person);
            const cv::Rect naresRoi = approximateNaresRegion(person);

            std::vector<cv::Mat> grayWideFrames;
            grayWideFrames.reserve(wideFrames.size());
            for (const auto& f : wideFrames) {
                cv::Mat gray;
                cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY);
                grayWideFrames.push_back(gray);
            }

            const auto opticalFlowResp =
                estimateRespirationFromOpticalFlow(grayWideFrames, torsoRoi, frameRateHz);
            const auto thermalResp =
                estimateRespirationFromThermal(thermalFrames, naresRoi, frameRateHz);
            const auto fusedResp = fuseRespirationEstimates(opticalFlowResp, thermalResp);

            VitalSigns vitals;
            vitals.respiratoryRateValid = fusedResp.valid;
            vitals.respiratoryRateBpm = fusedResp.ratePerMinute;

            cv::Rect face;
            RppgResult rppg;
            const bool faceFound = detector.ok() && detector.detectFace(zoomFrames.front(), face);
            if (faceFound) {
                rppg = estimateHeartRateAndPerfusion(zoomFrames, face, frameRateHz);
            }

            // Per the report: fall back to the core-to-periphery
            // thermal gradient when the optical (rPPG) signal quality
            // is degraded -- approximated here as "no face found" or a
            // low-quality/absent periodicity estimate.
            constexpr double kMinRppgQuality = 0.4;
            const bool rppgUsable = faceFound && rppg.periodicity.valid &&
                                     rppg.periodicity.quality >= kMinRppgQuality;

            if (rppgUsable) {
                vitals.heartRateValid = true;
                vitals.heartRateBpm = rppg.periodicity.ratePerMinute;
            }

            if (rppgUsable && rppg.perfusionIndexValid) {
                vitals.perfusionIndex = rppg.perfusionIndex;
                vitals.perfusionIndexValid = true;
                vitals.perfusionFromThermalFallback = false;
            } else {
                const cv::Rect peripheryRoi = approximatePeripheryRegion(person);
                const double gradient =
                    computeCoreToPeripheryGradient(thermalFrames.front(), torsoRoi, peripheryRoi);
                vitals.perfusionIndex = gradient;
                vitals.perfusionIndexValid = true;
                vitals.perfusionFromThermalFallback = true;
            }

            std::cout << vitalSignsToJson(vitals, windowStartSec) << std::endl;
        }

        windowStartSec += static_cast<double>(wideFrames.size()) / frameRateHz;
        ++windowIdx;
        wideFrames.clear();
        zoomFrames.clear();
        thermalFrames.clear();
    }

    return 0;
}
