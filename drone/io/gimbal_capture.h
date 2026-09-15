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
// Capture interface for the report's "quad-sensor gimbal" (a Zenmuse
// H20T: wide, zoom, and 640x512 radiometric thermal, plus a laser
// rangefinder the video-to-knowledge pipeline doesn't use).
//
// NOT IMPLEMENTED HERE: a live backend against DJI's Payload SDK and
// actual H20T hardware is intentionally left as a follow-up, the same
// call made for the microphone array and loudspeaker in the drone
// speech-to-knowledge work: without the gimbal and an aircraft to test
// against, a "live" backend would just be unverified code claiming to
// work. VideoReplayCapture below implements this same interface against
// three video files instead, so the rest of the pipeline (detection,
// optical flow, thermal fusion, rPPG) is real and testable end-to-end
// today, and a live backend can be dropped in later without touching
// any caller.
#pragma once

#include <opencv2/core.hpp>

namespace drone::io {

struct GimbalFrameSet {
    cv::Mat wideBgr;
    cv::Mat zoomBgr;

    // Single-channel. Whatever backend supplies this is expected to
    // have already reduced the sensor's native representation (e.g.
    // DJI's proprietary R-JPEG radiometric format) to a plain
    // intensity image -- see VideoReplayCapture's caveat about that.
    cv::Mat thermal;
};

class GimbalCapture {
public:
    virtual ~GimbalCapture() = default;

    // Reads the next time-synchronized frame set. Returns false once
    // any one of the three streams is exhausted.
    virtual bool readNext(GimbalFrameSet& out) = 0;

    virtual double frameRateHz() const = 0;
};

}  // namespace drone::io
