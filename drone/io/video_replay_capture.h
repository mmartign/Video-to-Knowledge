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
// A GimbalCapture backend that replays three pre-recorded video files
// (wide, zoom, thermal) in lockstep, for offline testing and
// simulation without the physical gimbal.
//
// Caveat: the "thermal" input is read as an ordinary video file
// (whatever OpenCV's VideoCapture backend can decode) and converted to
// grayscale -- not DJI's proprietary R-JPEG radiometric format the
// actual H20T produces. A real integration would need to decode that
// format (or work from whatever intensity-mapped export the gimbal SDK
// provides) before frames reach this interface; see drone/README.md.
#pragma once

#include "gimbal_capture.h"

#include <opencv2/videoio.hpp>

#include <string>

namespace drone::io {

class VideoReplayCapture : public GimbalCapture {
public:
    VideoReplayCapture(
        const std::string& widePath,
        const std::string& zoomPath,
        const std::string& thermalPath);

    bool ok() const { return ok_; }

    bool readNext(GimbalFrameSet& out) override;
    double frameRateHz() const override { return frameRateHz_; }

private:
    cv::VideoCapture wide_;
    cv::VideoCapture zoom_;
    cv::VideoCapture thermal_;
    double frameRateHz_ = 0.0;
    bool ok_ = false;
};

}  // namespace drone::io
