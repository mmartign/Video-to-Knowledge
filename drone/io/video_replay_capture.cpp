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
#include "video_replay_capture.h"

#include <opencv2/imgproc.hpp>

namespace drone::io {

VideoReplayCapture::VideoReplayCapture(
    const std::string& widePath,
    const std::string& zoomPath,
    const std::string& thermalPath)
    : wide_(widePath), zoom_(zoomPath), thermal_(thermalPath)
{
    ok_ = wide_.isOpened() && zoom_.isOpened() && thermal_.isOpened();
    if (ok_) {
        // Use the wide stream's rate as the reference; a real gimbal's
        // three sensors would be genuinely synchronized, which three
        // independently-encoded test files are not guaranteed to be.
        frameRateHz_ = wide_.get(cv::CAP_PROP_FPS);
    }
}

bool VideoReplayCapture::readNext(GimbalFrameSet& out)
{
    if (!ok_) {
        return false;
    }

    cv::Mat wideFrame, zoomFrame, thermalFrame;
    if (!wide_.read(wideFrame) || !zoom_.read(zoomFrame) || !thermal_.read(thermalFrame)) {
        return false;
    }

    out.wideBgr = wideFrame;
    out.zoomBgr = zoomFrame;
    if (thermalFrame.channels() > 1) {
        cv::cvtColor(thermalFrame, out.thermal, cv::COLOR_BGR2GRAY);
    } else {
        out.thermal = thermalFrame;
    }
    return true;
}

}  // namespace drone::io
