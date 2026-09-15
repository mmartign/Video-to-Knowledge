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
#include "optical_flow_respiration.h"

#include <opencv2/video/tracking.hpp>

namespace drone::video_core {

std::vector<double> extractThoraxDisplacementSignal(
    const std::vector<cv::Mat>& grayFrames,
    const cv::Rect& torsoRoi)
{
    std::vector<double> signal;
    if (grayFrames.size() < 2) {
        return signal;
    }
    signal.reserve(grayFrames.size() - 1);

    const cv::Rect frameBounds(0, 0, grayFrames.front().cols, grayFrames.front().rows);
    const cv::Rect roi = torsoRoi & frameBounds;
    if (roi.area() <= 0) {
        return signal;
    }

    cv::Mat flow;
    for (size_t i = 1; i < grayFrames.size(); ++i) {
        cv::calcOpticalFlowFarneback(
            grayFrames[i - 1], grayFrames[i], flow,
            /*pyr_scale=*/0.5, /*levels=*/3, /*winsize=*/15,
            /*iterations=*/3, /*poly_n=*/5, /*poly_sigma=*/1.2, /*flags=*/0);

        const cv::Mat flowRoi = flow(roi);
        double sumY = 0.0;
        for (int r = 0; r < flowRoi.rows; ++r) {
            for (int c = 0; c < flowRoi.cols; ++c) {
                sumY += static_cast<double>(flowRoi.at<cv::Point2f>(r, c).y);
            }
        }
        signal.push_back(sumY / static_cast<double>(flowRoi.total()));
    }

    return signal;
}

PeriodicityEstimate estimateRespirationFromOpticalFlow(
    const std::vector<cv::Mat>& grayFrames,
    const cv::Rect& torsoRoi,
    double frameRateHz)
{
    const auto signal = extractThoraxDisplacementSignal(grayFrames, torsoRoi);
    return estimatePeriodicity(
        signal, frameRateHz,
        kMinRespirationRateBpm / 60.0, kMaxRespirationRateBpm / 60.0);
}

}  // namespace drone::video_core
