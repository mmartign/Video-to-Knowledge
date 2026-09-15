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
// The report specifies "a quantized detector on the wide-camera feed to
// isolate the upper torso and facial plane" -- a trained model. This
// environment has no such model (no training data, no training
// pipeline, same situation as the drone speech-to-knowledge work's
// "convolutional classifier for non-verbal distress"). What's here
// instead is a real, working classical-CV stand-in behind the same
// interface a trained detector would sit behind: OpenCV's built-in HOG
// person detector for the torso (wide camera) and a Haar cascade for
// the facial plane (zoom camera). Both are genuine, testable detectors
// -- not stubs -- just not the report's quantized model.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <string>

namespace drone::video_core {

class CasualtyDetector {
public:
    // `faceCascadePath` must point at a Haar cascade XML file (e.g.
    // OpenCV's bundled haarcascade_frontalface_default.xml -- see
    // drone/README.md for where that typically lives). Construction
    // never throws; check ok() before use.
    explicit CasualtyDetector(const std::string& faceCascadePath);

    bool ok() const { return faceCascadeLoaded_; }

    // Detects the most prominent person's full bounding box in a
    // wide-camera BGR frame via OpenCV's default HOG people detector.
    // Returns false if nothing was detected.
    bool detectPerson(const cv::Mat& wideFrameBgr, cv::Rect& outPersonRoi) const;

    // Detects the most prominent person and returns the upper fraction
    // of their box as an "upper torso" approximation, a person
    // detector being what's available without a torso-specific model.
    // Equivalent to detectPerson() followed by torsoRegionFromPerson().
    bool detectTorso(const cv::Mat& wideFrameBgr, cv::Rect& outTorsoRoi) const;

    // Detects the largest face in a zoom-camera BGR frame via a Haar
    // cascade. Returns false if ok() is false or nothing was detected.
    bool detectFace(const cv::Mat& zoomFrameBgr, cv::Rect& outFaceRoi) const;

private:
    mutable cv::HOGDescriptor hog_;
    mutable cv::CascadeClassifier faceCascade_;  // detectMultiScale() isn't const
    bool faceCascadeLoaded_ = false;
};

// The upper fraction of a full-body detection approximating "upper
// torso" (head + shoulders + chest, excluding legs) -- see
// CasualtyDetector::detectTorso().
cv::Rect torsoRegionFromPerson(const cv::Rect& personRoi);

// A coarse geometric guess at where a nares ROI would fall within a
// full-body detection on the *same* wide/thermal-registered frame (top
// of the box, horizontally centered), for thermal respiration
// tracking. There's no separate nares/head detector or cross-sensor
// registration here -- see thermal_respiration.h and drone/README.md.
cv::Rect approximateNaresRegion(const cv::Rect& personRoi);

}  // namespace drone::video_core
