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
// A single second-order IIR filter (biquad) in direct form I, per the
// standard RBJ Audio EQ Cookbook design formulas. Used to band-limit
// the facial rPPG signal to the plausible heart-rate frequency range
// before periodicity estimation (see photoplethysmography.h).
#pragma once

namespace drone::video_core {

struct Biquad {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;  // a0 normalized to 1
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;

    static Biquad makeLowpass(double cutoffHz, double sampleRateHz, double q);
    static Biquad makeHighpass(double cutoffHz, double sampleRateHz, double q);

    double process(double input);
};

}  // namespace drone::video_core
