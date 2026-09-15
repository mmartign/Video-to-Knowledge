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
// The structured output of the video-to-knowledge pipeline: per the
// report, "Respiratory rate is recovered from optical-flow displacement
// of the thorax fused with thermal intensity change at the nares; heart
// rate and a perfusion index are obtained from facial remote
// photoplethysmography on the zoom camera. When optical signal quality
// is degraded by dust or vasoconstriction, the pipeline falls back to
// the core-to-periphery thermal gradient."
#pragma once

namespace drone::video_core {

struct VitalSigns {
    double respiratoryRateBpm = 0.0;
    bool respiratoryRateValid = false;

    double heartRateBpm = 0.0;
    bool heartRateValid = false;

    // Unitless AC/DC ratio of the pulsatile facial signal when rPPG
    // succeeded, or a normalized core-to-periphery thermal gradient
    // when it fell back -- see perfusionFromThermalFallback.
    double perfusionIndex = 0.0;
    bool perfusionIndexValid = false;

    // True if perfusionIndex came from the thermal-gradient fallback
    // rather than facial rPPG (i.e. optical signal quality was too low
    // -- dust obscuring the face, or vasoconstriction flattening the
    // pulsatile signal).
    bool perfusionFromThermalFallback = false;
};

}  // namespace drone::video_core
