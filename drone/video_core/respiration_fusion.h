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
// Fuses the optical-flow-based and thermal-based respiratory rate
// estimates into one, per the report's "optical-flow displacement of
// the thorax fused with thermal intensity change at the nares" --
// unlike heart-rate/perfusion, this is a fusion of two always-computed
// estimates, not a primary-with-fallback.
#pragma once

#include "periodicity.h"

namespace drone::video_core {

// Combines two independent respiratory rate estimates:
// - both valid: quality-weighted average of the two rates (the
//   higher-quality estimate contributes more), UNLESS they disagree by
//   more than a plausible single-breath margin, in which case the
//   higher-quality one alone is trusted rather than averaging two
//   estimates that likely aren't tracking the same breath;
// - exactly one valid: that one, unchanged;
// - neither valid: invalid.
PeriodicityEstimate fuseRespirationEstimates(
    const PeriodicityEstimate& fromOpticalFlow,
    const PeriodicityEstimate& fromThermal);

}  // namespace drone::video_core
