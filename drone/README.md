<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Drone Video-to-Knowledge

An implementation of the **video-to-knowledge** pipeline from the
companion-computer architecture described in the project's Innovation
Report (airborne edge triage for earthquake mass-casualty incidents):

> Video-to-knowledge runs a quantized detector on the wide-camera feed to
> isolate the upper torso and facial plane. Respiratory rate is recovered
> from optical-flow displacement of the thorax fused with thermal
> intensity change at the nares; heart rate and a perfusion index are
> obtained from facial remote photoplethysmography on the zoom camera.
> When optical signal quality is degraded by dust or vasoconstriction,
> the pipeline falls back to the core-to-periphery thermal gradient.

This directory implements exactly that. It does **not** implement
speech-to-knowledge (maintained in a separate repository) or the
cross-pipeline rule-based triage aggregator/telemetry/flight-control —
those are out of scope here.

## Scope and what's real vs. a documented stand-in

Everything below is real, working code unless flagged otherwise — no
silent stubs. Where the report specifies a trained model or hardware/SDK
this environment doesn't have, that's called out explicitly.

**"a quantized detector ... to isolate the upper torso and facial
plane"** — `video_core/casualty_detector.h` is explicitly **not** that.
A quantized neural detector needs training data and a training pipeline
this environment has neither of (the same situation as the drone
speech-to-knowledge work's "convolutional classifier for non-verbal
distress"). What's there instead is a real, working classical-CV
stand-in behind the same interface a trained detector would sit behind:
OpenCV's built-in HOG person detector for the torso (wide camera), and a
Haar cascade for the facial plane (zoom camera). Both are genuine,
testable detectors, just not the report's quantized model. There's also
no separate nares/head detector, so `approximateNaresRegion()` is a
coarse geometric guess within the same person detection (documented in
its own header comment) rather than a registered landmark location.

**Respiratory rate** — real, and matches the report's own description
closely: `video_core/optical_flow_respiration.h` recovers thorax
displacement via OpenCV dense optical flow (Farneback) on the torso ROI;
`video_core/thermal_respiration.h` recovers the nares thermal-intensity
signal the same way; `video_core/periodicity.h` (shared with heart rate
below) turns either 1D signal into a rate via normalized autocorrelation;
`video_core/respiration_fusion.h` combines the two into one estimate,
quality-weighted, per "fused with".

**Heart rate and perfusion index** — real: `video_core/photoplethysmography.h`
tracks the face ROI's mean green-channel intensity (green carries the
strongest plethysmographic signal), band-pass filters it to the
plausible heart-rate range, and recovers a rate via the same periodicity
estimator, plus a perfusion index from the filtered (AC) / raw-mean (DC)
ratio.

**Thermal-gradient fallback** — real: `video_core/thermal_gradient.h`
computes a core-minus-periphery mean thermal intensity, used for the
perfusion index whenever rPPG isn't usable (no face found, or the
recovered periodicity's quality is below a threshold — the closest
proxy available here to the report's "optical signal quality is
degraded by dust or vasoconstriction", since there's no independent
signal-quality sensor). This is a relative, uncalibrated proxy, not in
the same units as the rPPG-derived AC/DC ratio — see the comment in
`vital_signs.h`.

**The Zenmuse H20T gimbal** — `io/gimbal_capture.h` defines the capture
interface (three synchronized streams: wide, zoom, thermal); no live
backend against DJI's Payload SDK and actual hardware is implemented
(no gimbal or aircraft here to build and verify one against, the same
call made for the ReSpeaker array and Zenmuse V1 loudspeaker in the
speech-to-knowledge work). `io/video_replay_capture.h` implements the
same interface against three video files instead, so the rest of the
pipeline is real and testable end-to-end today. One caveat specific to
this interface: the "thermal" stream is read and decoded as an ordinary
video file, not DJI's proprietary R-JPEG radiometric format the H20T
actually produces — a real integration needs to decode that (or work
from whatever intensity-mapped export the gimbal SDK provides) before
frames reach this interface.

## Architecture

```
video_core/            Pure(ish) CV logic -- depends on OpenCV
                       (unavoidable for image/video processing) but not
                       on the capture/network-facing io/ layer. Unit
                       tested with synthetic in-memory frames, no
                       camera/gimbal/video files needed.
  vital_signs.h            VitalSigns result struct
  periodicity.{h,cpp}      shared autocorrelation-based rate estimator
  optical_flow_respiration.{h,cpp}  thorax displacement -> respiratory rate
  thermal_respiration.{h,cpp}       nares thermal signal -> respiratory rate
  respiration_fusion.{h,cpp}        combines the two above
  biquad.{h,cpp}            shared IIR biquad (rPPG band-pass)
  photoplethysmography.{h,cpp}      facial rPPG -> heart rate + perfusion
  thermal_gradient.{h,cpp}          core-to-periphery fallback perfusion proxy
  casualty_detector.{h,cpp}         torso/face detection (see above)

io/                    Hardware-facing -- not unit tested for the same
                       reason realtime_video_pipeline.cpp in the parent
                       project isn't, except video_replay_capture.{h,cpp},
                       which needs no hardware and is tested (fixture
                       videos are generated on the fly with
                       cv::VideoWriter, not checked in as binary data).
  gimbal_capture.h                  capture interface (no live backend)
  video_replay_capture.{h,cpp}      video-file capture backend

main_video_to_knowledge.cpp   Wires it all together over three video files.
```

Because there's no live gimbal backend, `drone_video_to_knowledge` runs
in **offline/replay mode**: it takes three pre-recorded video files, not
a live feed, and prints one JSON result per analysis window.

## Building and testing

Self-contained CMake project:

```sh
cmake -S drone -B drone/build
cmake --build drone/build
ctest --test-dir drone/build --output-on-failure
```

Requires OpenCV on the host (core, imgproc, video, videoio, objdetect);
nlohmann/json and Catch2 are fetched automatically. Set
`-DDRONE_BUILD_TESTS=OFF` to skip building tests. One test
(`CasualtyDetector loads a real Haar cascade...`) needs OpenCV's bundled
cascade data at a conventional install path and passes trivially without
testing anything, rather than failing, if none is found on the host --
e.g. on the CI image, whose `libopencv-dev` doesn't ship that data.

## Running

```sh
cmake --build drone/build
./drone/build/drone_video_to_knowledge <wide.mp4> <zoom.mp4> <thermal.mp4> \
    --face-cascade <path/to/haarcascade_frontalface_default.xml> [--window-seconds 10]
```

The three video files should be the same length and roughly the same
frame rate; `VideoReplayCapture` uses the wide stream's reported rate as
the reference for all three. Common locations for OpenCV's bundled Haar
cascade (installed with OpenCV itself, no separate download):
Homebrew — `$(brew --prefix opencv)/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`;
Debian/Ubuntu — `/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`.

Each analysis window prints one JSON line to stdout, e.g.:

```json
{"window_start_sec":0.0,"respiratory_rate_bpm":16.2,"heart_rate_bpm":78.4,"perfusion_index":0.031,"perfusion_from_thermal_fallback":false}
```

`respiratory_rate_bpm`, `heart_rate_bpm`, and `perfusion_index` are
`null` when that particular signal couldn't be recovered for the
window (e.g. no casualty detected, or the periodicity estimator didn't
clear its quality threshold).
