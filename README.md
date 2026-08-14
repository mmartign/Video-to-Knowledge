<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# SI-Watcher  
### Real-Time Video-to-Knowledge Engine for Edge AI
### *"Video-to-Knowledge, not Video-to-Identity."*

[![CI](https://github.com/mmartign/Video-to-Knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/mmartign/Video-to-Knowledge/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Security Policy](https://img.shields.io/badge/Security-Policy-brightgreen.svg)](SECURITY.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**SI-Watcher** is an edge-first, real-time **video-to-knowledge pipeline** designed to transform live or recorded video streams into **structured, actionable insights** using **multimodal Generative AI**—securely, locally, and with ultra-low latency.

The system has been **explicitly designed for healthcare applications**, where privacy, reliability, and real-time decision support are critical, and has been **tested and validated with the MedGemma-1.5:4b multimodal model** for medical-grade vision-language inference.

> **Design goals:** low latency, privacy-first processing, deterministic behavior, and full cloud independence.

---

## What SI-Watcher does

- **Ingests video** from live cameras (by device index), network streams (RTSP/HTTP), or recorded media files  
- **Samples frames** at configurable, fixed-interval triggers  
- Encodes sampled frames as JPEG data URLs and forwards them to a **multimodal GenAI model** with a configurable text prompt  
- Logs **timestamped model responses** to stdout in (near) real time, directly on edge hardware  

---

## Core capabilities

- Live video ingestion (RTSP/HTTP streams, USB/camera device indices, video files)  
- Fixed-interval frame sampling with automatic catch-up under load  
- Single-slot job queue: newer frames silently overwrite stale pending inference work, keeping latency bounded  
- Multimodal GenAI inference via any **OpenAI-compatible vision API**  
- Optional real-time local preview window (OpenCV `imshow`)  
- Edge-first architecture — runs on a single industrial or medical-grade PC  
- Cross-platform: Linux and Windows supported  
- Fully open-source and highly configurable via CLI and INI file  

---

## Healthcare-first by design

SI-Watcher was architected with **healthcare environments** in mind, including:

- **Patient privacy preservation** (no mandatory cloud dependency)  
- **On-premise deployment** in hospitals, clinics, and laboratories  
- **Low-latency decision support** for time-critical scenarios  
- **Auditability**: each inference output is tagged with both wall-clock acquisition time and encoded media timeline time  

The pipeline has been **tested with MedGemma-1.5:4b**, confirming compatibility with medical multimodal models for observational analysis, contextual interpretation, and structured reporting.

> While healthcare is a primary focus, the architecture remains domain-agnostic and adaptable to other safety-critical environments.

---

## Privacy-by-Design and No Biometric Identification

> **"Understand the situation, not the identity."**  
> **"Video-to-Knowledge, not Video-to-Identity."**

SI-Watcher and the underlying Video-to-Knowledge technology are designed to extract contextual and operational knowledge from video streams **without performing facial recognition, biometric identification, or biometric categorization** of individuals. The system focuses on events, situations, objects, activities, safety conditions, and process observations — not on identifying people.

This design choice supports a **privacy-by-design approach** and reduces regulatory risks associated with biometric processing under the **EU AI Act** and **GDPR**. By analyzing *what is happening* rather than *who a person is*, SI-Watcher enables organizations to obtain operational insights while minimizing the collection and processing of personal data.

Key characteristics:

- No facial recognition  
- No biometric identification  
- No biometric categorization  
- No creation of biometric databases  
- Focus on events, behaviors, safety, and operational conditions  
- Edge-based processing to reduce data exposure  
- Human oversight of AI-generated observations  

As a result, SI-Watcher is positioned as a **Video-to-Knowledge platform** for situational awareness and operational intelligence — not as a surveillance or identity-tracking system.

---

## Key innovations

- Multimodal GenAI applied directly to sampled video frames via inline JPEG data URLs  
- **Single-slot inference queue**: freshness is prioritised over completeness — slow inference never causes queue backlog  
- Aspect-ratio-preserving resize using `max-dim` constraint before JPEG encoding  
- Accurate **dual-timestamp logging**: wall time (program uptime) and media position (encoded timeline) are tracked independently, preventing drift during file playback  
- **Smart base-time resolution** for recorded files: tries `--predefined_start_time`, then `ffprobe start_time_realtime`, then `ffprobe creation_time`, then application start — in that order  
- Bounded-exponential backoff reconnection for live streams (250 ms → 500 ms → 1000 ms → 2000 ms, capped)  
- Compatible with any **OpenAI-compatible** vision API response shape (standard `choices[0].message.content`, content arrays, and common wrapper variants)  
- Non-blocking GUI: `waitKey(1)` allows `q`/`Esc` to quit without stalling the capture or inference pipeline  

---

## Why deploy on the edge?

- **No cloud dependency**  
- **Privacy-first**: sensitive visual data remains local  
- **Ultra-low latency** for real-time interpretation  
- **Offline-capable** and resilient to poor connectivity  

---

## Architecture

The pipeline runs four concurrent components on a single edge PC:

```
┌──────────────┐    latest frame     ┌──────────────────┐
│ Capture      │ ─────────────────▶  │ Scheduling loop  │
│ Thread       │   (mutex-protected) │ (main thread)    │
│              │                     │                  │
│ OpenCV read  │                     │ Fires at fixed   │
│ Reconnect    │                     │ intervals;       │
│ backoff      │                     │ overwrites slot  │
└──────────────┘                     └────────┬─────────┘
                                              │ single-slot job
                                              ▼
                                     ┌──────────────────┐
                                     │ Worker Thread    │
                                     │                  │
                                     │ Resize → JPEG    │
                                     │ → base64 → API   │
                                     │ → stdout log     │
                                     └──────────────────┘
```

1. **Capture thread** — reads frames from OpenCV continuously; stores only the latest frame in a mutex-protected slot; handles reconnection for live streams with exponential backoff  
2. **Scheduling loop (main thread)** — fires at fixed `--interval` cadence; copies the latest frame into the single pending inference job; also drives the optional GUI preview  
3. **Worker thread** — waits for a pending job, encodes the frame as JPEG, base64-encodes it into a data URL, sends it to the model via the OpenAI-compatible API, and prints the response to stdout  
4. **GUI (optional)** — non-blocking `imshow`/`waitKey(1)` preview; disabled with `--no-gui`  

---

## Configuration

### INI file (required)

The API connection is loaded from an INI file (default: `config.ini`).

```ini
[openai]
base_url    = http://localhost:11434/v1/   ; or any OpenAI-compatible endpoint
api_key     = your-api-key-here
vmodel_name = medgemma-15:4b              ; any vision-capable model name
```

All three keys are required. The application will print a clear error and exit if any are missing.

### Command-line options

```
Usage: <program> <video_or_stream_uri> [config.ini] [options]

Positional arguments:
  <video_or_stream_uri>          RTSP/HTTP stream URL, camera device index (e.g. 0), or video file path
  [config.ini]                   Path to INI config file (default: config.ini)

Options:
  --interval <sec>               Frame sampling interval in seconds (default: 10; minimum: 0.1)
  --max-dim <px>                 Resize frames so max(width, height) ≤ px before encoding (default: 1024; 0 = disabled)
  --jpeg-quality <1-100>         JPEG encoding quality (default: 85)
  --prompt <text>                Text prompt sent to the model with each frame (default: "Analyze this frame.")
  --no-gui                       Disable the OpenCV imshow preview window
  --reconnect-sec <sec>          Seconds to attempt reconnection before giving up on a live stream (default: 5; 0 = no retry)
  --predefined_start_time "YYYY-mm-dd HH:MM:SS"
                                 Override the base datetime for media file timestamp calculations
  --help / -h                    Print usage and exit
```

---

## Output format

Each inference result is printed to **stdout** with timing context:

```
[2026-04-27 10:15:30] media-time=30.000s  <model response text>
```

For live streams, the log line uses acquisition time and also includes `encoded-at=<timestamp>`. For file playback, the primary timestamp is derived from the encoded media timeline, anchored to the resolved base time.

Wall time and media position are also appended to the prompt sent to the model:

```
Analyze this frame. Wall time: 30.000s; media position: 30.000s; interval #3
```

---

## Dependencies

| Library | Purpose |
|---|---|
| [OpenCV](https://opencv.org/) | Video capture, frame decoding, JPEG encoding, GUI preview |
| [openai-cpp](https://github.com/olrea/openai-cpp) | OpenAI-compatible API client (`openai::chat().create(...)`) |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON serialisation of API request/response |
| ffprobe (optional, runtime) | Probing encoded `start_time_realtime` and `creation_time` from media files |

---

## Building and testing

The build is CMake-based. OpenCV and libcurl must be installed on the host;
[nlohmann/json](https://github.com/nlohmann/json) and the
[Catch2](https://github.com/catchorg/Catch2) test framework are fetched
automatically. The [openai-cpp](https://github.com/olrea/openai-cpp) headers
are expected as a sibling checkout at `../openai-cpp` (only needed to build
`realtime_video_pipeline.exe`, not the tests).

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Dependency-light logic (INI/CLI parsing, string/date helpers, API response
parsing) lives in `pipeline_core.{h,cpp}` and is covered by the unit test
suite in `tests/`. Set `-DVTK_BUILD_TESTS=OFF` to skip building tests. A
[GitHub Actions workflow](.github/workflows/ci.yml) builds and runs the test
suite on every push and pull request.

### Relevance to EU AI Act obligations

This is **not** a legal compliance assessment, and using SI-Watcher does not
by itself make any given deployment compliant with the EU AI Act (Regulation
(EU) 2024/1689). Risk classification depends on the specific deployment's
intended purpose and is the deployer's/provider's responsibility.

Where SI-Watcher is used within a system that falls under the Act's
high-risk provisions (Chapter III — e.g. via Annex III, or as a safety
component of a product already covered by Annex I legislation such as the
Medical Device Regulation), the automated test suite and CI pipeline in this
repository provide supporting technical evidence for two obligations placed
on providers of high-risk AI systems:

- **Article 15 (accuracy, robustness and cybersecurity)** — high-risk
  systems must "achieve an appropriate level of accuracy, robustness, and
  cybersecurity, and perform consistently in those respects throughout
  their lifecycle." The unit test suite (`tests/`) and the CI workflow that
  runs it on every change give a repeatable, automated check of that
  consistency for the pipeline's core logic (config/CLI parsing, timestamp
  handling, API response parsing).
- **Article 17 (quality management system)** — providers must document
  "testing and validation procedures" as part of a quality management
  system covering the AI system's lifecycle. `.github/workflows/ci.yml` and
  `tests/test_pipeline_core.cpp` are one concrete, version-controlled
  instance of such a procedure.

These practices address the *testing and robustness* dimension of the Act.
They do not, on their own, satisfy data governance (Article 10), risk
management (Article 9), technical documentation (Article 11), human
oversight (Article 14), or conformity assessment/CE marking obligations,
all of which remain the deployer's/provider's responsibility. See also
[Privacy-by-Design and No Biometric Identification](#privacy-by-design-and-no-biometric-identification)
above.

---

## Example use case (healthcare monitoring)

1. A camera observes a clinical or laboratory environment  
2. SI-Watcher samples a frame every **N seconds** (configurable via `--interval`)  
3. The frame is JPEG-encoded, base64-embedded, and sent to a multimodal model (e.g. **MedGemma-1.5:4b**) with a structured prompt  
4. The model's response is printed to stdout with a wall-clock and media-position timestamp  
5. Clinicians or operators remain in the loop for validation and action  

---

## Potential application domains

- **Healthcare and clinical observation**  
- Medical device and procedure monitoring  
- Laboratory and research environments  
- Industrial monitoring and inspections  
- Safety monitoring and situational awareness  
- Manufacturing quality control  

---

## Source heuristics

The application automatically distinguishes between **live streams/cameras** and **media files**:

- A numeric-only source string (e.g. `0`, `1`, `10`) is treated as a **camera device index**  
- Any other string is opened as a URI or file path  
- If `CAP_PROP_FRAME_COUNT` is finite and positive, and the source is not a camera index, it is treated as a **media file** (enabling real-time playback pacing and encoded-timeline timestamping)  

---

📄 **License**
This project is released under the GNU Affero General Public License v3.0 (AGPL-3.0-or-later).
You may copy, modify, and redistribute it under the AGPL terms, including source disclosure obligations for network use.
See the [LICENSE](LICENSE) file for the full text and terms.

---

## 🏢 About Spazio IT

Spazio - IT Soluzioni Informatiche s.a.s.\
via Manzoni 40\
46051 San Giorgio Bigarello\
Italy

https://spazioit.com

Part of the **OR-Edge Project** — AI-powered solutions for medical edge environments.

---

## ⚠ Disclaimer

This software is provided **without warranty**.\
It is intended for research, validation, and controlled medical IT environments.\
It does not replace certified medical decision systems.
