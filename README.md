# SI-Watcher  
### Real-Time Video-to-Knowledge Engine for Edge AI

**SI-Watcher** is an edge-first, real-time **video-to-knowledge pipeline** designed to transform live or recorded video streams into **structured, actionable insights** using **multimodal Generative AI**—securely, locally, and with ultra-low latency.

The system has been **explicitly designed for healthcare applications**, where privacy, reliability, and real-time decision support are critical, and has been **tested and validated with the MedGemma-1.5:4b multimodal model** for medical-grade vision-language inference.

> **Design goals:** low latency, privacy-first processing, deterministic behavior, and full cloud independence.

---

## What SI-Watcher does

- **Ingests video** from live cameras, network streams, or recorded files  
- **Samples frames** at configurable, time-based intervals (trigger-style sampling)  
- Performs **context-aware vision analysis** using multimodal GenAI (images + prompts)  
- Generates **structured outputs** in (near) real time directly on edge hardware  

---

## Core capabilities

- Live video ingestion (RTSP/HTTP streams, USB cameras, video files)  
- Adaptive, time-based frame sampling  
- Multimodal GenAI inference (vision + language)  
- Edge-first architecture (single industrial or medical-grade PC)  
- Fully open-source and highly configurable  

---

## Healthcare-first by design

SI-Watcher was architected with **healthcare environments** in mind, including:

- **Patient privacy preservation** (no mandatory cloud dependency)  
- **On-premise deployment** in hospitals, clinics, and laboratories  
- **Low-latency decision support** for time-critical scenarios  
- **Auditability and structured outputs** suitable for clinical workflows  

The pipeline has been **tested with MedGemma-1.5:4b**, confirming compatibility with medical multimodal models for observational analysis, contextual interpretation, and structured reporting.

> While healthcare is a primary focus, the architecture remains domain-agnostic and adaptable to other safety-critical environments.

---

## Key innovations

- Multimodal GenAI applied directly to sampled video frames  
- Non-blocking, resilient pipeline with frame dropping under load  
- Edge-optimized JPEG encoding with adaptive resizing  
- Robust live-stream reconnection handling  
- CLI-first, automation-friendly design  

---

## Why deploy on the edge?

- **No cloud dependency**  
- **Privacy-first**: sensitive visual data remains local  
- **Ultra-low latency** for real-time interpretation  
- **Offline-capable** and resilient to poor connectivity  

---

## Technical highlights

- OpenCV-based video capture and rendering  
- Multithreaded pipeline (capture, sampling, inference, UI/output)  
- Adaptive JPEG compression and resizing  
- OpenAI-compatible multimodal vision API integration  
- Production-grade error handling and recovery  

---

## Example use case (healthcare monitoring)

1. A camera observes a clinical or laboratory environment  
2. SI-Watcher samples a frame every **N seconds**  
3. A multimodal GenAI model (e.g. **MedGemma-15:4b**) interprets visual context  
4. Structured insights are generated (events, summaries, confidence scores)  
5. Clinicians or operators remain in the loop for validation and action  

---

## Potential application domains

- **Healthcare and clinical observation**  
- Medical device and procedure monitoring  
- Laboratory and research environments  
- Industrial monitoring and inspections  
- Smart surveillance and safety systems  
- Manufacturing quality control  

---

## Architecture (high level)

A typical deployment runs on a single edge or industrial PC:

1. **Capture** → OpenCV video ingestion with reconnection handling  
2. **Sampler** → time-based triggers with optional frame dropping  
3. **Preprocessing** → resize and JPEG encode for efficiency  
4. **Inference** → multimodal GenAI (OpenAI-compatible API)  
5. **Output** → structured events (log, file, or API sink)  

---

## Configuration highlights

- `--source` — RTSP/HTTP stream, device index, or video file  
- `--interval` — frame sampling cadence (seconds)  
- `--max-width / --max-height` or `--resize` — preprocessing constraints  
- `--drop-frames` — maintain responsiveness under load  
- `--vision-endpoint` + `--api-key` — OpenAI-compatible vision API  

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

Part of the **OR-Edge Project** --- AI-powered solutions for medical
edge environments.

---

## ⚠ Disclaimer

This software is provided **without warranty**.\
It is intended for research, validation, and controlled medical IT
environments.\
It does not replace certified medical decision systems.
