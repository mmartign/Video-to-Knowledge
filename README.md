# SI-Watcher — Real-Time Video-to-Knowledge Engine on the Edge

SI-Watcher is an edge-first, real-time **video-to-knowledge** pipeline that ingests live or recorded video streams, periodically samples frames, and uses **multimodal Generative AI** to extract **structured insights** locally and securely.

> **Design goals:** low-latency decisions, privacy-first processing, and cloud-independence. 

---

## What it does

- **Ingests** live video (cameras/streams) or video files  
- **Samples frames** at configurable, time-based intervals (trigger-style sampling)  
- Runs **contextual vision analysis** with multimodal GenAI (images + prompts)  
- Produces **structured output** in (near) real time on an edge device  

---

## Core capabilities

- Live video ingestion (cameras, streams, video files)  
- Adaptive frame sampling (time-based triggers)  
- Contextual vision analysis (GenAI with images + prompts)  
- Edge-first architecture (single industrial PC)  
- Open-source and configurable  

---

## Key innovations

- Multimodal GenAI applied directly to video frames 
- Non-blocking pipeline with frame dropping for responsiveness 
- Edge-optimized JPEG encoding and adaptive resizing 
- Seamless live-stream reconnection handling 
- CLI-first, automation-friendly design 

---

## Why edge deployment?

- No dependency on cloud   
- Privacy-first: sensitive data stays local   
- Ultra-low latency for real-time decisions   
- Works offline / low-connectivity environments   

---

## Technical highlights

- OpenCV-based capture and rendering  
- Multithreaded pipeline (capture, inference, UI)  
- JPEG compression with adaptive resizing  
- OpenAI-compatible vision API integration  
- Production-grade error handling  

---

## Example use case (industrial monitoring)

1. A camera observes an industrial process  
2. SI-Watcher samples a frame every **N seconds**  
3. GenAI interprets visual context (e.g., detects an anomaly)  
4. Structured output is logged or forwarded via API  
5. A human operator remains in the loop  

---

## Potential application domains

- Industrial monitoring & inspections  
- Smart surveillance and safety systems  
- Manufacturing quality control  
- Research and laboratory observation  
- Remote site supervision  

---

## Architecture (high level)

A typical deployment runs on a single edge/industrial PC:

1. **Capture** (OpenCV) → read stream, handle reconnections  
2. **Sampler** → time-based triggers; may drop frames when overloaded  
3. **Preprocess** → resize + JPEG encode for efficiency  
4. **Inference** → multimodal GenAI call (OpenAI-compatible vision API)  
5. **Output** → structured events (log/file/API)  

---



### Configuration ideas

- `--source`: RTSP/HTTP stream, device index, or video file  
- `--interval`: sampling cadence (seconds)  
- `--max-width/--max-height` or `--resize`: preprocessing constraints  
- `--drop-frames`: keep pipeline responsive under load  
- `--vision-endpoint` + `--api-key`: OpenAI-compatible vision API integration  

---

## Output format (recommended)

While the deck states “structured insights”, a practical format is JSON Lines (`.jsonl`) with a schema like:

```json
{
  "ts": "2026-01-25T10:15:04Z",
  "source": "camera-1",
  "frame_id": 1842,
  "labels": ["anomaly"],
  "summary": "Possible blockage detected near conveyor intake.",
  "confidence": 0.78,
  "evidence": {
    "prompt": "…",
    "model": "…",
    "image": {
      "width": 1280,
      "height": 720
    }
  }
}
```

---

## Contributing

Contributions are welcome—especially around:
- new sampling triggers,
- additional output sinks (MQTT/REST/Webhooks),
- prompt packs for specific domains,
- benchmarking and edge optimizations.  

---


## Contacts

- GitHub: `github.com/mmartign/Video-to-Knowledge` 
- Contact: `info@spazioit.com`
- Website: `spazioit.com`  
