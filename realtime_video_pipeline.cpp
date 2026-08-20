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
#include <opencv2/opencv.hpp>
#include <openai.hpp>
#include <nlohmann/json.hpp>

#include "pipeline_core.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using namespace pipeline_core;

// Single-slot job exchanged between the main thread and the inference worker.
//
// Design choice:
// - We keep only one pending job.
// - Newer frames overwrite older pending work.
// - This keeps the pipeline responsive for live streams and prevents latency
//   from growing without bound under slow inference.
struct PendingJob {
    cv::Mat frame;
    double wallTimeSec = 0.0;   // elapsed program time when the trigger fired
    double mediaPosSec = 0.0;   // position in the media timeline, if known
    int triggerIdx = 0;
    bool has = false;
    bool stop = false;
};

//------------------------------------------------------------------------------
// Utility helpers
//------------------------------------------------------------------------------

// Simple RAII joiner to ensure threads are joined on all exit paths.
class ThreadJoiner {
public:
    explicit ThreadJoiner(std::thread& t) noexcept : thread_(t) {}
    ThreadJoiner(const ThreadJoiner&) = delete;
    ThreadJoiner& operator=(const ThreadJoiner&) = delete;

    ~ThreadJoiner() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread& thread_;
};

// The following pure helpers now live in pipeline_core.h/.cpp so they can be
// unit tested without linking OpenCV/CURL/openai-cpp:
// printUsage, trimInPlace, parseIni, ensureTrailingSlash, endsWith,
// toLowerAscii, usesOpenWebUIChatEndpoint, loadOpenAIConfig, base64Encode,
// extractMessageText, safeLocalTime, safeUtcTime, formatDateTime,
// formatDateTimeNoConversion, parseDateTime, DateTimeParts, parseNDigits,
// isLeapYear, daysInMonth, parseDateTimeParts, daysFromCivil,
// parseDateTimeNoConversion, parseMetadataDateTime.

// Probe ffprobe start_time_realtime (microseconds since epoch) for media files.
static std::optional<std::chrono::system_clock::time_point>
probeFileEncodedTimelineStart(const std::string& path)
{
#ifdef _WIN32
    const std::string cmd =
        "ffprobe -v error -show_entries format=start_time_realtime:stream=start_time_realtime "
        "-of default=noprint_wrappers=1:nokey=1 \"" + path + "\" 2>nul";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    const std::string cmd =
        "ffprobe -v error -show_entries format=start_time_realtime:stream=start_time_realtime "
        "-of default=noprint_wrappers=1:nokey=1 \"" + path + "\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (pipe == nullptr) {
        return std::nullopt;
    }

    std::string output;
    char buffer[256];
    while (std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr) {
        output += buffer;
    }

#ifdef _WIN32
    const int rc = _pclose(pipe);
#else
    const int rc = pclose(pipe);
#endif
    if (rc != 0 || output.empty()) {
        return std::nullopt;
    }

    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        if (!trimInPlace(line)) {
            continue;
        }
        try {
            const long long micros = std::stoll(line);
            if (micros <= 0) {
                continue;
            }
            return std::chrono::system_clock::time_point{
                std::chrono::microseconds(micros)};
        } catch (...) {
        }
    }
    return std::nullopt;
}

// Probe ffprobe creation_time tags and parse to local system_clock time.
static std::optional<std::chrono::system_clock::time_point>
probeFileEncodedStartTime(const std::string& path)
{
#ifdef _WIN32
    const std::string cmd =
        "ffprobe -v error -show_entries "
        "format_tags=creation_time:stream_tags=creation_time "
        "-of default=noprint_wrappers=1:nokey=1 \"" + path + "\" 2>nul";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    const std::string cmd =
        "ffprobe -v error -show_entries "
        "format_tags=creation_time:stream_tags=creation_time "
        "-of default=noprint_wrappers=1:nokey=1 \"" + path + "\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (pipe == nullptr) {
        return std::nullopt;
    }

    std::string output;
    char buffer[256];
    while (std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr) {
        output += buffer;
    }

#ifdef _WIN32
    const int rc = _pclose(pipe);
#else
    const int rc = pclose(pipe);
#endif
    if (rc != 0 || output.empty()) {
        return std::nullopt;
    }

    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        std::chrono::system_clock::time_point parsed{};
        if (parseMetadataDateTime(line, parsed)) {
            return parsed;
        }
    }
    return std::nullopt;
}

//------------------------------------------------------------------------------
// Frame/image helpers
//------------------------------------------------------------------------------

// Resize a frame so that max(width, height) <= maxDim, preserving aspect ratio.
// If maxDim <= 0, resizing is disabled.
static cv::Mat resizeMaxDim(const cv::Mat& frame, int maxDim)
{
    if (maxDim <= 0 || frame.empty()) {
        return frame;
    }

    const int w = frame.cols;
    const int h = frame.rows;
    const int m = (w > h) ? w : h;
    if (m <= maxDim) {
        return frame;
    }

    const double scale = static_cast<double>(maxDim) / static_cast<double>(m);
    const int nw = std::max(1, static_cast<int>(std::lround(w * scale)));
    const int nh = std::max(1, static_cast<int>(std::lround(h * scale)));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    return resized;
}

// Open either a camera index or a URI/file path.
static bool openCapture(cv::VideoCapture& cap, const std::string& src)
{
    if (isCameraIndexSource(src)) {
        try {
            const int index = std::stoi(src);

            // Request a specific platform backend instead of the default
            // CAP_ANY, which makes OpenCV probe backends in priority
            // order. On some platform/camera-driver combinations, a
            // backend tried before reaching a working one can block
            // indefinitely instead of failing fast, hanging cap.open()
            // with no diagnostic output.
#if defined(_WIN32)
            return cap.open(index, cv::CAP_DSHOW);
#elif defined(__APPLE__)
            return cap.open(index, cv::CAP_AVFOUNDATION);
#elif defined(__linux__)
            return cap.open(index, cv::CAP_V4L2);
#else
            return cap.open(index);
#endif
        } catch (...) {
            return false;
        }
    }
    return cap.open(src);
}

//------------------------------------------------------------------------------
// OpenAI request
//------------------------------------------------------------------------------

// Encode a frame as JPEG, wrap it into a data URL, and send it to the model.
//
// We include both:
// - wall time: how long the program has been running
// - media position: where the frame sits in the media timeline
//
// Keeping those separate matters for offline file playback, where media time
// should not drift if inference becomes slower or faster.
static bool sendFrameToOpenAI(
    const cv::Mat& frame,
    double wallTimeSec,
    double mediaPosSec,
    int triggerIdx,
    const OpenAIConfig& cfg,
    const std::string& prompt,
    int maxDim,
    int jpegQuality)
{
    cv::Mat resized = resizeMaxDim(frame, maxDim);

    std::vector<uchar> buffer;
    std::vector<int> params;
    if (jpegQuality > 0 && jpegQuality <= 100) {
        params = {cv::IMWRITE_JPEG_QUALITY, jpegQuality};
    }

    if (!cv::imencode(".jpg", resized, buffer, params)) {
        std::cerr << "[ERROR] Interval #" << triggerIdx
                  << " failed to encode frame to JPEG\n";
        return false;
    }

    const std::string dataUrl = "data:image/jpeg;base64," + base64Encode(buffer);

    std::ostringstream promptStream;
    promptStream
        << prompt
        << " Wall time: " << std::fixed << std::setprecision(3) << wallTimeSec << "s;"
        << " media position: " << std::fixed << std::setprecision(3) << mediaPosSec << "s;"
        << " interval #" << triggerIdx;

    json body = {
        {"model", cfg.vmodelName},
        {"messages", json::array({
            {
                {"role", "user"},
                {"content", json::array({
                    {{"type", "text"}, {"text", promptStream.str()}},
                    {{"type", "image_url"}, {"image_url", {{"url", dataUrl}}}}
                })}
            }
        })},
        {"stream", false}
    };

    if (usesOpenWebUIChatEndpoint(cfg.baseUrl)) {
        const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string requestId =
            "realtime-video-" + std::to_string(nowMs) + "-" + std::to_string(triggerIdx);

        // Open WebUI's /api chat route assumes metadata.chat_id is a string
        // during response post-processing. Supplying a local chat id avoids a
        // server-side None.startswith() failure for direct API callers.
        body["chat_id"] = "local:" + requestId;
        body["id"] = "msg-" + requestId;
    }

    try {
        // A response that looks undispatched (see looksUndispatched()) is
        // the signature of a backend orchestration bug rather than a
        // model that ran and legitimately produced no text -- observed
        // with an Open WebUI backend returning an HTTP 200 stub
        // completion (empty content, all-zero usage, no echoed model)
        // without ever calling the underlying model. That's plausibly
        // transient, so retry a couple of times before giving up.
        constexpr int kMaxAttempts = 3;
        constexpr auto kRetryDelay = std::chrono::milliseconds(300);

        json chat;
        std::string message;
        bool undispatched = false;

        for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
            chat = openai::chat().create(body);
            message = extractMessageText(chat);
            if (!message.empty()) {
                break;
            }

            undispatched = looksUndispatched(chat, cfg.vmodelName);
            if (!undispatched || attempt == kMaxAttempts) {
                break;
            }

            std::cerr << "[WARN] Interval #" << triggerIdx << " attempt " << attempt
                      << "/" << kMaxAttempts << ": backend didn't dispatch the request "
                         "to any model (echoed model=\"" << chat.value("model", std::string())
                      << "\"); retrying...\n";
            std::this_thread::sleep_for(kRetryDelay);
        }

        if (!message.empty()) {
            std::cout << message << std::endl;
            return true;
        }

        std::cout << "(no text content)" << std::endl;

        // extractMessageText() didn't recognize any of the response
        // shapes it knows about (or the model/backend genuinely returned
        // an empty message). Dump the raw response so this is
        // diagnosable without reproducing the request by hand; capped to
        // avoid flooding stderr if a backend ever echoes something large
        // (e.g. the request body) back in the response.
        std::string dump = chat.dump();
        constexpr size_t kMaxDumpLen = 4000;
        if (dump.size() > kMaxDumpLen) {
            dump.resize(kMaxDumpLen);
            dump += " ...[truncated]";
        }

        if (undispatched) {
            std::cerr << "[ERROR] Interval #" << triggerIdx
                      << ": backend accepted the request but doesn't appear to have "
                         "run any model after " << kMaxAttempts << " attempt(s) "
                         "(requested model=\"" << cfg.vmodelName
                      << "\", server echoed model=\"" << chat.value("model", std::string()) << "\"). "
                         "This almost always means \"" << cfg.vmodelName
                      << "\" isn't a model name the backend at " << cfg.baseUrl
                      << " recognizes -- double-check vmodel_name in config.ini "
                         "exactly matches an available, vision-capable model on "
                         "that server. Raw response: " << dump << "\n";
        } else {
            std::cerr << "[WARN] Interval #" << triggerIdx
                      << " got an empty message; raw response: " << dump << "\n";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] OpenAI request failed for interval #"
                  << triggerIdx << ": " << e.what() << "\n";
        return false;
    }
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

// Set by the SIGINT/SIGTERM handler below (e.g. Ctrl+C, or `kill`/`killall`,
// which sends SIGTERM by default) and polled by the main loop and capture
// thread. File-scope so the signal handler -- which must be a plain
// function, not a capturing lambda -- can reach it. This lets the process
// run its normal shutdown path (releasing the camera, closing the GUI
// window, joining threads) instead of dying mid-syscall and potentially
// leaving the camera device locked for the next run.
//
// Note: this cannot help if the process is stuck in an uninterruptible
// native call (e.g. a wedged AVFoundation camera-enumeration call on
// macOS); SIGKILL (`kill -9`) bypasses this entirely by design, since no
// user-space handler can intercept it.
static std::atomic<bool> running{true};

static void handleShutdownSignal(int /*signum*/)
{
    // Async-signal-safe: touches only a lock-free atomic<bool>.
    running.store(false);
}

// Entry point wiring capture, scheduling, and inference worker threads.
int main(int argc, char** argv)
{
    std::signal(SIGINT, handleShutdownSignal);
    std::signal(SIGTERM, handleShutdownSignal);

    ProgramOptions options;
    if (!parseCommandLine(argc, argv, options)) {
        return 1;
    }

    OpenAIConfig cfg;
    if (!loadOpenAIConfig(options.configPath, cfg)) {
        return 1;
    }

    // Log the effective non-secret configuration for easier troubleshooting.
    std::cerr << "[INFO] OpenAI base URL: " << cfg.baseUrl << "\n";
    std::cerr << "[INFO] Vision model: " << cfg.vmodelName << "\n";
    std::cerr << "[INFO] Source: " << options.src << "\n";

    try {
        openai::start(cfg.apiKey, "", true, cfg.baseUrl);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize OpenAI client: "
                  << e.what() << "\n";
        return 1;
    }

    // Printed before the (potentially slow, and on some platform/backend
    // combinations previously hang-prone -- see openCapture()) blocking
    // call below, so a stall is immediately visible as "stuck opening the
    // source" instead of looking identical to complete silence.
    std::string openingMsg = "[INFO] Opening video source...";
#if defined(__APPLE__)
    // macOS grants camera access per-executable-path, so a freshly built
    // binary at a new path gets its own first-run permission prompt. If
    // that dialog doesn't get focus (hidden behind another window/Space),
    // the underlying AVFoundation call blocks indefinitely waiting for a
    // decision that never comes -- indistinguishable from a hang unless
    // the user knows to look for it.
    if (isCameraIndexSource(options.src)) {
        openingMsg +=
            " (macOS may show a camera permission dialog on first run for "
            "this exact binary -- if this hangs here, check for a hidden "
            "dialog, or System Settings > Privacy & Security > Camera)";
    }
#endif
    std::cerr << openingMsg << "\n";

    cv::VideoCapture cap;
    if (!openCapture(cap, options.src) || !cap.isOpened()) {
        std::cerr << "[ERROR] Could not open source\n";
        return 1;
    }

    // Request a small internal buffer to reduce lag on live sources.
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // Heuristic:
    // - if frame count is finite and > 0, and the source is not a camera index,
    //   treat it as a media file
    // - otherwise assume a live-ish source
    const double frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
    const bool likelyFile =
        std::isfinite(frameCount) &&
        frameCount > 0.0 &&
        !isCameraIndexSource(options.src);

    const auto applicationStartTime = std::chrono::system_clock::now();

    // Base timestamp used to map media position -> absolute datetime.
    //
    // Priority:
    // 1. explicit --predefined_start_time
    // 2. encoded timeline start (start_time_realtime)
    // 3. encoded start from media metadata (creation_time)
    // 4. application start time
    std::chrono::system_clock::time_point fileBaseTime =
        applicationStartTime;

    if (likelyFile) {
        if (options.hasPredefinedStartTime) {
            fileBaseTime = options.predefinedStartTime;
        } else {
            const auto timeline = probeFileEncodedTimelineStart(options.src);
            if (timeline.has_value()) {
                fileBaseTime = *timeline;
            } else {
                const auto probed = probeFileEncodedStartTime(options.src);
                if (probed.has_value()) {
                    fileBaseTime = *probed;
                }
            }
        }
    }

    // Shared latest frame state.
    //
    // The capture thread continuously updates this.
    // The main loop periodically samples it and overwrites the single pending
    // inference job.
    std::mutex frameMtx;
    cv::Mat latestFrame;
    double latestMediaPosSec = 0.0;

    // Shared job state for the worker thread.
    std::mutex jobMtx;
    std::condition_variable jobCv;
    PendingJob pending;

    //--------------------------------------------------------------------------
    // Worker thread
    //
    // Responsibilities:
    // - wait for a pending job
    // - process only the newest job
    // - send frame to the model
    //
    // Important:
    // For file playback, encoded timestamps are derived from mediaPosSec,
    // not wallTimeSec. This avoids drift when inference is slower than real
    // time or when file playback timing varies.
    //--------------------------------------------------------------------------
    std::thread worker([&] {
        while (true) {
            PendingJob job;
            {
                std::unique_lock<std::mutex> lk(jobMtx);
                jobCv.wait(lk, [&] { return pending.stop || pending.has; });

                if (pending.stop) {
                    break;
                }

                job.wallTimeSec = pending.wallTimeSec;
                job.mediaPosSec = pending.mediaPosSec;
                job.triggerIdx = pending.triggerIdx;
                pending.frame.copyTo(job.frame);

                // Clear the pending slot immediately so that newer work can be
                // scheduled while inference is running.
                pending.has = false;
            }

            if (job.frame.empty()) {
                continue;
            }

            const std::string acquisitionTag = formatDateTime(
                addSecondsToTimePoint(applicationStartTime, job.wallTimeSec));

            std::string mediaTag;
            if (likelyFile) {
                const auto mediaTime =
                    addSecondsToTimePoint(fileBaseTime, job.mediaPosSec);
                mediaTag = options.hasPredefinedStartTime
                    ? formatDateTimeNoConversion(mediaTime)
                    : formatDateTime(mediaTime);
            } else {
                mediaTag = acquisitionTag;
            }

            const bool useEncodedTimelineTag = likelyFile;
            const std::string& logTimestamp = useEncodedTimelineTag
                                                  ? mediaTag
                                                  : acquisitionTag;

            // For files we log encoded media timeline time; for live sources we
            // fall back to acquisition time because no stable encoded timeline exists.
            std::cout << logTimestamp
                      << " media-time=" << std::fixed << std::setprecision(3)
                      << job.mediaPosSec << "s";
            if (!useEncodedTimelineTag) {
                std::cout << " encoded-at=" << mediaTag;
            }
            std::cout << "  ";

            sendFrameToOpenAI(
                job.frame,
                job.wallTimeSec,
                job.mediaPosSec,
                job.triggerIdx,
                cfg,
                options.prompt,
                options.maxDim,
                options.jpegQuality);
        }
    });
    ThreadJoiner workerJoiner(worker);

    //--------------------------------------------------------------------------
    // Capture thread
    //
    // Responsibilities:
    // - continuously read frames from OpenCV
    // - keep only the latest frame
    // - for files, pace playback approximately in real time
    // - for streams, attempt reconnects on transient failures
    //
    // Notes:
    // - For media files, we prefer CAP_PROP_POS_MSEC as the media timeline.
    // - If POS_MSEC is unavailable, we fall back to FPS-derived progression.
    //--------------------------------------------------------------------------
    std::thread captureThread([&] {
        cv::Mat f;
        auto lastOk = std::chrono::steady_clock::now();

        const double fps = cap.get(cv::CAP_PROP_FPS);
        const bool hasValidFps = std::isfinite(fps) && fps > 1e-6;
        const double frameDuration = hasValidFps ? (1.0 / fps) : 0.0;

        const auto playbackStart = std::chrono::steady_clock::now();
        double fallbackPosSec = 0.0;
        int reconnectAttempt = 0;

        while (running.load()) {
            if (cap.read(f) && !f.empty()) {
                reconnectAttempt = 0;
                lastOk = std::chrono::steady_clock::now();

                double mediaPosSec = fallbackPosSec;

                if (likelyFile) {
                    const double posMsec = cap.get(cv::CAP_PROP_POS_MSEC);

                    if (std::isfinite(posMsec) && posMsec >= 1e-3) {
                        mediaPosSec = posMsec / 1000.0;
                        fallbackPosSec = mediaPosSec;
                    } else if (hasValidFps) {
                        fallbackPosSec += frameDuration;
                        mediaPosSec = fallbackPosSec;
                    }

                    // Throttle file reading to approximately real-time playback
                    // based on media position, not on loop speed.
                    if (mediaPosSec > 0.0) {
                        const auto targetTime =
                            playbackStart + std::chrono::duration<double>(mediaPosSec);
                        const auto now = std::chrono::steady_clock::now();
                        if (targetTime > now) {
                            std::this_thread::sleep_for(targetTime - now);
                        }
                    }
                } else {
                    // For streams/cameras, media position is less meaningful.
                    // Use POS_MSEC only if the backend provides something sane.
                    const double posMsec = cap.get(cv::CAP_PROP_POS_MSEC);
                    if (std::isfinite(posMsec) && posMsec >= 0.0) {
                        mediaPosSec = posMsec / 1000.0;
                    } else {
                        mediaPosSec = 0.0;
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(frameMtx);
                    f.copyTo(latestFrame);
                    latestMediaPosSec = mediaPosSec;
                }
                continue;
            }

            // For files, EOF/failure is expected termination.
            if (likelyFile) {
                running.store(false);
                break;
            }

            // For streams/cameras, treat failures as transient up to a limit.
            const auto now = std::chrono::steady_clock::now();
            const double downFor =
                std::chrono::duration<double>(now - lastOk).count();

            if (options.reconnectSec <= 0 ||
                downFor > static_cast<double>(options.reconnectSec)) {
                std::cerr << "[ERROR] Stream read failed for >"
                          << options.reconnectSec << "s; stopping.\n";
                running.store(false);
                break;
            }

            // Bounded exponential backoff:
            // 250, 500, 1000, 2000, 2000, ...
            ++reconnectAttempt;
            const int backoffMs =
                std::min(2000, 250 * (1 << std::min(reconnectAttempt - 1, 3)));

            std::cerr << "[WARN] Stream read failed; attempting reconnect in "
                      << backoffMs << " ms...\n";

            cap.release();
            std::this_thread::sleep_for(std::chrono::milliseconds(backoffMs));

            if (!running.load()) {
                break;
            }

            if (!openCapture(cap, options.src) || !cap.isOpened()) {
                continue;
            }

            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            lastOk = std::chrono::steady_clock::now();
        }
    });
    ThreadJoiner captureJoiner(captureThread);

    //--------------------------------------------------------------------------
    // Main scheduling loop
    //
    // Responsibilities:
    // - periodically sample the newest available frame
    // - overwrite the pending single-slot inference job
    // - optionally display the current frame in a GUI window
    //--------------------------------------------------------------------------
    const auto t0 = std::chrono::steady_clock::now();
    double nextTrigger = 0.0;
    int triggerIdx = 0;

    while (running.load()) {
        const auto tNow = std::chrono::steady_clock::now();
        const double wallSec = std::chrono::duration<double>(tNow - t0).count();

        // Fire at fixed intervals.
        //
        // We overwrite the pending job rather than queueing indefinitely,
        // because freshness matters more than completeness for this workload.
        if (wallSec >= nextTrigger) {
            cv::Mat frameCopy;
            double mediaPosSec = 0.0;

            {
                std::lock_guard<std::mutex> lock(frameMtx);
                if (!latestFrame.empty()) {
                    latestFrame.copyTo(frameCopy);
                }
                mediaPosSec = latestMediaPosSec;
            }

            if (!frameCopy.empty()) {
                {
                    std::lock_guard<std::mutex> lk(jobMtx);
                    pending.frame = frameCopy;
                    pending.wallTimeSec = wallSec;
                    pending.mediaPosSec = mediaPosSec;
                    pending.triggerIdx = triggerIdx++;
                    pending.has = true;
                }
                jobCv.notify_one();

                // If the main loop was delayed, catch up by advancing the next
                // trigger beyond the current wall time.
                while (wallSec >= nextTrigger) {
                    nextTrigger += options.intervalSec;
                }
            }
        }

        // Optional local preview window.
        //
        // This is deliberately non-blocking: waitKey(1) allows GUI events to be
        // processed without stalling capture/inference scheduling.
        if (options.guiEnabled) {
            cv::Mat toShow;
            {
                std::lock_guard<std::mutex> lock(frameMtx);
                if (!latestFrame.empty()) {
                    latestFrame.copyTo(toShow);
                }
            }

            if (!toShow.empty()) {
                cv::imshow("Live", toShow);
                const int key = cv::waitKey(1);
                if (key == 'q' || key == 27) {
                    running.store(false);
                }
            }
        }

        // Small sleep to avoid a busy-spin main loop.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    //--------------------------------------------------------------------------
    // Shutdown
    //
    // Signal the worker explicitly instead of faking a pending job.
    //--------------------------------------------------------------------------
    {
        std::lock_guard<std::mutex> lk(jobMtx);
        pending.stop = true;
        pending.has = false;
    }
    jobCv.notify_one();

    cap.release();

    if (options.guiEnabled) {
        cv::destroyAllWindows();
    }

    return 0;
}
