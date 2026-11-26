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
#include <utility>
#include <vector>

using json = nlohmann::json;
using namespace pipeline_core;

// Single-slot job exchanged between the main scheduling loop and the worker.
//
// Design choice:
// - Only one inference call may execute at a time (`inFlight`).
// - While a job is pending or in flight, new interval ticks are skipped so
//   requests never pile up in the backend queue.
struct PendingJob {
    cv::Mat frame;
    double wallTimeSec = 0.0;   // elapsed program time when the trigger fired
    double mediaPosSec = 0.0;   // position in the media timeline, if known
    int triggerIdx = 0;
    bool has = false;           // true when a job is waiting for the worker
    bool inFlight = false;      // true while an inference call is active
    bool stop = false;          // signals the worker thread to exit
};

//------------------------------------------------------------------------------
// Utility helpers
//------------------------------------------------------------------------------

// RAII wrapper to guarantee std::thread objects are joined on all exit paths.
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

// Format a duration in seconds as "HH:MM:SS.mmm (Xs)" so the LLM does not
// have to perform error-prone mental division on raw second counts.
static std::string formatDurationWithHours(double totalSec)
{
    if (!std::isfinite(totalSec) || totalSec < 0.0) {
        totalSec = 0.0;
    }

    const long long totalMs = std::llround(totalSec * 1000.0);
    const long long hours = totalMs / 3600000;
    const long long minutes = (totalMs % 3600000) / 60000;
    const long long seconds = (totalMs % 60000) / 1000;
    const long long millis = totalMs % 1000;

    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(2) << hours << ":"
        << std::setw(2) << minutes << ":"
        << std::setw(2) << seconds << "."
        << std::setw(3) << millis
        << " (" << std::fixed << std::setprecision(3) << totalSec << "s)";
    return oss.str();
}

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

// Probe ffprobe creation_time metadata tags and parse into a system_clock time_point.
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

// Downscale a frame so max(width, height) <= maxDim while preserving aspect ratio.
// Returns the original frame unmodified if maxDim <= 0 or already within bounds.
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

// Open either a numeric camera index (using an explicit platform backend to
// avoid slow/hanging CAP_ANY probing) or a file path / stream URI.
static bool openCapture(cv::VideoCapture& cap, const std::string& src)
{
    if (isCameraIndexSource(src)) {
        try {
            const int index = std::stoi(src);
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
// OpenAI response parsing & request handling
//------------------------------------------------------------------------------

// Extract the assistant's text from `choices[0].message.content` (supporting
// both plain string and segmented array formats), falling back to
// pipeline_core::extractMessageText() for non-standard server envelopes.
static std::string extractChatContent(const json& chat)
{
    if (chat.contains("choices") && chat["choices"].is_array() && !chat["choices"].empty()) {
        const auto& choice0 = chat["choices"][0];
        if (choice0.is_object() && choice0.contains("message") && choice0["message"].is_object()) {
            const auto& msg = choice0["message"];
            if (msg.contains("content")) {
                const auto& content = msg["content"];
                if (content.is_string()) {
                    std::string text = content.get<std::string>();
                    trimInPlace(text);
                    if (!text.empty()) {
                        return text;
                    }
                } else if (content.is_array()) {
                    std::string combined;
                    for (const auto& part : content) {
                        if (part.is_object() &&
                            part.value("type", "") == "text" &&
                            part.contains("text") &&
                            part["text"].is_string()) {
                            if (!combined.empty()) {
                                combined += "\n";
                            }
                            combined += part["text"].get<std::string>();
                        }
                    }
                    trimInPlace(combined);
                    if (!combined.empty()) {
                        return combined;
                    }
                }
            }
        }
    }

    std::string fallback = extractMessageText(chat);
    trimInPlace(fallback);
    return fallback;
}

// Encode a frame as a base64 JPEG data URL, send it to the configured chat
// completions endpoint, and write the extracted content field to stdout.
static bool sendFrameToOpenAI(
    const cv::Mat& frame,
    bool isFileSource,
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

    // For recorded files, pass only the media timeline position (formatted as
    // HH:MM:SS.mmm alongside seconds) so decoding drift never misleads the
    // model into thinking the frame came from beyond the end of the file.
    // For live streams/cameras without a fixed timeline, include wall time.
    std::ostringstream promptStream;
    promptStream << prompt;
    if (isFileSource) {
        promptStream << " Media position: " << formatDurationWithHours(mediaPosSec) << ";";
    } else {
        promptStream << " Elapsed stream time: " << formatDurationWithHours(wallTimeSec) << ";";
        if (mediaPosSec > 0.0) {
            promptStream << " media position: " << formatDurationWithHours(mediaPosSec) << ";";
        }
    }
    promptStream << " interval #" << triggerIdx;

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

    try {
        // Retry with exponential backoff only if the backend returns an empty
        // "undispatched" stub (e.g., when Ollama/Open WebUI drops a request
        // while busy loading or running a prior generation).
        constexpr int kMaxAttempts = 3;
        constexpr auto kInitialRetryDelay = std::chrono::seconds(1);

        json chat;
        std::string message;
        bool undispatched = false;
        auto retryDelay = kInitialRetryDelay;

        for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
            chat = openai::chat().create(body);
            message = extractChatContent(chat);
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
                      << "\"); retrying in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(retryDelay).count()
                      << "ms...\n";
            std::this_thread::sleep_for(retryDelay);
            retryDelay *= 2;
        }

        // Flush stdout explicitly so output appears immediately even when
        // piped or redirected to a file.
        if (!message.empty()) {
            std::cout << message << '\n' << std::flush;
            return true;
        }

        std::cout << "(no text content)\n" << std::flush;

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
                         "Verify vmodel_name in config.ini matches an available vision model "
                         "on " << cfg.baseUrl << ". Raw response: " << dump << "\n";
        } else {
            std::cerr << "[WARN] Interval #" << triggerIdx
                      << " got an empty content field; raw response: " << dump << "\n";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] OpenAI request failed for interval #"
                  << triggerIdx << ": " << e.what() << "\n";
        return false;
    }
}

//------------------------------------------------------------------------------
// Signal handling & entry point
//------------------------------------------------------------------------------

// Global flag cleared by SIGINT/SIGTERM to trigger clean thread & device shutdown.
static std::atomic<bool> running{true};

static void handleShutdownSignal(int /*signum*/)
{
    running.store(false);
}

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

    std::string openingMsg = "[INFO] Opening video source...";
#if defined(__APPLE__)
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

    // Keep internal capture buffering minimal to reduce latency on live sources.
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // Finite positive frame count on a non-camera source indicates a media file.
    const double frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
    const bool likelyFile =
        std::isfinite(frameCount) &&
        frameCount > 0.0 &&
        !isCameraIndexSource(options.src);

    const auto applicationStartTime = std::chrono::system_clock::now();

    // Base timestamp used to map media position to wall-clock datetime:
    // 1. --predefined_start_time
    // 2. container/stream start_time_realtime
    // 3. creation_time metadata tag
    // 4. application start time
    std::chrono::system_clock::time_point fileBaseTime = applicationStartTime;

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

    // Latest captured frame shared between captureThread and the main loop.
    std::mutex frameMtx;
    cv::Mat latestFrame;
    double latestMediaPosSec = 0.0;

    // Single-slot job state shared between the main loop and the inference worker.
    std::mutex jobMtx;
    std::condition_variable jobCv;
    PendingJob pending;

    //--------------------------------------------------------------------------
    // Inference worker thread
    //
    // Executes one vision-language request at a time. While a request is in
    // flight, `pending.inFlight` remains true so the scheduler will not start
    // or queue another request until this one completes.
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

                pending.has = false;
                pending.inFlight = true;
            }

            // Guarantee `inFlight` is cleared when this iteration finishes,
            // even if frame validation or network calls exit early.
            struct InFlightGuard {
                std::mutex& mtx;
                bool& flag;
                ~InFlightGuard() {
                    std::lock_guard<std::mutex> lk(mtx);
                    flag = false;
                }
            } guard{jobMtx, pending.inFlight};

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

            std::cout << logTimestamp
                      << " media-time=" << std::fixed << std::setprecision(3)
                      << job.mediaPosSec << "s";
            if (!useEncodedTimelineTag) {
                std::cout << " encoded-at=" << mediaTag;
            }
            std::cout << "  " << std::flush;

            sendFrameToOpenAI(
                job.frame,
                likelyFile,
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
    // Continuously reads frames, paces file playback to real time, reconnects
    // live streams on transient read failures, and publishes the newest frame.
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

                    // Pace file decoding to roughly match real-time playback.
                    if (mediaPosSec > 0.0) {
                        const auto targetTime =
                            playbackStart + std::chrono::duration<double>(mediaPosSec);
                        const auto now = std::chrono::steady_clock::now();
                        if (targetTime > now) {
                            std::this_thread::sleep_for(targetTime - now);
                        }
                    }
                } else {
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

            // End-of-file for recorded media terminates the application cleanly.
            if (likelyFile) {
                running.store(false);
                break;
            }

            // Live stream / camera failure: retry with bounded backoff up to reconnectSec.
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
    // Main scheduling & UI loop
    //
    // Fires at fixed intervals (`options.intervalSec`). If the worker is still
    // processing the previous frame (`pending.has || pending.inFlight`), the
    // trigger is skipped so a new analysis is never started prematurely.
    //--------------------------------------------------------------------------
    const auto t0 = std::chrono::steady_clock::now();
    double nextTrigger = 0.0;
    int triggerIdx = 0;

    while (running.load()) {
        const auto tNow = std::chrono::steady_clock::now();
        const double wallSec = std::chrono::duration<double>(tNow - t0).count();

        if (wallSec >= nextTrigger) {
            // Advance the schedule immediately so skipped ticks do not spin.
            while (wallSec >= nextTrigger) {
                nextTrigger += options.intervalSec;
            }

            bool workerBusy = false;
            {
                std::lock_guard<std::mutex> lk(jobMtx);
                workerBusy = pending.has || pending.inFlight;
            }

            if (!workerBusy) {
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
                    bool scheduled = false;
                    {
                        std::lock_guard<std::mutex> lk(jobMtx);
                        if (!pending.has && !pending.inFlight) {
                            pending.frame = std::move(frameCopy);
                            pending.wallTimeSec = wallSec;
                            pending.mediaPosSec = mediaPosSec;
                            pending.triggerIdx = triggerIdx++;
                            pending.has = true;
                            scheduled = true;
                        }
                    }
                    if (scheduled) {
                        jobCv.notify_one();
                    }
                }
            }
        }

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

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    //--------------------------------------------------------------------------
    // Clean shutdown
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