// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// Copyright (C) 2026 Spazio IT
// Spazio - IT Soluzioni Informatiche s.a.s.
// via Manzoni 40
// 46051 San Giorgio Bigarello
// https://spazioit.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see https://www.gnu.org/licenses/.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <openai.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct OpenAIConfig {
    std::string baseUrl;
    std::string apiKey;
    std::string vmodelName;
};

static std::map<std::string, std::string> parseIni(const std::string& filename)
{
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    if (!file.is_open()) return config;

    std::string line, section;
    while (std::getline(file, line)) {
        const size_t commentPos = line.find_first_of(";#");
        if (commentPos != std::string::npos) line = line.substr(0, commentPos);

        const auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line.erase(0, start);
        const auto end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) line.erase(end + 1);

        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            continue;
        }

        const size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue;

        std::string key = line.substr(0, eqPos);
        std::string value = line.substr(eqPos + 1);
        key.erase(0, key.find_first_not_of(" \t\r\n"));
        key.erase(key.find_last_not_of(" \t\r\n") + 1);
        value.erase(0, value.find_first_not_of(" \t\r\n"));
        value.erase(value.find_last_not_of(" \t\r\n") + 1);

        if (!section.empty()) key = section + "." + key;
        config[key] = value;
    }
    return config;
}

static std::string ensureTrailingSlash(const std::string& url)
{
    if (url.empty() || url.back() == '/') return url;
    return url + "/";
}

static bool loadOpenAIConfig(const std::string& path, OpenAIConfig& cfg)
{
    const auto config = parseIni(path);
    auto getValue = [&](const std::string& key, std::string& out) {
        const auto it = config.find(key);
        if (it != config.end()) out = it->second;
    };

    getValue("openai.base_url", cfg.baseUrl);
    getValue("openai.api_key", cfg.apiKey);
    getValue("openai.vmodel_name", cfg.vmodelName);

    std::vector<std::string> missing;
    if (cfg.baseUrl.empty()) missing.push_back("openai.base_url");
    if (cfg.apiKey.empty()) missing.push_back("openai.api_key");
    if (cfg.vmodelName.empty()) missing.push_back("openai.vmodel_name");

    if (!missing.empty()) {
        std::cerr << "[ERROR] Missing config values in " << path << ":";
        for (const auto& key : missing) std::cerr << ' ' << key;
        std::cerr << "\n";
        return false;
    }

    cfg.baseUrl = ensureTrailingSlash(cfg.baseUrl);
    return true;
}

static std::string base64Encode(const std::vector<uchar>& data)
{
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i + 2 < data.size()) {
        const uint32_t triple =
            (static_cast<uint32_t>(data[i]) << 16) |
            (static_cast<uint32_t>(data[i + 1]) << 8) |
            static_cast<uint32_t>(data[i + 2]);
        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(table[(triple >> 6) & 0x3F]);
        encoded.push_back(table[triple & 0x3F]);
        i += 3;
    }

    const size_t rem = data.size() - i;
    if (rem == 1) {
        const uint32_t triple = (static_cast<uint32_t>(data[i]) << 16);
        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back('=');
        encoded.push_back('=');
    } else if (rem == 2) {
        const uint32_t triple =
            (static_cast<uint32_t>(data[i]) << 16) |
            (static_cast<uint32_t>(data[i + 1]) << 8);
        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(table[(triple >> 6) & 0x3F]);
        encoded.push_back('=');
    }

    return encoded;
}

static std::string extractMessageText(const json& response)
{
    const auto choicesIt = response.find("choices");
    if (choicesIt == response.end() || !choicesIt->is_array() || choicesIt->empty()) {
        return {};
    }

    const auto& first = (*choicesIt)[0];
    const auto messageIt = first.find("message");
    if (messageIt == first.end()) return {};

    const auto contentIt = messageIt->find("content");
    if (contentIt == messageIt->end()) return {};

    if (contentIt->is_string()) {
        return contentIt->get<std::string>();
    }

    if (contentIt->is_array()) {
        std::string combined;
        for (const auto& part : *contentIt) {
            const auto textIt = part.find("text");
            if (textIt != part.end() && textIt->is_string()) {
                combined += textIt->get<std::string>();
            }
        }
        return combined;
    }

    return {};
}

static cv::Mat resizeMaxDim(const cv::Mat& frame, int maxDim)
{
    if (maxDim <= 0) return frame;
    if (frame.empty()) return frame;

    const int w = frame.cols;
    const int h = frame.rows;
    const int m = (w > h) ? w : h;
    if (m <= maxDim) return frame;

    const double scale = static_cast<double>(maxDim) / static_cast<double>(m);
    const int nw = std::max(1, static_cast<int>(std::lround(w * scale)));
    const int nh = std::max(1, static_cast<int>(std::lround(h * scale)));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    return resized;
}

static bool sendFrameToOpenAI(
    const cv::Mat& frame,
    double wallTimeSec,
    int triggerIdx,
    const OpenAIConfig& cfg,
    const std::string& prompt,
    int maxDim,
    int jpegQuality
) {
    cv::Mat resized = resizeMaxDim(frame, maxDim);

    std::vector<uchar> buffer;
    std::vector<int> params;
    if (jpegQuality > 0 && jpegQuality <= 100) {
        params = {cv::IMWRITE_JPEG_QUALITY, jpegQuality};
    }
    if (!cv::imencode(".jpg", resized, buffer, params)) {
        std::cerr << "[ERROR] Trigger #" << triggerIdx << " failed to encode frame to JPEG\n";
        return false;
    }

    const std::string dataUrl = "data:image/jpeg;base64," + base64Encode(buffer);

    json body = {
        {"model", cfg.vmodelName},
        {"messages", json::array({
            {
                {"role", "user"},
                {"content", json::array({
                    {{"type", "text"}, {"text", prompt + " Wall time: " + std::to_string(wallTimeSec) + "s; trigger #" + std::to_string(triggerIdx)}},
                    {{"type", "image_url"}, {"image_url", {{"url", dataUrl}}}}
                })}
            }
        })},
        {"stream", false}
    };

    try {
        auto chat = openai::chat().create(body);
        const std::string message = extractMessageText(chat);
        if (!message.empty()) {
            std::cout << "[AI RESPONSE #" << triggerIdx << "] " << message << std::endl;
        } else {
            std::cout << "[AI RESPONSE #" << triggerIdx << "] (no text content)" << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] OpenAI request failed for trigger #" << triggerIdx << ": " << e.what() << "\n";
        // If you want deeper debugging, uncomment:
        // std::cerr << "[DEBUG] Raw response json: " << chat.dump(2) << "\n";
        return false;
    }
}

static void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " <video_or_stream_uri> [config.ini] [options]\n"
        << "Options (defaults match current hardcoded behavior):\n"
        << "  --interval <sec>        Trigger interval in seconds (default 10)\n"
        << "  --max-dim <px>          Resize frames so max(width,height)<=px (default 1024)\n"
        << "  --jpeg-quality <1-100>  JPEG quality (default 85)\n"
        << "  --prompt <text>         Prompt prefix (default: \"Analyze this frame.\")\n"
        << "  --no-gui                Disable OpenCV imshow/waitKey\n"
        << "  --reconnect-sec <sec>   Reconnect window for live streams (default 5)\n";
}

static bool isSingleDigitIndex(const std::string& s)
{
    return s.size() == 1 && std::isdigit(static_cast<unsigned char>(s[0]));
}

static bool openCapture(cv::VideoCapture& cap, const std::string& src)
{
    if (isSingleDigitIndex(src)) {
        return cap.open(std::stoi(src));
    }
    return cap.open(src);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string src = argv[1];

    std::string configPath = "config.ini";
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
        configPath = argv[2];
    }

    // Defaults (keep old behavior as defaults where applicable)
    double intervalSec = 10.0;
    int maxDim = 1024;
    int jpegQuality = 85;
    std::string prompt = "Analyze this frame.";
    bool guiEnabled = true;
    int reconnectSec = 5;

    // Parse options (after src and optional config.ini)
    int optStart = 2;
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) optStart = 3;

    for (int i = optStart; i < argc; ++i) {
        std::string a = argv[i];
        auto needVal = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "--interval") {
            const char* v = needVal("--interval");
            if (!v) return 1;
            intervalSec = std::max(0.1, std::stod(v));
        } else if (a == "--max-dim") {
            const char* v = needVal("--max-dim");
            if (!v) return 1;
            maxDim = std::max(0, std::stoi(v));
        } else if (a == "--jpeg-quality") {
            const char* v = needVal("--jpeg-quality");
            if (!v) return 1;
            jpegQuality = std::stoi(v);
            if (jpegQuality < 1 || jpegQuality > 100) {
                std::cerr << "[ERROR] --jpeg-quality must be 1..100\n";
                return 1;
            }
        } else if (a == "--prompt") {
            const char* v = needVal("--prompt");
            if (!v) return 1;
            prompt = v;
        } else if (a == "--no-gui") {
            guiEnabled = false;
        } else if (a == "--reconnect-sec") {
            const char* v = needVal("--reconnect-sec");
            if (!v) return 1;
            reconnectSec = std::max(0, std::stoi(v));
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "[ERROR] Unknown option: " << a << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    OpenAIConfig cfg;
    if (!loadOpenAIConfig(configPath, cfg)) {
        return 1;
    }

    try {
        openai::start(cfg.apiKey, "", true, cfg.baseUrl);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize OpenAI client: " << e.what() << "\n";
        return 1;
    }

    cv::VideoCapture cap;
    if (!openCapture(cap, src) || !cap.isOpened()) {
        std::cerr << "Error: could not open source\n";
        return 1;
    }

    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // Heuristic: if frame count is known and >0, it's likely a file.
    // For files, end-of-stream is expected; for streams, try reconnect briefly.
    const double frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
    const bool likelyFile = std::isfinite(frameCount) && frameCount > 0.0 && !isSingleDigitIndex(src);

    std::mutex frameMtx;
    cv::Mat latestFrame;
    std::atomic<bool> running{true};

    // Pending (single-slot) inference request: newest wins.
    struct PendingJob {
        cv::Mat frame;
        double wallTimeSec = 0.0;
        int triggerIdx = 0;
        bool has = false;
    };

    std::mutex jobMtx;
    std::condition_variable jobCv;
    PendingJob pending;

    // Inference worker thread (keeps UI responsive; drops older pending jobs)
    std::thread worker([&]{
        while (running.load()) {
            PendingJob job;
            {
                std::unique_lock<std::mutex> lk(jobMtx);
                jobCv.wait(lk, [&]{ return !running.load() || pending.has; });
                if (!running.load()) break;

                // Move newest job out; clear pending slot
                job.wallTimeSec = pending.wallTimeSec;
                job.triggerIdx = pending.triggerIdx;
                pending.frame.copyTo(job.frame);
                pending.has = false;
            }

            if (!job.frame.empty()) {
                std::cout << "[TRIGGER #" << job.triggerIdx << "] wall t=" << job.wallTimeSec
                          << "s  frame=" << job.frame.cols << "x" << job.frame.rows << std::endl;

                sendFrameToOpenAI(job.frame, job.wallTimeSec, job.triggerIdx, cfg, prompt, maxDim, jpegQuality);
            }
        }
    });

    // Capture thread: keep only the most recent frame; reconnect briefly for streams
    std::thread captureThread([&]{
        cv::Mat f;
        auto lastOk = std::chrono::steady_clock::now();
        const double fps = cap.get(cv::CAP_PROP_FPS);
        const bool hasValidFps = std::isfinite(fps) && fps > 1e-6;
        const double frameDuration = hasValidFps ? (1.0 / fps) : 0.0;
        const auto playbackStart = std::chrono::steady_clock::now();
        double fallbackPosSec = 0.0;

        while (running.load()) {
            if (cap.read(f) && !f.empty()) {
                {
                    std::lock_guard<std::mutex> lock(frameMtx);
                    f.copyTo(latestFrame);
                }
                lastOk = std::chrono::steady_clock::now();

                // For files, throttle reads to approximate real-time playback
                if (likelyFile) {
                    double targetSec = 0.0;
                    const double posMsec = cap.get(cv::CAP_PROP_POS_MSEC);
                    if (std::isfinite(posMsec) && posMsec >= 1e-3) {
                        fallbackPosSec = posMsec / 1000.0;
                        targetSec = fallbackPosSec;
                    } else if (hasValidFps) {
                        fallbackPosSec += frameDuration;
                        targetSec = fallbackPosSec;
                    }

                    if (targetSec > 0.0) {
                        const auto targetTime = playbackStart + std::chrono::duration<double>(targetSec);
                        const auto now = std::chrono::steady_clock::now();
                        if (targetTime > now) {
                            std::this_thread::sleep_for(targetTime - now);
                        }
                    }
                }
                continue;
            }

            if (likelyFile) {
                running.store(false);
                break;
            }

            // Stream/camera: transient failure - try reconnect for reconnectSec seconds
            const auto now = std::chrono::steady_clock::now();
            const double downFor = std::chrono::duration<double>(now - lastOk).count();

            if (reconnectSec <= 0 || downFor > static_cast<double>(reconnectSec)) {
                std::cerr << "[ERROR] Stream read failed for >" << reconnectSec
                          << "s; stopping.\n";
                running.store(false);
                break;
            }

            std::cerr << "[WARN] Stream read failed; attempting reconnect...\n";
            cap.release();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (!openCapture(cap, src) || !cap.isOpened()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        }
    });

    const auto t0 = std::chrono::steady_clock::now();
    double nextTrigger = intervalSec;
    int triggerIdx = 0;

    while (running.load()) {
        const auto tNow = std::chrono::steady_clock::now();
        const double wallSec = std::chrono::duration<double>(tNow - t0).count();

        // Trigger every intervalSec; queue newest only
        if (wallSec >= nextTrigger) {
            cv::Mat frameCopy;
            {
                std::lock_guard<std::mutex> lock(frameMtx);
                if (!latestFrame.empty()) latestFrame.copyTo(frameCopy);
            }

            if (!frameCopy.empty()) {
                {
                    std::lock_guard<std::mutex> lk(jobMtx);
                    pending.frame = frameCopy;          // overwrite slot
                    pending.wallTimeSec = wallSec;
                    pending.triggerIdx = triggerIdx++;
                    pending.has = true;
                }
                jobCv.notify_one();
            }

            while (wallSec >= nextTrigger) nextTrigger += intervalSec;
        }

        // Optional display (non-blocking)
        if (guiEnabled) {
            cv::Mat toShow;
            {
                std::lock_guard<std::mutex> lock(frameMtx);
                if (!latestFrame.empty()) latestFrame.copyTo(toShow);
            }
            if (!toShow.empty()) {
                cv::imshow("Live", toShow);
                int key = cv::waitKey(1);
                if (key == 'q' || key == 27) running.store(false);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Shutdown
    {
        std::lock_guard<std::mutex> lk(jobMtx);
        pending.has = true; // wake worker
    }
    jobCv.notify_one();

    if (captureThread.joinable()) captureThread.join();
    if (worker.joinable()) worker.join();

    cap.release();
    if (guiEnabled) cv::destroyAllWindows();
    return 0;
}
