// -*- coding: utf-8 -*-
//
// This file is part of the Spazio IT Video-to-Knowledge project.
//
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Test-only helpers shared across pipeline_core's unit test files.
#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace pipeline_core_test {

// Writes `contents` to a uniquely-named temp file and returns its path.
// The file is removed when the returned guard goes out of scope.
class TempFile {
public:
    explicit TempFile(const std::string& contents)
        : path_(std::filesystem::temp_directory_path() /
                ("pipeline_core_test_" + std::to_string(counter_++) + ".ini"))
    {
        std::ofstream out(path_);
        out << contents;
    }

    ~TempFile() { std::filesystem::remove(path_); }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
    static inline int counter_ = 0;
};

// Builds an argv-style char** from a list of strings for parseCommandLine tests.
class Argv {
public:
    explicit Argv(std::vector<std::string> args) : args_(std::move(args))
    {
        for (auto& a : args_) {
            ptrs_.push_back(a.data());
        }
    }

    int argc() const { return static_cast<int>(ptrs_.size()); }
    char** argv() { return ptrs_.data(); }

private:
    std::vector<std::string> args_;
    std::vector<char*> ptrs_;
};

}  // namespace pipeline_core_test
