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
#include <iostream>

int main() {
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            std::cout << "Camera index " << i << " is available\n";
            cap.release();
        }
    }
    return 0;
}
