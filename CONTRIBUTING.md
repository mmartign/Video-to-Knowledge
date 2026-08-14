<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Contributing to SI-Watcher / Video-to-Knowledge

Thanks for your interest in contributing! This project welcomes issues,
bug reports, and pull requests.

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and requesting features

Please open a [GitHub issue](https://github.com/mmartign/Video-to-Knowledge/issues/new/choose)
using the appropriate template. Include:

- What you expected to happen vs. what actually happened
- Steps to reproduce (video/stream source type, relevant CLI options,
  `config.ini` shape with secrets redacted)
- Platform (Linux/Windows), compiler, and OpenCV version

For security vulnerabilities, please follow [SECURITY.md](SECURITY.md)
instead of opening a public issue.

## Development setup

The build is CMake-based. See
[Building and testing](README.md#building-and-testing) in the README for
the full dependency list and build commands. In short:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

`pipeline_core.{h,cpp}` holds the dependency-light logic (INI/CLI parsing,
string/date helpers, API response parsing) and has no OpenCV/CURL/openai-cpp
dependency — most logic changes should be testable there without needing a
camera, stream, or model endpoint. Add or update unit tests in
`tests/test_pipeline_core.cpp` alongside any change to that file.

## Making a change

1. Fork the repository and create a branch off `main`.
2. Make your change. Keep the existing file header convention (SPDX
   license identifier and copyright block) on new source files.
3. Add or update tests for any change to `pipeline_core.{h,cpp}`.
4. Make sure the project builds and `ctest` passes locally.
5. Open a pull request against `main`. CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))
   must pass before a PR can be merged.

## Coding conventions

- C++20, no compiler-specific extensions (`CMAKE_CXX_EXTENSIONS OFF`).
- Prefer adding new dependency-light logic to `pipeline_core` over the
  OpenCV/CURL/openai-cpp-dependent executables, so it stays unit-testable.
- Keep comments to the "why", not the "what" — the code should already say
  what it does.

## License

By contributing, you agree that your contributions will be licensed under
the [GNU Affero General Public License v3.0 (AGPL-3.0-or-later)](LICENSE),
the same license as the rest of the project.
