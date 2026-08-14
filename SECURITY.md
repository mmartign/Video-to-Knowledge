<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Security Policy

## Supported Versions

This project does not currently maintain separate release branches.
Security fixes are made against the latest commit on `main`; there is no
guarantee of backports to older commits.

| Version | Supported |
| --- | --- |
| `main` (latest) | ✅ |
| anything older | ❌ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private vulnerability reporting for this repository:

1. Go to the [Security tab](https://github.com/mmartign/Video-to-Knowledge/security/advisories/new).
2. Click **"Report a vulnerability"**.
3. Include a description of the issue, steps to reproduce, and, if
   applicable, the potential impact (e.g. exposure of `config.ini`
   credentials, unsafe handling of untrusted stream/media input).

If private reporting is unavailable to you, open a minimal, non-sensitive
issue asking a maintainer to open a private channel, without including
exploit details.

## What to expect

- We will acknowledge new reports as soon as reasonably possible.
- We will investigate and, where confirmed, work on a fix and coordinate a
  disclosure timeline with the reporter before any public details are
  published.

## Scope notes

SI-Watcher is designed to run on-premise/edge and to avoid biometric
identification by design (see
[Privacy-by-Design and No Biometric Identification](README.md#privacy-by-design-and-no-biometric-identification)
in the README). Reports related to that design goal (e.g. a code path that
would enable biometric identification contrary to the stated design) are
also in scope, even if not a traditional memory-safety or injection issue.
