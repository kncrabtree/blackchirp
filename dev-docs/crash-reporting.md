# Crash Reporting

End-user crash diagnostics for stripped release builds. The user-facing
description and the developer triage runbook are in the published
Sphinx docs (`doc/source/user_guide/crash_reports.rst` and
`doc/source/developer_guide/crash_handling.rst`); this dev-doc tracks
the remaining packaging-side work and a few decisions that did not
make it into the published docs.

## Implementation summary

Landed in `src/data/crashhandler.{h,cpp,_p.h}` plus per-platform
`crashhandler_unix.cpp` and `crashhandler_win.cpp`, with the user-
facing dialog at `src/gui/dialog/crashreportdialog.{h,cpp}`. Install +
reopen + setActiveExperiment are wired through `main.cpp`,
`BCSavePathWidget::save`, and `AcquisitionManager::beginExperiment` /
end-of-experiment. The handler is the documented exception to the
`bcLog`/`bcDebug` rule because signal context cannot allocate or take
Qt mutexes.

Key decisions:

- POSIX writes a text log via raw `write(2)` to a file descriptor
  opened in `CrashHandler::reopen`; it does not open the file from
  within the handler. Windows writes both a minidump (`.dmp`) and a
  text sidecar (`.log`) so triage tooling has the same header on both
  platforms.
- `<savePath>/log/crashes/` is created lazily from `reopen`, so a
  fresh user with no `log/` subtree still gets crash logs.
- Release builds add `-g` / `/Zi` (in the top-level `CMakeLists.txt`)
  so captured frames resolve. Stripping happens at install time in
  `cmake/Packaging.cmake`.
- The main `blackchirp` executable is non-PIE `EXEC`, so the captured
  `+0xoffset` is from a non-zero declared load base (0x400000). The
  developer triage page documents that `addr2line` wants the bracketed
  absolute PC for the main executable and the parenthesized offset for
  shared libraries.
- The `lastSeen` settings key owned by `CrashReportDialog` (group
  `BC::Key::CrashDialog::crashDialog`) records the most recent
  acknowledged crash timestamp, so a dismissed report does not
  re-prompt unless a still-newer crash arrives.
- The Windows code path is compile-only on this Linux host; it
  awaits the Windows CI job for true verification.

Verified on Linux against a deliberately-injected null pointer
dereference triggered from the About dialog: the resulting log
resolved cleanly via `addr2line -e blackchirp -f -C -i 0x<pc>` to the
expected source-line locations.

## Remaining work — CI symbol capture

In `.github/workflows/release.yml`, after `cmake --build` and before
`cpack`:

- Linux: `objcopy --only-keep-debug blackchirp blackchirp.debug` then
  upload `*.debug` via `actions/upload-artifact` with the maximum
  retention period (90 days minimum). Same for `blackchirp-viewer`.
- macOS: `dsymutil blackchirp -o blackchirp.dSYM` then upload the
  `.dSYM` bundles.
- Windows: the `.pdb` files are produced alongside the `.exe` by MSVC;
  upload them directly.

Symbol artifacts are workflow artifacts, **not** release assets — they
contain enough to reverse-engineer internals and are bulky. Developers
fetch them via `gh run download <run-id>` when triaging.

The release job should also write a small `symbols-manifest.json`
listing artifact name / SHA-256 / git SHA so a developer with a crash
log's embedded SHA can identify the right artifact run without
browsing the Actions UI.

This work edits the same workflow file as `packaging-and-ci.md`'s
remaining verification, so it should land afterward to avoid merge
churn.

## Open questions

- **Out-of-process handler.** Breakpad / Crashpad survive heap
  corruption that an in-process handler does not. Worth revisiting
  only if the in-process handler proves unreliable in field use; the
  added build-system complexity is not justified up front for a
  research instrument app.
- **Symbol storage longevity.** GitHub workflow artifacts cap at ~90
  days. Crashes against older releases will lose easy symbol access.
  If long-tail support matters, a follow-up project is to publish the
  symbols to a private S3 bucket or attach them to releases as
  password-protected ZIPs.
- **Privacy review.** The crash log includes the active experiment
  number and savePath, both of which are user data. The user-facing
  dialog discloses the file's contents and lets the user open it in a
  text editor before deciding to send; the user-guide page reinforces
  that no acquired data is ever included.
