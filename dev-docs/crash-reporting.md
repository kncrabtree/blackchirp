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

## CI symbol capture

`.github/workflows/release.yml` has a `Capture symbols` step in each
build job that runs immediately after `cmake --build`, before
stripping happens during `cpack`:

- **Linux** (deb, rpm, appimage): `objcopy --only-keep-debug` extracts
  per-binary `.debug` files into `symbols/`.
- **macOS**: `dsymutil` produces `.dSYM` bundles (the inner DWARF blob
  is what's hashed in the manifest).
- **Windows**: MSVC writes `.pdb` files alongside `.exe` because
  `CMakeLists.txt` adds `/Zi /DEBUG` for Release builds. The capture
  step renames them from `blackchirp-<ver>.pdb` to `blackchirp.pdb` so
  the manifest's `name` matches the binary the user invokes.

Each job also writes a `symbols-manifest.json` in the same `symbols/`
dir containing the platform tag, the workflow's git SHA, the run ID,
and a list of `{name, sha256}` entries. Symbols + manifest upload as
a separate `blackchirp-symbols-<platform>` artifact with 90-day
retention.

Triage flow: a crash log embeds `BC_BUILD_VERSION` (the git SHA at
build time). To resolve a crash:

```bash
git_sha=$(awk '/BuildVersion:/{print $2; exit}' crash.log)
run_id=$(gh run list --workflow=release.yml --commit=$git_sha \
                      --json databaseId --jq '.[0].databaseId')
gh run download $run_id --name blackchirp-symbols-<platform>
```

Then resolve frames against the downloaded `.debug` / `.dSYM` / `.pdb`
files. Symbol artifacts are intentionally NOT release assets — they're
bulky and enable reverse engineering of internals.

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
