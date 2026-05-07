# Crash Reporting

Reference for capturing diagnostic information when a release build of
`blackchirp` or `blackchirp-viewer` crashes on a user's machine. Goal:
when a user reports "Blackchirp crashed", a developer can resolve the
crash to a function and source line without having to reproduce it.

## Strategy

Two design decisions drive the rest of the plan:

1. **Symbols stay developer-side.** Release binaries are stripped (so
   the user gets the same ~8 MB executable they get today), and the
   `.debug` / `.pdb` / `.dSYM` companion files are kept on the build
   server only — attached as workflow artifacts on the GitHub release
   workflow. Users never download symbols. Developers resolve raw
   addresses from a sent crash log against the matching companion file
   identified by the embedded git SHA.
2. **Platform-native crash artifacts.** POSIX (Linux + macOS) emits a
   text crash log with a `std::stacktrace` and process metadata.
   Windows emits a minidump (`.dmp`) plus a small text sidecar.
   Minidumps are the standard Windows debugger artifact and let
   developers load the full process state in WinDbg / Visual Studio
   with the matching `.pdb`; reducing the Windows path to text would
   discard most of that value.

C++23's `<stacktrace>` is used directly on POSIX rather than pulling
in `backward-cpp`. The standard library produces `function + file:line`
output matching what backward-cpp would give for a stripped binary on
a user machine (no source context either way), so the third-party
dependency does not earn its keep here.

## Crash artifacts

| Platform | Artifact                                            | Resolved with                       |
| -------- | --------------------------------------------------- | ----------------------------------- |
| Linux    | `crash-<timestamp>-<sha>.log` (text + stacktrace)   | `addr2line` on the kept `.debug`    |
| macOS    | `crash-<timestamp>-<sha>.log` (text + stacktrace)   | `atos` on the kept `.dSYM`          |
| Windows  | `crash-<timestamp>-<sha>.dmp` + `.log` sidecar      | WinDbg / VS on the kept `.pdb`      |

Crash artifacts land under `<savePath>/log/crashes/` (sibling of the
existing `log/` tree managed by `BlackchirpCSV::logDir()`). Keeping
them on disk inside the user's chosen save path ensures they survive
restarts and are easy for a user to attach to an email.

The text log contains:

- Build identity: `BC_MAJOR.MINOR.PATCH-RELEASE` plus `BC_BUILD_VERSION`
  (the git SHA already configured in the top-level `CMakeLists.txt`).
- Platform: OS name and version, Qt runtime version, CPU arch.
- Signal / exception code and the faulting address.
- Stack trace as `module+0xoffset` pairs (offsets relative to module
  load base, not absolute, so ASLR does not disturb resolution).
- Active experiment number, if one is running (sampled from a
  `std::atomic<int>` written by the acquisition thread).

## File layout

New files under `src/data/`:

```
src/data/crashhandler.h        # public interface (install/uninstall)
src/data/crashhandler.cpp      # cross-platform glue (file paths, formatting)
src/data/crashhandler_unix.cpp # signal handlers, std::stacktrace
src/data/crashhandler_win.cpp  # SetUnhandledExceptionFilter + MiniDumpWriteDump
```

CMake selects `_unix.cpp` or `_win.cpp` from `BlackchirpData.cmake` via
`if(WIN32) ... else()`. The handler is installed once from `main.cpp`
immediately after `QApplication` construction so it covers as much of
startup as practical.

The crash handler is the **one explicit exception** to the
`bcLog` / `bcDebug` / `bcWarn` / `bcError` rule documented in
`src/AGENTS.md`. Signal handler context cannot allocate, take Qt
mutexes, or touch `LogHandler`. The handler writes via raw `write(2)`
to a file descriptor opened at startup.

## Implementation: POSIX (`crashhandler_unix.cpp`)

- Open `crash-<timestamp>-<sha>.log` at install time so no `open(2)`
  is needed inside the handler.
- Allocate a `sigaltstack` buffer (~64 KiB) so a SIGSEGV from stack
  overflow still has stack to run on.
- `sigaction` for SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS with
  `SA_SIGINFO | SA_ONSTACK`.
- In the handler:
  1. Format and write process metadata (synchronously, raw `write`).
  2. Call `std::stacktrace::current()` and write each
     `stacktrace_entry`'s `native_handle()` resolved through `dladdr`
     to `module+0xoffset`.
  3. `fsync` the log fd.
  4. Restore the default disposition for the signal and re-raise it
     so the kernel still produces a core dump for users with
     `ulimit -c` configured.
- `std::set_terminate` is installed for unhandled C++ exceptions; in
  that path `std::stacktrace` may safely allocate.

GCC libstdc++ requires `-lstdc++exp` (libstdc++ 13+) or
`-lstdc++_libbacktrace` (older) to provide stacktrace symbol resolution.
Add the link via `target_link_libraries` in `BlackchirpData.cmake`,
gated on compiler / version detection.

**Async-signal-safety caveat (acknowledged):** `std::stacktrace::current`
allocates, so calling it from a signal handler is technically UB. In
practice this is the same trade-off every native crash logger makes,
and the alternative is to die without diagnostics. Heap corruption that
defeats the in-process handler is a known, accepted failure mode; the
re-raise step ensures the OS still gets a chance to write a core file.

## Implementation: Windows (`crashhandler_win.cpp`)

- `SetUnhandledExceptionFilter` for the top-level filter.
- Optional: `AddVectoredExceptionHandler` to catch exceptions before
  any `__try`/`__except` frames swallow them. Skipped initially —
  Blackchirp does not use SEH directly.
- Install MS C runtime auxiliary handlers:
  `_set_purecall_handler`, `_set_invalid_parameter_handler`,
  `signal(SIGABRT, ...)` for `abort()` paths.
- In the filter:
  1. `CreateFileW` the `.dmp` path.
  2. `MiniDumpWriteDump` with `MiniDumpWithDataSegs |
     MiniDumpWithThreadInfo | MiniDumpWithProcessThreadData`.
  3. Write a small text sidecar with build identity + exception
     summary (mirroring the POSIX log so triage tooling is the same).
  4. Return `EXCEPTION_EXECUTE_HANDLER`.
- Link `dbghelp.lib`. The MSVC runtime ships `<stacktrace>` since VS
  2022 17.4; it can supplement the minidump with a quick text trace
  in the sidecar but is not strictly necessary.

## Build / CI changes

- **Build flags.** Release builds add `-g` (or MSVC's `/Zi`) so
  debug info is generated even at `-O3`. Stripping happens at install
  time in `cmake/Packaging.cmake` (already configured); the change is
  to capture the symbols before that strip step rather than discarding
  them.
- **Symbol capture in CI.** In `.github/workflows/release.yml`, after
  `cmake --build` and before `cpack`:
  - Linux: `objcopy --only-keep-debug blackchirp blackchirp.debug`
    then upload `*.debug` via `actions/upload-artifact` with a long
    retention period (90 days minimum, prefer max). Same for
    `blackchirp-viewer`.
  - macOS: `dsymutil blackchirp -o blackchirp.dSYM` then upload the
    `.dSYM` bundles.
  - Windows: the `.pdb` files are produced alongside the `.exe` by
    MSVC; upload them directly.
- **Symbol artifacts are workflow artifacts, not release assets.**
  They contain enough to reverse-engineer internals and are bulky;
  keeping them off the public release page is safer and clutter-free.
  Developers fetch them via `gh run download <run-id>` when triaging.
- **Per-release manifest.** The release job writes a small
  `symbols-manifest.json` listing artifact name / SHA-256 / git SHA so
  a developer with a crash log's embedded SHA can identify the right
  artifact run without browsing the Actions UI.

## Developer triage workflow

When a user emails a crash log:

```bash
# Read the embedded git SHA from the log header.
gh run download --name symbols-linux-<sha>
addr2line -e blackchirp.debug -f -C -i 0x12ab4 0x9f30 ...
```

For Windows, open the `.dmp` in WinDbg / Visual Studio with the matching
`.pdb` on the symbol path. For macOS, `atos -o blackchirp.dSYM -l <load_addr> 0x...`.

## User-facing follow-up

After the handler infrastructure works:

- On startup, scan `<savePath>/log/crashes/` for artifacts newer than
  the previous successful exit (timestamp on a separate sentinel file).
- If any are found, show a non-blocking dialog offering to open the
  crash directory or copy a `mailto:` URL pre-filled with the build
  identity. Do not auto-send anything; users in research environments
  may be on networks where outbound traffic is restricted.
- Add a "Help → Send Crash Report…" action that opens the most recent
  artifact in a file manager.

## Phasing

| Phase | Scope                                                              | Gates                            |
| ----- | ------------------------------------------------------------------ | -------------------------------- |
| 1     | POSIX handler, log format, manual install in `main.cpp`            | Manual SIGSEGV in dev build      |
| 2     | Windows handler + minidump path                                    | Manual access violation in MSVC  |
| 3     | CI symbol capture in `release.yml`; manifest                       | Packaging workflow stable        |
| 4     | Startup detection of prior crashes; user dialog                    | Settings + dialog UX review      |
| 5     | User-guide page in Sphinx docs; developer guide for triage         | Other phases shipped             |

Phases 1–3 are independent of the existing pre-release packaging
work but should land after `packaging-and-ci.md`'s remaining
verification, since the symbol-capture step modifies the same
workflow file.

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
  dialog should disclose the file's contents and let the user open it
  in a text editor before deciding to send.
