# Packaging — Alpha Prep TODOs

Ephemeral scratchpad for the v2.0.0-alpha release. The durable
architecture reference (CPack/linuxdeploy strategy, Qt/Qwt sourcing
matrix, signing and provenance, AppImage glibc floor, non-intuitive
constructions, etc.) lives in the developer guide at
`doc/source/developer_guide/packaging.rst` — read that first if you
need the "how it works." This file is just the running checklist of
work items that block the alpha tag, and gets purged once the alpha
ships.

## Next steps before alpha tag

1. **Manual clean-VM smoke test on macOS and Windows.** The Linux
   platforms (DEB on Ubuntu LTS, RPM on openSUSE Tumbleweed and
   Fedora, AppImage on a non-Qt6-distro host) have all been
   manually verified; macOS and Windows are the remaining gap. The
   in-CI `--version` exercises only `main()` entry and the dynamic
   loader; the entire Qt runtime + `MainWindow` constructor is
   uncovered. Per-platform: launch the installed .app/.exe, see the
   main window draw, dismiss it. ~10 minutes per platform. What the
   manual pass catches that CI does not:
   - Qt platform plugin load (`qwindows` / cocoa) inside the
     `QApplication` ctor.
   - Deferred plugin loads on first paint (`imageformats/`,
     `iconengines/qsvgicon`, `tls/qschannelbackend`,
     `styles/qmodernwindowsstyle`).
   - OpenGL context creation when qwt's plot is first
     instantiated (linking `Qt6OpenGL.dll` ≠ initializing it).
   - Release-build (-O3) codepath through GUI init — local builds
     here are Debug, CI Release-builds but doesn't exercise GUI;
     any UB-at-O3 bug is unobserved.
   - First-run dialog (`savePath` empty → `ApplicationConfigDialog`
     + `RuntimeHardwareConfigDialog`); `.ui` forms loaded by uic.
   - Gatekeeper / SmartScreen UX on unsigned binaries; what the
     user has to click through after downloading.
   - `QSettings` against a fresh user profile (HKCU / plist
     creation).
   - The crash handler itself (`CrashHandler::install`,
     `MiniDumpWriteDump`, signal handlers); never run in Release
     because `--version` early-returns before it.

   Acceptance criteria per platform:
   - `.dmg` on a clean macOS install — both `.app`s launch and
     find bundled libqwt (`otool -L` shows `@executable_path/...`).
   - NSIS `.exe` on a clean Windows install — main window opens
     without `STATUS_DLL_NOT_FOUND` from a transitive Qt6OpenGL
     load.

2. **Cut the alpha release.** Tag `v2.0.0-alpha`, push; the
   workflow's `release: published` trigger fires across all five
   platforms and uploads artifacts to the GitHub release. The three
   pre-release notices added in `f0a8596a` (README,
   `doc/source/index.rst`, `doc/source/python.rst`) stay through
   alpha / beta / rc — they go away only at the `v2.0.0` release.

3. **Follow-up not blocking the alpha** (deferred):
   - VC++ Redistributable distribution for end users on a stock
     Windows install. Smoke-runners-with-VS-installed happens to
     mask this; the clean-VM Windows test in item 1 will surface
     it. Options: NSIS auto-run of `vc_redist.x64.exe`, or document
     it as a prerequisite. The clean-VM pass decides which.
   - Apple Developer ID + notarization and Windows Authenticode
     signing. Both cost money (~$99/yr Apple, $200–$500/yr code
     signing CA). Defer until budget exists or a sponsor offers a
     cert; Gatekeeper / SmartScreen click-through is the stand-in
     for alpha.
