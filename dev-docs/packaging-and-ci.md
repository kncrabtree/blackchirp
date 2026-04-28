# Packaging and Binary Generation

Set up cross-platform binary packaging for Blackchirp via CPack and GitHub
Actions. Binaries are built on demand (manual dispatch or release tag), not on
every push.

## Synopsis

`cmake/Packaging.cmake` already drafts CPack generators for every target
platform, but several pieces are missing or incorrect for a working release
pipeline:

- The Eigen3 dependency is not declared in CMake; the build only succeeds on
  Linux because `/usr/include/eigen3` is on gcc's implicit search path.
- Several packaging asset files referenced by the CMake scripts do not exist
  (macOS `Info.plist`, Linux Debian maintainer scripts, application icons,
  `.desktop` template).
- No Qt deployment step (`windeployqt`/`macdeployqt`) is invoked, so packaged
  Windows/macOS binaries would not run on a clean machine.
- The macOS DragNDrop generator is misconfigured (uses `CPACK_BUNDLE_*`, which
  applies to a different generator).
- Component names are case-mismatched between `CPACK_COMPONENTS_ALL` and the
  per-target `install(... COMPONENT ...)` calls.
- RPM dependencies are hard-coded with Fedora package names that do not match
  openSUSE; auto-derivation should be used instead.
- No `.github/workflows/` directory exists.

The Eigen3 dependency itself is small (~25 LOC across `analysis.{h,cpp}` and
`peakfinder.{h,cpp}` plus one call site in `liftrace.cpp`) and could be
replaced with GSL, but Eigen is header-only and trivial to install on every
CI runner. Refactoring carries numerical risk on signal-processing code; the
plan keeps Eigen and adds it as a proper CMake dependency. A future refactor
to drop it remains straightforward if desired.

## Linux Packaging Strategy

The matrix produced by the release pipeline:

| Format    | Target Audience                                  |
| --------- | ------------------------------------------------ |
| `.rpm`    | openSUSE (primary), Fedora, RHEL — auto-deps     |
| `.deb`    | Debian, Ubuntu, Mint — `shlibdeps` auto-derived  |
| AppImage  | Universal fallback (Arch, NixOS, anything else)  |
| `.tar.gz` | Source-style binary tarball                      |

Snap and Flatpak are intentionally excluded: their sandboxing models conflict
with serial-port and USB hardware access, which is core to Blackchirp's
purpose.

## Implementation Plan

### 1. Fix Eigen3 dependency declaration

- Add `find_package(Eigen3 3.3 REQUIRED NO_MODULE)` to the top-level
  `CMakeLists.txt`.
- Link `Eigen3::Eigen` `PUBLIC` on the `blackchirp-data` target, since
  `analysis.h` exposes `Eigen::MatrixXd` in its public API.
- Existing `#include <eigen3/Eigen/...>` lines may stay; the imported target
  sets the correct include root either way.

### 2. Repair `cmake/Packaging.cmake`

- Remove the explicit `CPACK_RPM_PACKAGE_REQUIRES` block; rely on
  `CPACK_RPM_PACKAGE_AUTOREQ ON` (already set) so the RPM works on both
  openSUSE and Fedora-family distros without naming drift.
- Optionally set `CPACK_RPM_PACKAGE_RELOCATABLE ON`.
- Resolve the `CPACK_COMPONENTS_ALL applications libraries development`
  (lowercase) vs. `install(... COMPONENT Applications/Libraries/Development)`
  (TitleCase) mismatch — choose one casing and apply consistently.
- Remove the `CPACK_BUNDLE_*` variables (they apply to the `Bundle` generator,
  not `DragNDrop`) and instead set `MACOSX_BUNDLE` properties on the main
  `blackchirp` target (the viewer target already has this at
  `BlackchirpViewerApplication.cmake:104-109`).
- Trim `CPACK_DEBIAN_PACKAGE_DEPENDS` to bare essentials and lean on
  `CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON` (already set).

### 3. Create missing packaging assets

Add a `packaging/` directory at the repo root with:

- `packaging/macos/Info.plist` — main application bundle plist
- `packaging/macos/ViewerInfo.plist` — viewer bundle plist (referenced at
  `cmake/BlackchirpViewerApplication.cmake:105`)
- `packaging/linux/postinst` — Debian post-install (refresh icon/desktop cache)
- `packaging/linux/prerm` — Debian pre-remove (cleanup)
- `packaging/blackchirp.desktop.in` — Linux desktop entry template

Add an `icons/` directory at the repo root with:

- `icons/blackchirp.icns` — generated from `src/resources/icons/bc_logo_large.png`

### 4. Add Qt deployment for Windows and macOS

- Invoke `windeployqt` on Windows and `macdeployqt` on macOS as a post-install
  step or via CPack's `_deploy_runtime_dependencies` mechanism, so that all Qt
  libraries, plugins, and Qwt are bundled into the package.
- Verify packaged binaries launch on a clean (non-developer) machine.

### 5. Stand up GitHub Actions release workflow

Create `.github/workflows/release.yml` triggered by `workflow_dispatch` and
`release: published` events (per the roadmap requirement: "on demand, not on
every push").

Job matrix:

| Runner                                         | Output                                                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------- |
| `ubuntu-22.04`                                 | `.deb` (oldest LTS for glibc compatibility)                             |
| `ubuntu-22.04`                                 | `.AppImage` (separate job, uses `linuxdeploy` + `linuxdeploy-plugin-qt`)|
| `opensuse/leap` (container on `ubuntu-latest`) | `.rpm`                                                                  |
| `macos-latest`                                 | `.dmg`, `.tar.gz`                                                       |
| `windows-latest`                               | NSIS installer, `.zip`                                                  |

Per-platform install steps:

- Qt6 via `jurplel/install-qt-action` (or `aqtinstall` directly).
- GSL via system package manager (`apt`, `brew`, vcpkg).
- Eigen3 via system package manager.
- **Qwt is the schedule risk**: no reliable Homebrew formula or vcpkg port for
  Qt6. Build from source per platform; cache the build artifact between runs.

Each job runs `cmake → make → ctest → cpack` and uploads the resulting
package(s) as workflow artifacts. On `release: published` events, attach
artifacts to the GitHub release.

### 6. Documentation cleanup

`CLAUDE.md` still references `cmake/HardwareConfig.cmake` as an auto-created
config file; that file was removed in commit `4d27ca0a`. Update the
"Configuration files" subsection accordingly.

## Verification

- `cmake . -B build/Desktop-Release/ -DCMAKE_BUILD_TYPE=Release` succeeds with
  a fresh build directory and Eigen3 properly discovered.
- `cpack` from the build directory produces the expected file types per
  platform.
- Generated `.rpm` installs cleanly on openSUSE Tumbleweed; `rpm -qpR` shows
  reasonable auto-derived requirements.
- Generated `.deb` installs cleanly on Ubuntu LTS.
- AppImage launches on a Linux distro without Blackchirp's deps installed
  (verify by running it inside a minimal container).
- Windows/macOS packages launch on a clean VM with no developer tooling.
- GitHub Actions `workflow_dispatch` produces all artifacts in a single run
  without manual intervention.
