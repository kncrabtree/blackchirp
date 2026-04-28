# Packaging and Binary Generation

Set up cross-platform binary packaging for Blackchirp via CPack and GitHub
Actions. Binaries are built on demand (manual dispatch or release tag), not
on every push.

## Linux Packaging Strategy

| Format    | Target Audience                                  |
| --------- | ------------------------------------------------ |
| `.rpm`    | openSUSE (primary), Fedora, RHEL — auto-deps     |
| `.deb`    | Debian, Ubuntu, Mint — `shlibdeps` auto-derived  |
| AppImage  | Universal fallback (Arch, NixOS, anything else)  |
| `.tar.gz` | Generic binary tarball                           |

Snap and Flatpak are intentionally excluded: their sandboxing models conflict
with serial-port and USB hardware access, which is core to Blackchirp's
purpose.

## Status

### Done

- **CMake dependency declared for Eigen3.** `find_package(Eigen3)` plus
  `Eigen3::Eigen` linked PUBLIC on `blackchirp-data`. Eigen kept as-is
  (header-only, trivial to install per-platform); refactor to GSL deferred.
- **`cmake/Packaging.cmake` repaired.** Hard-coded distro package names
  removed in favour of `dpkg-shlibdeps` / RPM `AUTOREQ`. RPM marked
  relocatable. Component-name casing aligned (`Applications`, `Libraries`,
  `Development`). Misapplied `CPACK_BUNDLE_*` removed; macOS metadata moved
  to per-target `MACOSX_BUNDLE_*` properties on both apps.
- **Packaging assets created.** `packaging/macos/{Info,ViewerInfo}.plist`,
  `packaging/linux/{postinst,prerm}` (executable bit set),
  `packaging/blackchirp.desktop.in`, and `icons/blackchirp.icns` (multi-res,
  generated via `icnsutil` from `bc_logo_large.png`). The `.icns` is wired
  into both bundles via `MACOSX_PACKAGE_LOCATION = "Resources"`.
- **Pre-existing CPack-blocking bug fixed.** `include(GNUInstallDirs)` moved
  early in the top-level `CMakeLists.txt` so subdirectory `install()` rules
  see `CMAKE_INSTALL_DATADIR` at registration time. Without this, the Python
  hardware templates were registered with an absolute `/blackchirp`
  destination, breaking every CPack generator.
- **Versions bumped** to 2.0.0-alpha for both apps; package vendor and macOS
  bundle copyright set to "Kyle N. Crabtree".
- **Verified locally on openSUSE:** TGZ, RPM, and DEB generators all produce
  installable packages. RPM auto-derived requirements look correct
  (`libQt6*.so.6`, `libgsl.so.28`, `libgslcblas.so.0`, etc.). DEB
  auto-derivation cannot be validated on openSUSE because `dpkg-shlibdeps`
  needs Debian's `*.shlibs` database; this will populate correctly on an
  Ubuntu CI runner.

### Remaining work (handoff)

1. **Qt deployment for Windows and macOS.** `windeployqt` and `macdeployqt`
   are not invoked anywhere. Without them, packaged Windows/macOS binaries
   cannot launch on a clean machine. Wire these into `install()` rules or
   into the CI workflow as a post-build step. Verify with a clean VM.
2. **GitHub Actions release workflow.** No `.github/workflows/` directory
   exists. Create `release.yml` triggered by `workflow_dispatch` and
   `release: published`. Job matrix:

   | Runner                                         | Output                                                                  |
   | ---------------------------------------------- | ----------------------------------------------------------------------- |
   | `ubuntu-22.04`                                 | `.deb` (oldest LTS for glibc compatibility)                             |
   | `ubuntu-22.04`                                 | `.AppImage` (separate job, `linuxdeploy` + `linuxdeploy-plugin-qt`)     |
   | `opensuse/leap` (container on `ubuntu-latest`) | `.rpm`                                                                  |
   | `macos-latest`                                 | `.dmg`, `.tar.gz`                                                       |
   | `windows-latest`                               | NSIS installer, `.zip`                                                  |

   Per-platform installs: Qt6 via `jurplel/install-qt-action`, GSL/Eigen3
   via the system package manager. **Qwt is the schedule risk** — no
   reliable Homebrew formula or vcpkg port for Qt6; build from source per
   platform and cache the artifact between runs.

   Each job: `cmake → cmake --build → ctest → cpack`, upload the package(s)
   as workflow artifacts; on `release: published`, attach to the release.

3. **Package size sanity check.** The Debug-build RPM/DEB came in at ~190 MB
   because the `Development` component ships static libs and headers
   (~150 MB executables alone in Debug). Worth checking with a Release build
   whether splitting the runtime and development components into separate
   packages (e.g. `blackchirp` vs `blackchirp-devel`) makes sense before the
   first public release.

4. **Verification once CI is up.**
   - `.rpm` installs cleanly on openSUSE Tumbleweed; `rpm -qpR` shows
     reasonable auto-derived requirements.
   - `.deb` installs cleanly on Ubuntu LTS; `dpkg -I` shows non-empty
     `Depends:`.
   - AppImage launches on a Linux distro without Blackchirp's deps
     installed (test in a minimal container).
   - Windows/macOS packages launch on a clean VM with no developer tooling.
   - `workflow_dispatch` produces all artifacts in a single run.

## Notes for the next session

- The packaging-blocking bugs are all fixed; CPack works end-to-end on
  Linux. The remaining work is **CI wiring and Qt redistributable
  bundling**, not cmake repair.
- Recent commits on this branch: `f3ab9b15` (Eigen3 wiring) and `55257617`
  (CPack overhaul + asset creation + version bump).
- `icnsutil` was used locally to generate `icons/blackchirp.icns` from the
  existing `src/resources/icons/bc_logo_large.png`. The `.icns` is checked
  in; no regeneration needed unless the source logo changes.
- Eigen3 is currently pinned with no version requirement. The system Eigen
  on this dev box is 5.0.1, and the `find_package` call had to drop the
  `3.3` minimum because Eigen 5's CMake config rejects lower-version
  requests. If a CI runner ships Eigen 3.x and we want to enforce it, the
  pin can be reintroduced with `find_package(Eigen3 3.3...<6 REQUIRED)`.
