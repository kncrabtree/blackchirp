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
- **Qt deployment wired for Windows and macOS.** New `cmake/QtDeployment.cmake`
  exposes `blackchirp_deploy_qt(<target>)`, which locates `windeployqt` /
  `macdeployqt` from the resolved `Qt6::qmake` and registers an
  `install(CODE)` hook that runs the right tool against the staged binary.
  Both `blackchirp` and `blackchirp-viewer` call it after their
  `install(TARGETS)` rule, and macOS `BUNDLE DESTINATION` is normalized to
  `.` so the `.app` lands at the install-prefix root for DragNDrop. No-op on
  Linux. Still needs clean-VM verification once CI is up.
- **Release workflow drafted.** `.github/workflows/release.yml` with five
  jobs: `linux-deb` (`ubuntu-latest`), `linux-rpm`
  (`opensuse/leap:16.0` container), `linux-appimage` (`ubuntu-latest`,
  `linuxdeploy` + `linuxdeploy-plugin-qt`), `macos-dmg` (`macos-latest`),
  and `windows-nsis` (`windows-latest`, MSVC, NSIS via Chocolatey). Qt
  6.9.1 via `jurplel/install-qt-action`; Qwt 6.3.0 built from the
  SourceForge tarball with a per-OS `actions/cache` keyed on Qt and Qwt
  version. Triggers: `workflow_dispatch` (with per-platform boolean
  inputs) and `release: published`. On release events each job uses
  `gh release upload --clobber` to attach its artifacts.

### Remaining work (handoff)

1. **First-run dispatch verification.** The workflow is unverified end-to-end;
   trigger jobs one platform at a time via `workflow_dispatch` (the boolean
   inputs let you run one platform per dispatch). Likely first-run friction
   points:
   - **AppImage**: `linuxdeploy-plugin-qt` requires `qmake` on PATH and may
     trip on Qwt's libdir; pass `EXTRA_PLATFORM_PLUGINS` / `LD_LIBRARY_PATH`
     adjustments if it can't resolve `libqwt`. The `staged AppDir` step
     currently installs only the `Applications` and `Libraries` components
     — desktop-file path may need a tweak if the cmake install rule changes.
   - **macOS**: `cpack -G DragNDrop` invokes `macdeployqt` via the
     `install(CODE)` hook from `QtDeployment.cmake`; if Qwt's `.dylib`
     lives outside the bundle's framework search paths, macdeployqt's
     library-resolution may fail. May need to add `-libpath=` or copy the
     Qwt `.dylib` into the bundle pre-deploy.
   - **Windows**: NSIS comes from Chocolatey; vcpkg supplies GSL and Eigen.
     `windeployqt` runs through `install(CODE)`. If the resulting
     installer is missing OpenSSL or platform plugins, revisit
     `windeployqt`'s flag set in `QtDeployment.cmake`.
   - **openSUSE container**: Leap 16.0's Qt6 packages should suffice for
     building Qwt; `gh` CLI is not preinstalled, so the release-upload
     step has a fallback to `curl` against the GitHub uploads API.

2. **Package size sanity check.** The Debug-build RPM/DEB came in at ~190 MB
   because the `Development` component ships static libs and headers
   (~150 MB executables alone in Debug). Worth checking with a Release build
   whether splitting the runtime and development components into separate
   packages (e.g. `blackchirp` vs `blackchirp-devel`) makes sense before the
   first public release.

3. **Verification once CI is up.**
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
  Linux, Qt redistributable bundling is wired into the install rules for
  Windows/macOS, and a five-job CI release workflow is drafted. The
  remaining work is **dispatching the workflow per-platform and fixing
  first-run issues**, not cmake repair.
- Use `workflow_dispatch` with one platform's boolean input enabled at a
  time to iterate without burning all five runners on every push.
- The Qt deploy hook runs at `cmake --install` / `cpack` time and depends
  on `windeployqt` / `macdeployqt` being on PATH (or in the Qt6 bin dir
  resolved from `Qt6::qmake`'s `IMPORTED_LOCATION`). On Linux it is a no-op.
- Recent commits on this branch: `f3ab9b15` (Eigen3 wiring), `55257617`
  (CPack overhaul + asset creation + version bump), `ac7928b4` (Qt deploy
  install hooks).
- `icnsutil` was used locally to generate `icons/blackchirp.icns` from the
  existing `src/resources/icons/bc_logo_large.png`. The `.icns` is checked
  in; no regeneration needed unless the source logo changes.
- Eigen3 is currently pinned with no version requirement. The system Eigen
  on this dev box is 5.0.1, and the `find_package` call had to drop the
  `3.3` minimum because Eigen 5's CMake config rejects lower-version
  requests. If a CI runner ships Eigen 3.x and we want to enforce it, the
  pin can be reintroduced with `find_package(Eigen3 3.3...<6 REQUIRED)`.
