# Packaging and CI

Reference for Blackchirp's binary packaging and release-build CI. Binaries are
generated on demand (manual workflow dispatch or release tag), not on every
push.

## Strategy

Cross-platform binary distribution is driven by **CPack** for the per-platform
package formats (DEB, RPM, DMG, NSIS, TGZ, ZIP) and by **linuxdeploy** for the
universal Linux AppImage. A single GitHub Actions workflow drives all five
platforms.

### Linux package matrix

| Format    | Target audience                                  |
| --------- | ------------------------------------------------ |
| `.rpm`    | openSUSE (primary), Fedora, RHEL — auto-deps     |
| `.deb`    | Debian, Ubuntu, Mint — `shlibdeps` auto-derived  |
| AppImage  | Universal fallback (Arch, NixOS, anything else)  |
| `.tar.gz` | Generic binary tarball                           |

Snap and Flatpak are intentionally excluded: their sandboxing models conflict
with serial-port and USB hardware access, which is core to Blackchirp's
purpose.

### Packages contain only `Applications`

`CPACK_COMPONENTS_ALL` is restricted to the `Applications` component. All
`blackchirp-*` libraries are STATIC and linked into the two executables, so
the `Libraries` (`.a` archives) and `Development` (headers + CMake export
files) install rules are dev-only. They remain wired into `cmake --install`
for source-tree workflows but do not ship in binary packages. With Release
builds and stripping, this yields packages in the 4–9 MB range (RPM 4.2 MB,
DEB 8.2 MB measured on openSUSE) instead of ~190 MB Debug-with-static-libs.

### Qt redistribution

- **Linux**: Qt is resolved at install time via the distro's package manager;
  `dpkg-shlibdeps` (DEB) and RPM `AUTOREQ` derive the dependency list from
  linked `.so` files automatically.
- **Windows / macOS**: `windeployqt` / `macdeployqt` run as install hooks via
  `cmake/QtDeployment.cmake`, copying Qt frameworks/DLLs and platform plugins
  into the staged install tree before CPack zips it up.
- **AppImage**: `linuxdeploy-plugin-qt` walks the executable's library
  closure and bundles everything into the AppImage.

## Key files

### `cmake/Packaging.cmake`

CPack configuration: package metadata, per-OS generator selection
(`DEB;RPM;TGZ` on Linux, `DragNDrop;TGZ` on macOS, `NSIS;ZIP` on Windows),
component definitions, and platform-specific knobs (DEB `SHLIBDEPS`, RPM
`AUTOREQ` and `RELOCATABLE`, NSIS shortcuts and uninstall, DMG volume name).
Strips binaries in Release. Drives the `package-all`, `package-deb`,
`package-rpm`, `package-nsis`, `package-dmg` custom targets.

### `cmake/QtDeployment.cmake`

Provides `blackchirp_deploy_qt(<target>)` — locates `windeployqt` /
`macdeployqt` from `Qt6::qmake`'s `IMPORTED_LOCATION` and registers an
`install(CODE)` hook that runs the right tool against the installed binary.
Called once per app from `BlackchirpApplication.cmake` and
`BlackchirpViewerApplication.cmake`. No-op on Linux.

### `cmake/BlackchirpApplication.cmake` / `cmake/BlackchirpViewerApplication.cmake`

Per-app target wiring. Each sets `MACOSX_BUNDLE_*` properties (info plist,
copyright, icon, version), installs to `BUNDLE DESTINATION .` (DragNDrop DMG
convention) and `RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}`, then calls
`blackchirp_deploy_qt(<target>)`.

### `packaging/`

| Path                                    | Role                                                |
| --------------------------------------- | --------------------------------------------------- |
| `packaging/macos/Info.plist`            | Bundle metadata for `blackchirp.app`                |
| `packaging/macos/ViewerInfo.plist`      | Bundle metadata for `blackchirp-viewer.app`         |
| `packaging/linux/postinst`              | `update-desktop-database` after DEB install         |
| `packaging/linux/prerm`                 | `update-desktop-database` cleanup before DEB remove |
| `packaging/blackchirp.desktop.in`       | XDG desktop file (substituted via `configure_file`) |
| `icons/blackchirp.icns`                 | Multi-resolution macOS bundle icon                  |
| `src/resources/icons/bc_logo_large.png` | Source logo; used for Linux pixmap install          |

### `.github/workflows/release.yml`

Five jobs:

| Job              | Runner                          | Output                 |
| ---------------- | ------------------------------- | ---------------------- |
| `linux-deb`      | `ubuntu-latest`                 | `.deb`                 |
| `linux-rpm`      | `opensuse/leap:16.0` container  | `.rpm`                 |
| `linux-appimage` | `ubuntu-latest` + `linuxdeploy` | `.AppImage`            |
| `macos-dmg`      | `macos-latest`                  | `.dmg` + `.tar.gz`     |
| `windows-nsis`   | `windows-latest` (MSVC)         | `.exe` (NSIS) + `.zip` |

Triggers: `workflow_dispatch` (with per-platform boolean inputs for
single-job iteration) and `release: published`. Each job: install system
deps → install Qt 6.9.1 via `jurplel/install-qt-action@v4` → restore-or-build
Qwt 6.3.0 from the SourceForge tarball (cached by OS + Qt + Qwt version) →
`cmake → cmake --build → ctest → cpack` (or `linuxdeploy` for AppImage) →
`actions/upload-artifact` → on release events, `gh release upload --clobber`.

## Non-intuitive constructions

- **`include(GNUInstallDirs)` is called early in the top-level
  `CMakeLists.txt`**, before any subdirectory `install()` rules. Without
  this, subdirectory rules fall back to absolute paths (e.g.,
  `/blackchirp` for Python templates), which breaks every CPack generator.
- **macOS bundle metadata lives on the executable targets**, set via
  `MACOSX_BUNDLE_*` target properties. The DragNDrop CPack generator picks
  them up automatically; do not use `CPACK_BUNDLE_*` (those apply to the
  separate Bundle generator and were causing trouble).
- **Both apps install with `BUNDLE DESTINATION .`** so the `.app` lands at
  the install-prefix root. This matches the DragNDrop DMG layout (drag the
  `.app` straight onto the `Applications` shortcut) and is the path
  `blackchirp_deploy_qt` looks for at install time.
- **Distro package names are not hard-coded.** DEB dependencies come from
  `dpkg-shlibdeps` (`CPACK_DEBIAN_PACKAGE_SHLIBDEPS=ON`); RPM dependencies
  come from RPM's auto-requires (`CPACK_RPM_PACKAGE_AUTOREQ=ON`). This
  avoids drift between Ubuntu/Debian releases or between openSUSE/Fedora's
  qt6/gsl/qwt package names. DEB resolution requires building on a
  Debian-derivative because `shlibdeps` reads Debian's `*.shlibs` database;
  it cannot be validated on openSUSE.
- **Eigen3 has no version pin.** The dev-box system Eigen is 5.0.1 and its
  CMake config rejects pre-5 minimum-version requests. If a CI runner
  ships Eigen 3.x and we want to enforce it, change to
  `find_package(Eigen3 3.3...<6 REQUIRED)`.
- **`blackchirp.icns` was generated locally with `icnsutil`** from
  `src/resources/icons/bc_logo_large.png` and is checked in. Regenerate
  only if the source logo changes.
- **Qwt is built from source on every CI runner**, even on platforms with a
  system package, because there is no reliable Qt6 Qwt across Homebrew,
  vcpkg, and older Ubuntu/openSUSE releases. Per-OS `actions/cache` keyed
  on Qt + Qwt version amortizes the cost across runs.
- **The openSUSE container falls back to `curl` for release uploads** if
  the `gh` CLI is not preinstalled, because Leap's default repos don't
  always carry it.

## Remaining testing and verification

The cmake side is complete; what remains is exercising the workflow on real
runners and iterating on first-run issues.

### Iterate per-platform via workflow_dispatch

The `workflow_dispatch` inputs allow running one platform at a time. Recommended order:

1. **`linux-deb`** — fastest feedback loop; validates trigger / cache /
   upload-artifact / `gh release upload` mechanics.
2. **`linux-rpm`** — exercises the `opensuse/leap:16.0` container path.
3. **`linux-appimage`** — `linuxdeploy-plugin-qt` is the most fragile
   piece; expect to tune library/plugin search paths.
4. **`macos-dmg`** — first real test of the `macdeployqt` install hook;
   may need to copy Qwt's `.dylib` into the bundle pre-deploy if
   `macdeployqt`'s library resolution misses it.
5. **`windows-nsis`** — first real test of the `windeployqt` install
   hook; if the installer is missing OpenSSL or platform plugins,
   revisit `windeployqt`'s flag set in `QtDeployment.cmake`.

### Acceptance criteria

- `.rpm` installs cleanly on openSUSE Tumbleweed; `rpm -qpR` shows
  reasonable auto-derived requirements (`libQt6*.so.6`, `libgsl.so.*`,
  `libgslcblas.so.*`).
- `.deb` installs cleanly on Ubuntu LTS; `dpkg -I` shows non-empty
  `Depends:`.
- AppImage launches on a Linux distro without Blackchirp's deps installed
  (test in a minimal container).
- Windows / macOS packages launch on a clean VM with no developer tooling
  installed.
- A single `workflow_dispatch` with all platforms enabled produces all
  artifacts in one run.
- Cutting a `release: published` event automatically attaches all
  artifacts to the release.

### Likely first-run friction points

- **AppImage**: `linuxdeploy-plugin-qt` requires `qmake` on PATH and may
  trip on Qwt's libdir. The staged AppDir step currently installs only the
  `Applications` component — if the desktop file or icon path diverges
  from `share/applications/blackchirp.desktop` and `share/pixmaps/blackchirp.png`,
  the `--desktop-file` / `--icon-file` arguments to `linuxdeploy` need to
  follow.
- **macOS Qwt linkage**: Qwt installs to `${{ github.workspace }}/qwt-install/lib`
  outside the bundle. `macdeployqt` may not relocate it; if so, copy
  `libqwt.dylib` into `Contents/Frameworks/` before the deploy hook runs,
  or add a `-libpath=` flag to the `macdeployqt` invocation in
  `QtDeployment.cmake`.
- **Windows runtime DLLs**: if NSIS-installed app fails to launch on a
  clean Windows VM, the missing piece is usually `vcruntime140.dll` /
  `msvcp140.dll`. Drop `--no-compiler-runtime` from the `windeployqt`
  flags in `QtDeployment.cmake` to bundle the MSVC redistributable.
- **openSUSE Leap 16.0 Qt6**: confirm Leap 16.0 ships Qt 6.x at a version
  compatible with C++23 features used in Blackchirp; if not, fall back to
  installing Qt via `jurplel/install-qt-action` inside the container (the
  action does support container runs).
