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

## Active debugging — 2026-05-08

Two commits land the workflow in its current state:

- `323a496f` — initial fix attempt after the all-red first run. Bumped
  `actions/checkout` and `actions/cache` to v5; added `qt6-svg-devel`
  to the RPM zypper list; switched the Windows download to `curl.exe`
  with a size sanity check; tried (incorrectly, as observed in the
  next run) to override `QWT_INSTALL_PREFIX` via an unscoped append
  in `qwtconfig.pri`.
- `4718bc57` — second attempt after observing that the RPM install
  paths still resolved to the upstream `/usr/local/qwt-6.3.0` default
  despite the append. Replaced the echo append with an
  indentation-aware sed that rewrites the in-block
  `QWT_INSTALL_PREFIX` assignments in place. Reverted the `qtsvg`
  addition from `install-qt-action` modules — QtSvg is part of Qt 6
  essentials, ships with qtbase, and aqtinstall's repo XML does not
  list it as a separately-installable module (the explicit name
  produced "package not found" errors on the four jobs that use the
  action). Added a sed delete of the
  `QWT_CONFIG += QwtExamples / QwtPlayground / QwtDesigner / QwtTests`
  lines so the Qwt build skips the heavy bits.

### What to verify on the next workflow_dispatch run

- Install paths in the Qwt build log should resolve to
  `${{ github.workspace }}/qwt-install/...`, **not**
  `/usr/local/qwt-6.3.0/...`. This is the cleanest signal that the
  in-place sed override took effect.
- The CMake configure step should report `Found QWT: ...` instead of
  the previous `Could NOT find QWT (missing: QWT_LIBRARY
  QWT_INCLUDE_DIR)` failure.
- The Qwt build should be noticeably faster — examples, playground,
  designer plugin, and tests are skipped.
- All five jobs should reach Build / Test / Package.

### Branches from here

- **If all five go green**: proceed to the smoke-test pass against
  the artifacts (per the acceptance criteria above), then add the
  symbol-capture step described in `dev-docs/crash-reporting.md`
  ("Remaining work — CI symbol capture") in a separate commit, then
  tag the alpha.
- **If install paths still write to `/usr/local/qwt-6.3.0`**: the
  in-place sed did not rewrite the relevant line, or qmake captures
  the value at parse time before any edit can reach the scope that
  matters. Inspect the actual `qwtconfig.pri` shipped in
  `qwt-6.3.0.tar.bz2` — the rewrite assumes the assignments live
  inside `unix { }` / `win32 { }` blocks with the upstream
  `QWT_INSTALL_PREFIX = /usr/local/qwt-$$QWT_VERSION` form. A
  temporary `cat qwtconfig.pri` debug step right after the sed
  invocation will confirm whether the line was rewritten as expected.
- **If new failures emerge on individual platforms**: cross-reference
  the friction-points list above. AppImage's `linuxdeploy-plugin-qt`,
  macOS's `macdeployqt` Qwt relocation, and Windows runtime DLL
  bundling are the documented likely-suspects for each.

## Active debugging — 2026-05-09

After the 05-08 attempt: install paths in the Qwt build log resolved
to `${{ github.workspace }}/qwt-install/...` as expected (the in-place
sed override worked), and the Qwt build itself completed without the
Examples / Playground / Designer / Tests subprojects. Failures shifted
downstream: all four jobs that use the GitHub Actions runners' from-
source Qwt install (Linux DEB, Linux AppImage, Linux RPM, Windows)
failed when compiling `blackchirp-gui` because the `<qwt6/...>`
includes did not resolve. macOS failed earlier, at CMake configure,
with `Could NOT find QWT`.

Diagnosis: the source uses openSUSE's header layout convention
(`#include <qwt6/qwt_plot.h>`), which assumes a `qwt6/` subdir is
present beneath some include-path entry. CI's flat from-source install
(`qwt-install/include/qwt.h`, no `qwt6/` subdir) does not satisfy
that. macOS additionally builds Qwt as a framework by default —
`qmake` activates `QWT_CONFIG += QwtFramework` when Qt itself is a
framework, which it always is on macOS — placing headers inside
`qwt-install/lib/qwt.framework/Headers/` and emitting no `libqwt.dylib`
that `find_library` can pick up.

The single fix attempt for this round (one commit, all five jobs):

- `release.yml` — add a second sed to each Qwt build step that
  rewrites `QWT_INSTALL_HEADERS` to `$${QWT_INSTALL_PREFIX}/include/qwt6`,
  so the from-source install lands under a `qwt6/` subdir matching
  what the source expects. Extend the existing `QWT_CONFIG +=`
  deletion regex to also strip `QwtFramework`, so macOS produces a
  flat `libqwt.dylib` and headers under `qwt-install/include/qwt6/`
  rather than inside a `.framework` bundle.
- `cmake/FindQWT.cmake` — add `qwt6` to `PATH_SUFFIXES`, and when
  `qwt.h` is found inside a directory whose basename is `qwt`,
  `qwt-qt6`, or `qwt6`, expose the parent directory as a second
  `INTERFACE_INCLUDE_DIRECTORIES` entry so `<qwt6/qwt_plot.h>`
  resolves through the parent.
- All five Qwt cache keys gain a `-v2` salt — the cached install trees
  from the previous round still have the flat header layout, and
  without invalidation the `if: cache-hit != 'true'` guard skips the
  rebuild that would now produce `qwt6/`-subdir headers.

Local sanity check: reconfiguring `build/Desktop-Debug` against the
openSUSE system Qwt resolves `QWT_INCLUDE_DIR` to
`/usr/include/qt6/qwt6` (parent: `/usr/include/qt6`); with the new
parent-dir export, `<qwt6/qwt_plot.h>` resolves through the parent on
the dev box too, where previously it resolved only through the
implicit system path.

### What to verify on the next workflow_dispatch run

- Each Qwt build step's install log writes headers to
  `qwt-install/include/qwt6/qwt.h` (and the `Qwt*` classincludes to
  the same dir), not flat at `qwt-install/include/qwt.h`.
- macOS Qwt build emits `qwt-install/lib/libqwt.6.3.0.dylib` (plus
  symlinks), not a `qwt.framework` bundle.
- CMake configure on all five jobs reports
  `Found QWT: ...libqwt... (found version "6.3.0")` and lists two
  include dirs: `…/qwt-install/include/qwt6` and
  `…/qwt-install/include`.
- The four GUI-failing jobs reach build / test / package; macOS
  reaches configure → build / test / package.

### After the qwt6/ commit (`fcac699b`) — follow-on failures

Once `fcac699b` landed and the from-source Qwt installs began landing
under `qwt-install/include/qwt6/`, four new failure modes surfaced
across the runners. Bundled into a single follow-on commit
(`1af2d31e`):

- **macOS, build step**: `<eigen3/Eigen/SVD>` and `<eigen3/Eigen/Core>`
  failed to resolve. Same root cause as the qwt6 issue (openSUSE-
  specific double-prefix include form), but only two source sites and
  the standard upstream Eigen convention is plain `<Eigen/SVD>`. Fixed
  at the source rather than via CMake gymnastics:
  `src/data/analysis/{analysis,peakfinder}.h` drop the `eigen3/`
  prefix. `find_package(Eigen3)` exposes the directory containing
  `Eigen/`, so the upstream form resolves on every platform.
- **Windows, Qwt build step**: nmake bailed with
  `Makefile.Release(2232) : fatal error U1001: syntax error : illegal
  character '{' in macro`. Cause: the new pwsh `-replace` rewrite for
  `QWT_INSTALL_HEADERS` used `$${QWT_INSTALL_PREFIX}` in the
  replacement, but PowerShell's `-replace` runs the replacement string
  through .NET regex substitution where `$$` collapses to a single
  literal `$`. So `qwtconfig.pri` got `${QWT_INSTALL_PREFIX}` (single
  dollar) instead of qmake's `$${QWT_INSTALL_PREFIX}`, and qmake passed
  it through to `Makefile.Release` literally — nmake then choked on
  the `{`. Fix: use `$$$$` in the pwsh replacement so `$$` lands in
  the file. Bumped the Windows cache key to `-v3` to invalidate any
  partially-saved cache from the failed `-v2` run.
- **AppImage, linuxdeploy step**:
  `ERROR: Could not find dependency: libQt6Concurrent.so.6`. Cause:
  `linuxdeploy` resolves an ELF's dependency closure via `ldd`, which
  needs Qt's lib dir on `LD_LIBRARY_PATH`. The existing env block only
  added `qwt-install/lib`. `install-qt-action@v4` exports
  `$QT_ROOT_DIR` to the job environment, so the fix is appending
  `$QT_ROOT_DIR/lib` to `LD_LIBRARY_PATH` in the Build AppImage step.
- **Linux DEB, package step**: `CPackDeb.cmake:888` aborted with the
  generic `Problem compressing the directory` after the file(1)
  classification phase. The actual trigger was the
  `BlackchirpXxxTargets-release.cmake` files in the staged tree —
  their file(1) output contains nested unescaped quotes
  (`"# Generated CMake target import file for configuration "Release".`)
  that CPack's list parsing in CMake 3.31 mishandles. The deeper
  problem is that the binary package was *also* shipping headers and
  static `.a` archives (109 MB RPM, observed) despite
  `CPACK_COMPONENTS_ALL = Applications`, because that variable is
  silently ignored for DEB/RPM unless `CPACK_<GEN>_COMPONENT_INSTALL`
  is enabled. Fix:
  - Set `CPACK_DEB_COMPONENT_INSTALL` and `CPACK_RPM_COMPONENT_INSTALL`
    in `cmake/Packaging.cmake`.
  - Override `CPACK_DEBIAN_APPLICATIONS_FILE_NAME` and
    `CPACK_RPM_APPLICATIONS_FILE_NAME` to keep the canonical
    `Blackchirp-<version>-<system>-<arch>.{deb,rpm}` form, since
    component-install otherwise appends `-Applications` to the file
    name.
  - Add the missing `COMPONENT Applications` clause to the python-
    template install in `cmake/BlackchirpHardware.cmake` — without it,
    component-install would silently drop the python templates from
    the package.

Local sanity check: `cmake --install build/Desktop-Debug --component
Applications --strip` produces exactly the expected file list (two
executables + symlinks, two `.desktop` files, twelve python
templates, hicolor icons, pixmap) totalling 15 MB from Debug binaries.
Release-stripped output should land in the doc's 4–9 MB target range.

### What to verify on the next workflow_dispatch run

- Same checks as the previous round (qwt6/ headers, libqwt.dylib on
  macOS, QWT found on configure) plus:
- macOS build reaches `cpack -G "DragNDrop;TGZ"` (eigen3 fix in place).
- Windows Qwt build completes (`nmake install` succeeds).
- AppImage `linuxdeploy --plugin qt` step succeeds without "Could not
  find dependency" errors.
- Linux DEB and RPM packages are 4–9 MB compressed, contain only the
  Applications component (no `usr/include`, no `usr/lib/lib*.a`, no
  `usr/lib/cmake/`), and `dpkg -I` / `rpm -qpR` show non-empty
  Depends.
- File names are `Blackchirp-2.0.0-Linux-x86_64.deb` and
  `…rpm` (no `-Applications` suffix).

### After the bundled follow-on commit (`1af2d31e`) — further failures

Once `1af2d31e` (eigen3 / pwsh `$$$$` / `LD_LIBRARY_PATH` /
component-install) landed, macOS got past CMake configure and started
building, then failed in `crashhandler_unix.cpp` (other runners still
in flight when this section was written; more entries may be appended
below, and SHAs filled in once the fixes are committed):

- **macOS, build step** (fix uncommitted as of writing):
  `crashhandler_unix.cpp:214` — `::sigemptyset(&sa.sa_mask)` failed
  with `expected unqualified-id`. Apple's `<signal.h>` defines
  `sigemptyset` as a function-like macro
  (`#define sigemptyset(set) (*(set) = 0, 0)`); the `::` qualifier
  can't precede a macro identifier. glibc declares it as a real
  function, which is why Linux builds were unaffected. Fix: drop the
  `::` qualifier — the unqualified name resolves to the macro on
  macOS and to the global function on Linux. Single-line edit, no
  cache invalidation needed.

- **AppImage, linuxdeploy step** (fix uncommitted as of writing):
  `ERROR: Icon AppDir/usr/share/pixmaps/blackchirp.png has invalid x
  resolution: 1024`. linuxdeploy's icon validator rejects anything
  outside the freedesktop hicolor size list, which tops out at
  512×512; the checked-in `icons/blackchirp.png` is 1024×1024 (sized
  to feed the macOS `.icns` and Windows `.ico` generators, both of
  which want a 1024-class master). Fix: point `--icon-file` in the
  AppImage step at `AppDir/usr/share/icons/hicolor/256x256/apps/blackchirp.png`
  instead of the legacy `share/pixmaps/` copy. The hicolor tree is
  already installed by `BlackchirpApplication.cmake` and 256×256 is
  on linuxdeploy's accepted size list. The `share/pixmaps` install
  stays in place for desktop-environment fallback consumers; only
  the AppImage build switches paths.

- **Linux DEB, package step** (fix uncommitted as of writing):
  `dpkg-shlibdeps: error: cannot find library libQt6Widgets.so.6
  needed by ./usr/bin/blackchirp-2.0.0 (RPATH: '')`. Cause:
  `install-qt-action` puts Qt under `$QT_ROOT_DIR/lib`, which is not
  any Debian package, so dpkg-shlibdeps both fails to locate the libs
  (no `LD_LIBRARY_PATH` in the package step) and would have no
  shlibs metadata to derive a Debian dependency from even if it
  found them. This is the friction point predicted in the
  follow-ups list above. Resolution: switch the deb job to system
  Qt + bundled Qwt, which is the only mix that lines up with what
  Debian/Ubuntu actually ship (no Qt6 Qwt in any apt repo as of
  2026-05; Ubuntu noble has Qt 6.4.2 in `qt6-base-dev`, matching the
  source's deliberate Qt-6.4 ceiling).
  - `release.yml` (deb job): drop `install-qt-action`; apt-install
    `qt6-base-dev libqt6serialport6-dev`; bump the Qwt cache key to
    `…-sysqt-v1` so the Qt-version-keyed entry from prior rounds is
    not reused; pass `-DCMAKE_INSTALL_PREFIX=/usr -DBC_BUNDLE_QWT=ON`
    on the configure line. Qwt itself is still built from source
    because no apt repo carries a Qt6-built Qwt.
  - `cmake/Packaging.cmake`: add `BC_BUNDLE_QWT` (default OFF). When
    set, install `libqwt*.so*` into
    `${CMAKE_INSTALL_LIBDIR}/blackchirp` under the Applications
    component, and add `INSTALL_RPATH "$ORIGIN/../<libdir>/blackchirp"`
    on `blackchirp` and `blackchirp-viewer`. The basename of the lib
    is derived from `QWT_LIBRARY` rather than hard-coded `libqwt.so`
    so the same logic works for distros that name the Qt6 build
    distinctly (e.g., openSUSE's `libqwt-qt6.so`). dpkg-shlibdeps
    needs no extra hint: with `--ignore-missing-info` (CPack's
    default) it follows the binary's RPATH to the bundled libqwt and
    silently skips the dpkg-metadata lookup, while still resolving
    Qt sonames through the system /usr/lib path.

- **Linux RPM, package step** (fix uncommitted as of writing): no
  observed failure on this round, but the from-source Qwt build was
  always a workaround for runners without a usable Qt6 Qwt.
  openSUSE Leap 16.0 has `libqwt6-qt6-6_3` + `qwt6-qt6-devel` in its
  standard repos — confirmed working on the dev box — so the
  Active-Debugging-2026-05-09 friction-points list called this out
  as a viable simplification. Doing it now alongside the deb fix
  keeps the deb and rpm paths conceptually aligned (system Qt +
  whatever Qwt the distro provides; from-source Qwt only where the
  distro lacks it). Resolution:
  - `release.yml` (rpm job): add `libqwt6-qt6-6_3 qwt6-qt6-devel` to
    the zypper install list; drop the Qwt cache restore, the
    from-source Qwt build step, the `-DQWT_ROOT=` /
    `-DCMAKE_PREFIX_PATH=` configure args, and the
    `LD_LIBRARY_PATH=…qwt-install/lib` env vars on the test and
    package steps. `BC_BUNDLE_QWT` stays OFF (default), so the rpm
    just `Requires: libqwt6-qt6-6_3` via AUTOREQ — the standard
    openSUSE/Fedora/RHEL story.

- **Windows, build step** (fix uncommitted as of writing):
  `fatal error C1083: Cannot open include file: 'gsl/gsl_fft_real.h'`
  while compiling `communicationprotocol.cpp` for the
  `blackchirp-test-hardware` target. The error chain runs through
  the public header `data/analysis/ftworker.h`, which `#include`s
  `<gsl/gsl_fft_real.h>`, `<gsl/gsl_interp.h>`, and
  `<gsl/gsl_spline.h>`. `cmake/BlackchirpData.cmake` linked GSL as
  PRIVATE, so the GSL include directory propagated only to
  blackchirp-data's own `.cpp` files. On Linux this was invisible
  because GSL lives at `/usr/include/gsl/` (always on the compiler's
  default search path); on Windows with vcpkg the include dir is at
  `C:/vcpkg/installed/x64-windows/include` and only reaches
  consumers through the IMPORTED `GSL::gsl` target's
  `INTERFACE_INCLUDE_DIRECTORIES`, which requires PUBLIC linkage.
  Fix: promote `GSL::gsl` and `GSL::gslcblas` to PUBLIC on
  blackchirp-data. No other library needed the change because
  ftworker.h is the only public header that exposes GSL. Verified
  the Linux build still links cleanly afterwards.

### Branches from here

- **All five green**: proceed to artifact smoke tests per the
  acceptance criteria, then add CI symbol capture
  (`crash-reporting.md` "Remaining work" section), then tag the alpha.
- **macOS misses libqwt.dylib in the bundle**: `macdeployqt` does not
  auto-relocate non-framework dylibs that live outside the bundle.
  Fix: copy `qwt-install/lib/libqwt*.dylib` into
  `<bundle>/Contents/Frameworks/` before the `blackchirp_deploy_qt()`
  install hook runs, or add `-libpath=qwt-install/lib` to the
  `macdeployqt` invocation in `cmake/QtDeployment.cmake`.
- **Windows installer launches but missing qwt.dll**: `windeployqt`
  only walks Qt-named DLLs. Add an explicit `install(FILES …)` for
  `qwt.dll` alongside the executable, or use
  `RUNTIME_DEPENDENCIES` on `install(TARGETS …)` to pull it in.
- **DEB shlibdeps fails on Qt libs**: dpkg-shlibdeps may not resolve
  Qt6 sonames installed by `install-qt-action` because they're not
  from any Debian package. If so, set
  `CPACK_DEBIAN_PACKAGE_SHLIBDEPS=OFF` and provide a manual
  `CPACK_DEBIAN_PACKAGE_DEPENDS` list. Trade-off: loses auto-deps.
- **Cache save warning in opensuse/leap container persists**: the
  `actions/cache` post-step running on the host can't always see
  paths that the container created via the workspace bind-mount.
  Currently a non-blocking warning (~60 s rebuild penalty next run);
  worth investigating if it persists across runs.
  - **Bypass option**: openSUSE ships Qwt6 in its standard repos
    (`libqwt6-qt6-6_3` runtime + `qwt6-qt6-devel` headers/CMake
    config — confirmed working on the dev box). Switching the
    `linux-rpm` job to `zypper -n install -y libqwt6-qt6-6_3
    qwt6-qt6-devel` and dropping the from-source Qwt build + cache
    plumbing entirely is the cleanest fix if the cache wrinkle keeps
    re-appearing. Other jobs need from-source Qwt because Ubuntu LTS
    and Homebrew don't ship a Qt6-compatible Qwt 6.x at a usable
    version, and Windows has no system package; the openSUSE
    container is the only runner that has a viable system option.

## Once builds are green: known follow-ups before alpha tag

- Symbol capture in CI (`crash-reporting.md` Remaining work section).
  Workflow artifact uploads of `.debug` / `.dSYM` / `.pdb` files plus
  a `symbols-manifest.json` for git-SHA → artifact mapping.
- Confirm `BC_RELEASE_VERSION` in `CMakeLists.txt` matches the chosen
  alpha tag string.
- Confirm `python/blackchirp/pyproject.toml` version (currently
  `0.1.0rc2`) aligns with the alpha story; `pip install --pre
  blackchirp` resolves to whatever pre-release version is published.
- Choose the alpha tag name (existing convention: `v1.0.0-release`,
  `v1.1.0-beta`, `v1.1.0-release` — natural extension is
  `v2.0.0-alpha`).
- After the alpha is tagged and the docs default flips, remove the
  three pre-release notices added in commit `f0a8596a` (README,
  `doc/source/index.rst`, `doc/source/python.rst`).
