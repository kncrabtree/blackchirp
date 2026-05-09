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
files) install rules are dev-only. `CPACK_DEB_COMPONENT_INSTALL` /
`CPACK_RPM_COMPONENT_INSTALL` are ON because `CPACK_COMPONENTS_ALL` is
silently ignored for DEB/RPM otherwise. With Release builds and stripping,
this yields packages in the 4–9 MB range on Linux.

### Qt and Qwt sourcing per job

| Job              | Qt                              | Qwt                           |
| ---------------- | ------------------------------- | ----------------------------- |
| `linux-deb`      | apt `qt6-base-dev` (Ubuntu)     | from-source + `BC_BUNDLE_QWT` |
| `linux-rpm`      | zypper `qt6-base-devel`         | zypper `qwt6-qt6-devel`       |
| `linux-appimage` | `install-qt-action` 6.9.1       | from-source                   |
| `macos-dmg`      | `install-qt-action` 6.9.1       | from-source                   |
| `windows-nsis`   | `install-qt-action` 6.9.1 MSVC  | from-source                   |

The deb job uses system Qt because `dpkg-shlibdeps` reads Debian's `*.shlibs`
database — Qt installed by `install-qt-action` would have no Debian shlibs
metadata and the package step would fail. Ubuntu LTS has no Qt6-built Qwt, so
the deb job builds Qwt from source and bundles `libqwt.so*` inside the
package via `BC_BUNDLE_QWT=ON` (see `cmake/Packaging.cmake`); the executables
get an `$ORIGIN/../<libdir>/blackchirp` RPATH and dpkg-shlibdeps follows
that to the bundled lib while resolving Qt sonames through `/usr/lib`. The
rpm job uses openSUSE's system Qwt (`libqwt6-qt6-6_3` + `qwt6-qt6-devel`),
so RPM's AUTOREQ derives `libqwt-qt6.so.*` automatically. The other three
jobs build Qwt from source because no reliable Qt6 Qwt exists on
Homebrew, vcpkg, or any LTS apt channel.

### Qt redistribution into the package

- **Linux deb / rpm**: distro package manager resolves Qt at install time
  via auto-derived `dpkg-shlibdeps` / RPM AUTOREQ. The deb job additionally
  ships `libqwt.so*` bundled (see above).
- **Windows / macOS**: `windeployqt` / `macdeployqt` run as install hooks via
  `cmake/QtDeployment.cmake`. macOS passes `-libpath=<qwt-install/lib>` so
  macdeployqt can locate the from-source libqwt by basename and bundle it
  into `Contents/Frameworks/` (qmake's macOS build leaves the dylib's
  install_name pointing at `/usr/lib/...` which doesn't exist on the runner).
- **AppImage**: `linuxdeploy-plugin-qt` walks the executable's library
  closure and bundles everything into the AppImage. `LD_LIBRARY_PATH`
  must include `$QT_ROOT_DIR/lib` and the from-source qwt-install/lib
  during the linuxdeploy step, otherwise `ldd` reports Qt sonames as
  unresolved.

## Key files

### `cmake/Packaging.cmake`

CPack configuration: package metadata, per-OS generator selection
(`DEB;RPM;TGZ` on Linux, `DragNDrop;TGZ` on macOS, `NSIS;ZIP` on Windows),
component definitions, and platform-specific knobs. Owns the `BC_BUNDLE_QWT`
option. Strips binaries in Release. Drives the `package-all`, `package-deb`,
`package-rpm`, `package-nsis`, `package-dmg` custom targets.

### `cmake/QtDeployment.cmake`

Provides `blackchirp_deploy_qt(<target>)` — locates `windeployqt` /
`macdeployqt` from `Qt6::qmake`'s `IMPORTED_LOCATION`, registers an
`install(CODE)` hook that runs the right tool against the installed binary.
On macOS additionally derives `-libpath=` from `QWT_LIBRARY` so the
from-source libqwt bundles correctly. No-op on Linux.

### `cmake/BlackchirpApplication.cmake` / `cmake/BlackchirpViewerApplication.cmake`

Per-app target wiring. Each sets `MACOSX_BUNDLE_*` properties (info plist,
copyright, icon, version), installs to `BUNDLE DESTINATION .` (DragNDrop DMG
convention) and `RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}`, then calls
`blackchirp_deploy_qt(<target>)`.

### `cmake/FindQWT.cmake`

When `qwt.h` is found inside a `qwt`/`qwt-qt6`/`qwt6` subdirectory, exposes
the parent as a second include path so `<qwt6/qwt_plot.h>` resolves
(openSUSE convention used in the source). On Windows, sets `QWT_DLL` on the
imported target's `INTERFACE_COMPILE_DEFINITIONS` when `qwt.dll` is present
next to the import lib, so MSVC consumers get `__declspec(dllimport)` on
exported static data members.

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

Five jobs (table above). Triggers: `workflow_dispatch` (with per-platform
boolean inputs for single-job iteration) and `release: published`.
Per-job: install system deps → install Qt (per the table) → restore-or-build
Qwt 6.3.0 (cached separately by OS) → `cmake → cmake --build → ctest →
cpack` (or `linuxdeploy` for AppImage) → `actions/upload-artifact` → on
release events, `gh release upload --clobber`.

The Qwt cache uses split `actions/cache/restore@v5` + `actions/cache/save@v5`
with the save step gated on `cache-hit != 'true'` and placed immediately
after Build Qwt. `actions/cache@v5` dropped `save-always`; the unified
post-step skips on job failure, which would waste the rebuild on retries.
The split pattern saves inline regardless of downstream failure.

## Non-intuitive constructions

- **`include(GNUInstallDirs)` is called early in the top-level
  `CMakeLists.txt`**, before any subdirectory `install()` rules. Otherwise
  subdirectory rules fall back to absolute paths (e.g., `/blackchirp` for
  Python templates), which breaks every CPack generator.
- **macOS bundle metadata lives on the executable targets**, set via
  `MACOSX_BUNDLE_*` target properties. The DragNDrop CPack generator picks
  them up automatically; do not use `CPACK_BUNDLE_*` (those apply to the
  separate Bundle generator).
- **Both apps install with `BUNDLE DESTINATION .`** so the `.app` lands at
  the install-prefix root. Matches the DragNDrop DMG layout (drag the `.app`
  straight onto Applications) and is the path `blackchirp_deploy_qt` looks
  for at install time.
- **No `CPACK_SET_DESTDIR` on Apple.** DESTDIR-style staging is right for
  DEB/RPM/IFW but for DragNDrop the `.app` *is* the unit of distribution and
  the package root is the install root; with DESTDIR=ON, CPack stages the
  `.app` under `${DESTDIR}/usr/local/blackchirp.app` and the deploy hook +
  DragNDrop file walk both miss it.
- **Distro package names are not hard-coded.** DEB dependencies come from
  `dpkg-shlibdeps`; RPM dependencies come from `AUTOREQ`. This avoids drift
  across Ubuntu/Debian releases or between openSUSE/Fedora's qt6/gsl/qwt
  package names.
- **Eigen3 has no version pin.** The dev-box system Eigen is 5.0.1 and its
  CMake config rejects pre-5 minimum-version requests. If a CI runner ships
  Eigen 3.x and we want to enforce it, change to `find_package(Eigen3
  3.3...<6 REQUIRED)`.
- **`blackchirp.icns` was generated locally with `icnsutil`** from
  `src/resources/icons/bc_logo_large.png` and is checked in. Regenerate only
  if the source logo changes.
- **AppImage icon lookup uses the hicolor 256×256 path**, not
  `share/pixmaps/blackchirp.png`. The pixmap is 1024×1024 (sized for
  `.icns` / `.ico` masters), and linuxdeploy's icon validator caps at
  512×512.

## Status — 2026-05-09

All five build jobs and all five smoke jobs are green on master.
Symbol capture lands in each build job and uploads as a separate
`blackchirp-symbols-<platform>` artifact (90-day retention). Remaining
work before the alpha tag is the manual clean-VM acceptance pass — the
in-CI smoke tests run `--version` only, which catches missing libs and
broken RPATHs but doesn't exercise the GUI startup path.

### Recent commits (most recent last)

- `1af2d31e` — Eigen3 include / pwsh `$$` / AppImage `LD_LIBRARY_PATH` /
  CPack component-install for DEB/RPM (so packages don't ship headers and
  static `.a` archives).
- `f9e92771` — macOS sigemptyset macro, AppImage icon path, **DEB switched
  to system Qt + `BC_BUNDLE_QWT`**, **RPM switched to system Qwt**, Windows
  GSL PUBLIC link.
- `1ad7bce1` — DEB QtSvg, Windows `INT_MAX`, QSettings ctor consistency
  across platforms (production now 0-arg, tests 2-arg with isolated org).
- `35ae1bcf` — `viewer-src/` `INT_MAX` site, macOS `setOrganizationDomain`
  removal, Qt 6.4 `QString::arg(QAnyStringView)` shim, `ChirpSegment` UB.
- `9ad298ba` — DEB `<QDialogButtonBox>` include, macOS `CPACK_SET_DESTDIR`
  drop, Windows `QWT_DLL` define, split Qwt cache restore/save.
- `f61ae8c8` — macOS `-libpath` for macdeployqt, DEB Qt 6.4
  `QString::operator+(QStringView)` shim, Windows NSIS forward-slash paths
  (CMP0010).
- `f84bf9e2` — DEB ninja `-k 0` keep-going; Qt 6.4 `invokeMethod`
  member-pointer-with-args sites converted to lambda form.
- `9307c968` — `--version` / `--help` (with Windows `AttachConsole`),
  per-platform symbol capture + `symbols-manifest.json`, five companion
  `*-smoke` jobs running `--version` against the freshly-installed
  package on a clean container or runner.

## Next steps before alpha tag

1. **Manual clean-VM smoke test.** The CI smoke jobs cover
   `--version`; full UI startup on a fresh OS install is still worth
   the spot-check.
   - `.rpm` on openSUSE Tumbleweed; `rpm -qpR` shows auto-derived
     requirements (`libQt6*.so.6`, `libgsl.so.*`, `libqwt-qt6.so.*`).
   - `.deb` on Ubuntu LTS; `dpkg -I` shows non-empty `Depends:`; the
     bundled libqwt resolves through the binary RPATH.
   - AppImage on a distro without Blackchirp's Qt6 packages installed.
   - `.dmg` on a clean macOS install; both `.app`s launch and find
     their bundled libqwt (`otool -L` shows `@executable_path/...`).
   - NSIS `.exe` on a clean Windows install.
2. **Confirm version strings.** `BC_RELEASE_VERSION` in `CMakeLists.txt`
   matches the chosen alpha tag string;
   `python/blackchirp/pyproject.toml` (currently `0.1.0rc2`) aligns
   with the alpha story.
3. **Choose the alpha tag.** Existing convention: `v1.0.0-release`,
   `v1.1.0-beta`, `v1.1.0-release`. Natural extension: `v2.0.0-alpha`.
4. **Cut the release**, verify all five artifacts attach, then remove
   the three pre-release notices added in `f0a8596a` (README,
   `doc/source/index.rst`, `doc/source/python.rst`).

The published documentation tracks the current shape of CI:

- `doc/source/developer_guide/build_system.rst` covers the per-job Qt
  and Qwt sourcing matrix, the `BC_BUNDLE_QWT` option, the macOS
  `-libpath` assumption, and the symbol-capture / smoke-test layout.
- `doc/source/developer_guide/crash_handling.rst` covers the crash-log
  → CI symbol-artifact triage flow (`gh run download …`).
