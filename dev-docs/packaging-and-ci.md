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
| `linux-rpm`      | zypper `qt6-base-devel`         | from-source + `BC_BUNDLE_QWT` |
| `linux-appimage` | `install-qt-action` 6.9.1       | from-source                   |
| `macos-dmg`      | `install-qt-action` 6.9.1       | from-source                   |
| `windows-nsis`   | `install-qt-action` 6.9.1 MSVC  | from-source                   |

The deb job uses system Qt because `dpkg-shlibdeps` reads Debian's `*.shlibs`
database — Qt installed by `install-qt-action` would have no Debian shlibs
metadata and the package step would fail. Ubuntu LTS has no Qt6-built Qwt, so
the deb job builds Qwt from source and bundles `libqwt.so*` inside the
package via `BC_BUNDLE_QWT=ON` (see `cmake/Packaging.cmake`); the executables
get an `$ORIGIN/../<libdir>/blackchirp` RPATH and dpkg-shlibdeps follows
that to the bundled lib while resolving Qt sonames through `/usr/lib`.

The rpm job follows the same bundle-Qwt pattern, but for a different reason.
openSUSE patches libqwt-qt6.so's SONAME to include the minor version
(`libqwt-qt6.so.6.3` rather than upstream's `.so.6`); RPM AUTOREQ records
the linked SONAME verbatim, and `libqwt-qt6.so.6.3` is unsatisfiable on
Fedora, RHEL, and any other RPM distro that ABI-tracks at the major level.
Bundling sidesteps the soname mismatch entirely — the resulting RPM has no
qwt dependency at all and installs cleanly on every RPM distro that has
Qt6. The other three jobs build Qwt from source because no reliable Qt6
Qwt exists on Homebrew, vcpkg, or any LTS apt channel.

### Two AppImages per release

The Linux AppImage job emits both `Blackchirp-x86_64.AppImage` (main
acquisition app) and `Blackchirp-Viewer-x86_64.AppImage` (viewer-only
entry point). Each is fully self-contained — bundled Qt/Qwt/GSL is the
size driver and is duplicated across the two — but the duplication is
deliberate: AppImage users are exactly the audience without a system
package manager that pulls in both binaries, so click-and-run
discoverability beats download efficiency. Users who do care about the
size run the viewer from inside the main AppImage via
`--appimage-mount` or `--appimage-extract`; the main AppImage bundles
both binaries internally.

The build runs `linuxdeploy` twice against two AppDir copies (the
plugin mutates the AppDir in place — RPATH patches, AppRun injection,
libdir cleanup — so a single tree cannot be reused). `OUTPUT=` pins
the viewer AppImage's filename because appimagetool would otherwise
mangle `Name=Blackchirp Viewer` to `Blackchirp_Viewer-x86_64.AppImage`
with an underscore, breaking the docs' `Blackchirp-Viewer-…` glob.

### AppImage glibc floor

The AppImage build pins `runs-on: ubuntu-22.04` rather than `ubuntu-latest`.
AppImages bundle Qt, Qwt, libgsl, and the rest of the executable's library
closure but **not** glibc / libm — those always come from the host loader.
Symbol versions picked up by bundled libraries at link time therefore
become a hard host-glibc minimum at run time: a `libgsl.so.27` built
against glibc 2.39 references `GLIBC_2.38`-versioned symbols in libm and
fails to load on any host older than that, defeating the AppImage's
"universal fallback" purpose. Building on `ubuntu-22.04` (glibc 2.35) caps
the floor at the LTS distros the AppImage exists to serve — Ubuntu 22.04+,
RHEL 9+, openSUSE Leap 15.5+, Debian 12+. The Qwt cache key bakes in the
runner codename (`jammy`) so a future runner upgrade auto-invalidates the
cache rather than poisoning the new build with stale glibc-2.39-linked
artifacts.

The companion `linux-appimage-smoke` job intentionally stays on
`ubuntu-latest` (newer than the build runner) to verify forward
compatibility. It cannot, by construction, catch backward-incompatibility
against an *older* host than the build runner — that gap is closed by the
manual clean-VM pass on a 22.04-class system.

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

### Signing and provenance

GPG signing covers the Linux artifacts; build-provenance attestations cover
all five platforms.

| Artifact          | Signature form                  | How users verify                                                                |
| ----------------- | ------------------------------- | ------------------------------------------------------------------------------- |
| `.rpm`            | embedded (rpmsign --addsign)    | `rpm --import …blackchirp-release.asc` → `rpm --checksig` / `zypper install`    |
| `.deb`            | detached `.asc`                 | `gpg --import …blackchirp-release.asc` → `gpg --verify Blackchirp-*.deb.asc`    |
| AppImage          | detached `.asc`                 | `gpg --verify Blackchirp-*.AppImage.asc Blackchirp-*.AppImage`                  |
| `.dmg` / `.exe`   | unsigned                        | (see attestation row)                                                           |
| any of the above  | GitHub build-provenance         | `gh attestation verify <file> --owner kncrabtree`                               |

The signing key is a 4096-bit RSA GPG key, ID `898734DF7EDBDE45`, dedicated
to release signing (no daily-use mail attached). Public key:
`packaging/blackchirp-release.asc`, also attached to every GitHub release
by the deb job. Private key + passphrase live in repo Actions secrets
(`GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`, `GPG_KEY_ID`); offline backup of the
secret key is the user's responsibility.

DEB and AppImage use detached `.asc` rather than embedded signing because
apt does not verify in-`.deb` signatures (the apt trust model signs the
repository's `Release` file, not individual `.deb`s) and AppImage's
appended-signature scheme has near-zero downstream consumer support; a
side-car `.asc` users verify with stock `gpg --verify` is the most
broadly-supported form. RPM uses embedded signing because that is exactly
what `rpm --checksig` and `zypper install` consult.

macOS DMG and Windows NSIS are not GPG-signed: the relevant signing for
those platforms is Apple Developer ID + notarization (~$99/yr) and
Authenticode (~$200–$500/yr), respectively. Self-signing on Windows makes
SmartScreen warn harder, not less; ad-hoc signing on macOS does not satisfy
Gatekeeper. Both are deferred until budget exists. Attestations apply to
those binaries regardless of OS-level signing.

`actions/attest-build-provenance@v2` produces a Sigstore-signed SLSA
provenance record proving each artifact was built by a specific workflow
run on a specific commit. Keyless via OIDC against Sigstore's Fulcio CA;
no key to manage. Verify with
`gh attestation verify Blackchirp-2.0.0.deb --owner kncrabtree`. Requires
top-level `permissions: id-token: write` and `attestations: write` (set in
the workflow header).

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

## Status

All five build jobs and all five `--version` smoke jobs are green
on master. Symbol capture lands per platform as a separate
`blackchirp-symbols-<platform>` artifact (90-day retention).
Crash-log → symbol-artifact triage is documented in
`doc/source/developer_guide/crash_handling.rst`; the per-job Qt /
Qwt sourcing matrix and packaging knobs are in
`doc/source/developer_guide/build_system.rst`.

The smoke layer covers "the dynamic loader resolves every
`NEEDED` entry and `main()` returns clean," but `--version`
early-returns before `QApplication` is constructed (added in
`22e99372` so headless Linux containers and deb-postinst scripts
could invoke it without `QT_QPA_PLATFORM=offscreen`). Everything
the `QApplication` ctor and `MainWindow` constructor exercise —
platform plugin discovery, deferred imageformat / iconengine /
TLS plugin loads, OpenGL context creation, the first-run
config-dialog `.ui` form, Gatekeeper / SmartScreen UX,
`QSettings` on a fresh user profile, the Release-build (-O3)
codepath through GUI init, the crash handler — is uncovered
until the manual clean-VM pass runs. See the rationale in **Next
steps**.

### Selected commits (most recent last)

The current shape of the pipeline came together over the
following commits. Each commit message has the full per-fix
detail; this list is just the navigation aid.

- `9307c968` — `--version` / `--help` flags, per-platform symbol
  capture + `symbols-manifest.json`, five companion `*-smoke`
  jobs.
- `22e99372` — five blockers from the first full smoke run: SLA
  on macOS DMG, `QApplication`-before-argv-parse on Linux,
  AppImage missing libGL on bare `ubuntu-latest`, Windows PDB
  filename mismatch (CMake `VERSION` is a no-op for `.exe` /
  `.pdb`), and AppleClang missing from the compiler-id guard
  (so macOS Release had no `-g` and dSYMs were 1.4 MB).
- `1da52d62` — `yes | hdiutil` was actively harmful once the SLA
  was suppressed; pwsh's `&` operator + GUI-subsystem `.exe`
  doesn't reliably wait, switched to
  `Start-Process -Wait -PassThru -RedirectStandardOutput`;
  `AttachConsole` gated on `GetFileType(stdout) == FILE_TYPE_UNKNOWN`
  so a parent's `-RedirectStandardOutput` doesn't get clobbered.
- `def09dc8` — `windeployqt` walks Qt module imports only;
  manually `install(FILES …)` for `qwt.dll` and the GSL DLLs.
- `c5d48569` / `2761e645` / `70df7339` — Windows smoke still red
  after the manual install: turned out `qwt.dll` itself imports
  `Qt6OpenGL.dll` / `Qt6OpenGLWidgets.dll`. Final form: a second
  `install(CODE)` block runs `windeployqt` against `qwt.dll`
  with `--dir` anchored on the install bin. Diagnostic upgrade
  in the smoke step (recursive listing on non-zero exit) is what
  made the Qt6OpenGL miss visible from CI logs alone.
- `e900a006` — `virtual ~OverlayBase = default;`, surfaced by
  AppleClang's `-Wdelete-non-abstract-non-virtual-dtor`. Currently
  safe (`shared_ptr` type-erases the deleter) but UB-in-waiting
  the moment anyone introduces `unique_ptr<OverlayBase>`.

## Next steps before alpha tag

1. **Manual clean-VM smoke test** — the load-bearing item.
   The in-CI `--version` exercises only `main()` entry and the
   dynamic loader; the entire Qt runtime + `MainWindow`
   constructor is uncovered. Per-platform: launch the installed
   .app/.exe/.AppImage, see the main window draw, dismiss it.
   ~10 minutes per platform. What the manual pass catches that
   CI does not:
   - Qt platform plugin load (`qwindows` / xcb / cocoa) inside
     the `QApplication` ctor.
   - Deferred plugin loads on first paint (`imageformats/`,
     `iconengines/qsvgicon`, `tls/qschannelbackend`,
     `styles/qmodernwindowsstyle`).
   - OpenGL context creation when qwt's plot is first
     instantiated (linking `Qt6OpenGL.dll` ≠ initializing it).
   - Release-build (-O3) codepath through GUI init —
     local builds here are Debug, CI Release-builds but doesn't
     exercise GUI; any UB-at-O3 bug is unobserved.
   - First-run dialog (`savePath` empty → `ApplicationConfigDialog`
     + `RuntimeHardwareConfigDialog`); `.ui` forms loaded by uic.
   - Gatekeeper / SmartScreen UX on unsigned binaries; what the
     user has to click through after downloading.
   - `QSettings` against a fresh user profile (HKCU / XDG / plist
     creation).
   - The crash handler itself (`CrashHandler::install`,
     `MiniDumpWriteDump`, signal handlers); never run in Release
     because `--version` early-returns before it.

   Per-platform target list:
   - `.rpm` on openSUSE Tumbleweed — `rpm -qpR` shows auto-derived
     requirements (`libQt6*.so.6`, `libgsl.so.*`, `libqwt-qt6.so.*`).
   - `.deb` on Ubuntu LTS — `dpkg -I` shows non-empty `Depends:`;
     bundled libqwt resolves through the binary RPATH.
   - AppImage on a distro without Blackchirp's Qt6 packages.
   - `.dmg` on a clean macOS install — both `.app`s launch and
     find bundled libqwt (`otool -L` shows `@executable_path/...`).
   - NSIS `.exe` on a clean Windows install — main window opens
     without `STATUS_DLL_NOT_FOUND` from a transitive Qt6OpenGL
     load.

2. **Verify the Windows ZIP and NSIS now ship only the
   Applications component on the next CI run.** With
   `CPACK_ARCHIVE_COMPONENT_INSTALL ON` and
   `CPACK_NSIS_COMPONENT_INSTALL ON` (committed alongside this
   doc update), the ZIP should drop from ~88 MB to a fraction of
   that, and `lib/`, `include/`, `share/blackchirp` Development /
   Libraries content should disappear from both `.zip` and
   `.exe`. Locally verified on Linux TGZ (4.3 MB output, only
   Applications staged); Windows-specific NSIS behavior in
   component mode needs the CI run to confirm. Smoke step's
   `Get-ChildItem -Recurse C:\Blackchirp` already catches any
   regression here for free.

3. **Confirm version strings line up with the alpha tag.**
   - `BC_RELEASE_VERSION` is plain `set("alpha")` in
     `CMakeLists.txt` (no longer `CACHE STRING`, so it always
     reflects source-truth — a stale local cache silently
     overrode this previously).
   - `BCV_RELEASE_VERSION` is also `"alpha"`.
   - `python/blackchirp/pyproject.toml` stays decoupled for
     alpha / beta / rc; sync to a PEP-440 release at v2.0.0.
   - `--version` output should read `blackchirp 2.0.0-alpha
     (build <sha>)` — verify locally before tagging.

4. **Tag scheme.** The historical `v1.0.0-release` /
   `v1.1.0-release` convention is non-standard SemVer:
   `-release` is treated as a *pre-release identifier*, so
   `v1.1.0-release` literally means "a pre-release of 1.1.0,"
   sorting **before** a bare `v1.1.0`. This works against
   tooling that auto-detects "the latest stable release" (RTD's
   `stable` pointer, GitHub's "Latest" badge, package-version
   checkers) and probably explains why
   `v1.1.0-release` did not auto-build a documentation version
   on Read the Docs.
   - **Recommended scheme:** `vX.Y.Z` for the actual release;
     `vX.Y.Z-alpha`, `vX.Y.Z-beta`, `vX.Y.Z-rc.1` for
     pre-releases. The `v` prefix is fine — RTD strips it for
     display and CMake's `PROJECT_VERSION` accepts the digits-only
     form.
   - **For this drop:** `v2.0.0-alpha` (or `v2.0.0-alpha.1` if
     multiple alphas are anticipated).
   - **For the eventual stable:** `v2.0.0` — *not*
     `v2.0.0-release`.

5. **Read the Docs auto-build configuration.** RTD's tag →
   version mapping is governed by **Automation Rules** in the
   project admin panel, not `.readthedocs.yaml`. The yaml
   controls *how* a version builds; *whether* a tag is auto-
   activated is project-admin only. Without an Activate-version
   rule, RTD discovers tags as **inactive** versions that must
   be activated manually — almost certainly why
   `v1.1.0-release` produced no doc build (the version exists,
   just hidden).
   - In RTD admin → Automation Rules, add: type **Activate
     version**, version type **Tag**, predicate `^v\d+\.\d+\.\d+(-(alpha|beta|rc)(\.\d+)?)?$`
     (or just `^v` to be permissive).
   - With SemVer-compliant tags, RTD's `stable` alias maps to
     the highest non-pre-release tag automatically. The legacy
     `-release` tags broke this; the recommended scheme above
     restores it.
   - One-time admin action; no repo change required.

6. **Cut the alpha release.** Tag `v2.0.0-alpha`, push; the
   workflow's `release: published` trigger fires across all
   five platforms and uploads artifacts to the GitHub release.
   The three pre-release notices added in `f0a8596a` (README,
   `doc/source/index.rst`, `doc/source/python.rst`) **stay
   through alpha / beta / rc** — they go away only at the
   v2.0.0 release.

7. **Follow-up not blocking the alpha** (deferred):
   - VC++ Redistributable distribution for end users on a stock
     Windows install. Smoke-runners-with-VS-installed happens
     to mask this; a clean-VM Windows test (item 1) will surface
     it. Options: NSIS auto-run of `vc_redist.x64.exe`, or
     document it as a prerequisite. The clean-VM pass decides
     which.
