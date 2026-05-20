.. index::
   single: packaging
   single: CPack
   single: linuxdeploy
   single: AppImage
   single: DEB
   single: RPM
   single: DMG
   single: NSIS
   single: BC_BUNDLE_QWT
   single: windeployqt
   single: macdeployqt
   single: signing; GPG
   single: signing; rpmsign
   single: signing; release key
   single: attestation; build provenance
   single: attestation; Sigstore
   single: GitHub Actions; release workflow
   single: blackchirp.asc

Packaging and Release CI
========================

This page is the contributor-facing reference for how Blackchirp's
binary releases are produced. The build plumbing — CMake modules, Qt
deployment hooks, and the CPack configuration that ties them together
— lives in `cmake/` and is invoked by a single GitHub Actions
workflow at ``.github/workflows/release.yml``. The user-facing install
and verification instructions are in
:doc:`/user_guide/installation`; this page sits one level below those
and assumes the workflow file is open alongside it.

Binaries are generated on demand. The workflow runs on
``workflow_dispatch`` (with per-platform boolean inputs for
single-job iteration) or on ``release: published``. It does not run
on every push — a typical PR exercises Qt-Test and the docs build but
produces no installers.

Strategy
--------

Cross-platform binary distribution is driven by **CPack** for the
per-platform formats (``.deb``, ``.rpm``, ``.dmg``, NSIS, ``.tgz``,
``.zip``) and by **linuxdeploy** for the universal Linux AppImage.
One GitHub Actions workflow drives all five platforms.

The Linux matrix is intentionally redundant:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Format
     - Target audience
   * - ``.rpm``
     - openSUSE (primary build host), Fedora, RHEL — deps auto-derived
       at build time
   * - ``.deb``
     - Debian, Ubuntu, Mint — ``dpkg-shlibdeps`` auto-derived
   * - AppImage
     - Universal fallback (Arch, NixOS, anything else); two AppImages
       per release — one entry point for the main app, one for the
       viewer
   * - ``.tar.gz``
     - Generic binary tarball for the manual-extraction case

Snap and Flatpak are intentionally excluded: their sandbox models
restrict the serial-port and USB hardware access that Blackchirp
requires for acquisition.

macOS ships ``.dmg`` (DragNDrop) and ``.tar.gz`` in two
architectures — Apple Silicon (``arm64``) and Intel (``x86_64``);
CMake does not produce universal binaries by default, so each
``runs-on`` runner emits a single-arch slice and the two together
cover the installed-Mac base. Windows ships an NSIS installer
(``.exe``) and a ``.zip``.

Components contain only ``Applications``
----------------------------------------

``CPACK_COMPONENTS_ALL`` is restricted to the ``Applications``
component. The ``blackchirp-*`` libraries are all ``STATIC`` and
linked into the two executables, so the ``Libraries`` (``.a``
archives) and ``Development`` (headers + CMake export files) install
rules are dev-only — useful for ``cmake --install`` in source-tree
workflows, useless in a binary package.

``CPACK_DEB_COMPONENT_INSTALL`` / ``CPACK_RPM_COMPONENT_INSTALL`` /
``CPACK_ARCHIVE_COMPONENT_INSTALL`` / ``CPACK_NSIS_COMPONENT_INSTALL``
are all ``ON``. Without them, ``CPACK_COMPONENTS_ALL`` is silently
ignored for those generators and every install rule (including the
static archives and headers) lands in the package. With Release
builds and stripping enabled, the Applications-only filter yields
packages in the 4–9 MB range on Linux and a Windows ``.zip`` that
dropped from ~88 MB to a small fraction of that.

Qt and Qwt sourcing per job
---------------------------

The build jobs source Qt and Qwt differently because each platform
imposes different constraints on what can be linked, packaged, and
deployed.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Job
     - Qt
     - Qwt
   * - ``linux-deb``
     - apt ``qt6-base-dev`` (Ubuntu)
     - from-source + ``BC_BUNDLE_QWT``
   * - ``linux-rpm``
     - zypper ``qt6-base-devel`` (openSUSE Leap)
     - from-source + ``BC_BUNDLE_QWT``
   * - ``linux-appimage``
     - ``install-qt-action`` 6.9.1
     - from-source
   * - ``macos-dmg`` (arm64, x86_64)
     - ``install-qt-action`` 6.9.1
     - from-source
   * - ``windows-nsis``
     - ``install-qt-action`` 6.9.1 MSVC
     - from-source

The deb job uses Ubuntu's system Qt because ``dpkg-shlibdeps`` reads
Debian's ``*.shlibs`` database to derive dependencies — Qt installed
by ``install-qt-action`` has no Debian shlibs metadata and the
``cpack -G DEB`` step fails. Ubuntu LTS does not ship a Qt6 build of
Qwt at all (only the Qt5-era qwt 6.1.4), so the deb job builds Qwt
from source and bundles ``libqwt.so*`` inside the package via
``BC_BUNDLE_QWT=ON``. The executables get an
``$ORIGIN/../<libdir>/blackchirp`` RPATH, and ``dpkg-shlibdeps``
follows it to the bundled lib while resolving Qt sonames through
``/usr/lib``.

The rpm job follows the same bundle-Qwt pattern, but for a different
reason. openSUSE patches ``libqwt-qt6.so``'s SONAME to include the
minor version (``libqwt-qt6.so.6.3`` rather than upstream's
``.so.6``); RPM AUTOREQ records the linked SONAME verbatim, and
``libqwt-qt6.so.6.3`` is unsatisfiable on Fedora, RHEL, and any other
RPM distro that ABI-tracks Qwt at the major level. Bundling sidesteps
the soname mismatch entirely — the resulting RPM has no Qwt
dependency at all and installs cleanly on every RPM distro that has
Qt6.

The other three jobs build Qwt from source because no reliable Qt6
Qwt exists on Homebrew, vcpkg, or any LTS apt channel.

The Qwt cache is split into ``actions/cache/restore@v5`` +
``actions/cache/save@v5`` rather than the unified ``actions/cache@v5``
because the unified action's implicit post-step skips on job
failure — wasting the rebuild on every retry. The split form gates
the save on ``cache-hit != 'true'`` and places it immediately after
the build step, so a downstream failure does not invalidate the
saved Qwt.

AppImage specifics
------------------

Two AppImages per release
~~~~~~~~~~~~~~~~~~~~~~~~~

The Linux AppImage job emits both ``Blackchirp-x86_64.AppImage`` and
``Blackchirp-Viewer-x86_64.AppImage``. Each is fully self-contained
— bundled Qt/Qwt/GSL is the size driver and is duplicated across the
two — but the duplication is deliberate: AppImage users are exactly
the audience without a system package manager that pulls in both
binaries, so click-and-run discoverability beats download
efficiency. The main AppImage bundles both binaries internally, so
users who care about the size can run the viewer from inside it via
``--appimage-mount`` or ``--appimage-extract`` (the recipe is in
:ref:`installation-appimage-viewer-from-main`).

The build runs ``linuxdeploy`` twice against two AppDir copies. The
plugin mutates the AppDir in place (RPATH patches, AppRun injection,
libdir cleanup), so a single tree cannot be reused for two outputs —
the Stage AppDir step does ``cp -a AppDir AppDir-viewer`` before
linuxdeploy runs. ``OUTPUT=`` is set explicitly for the viewer
build; without it, appimagetool would mangle
``Name=Blackchirp Viewer`` (with a space) to
``Blackchirp_Viewer-x86_64.AppImage`` (with an underscore), breaking
the docs' ``Blackchirp-Viewer-*`` glob.

glibc floor
~~~~~~~~~~~

The AppImage build pins ``runs-on: ubuntu-22.04`` rather than
``ubuntu-latest``. AppImages bundle Qt, Qwt, libgsl, and the rest of
the executable's library closure but **not** glibc / libm — those
always come from the host loader. Symbol versions picked up by
bundled libraries at link time therefore become a hard host-glibc
minimum at run time: a ``libgsl.so.27`` built against glibc 2.39
references ``GLIBC_2.38``-versioned symbols in libm and fails to
load on any host older than that, defeating the AppImage's
"universal fallback" purpose. Building on ``ubuntu-22.04`` (glibc
2.35) caps the floor at the LTS distros the AppImage exists to serve
— Ubuntu 22.04+, RHEL 9+, openSUSE Leap 15.5+, Debian 12+. The Qwt
cache key bakes in the runner codename (``jammy``) so a future
runner upgrade auto-invalidates the cache rather than poisoning a
new build with stale glibc-2.39-linked artifacts.

The companion ``linux-appimage-smoke`` job stays on
``ubuntu-latest`` (newer than the build runner) to verify forward
compatibility. It cannot, by construction, catch
backward-incompatibility against an *older* host than the build
runner — that gap is closed by the manual clean-VM pass on a
22.04-class system.

Qt redistribution into the package
----------------------------------

How Qt ends up alongside the binary at install time differs per
platform:

**Linux DEB and RPM**
   The distro package manager resolves Qt at install time via
   auto-derived ``dpkg-shlibdeps`` (DEB) or RPM AUTOREQ (RPM). Both
   jobs additionally ship ``libqwt.so*`` bundled (see above).

**Windows and macOS**
   ``windeployqt`` / ``macdeployqt`` run as ``install(CODE)`` hooks
   registered by ``cmake/QtDeployment.cmake``. macOS passes
   ``-libpath=<qwt-install/lib>`` so macdeployqt can locate the
   from-source ``libqwt`` by basename and bundle it into
   ``Contents/Frameworks/``; qmake's macOS build leaves the dylib's
   ``install_name`` pointing at ``/usr/lib/...``, which does not
   exist on the runner.

**AppImage**
   ``linuxdeploy-plugin-qt`` walks the executable's library closure
   and bundles everything into the AppImage. ``LD_LIBRARY_PATH``
   must include ``$QT_ROOT_DIR/lib`` and the from-source
   ``qwt-install/lib`` during the linuxdeploy step, otherwise
   ``ldd`` reports Qt sonames as unresolved and the plugin refuses
   to bundle them.

Windows additionally needs a hand-rolled second ``windeployqt``
invocation against ``qwt.dll``. The qmake-built ``qwt.dll`` imports
``Qt6OpenGL.dll`` / ``Qt6OpenGLWidgets.dll``, but the first
``windeployqt`` pass walks the executable's modules only and misses
those transitive dependencies. ``cmake/QtDeployment.cmake``
registers a separate ``install(CODE)`` block that runs
``windeployqt`` against ``qwt.dll`` with ``--dir`` anchored on the
install bin so the missing DLLs land alongside the executable.

Signing and provenance
----------------------

GPG signing covers the Linux artifacts; build-provenance
attestations cover all five platforms.

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Artifact
     - Signature form
     - How users verify
   * - ``.rpm``
     - embedded (``rpmsign --addsign``)
     - ``rpm --import …blackchirp.asc`` → ``rpm --checksig``
       / ``zypper install`` / ``dnf install``
   * - ``.deb``
     - detached ``.asc``
     - ``gpg --import …blackchirp.asc`` →
       ``gpg --verify Blackchirp-*.deb.asc``
   * - AppImage
     - detached ``.asc``
     - ``gpg --verify Blackchirp-*.AppImage.asc``
   * - ``.dmg``
     - ad-hoc codesign (``codesign --force --deep --sign -``)
     - launches on a clean Mac after ``xattr -d
       com.apple.quarantine`` (no notarization yet — see below)
   * - ``.exe``
     - unsigned
     - (build-provenance only — see below)
   * - any of the above
     - GitHub build-provenance
     - ``gh attestation verify <file> --owner kncrabtree``

The release key is a 4096-bit RSA GPG key, ID ``898734DF7EDBDE45``,
dedicated to release signing. Public key:
``packaging/blackchirp.asc``, also attached to every GitHub
release by the deb job and published on ``keys.openpgp.org``. The
private key and passphrase live in repository Actions secrets
(``GPG_PRIVATE_KEY``, ``GPG_PASSPHRASE``, ``GPG_KEY_ID``). Offline
backup of the secret key is the maintainer's responsibility — if
both the local keyring and the offline backup are lost, no future
release can be signed with a key the existing user base trusts.

DEB and AppImage use detached ``.asc`` rather than embedded signing
because apt does not verify in-``.deb`` signatures (the apt trust
model signs the repository's ``Release`` file, not individual
``.deb`` files) and AppImage's appended-signature scheme has
near-zero downstream consumer support; a side-car ``.asc`` users
verify with stock ``gpg --verify`` is the most broadly-supported
form. RPM uses embedded signing because that is exactly what
``rpm --checksig`` and ``zypper`` / ``dnf install`` consult at
install time.

Neither macOS nor Windows uses GPG signing. The Windows NSIS
installer is unsigned: Authenticode would need a purchased
certificate, and self-signing makes SmartScreen warn harder rather
than less.

The macOS bundle **is** ad-hoc codesigned. ``macdeployqt`` rewrites
``LC_LOAD_DYLIB`` / ``install_name`` entries and copies framework
dylibs into the bundle, which invalidates the linker-applied ad-hoc
signature on the arm64 main executable and the upstream signatures
on the copied frameworks; the kernel ``SIGKILL``\ s an
invalidly-signed arm64 binary outright. ``cmake/QtDeployment.cmake``
therefore runs ``codesign --force --deep --sign -`` over the whole
bundle *after* ``macdeployqt``, and the ``macos-smoke`` job verifies
it with ``codesign --verify --deep --strict``. This is an ad-hoc
signature, not Apple Developer ID notarization: Gatekeeper still
flags the downloaded app as "damaged" until notarization is in
place, and users clear that with ``xattr -d com.apple.quarantine``
(documented in :doc:`/user_guide/installation`). Build-provenance
attestations apply to every artifact regardless of OS-level signing.

The RPM-signing step writes ``rpm-sign`` macros into
``~/.rpmmacros`` so ``rpmsign --addsign`` can drive GPG
non-interactively. The passphrase is read from a temp file created
with ``umask 077`` and ``shred -u``-d by an EXIT trap. The
``%__gpg_sign_cmd`` macro replaces GPG's default pinentry with
``--pinentry-mode loopback --passphrase-file <file>`` so the
container has no TTY for prompts. After signing, ``rpm --checksig``
runs against the signed file, but only after ``rpm --import`` has
populated rpm's own keyring (separate from gpg's) with the public
key — without the import, ``--checksig`` reports
``SIGNATURES NOT OK`` even on a valid signature.

``actions/attest-build-provenance@v2`` produces a Sigstore-signed
SLSA provenance record proving each artifact was built by a specific
workflow run on a specific commit. Keyless via OIDC against
Sigstore's Fulcio CA; nothing to manage or rotate. The workflow
requires top-level ``permissions: id-token: write`` and
``attestations: write``; without these the action errors out at the
OIDC-token mint step.

CMake modules and packaging files
---------------------------------

``cmake/Packaging.cmake``
   CPack configuration: package metadata, per-OS generator selection
   (``DEB;RPM;TGZ`` on Linux, ``DragNDrop;TGZ`` on macOS,
   ``NSIS;ZIP`` on Windows), component definitions, and
   platform-specific knobs. Owns the ``BC_BUNDLE_QWT`` option, the
   Windows third-party-DLL bundling logic, the per-format file-name
   overrides, and the ``$ORIGIN``-relative RPATH that points the
   executables at the bundled libqwt on Linux. Strips binaries in
   Release. Drives the ``package-all``, ``package-deb``,
   ``package-rpm``, ``package-nsis``, ``package-dmg`` custom targets.

``cmake/QtDeployment.cmake``
   Provides ``blackchirp_deploy_qt(<target>)``. Locates
   ``windeployqt`` / ``macdeployqt`` from ``Qt6::qmake``'s
   ``IMPORTED_LOCATION`` and registers ``install(CODE)`` hooks that
   run the right tool against the installed binary at packaging
   time. macOS derives ``-libpath=`` from ``QWT_LIBRARY`` so the
   from-source libqwt bundles correctly, then — because the
   ``VERSION`` target property leaves ``Contents/MacOS/<target>`` a
   symlink and ``codesign`` refuses a symlinked main executable —
   collapses that symlink onto the versioned binary and runs the
   ad-hoc ``codesign --force --deep --sign -`` pass over the bundle.
   No-op on Linux.

``cmake/BlackchirpApplication.cmake`` / ``cmake/BlackchirpViewerApplication.cmake``
   Per-app target wiring. Each sets ``MACOSX_BUNDLE_*`` properties
   (info plist, copyright, icon, version), installs to
   ``BUNDLE DESTINATION .`` (DragNDrop DMG convention) and
   ``RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}``, then calls
   ``blackchirp_deploy_qt(<target>)``.

``cmake/FindQWT.cmake``
   When ``qwt.h`` is found inside a ``qwt`` / ``qwt-qt6`` / ``qwt6``
   subdirectory, exposes the parent as a second include path so
   ``<qwt6/qwt_plot.h>`` resolves (the openSUSE convention used in
   the source). On Windows, sets ``QWT_DLL`` on the imported
   target's ``INTERFACE_COMPILE_DEFINITIONS`` when ``qwt.dll`` is
   present next to the import lib so MSVC consumers get
   ``__declspec(dllimport)`` on exported static data members.

``packaging/``
   Per-platform packaging assets. The full per-file map:

   .. list-table::
      :header-rows: 1
      :widths: 45 55

      * - Path
        - Role
      * - ``packaging/macos/Info.plist``
        - Bundle metadata for ``blackchirp.app``
      * - ``packaging/macos/ViewerInfo.plist``
        - Bundle metadata for ``blackchirp-viewer.app``
      * - ``packaging/linux/postinst``
        - ``update-desktop-database`` after DEB install
      * - ``packaging/linux/prerm``
        - ``update-desktop-database`` cleanup before DEB remove
      * - ``packaging/blackchirp.desktop.in``
        - XDG desktop file for the main app (substituted via
          ``configure_file``)
      * - ``packaging/blackchirp-viewer.desktop.in``
        - XDG desktop file for the viewer
      * - ``packaging/blackchirp.asc``
        - Public half of the GPG release signing key, attached to
          every GitHub release by the deb job
      * - ``icons/blackchirp.icns``
        - Multi-resolution macOS bundle icon
      * - ``src/resources/icons/bc_logo_large.png``
        - Source logo; used for the Linux pixmap install

Workflow structure
------------------

``.github/workflows/release.yml`` defines five build jobs (one per
platform; the macOS job is a matrix over ``arm64`` and ``x86_64``,
so six job executions per full run) and the matching ``*-smoke``
jobs that install each artifact in a clean-ish environment and
confirm ``<binary> --version`` exits zero. The smoke layer catches
the common packaging-step regressions: missing bundled libs, broken
RPATHs, soname mismatches, the wrong Qt module set in the bundle.
It does *not* exercise GUI initialization — ``--version``
early-returns before ``QApplication`` is constructed so headless
containers can invoke it without ``QT_QPA_PLATFORM=offscreen``.

Per-job skeleton:

#. Install system dependencies (apt / zypper / brew / vcpkg /
   chocolatey).
#. Install Qt (per the sourcing matrix above).
#. Restore or build Qwt 6.3.0 from source (cached per OS).
#. ``cmake → cmake --build → ctest``.
#. ``cpack`` (or ``linuxdeploy`` for AppImage). On macOS the
   ``QtDeployment.cmake`` install hook ad-hoc codesigns the bundle
   here, after ``macdeployqt``; the matching ``macos-smoke`` job
   later runs ``codesign --verify --deep --strict``.
#. Sign Linux artifacts (detached for DEB/AppImage, embedded for
   RPM).
#. ``actions/attest-build-provenance@v2``.
#. ``actions/upload-artifact`` and, on release events,
   ``gh release upload --clobber``.

The rpm job runs inside ``opensuse/leap:16.0``. Container jobs have
a quirk: ``${{ github.workspace }}`` templates to the runner host's
path (e.g., ``/home/runner/work/...``), which does not exist on the
container's bind mount. Only the ``$GITHUB_WORKSPACE`` env var
points at the mounted workspace (e.g., ``/__w/blackchirp/blackchirp``).
Writes to the host-path string from inside the container land on the
container's local filesystem and are invisible to the host — including
the ``actions/cache`` action that runs on the host. Container-side
script paths in the rpm job therefore reference ``$GITHUB_WORKSPACE``;
only actions running on the host (``actions/cache/*``) use the
templated host path.

Crash-log → symbol-artifact triage is documented separately in
:doc:`crash_handling`. Each build job uploads
``blackchirp-symbols-<platform>`` alongside the package (90-day
retention) and embeds the workflow ``run_id`` plus ``git_sha`` in a
``symbols-manifest.json`` inside the artifact, so a triager
downloading a crash log can locate the matching debug-info bundle
without guessing.

Non-intuitive constructions
---------------------------

A handful of choices in the packaging plumbing exist for non-obvious
reasons and should not be "tidied up" without rediscovering why they
look the way they do.

* ``include(GNUInstallDirs)`` is called early in the top-level
  ``CMakeLists.txt``, before any subdirectory ``install()`` rule.
  Otherwise subdirectory rules fall back to absolute paths (e.g.,
  ``/blackchirp`` for the Python templates), which breaks every
  CPack generator.
* macOS bundle metadata lives on the executable targets, set via
  ``MACOSX_BUNDLE_*`` target properties. The DragNDrop CPack
  generator picks them up automatically; do *not* use
  ``CPACK_BUNDLE_*``, which applies to the separate ``Bundle``
  generator and silently ignores ``MACOSX_BUNDLE_*``.
* Both apps install with ``BUNDLE DESTINATION .`` so the ``.app``
  lands at the install-prefix root. Matches the DragNDrop DMG
  layout (drag the ``.app`` straight onto Applications) and is the
  path ``blackchirp_deploy_qt`` looks for at install time.
* ``CPACK_SET_DESTDIR`` is left unset on Apple. DESTDIR-style
  staging is right for DEB/RPM/IFW, but for DragNDrop the ``.app``
  *is* the unit of distribution and the package root is the install
  root; with ``DESTDIR=ON``, CPack stages the ``.app`` under
  ``${DESTDIR}/usr/local/blackchirp.app`` and both the deploy hook
  and the DragNDrop file walk miss it.
* ``MACOSX_DEPLOYMENT_TARGET`` is pinned to ``13.3`` on both macOS
  jobs, not left at the runner default. Apple's libc++ marks the
  floating-point ``std::to_chars`` overloads ``introduced=13.3``, so
  a lower target fails to compile the shortest-roundtrip formatting
  in ``src/gui/util/numericformat.cpp`` and
  ``src/gui/widget/scientificspinbox.cpp``. 13.3 is therefore the
  binary's minimum macOS; the user-facing per-artifact minimum-OS
  table in :doc:`/user_guide/installation` is the single source of
  truth for the published floors.
* Distro package names are not hard-coded. DEB dependencies come
  from ``dpkg-shlibdeps``; RPM dependencies come from ``AUTOREQ``.
  Hard-coded ``Depends:`` lines drift across Ubuntu/Debian releases
  and across openSUSE/Fedora's qt6/gsl/qwt package names;
  auto-derivation tracks each distro's actual shipped sonames.
* Eigen3 has no version pin. The development-box system Eigen is
  5.0.1 and its CMake config rejects pre-5 minimum-version
  requests. If a CI runner ships Eigen 3.x and a minimum needs to
  be enforced, change ``find_package(Eigen3 REQUIRED)`` to
  ``find_package(Eigen3 3.3...<6 REQUIRED)``.
* ``blackchirp.icns`` was generated locally with ``icnsutil`` from
  ``src/resources/icons/bc_logo_large.png`` and is checked in.
  Regenerate only if the source logo changes.
* AppImage icon lookup uses the hicolor 256×256 path, not
  ``share/pixmaps/blackchirp.png``. The pixmap is 1024×1024 (sized
  for ``.icns`` / ``.ico`` masters), and linuxdeploy's icon
  validator caps at 512×512.
