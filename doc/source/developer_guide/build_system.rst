.. index::
   single: build system
   single: CMake
   single: BuildConfig.cmake
   single: hw_base.h
   single: hw_impl.h
   single: hw_h.h
   single: hardware aggregator headers
   single: AUTOMOC
   single: CPack
   single: blackchirp_deploy_qt
   single: tests; cmake target
   single: documentation; cmake target

Build System and Project Layout
===============================

This page is the contributor-facing tour of Blackchirp's CMake setup —
the layout of the ``cmake/*.cmake`` modules, the user-facing build
options, the auto-generated hardware aggregator headers that hold the
runtime registration model together, the test infrastructure, the
documentation targets, and the CPack-based packaging path. The
user-facing source-build steps live on :doc:`/user_guide/installation`;
this page is one level deeper and assumes you are reading the cmake
files alongside it.

Hardware *selection* is not part of the build. Every hardware
implementation in ``src/hardware/`` is always compiled into the
``blackchirp-hardware`` library; the active set is decided at run time
by the registry and profile system. There is no compile-time flag that
filters drivers in or out.

CMake module map
----------------

The top-level ``CMakeLists.txt`` sets project-wide options, finds the
external dependencies (Qt6, GSL, Eigen3, Qwt), then ``include()``-s one
``cmake/*.cmake`` module per logical layer. Each module is
self-contained, guarded against double-inclusion, and produces exactly
one library or executable target.

``BlackchirpData.cmake`` → ``blackchirp-data`` (STATIC)
   Data model, analysis (FT, peak finder), experiment configuration and
   validation, loadout system, file parsers, overlay processing, storage
   classes (``BlackchirpCSV``, ``SettingsStorage``, ``HeaderStorage``,
   ``FidStorageBase``), and logging. Linked into both the main app and
   the viewer.

``BlackchirpHardware.cmake`` → ``blackchirp-hardware`` (STATIC)
   All hardware base classes, every concrete implementation,
   communication protocols (``rs232``, ``tcp``, ``virtual``, ``gpib``,
   ``custom``), the registration machinery
   (:cpp:class:`HardwareManager`, :cpp:class:`HardwareRegistry`,
   :cpp:class:`HardwareProfileManager`,
   :cpp:class:`RuntimeHardwareConfig`), the Python trampolines, and the
   vendor library wrappers. Skipped when ``BC_BUILD_VIEWER_ONLY=ON``.

``BlackchirpGui.cmake`` → ``blackchirp-gui`` (STATIC)
   Full Qt Widgets layer: main window, dialogs, experiment-setup wizard
   pages, plots, overlay widgets, theme code. Main app only.

``BlackchirpViewerGui.cmake`` → ``blackchirp-viewer-gui`` (STATIC)
   Lighter GUI subset for the viewer — plotting and experiment
   inspection without any hardware dependency. Always built.

``BlackchirpApplication.cmake`` → ``blackchirp`` (executable)
   Glues data + GUI + hardware together, compiles ``main.cpp`` and the
   acquisition layer (``AcquisitionManager``, ``BatchManager`` and
   friends), wires Qt resources, and registers
   ``blackchirp_deploy_qt(blackchirp)``. Skipped when
   ``BC_BUILD_VIEWER_ONLY=ON``.

``BlackchirpViewerApplication.cmake`` → ``blackchirp-viewer`` (executable)
   Viewer entry point; links data + viewer-gui only and registers
   ``blackchirp_deploy_qt(blackchirp-viewer)``.

``BlackchirpDocumentation.cmake`` → ``docs``, ``doxygen`` (custom targets)
   Sphinx HTML build and Doxygen XML/HTML build. Wired only when
   ``BC_BUILD_DOCUMENTATION=ON``.

``Packaging.cmake`` → CPack configuration
   Per-platform generator selection (``DEB;RPM;TGZ`` on Linux,
   ``DragNDrop;TGZ`` on macOS, ``NSIS;ZIP`` on Windows), component
   restriction to ``Applications``, and the ``package-deb``,
   ``package-rpm``, ``package-dmg``, ``package-nsis``, and
   ``package-all`` custom targets.

``QtDeployment.cmake`` → ``blackchirp_deploy_qt(<target>)``
   Install hook that runs ``windeployqt`` (Windows) or ``macdeployqt``
   (macOS) against the installed binary so the CPack package is
   self-contained. No-op on Linux, where Qt comes from the system
   package manager.

``FindQWT.cmake`` → ``QWT::QWT`` (imported)
   Custom finder for the Qwt scientific-plotting library. No Qt6 Qwt
   Config file ships with most distributions, so this module probes a
   list of common include and lib paths, falls back to ``pkg-config``,
   and exposes ``QWT::QWT``.

Configuring a build
-------------------

Build directories live under ``build/`` inside the source tree (so the
debug, release, and test trees are easy to find from the project root):

.. code-block:: bash

   cmake . -B build/Desktop-Debug/
   cmake --build build/Desktop-Debug/ -j$(nproc)

   cmake . -B build/Desktop-Release/ -DCMAKE_BUILD_TYPE=Release
   cmake --build build/Desktop-Release/ -j$(nproc)

Use ``cmake --build`` rather than ``make -C``. CMake regenerates the
build system mid-invocation when ``CMakeLists.txt`` or any included
``.cmake`` module changes; ``make`` aborts in that case with
"No rule to make target 'CMakeFiles/Makefile2'", whereas ``cmake
--build`` re-invokes the generator cleanly.

The default build type is Debug. A Release build switches on ``-O3``
(GCC/Clang) or ``/O2`` (MSVC) and suppresses ``qDebug()`` output.

The ``BuildConfig.cmake`` user-options file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User-facing build options are not declared in ``CMakeLists.txt``
directly — they live in ``cmake/BuildConfig.cmake``. This file is
**git-ignored**, so changes survive ``git pull``. On the first cmake
configure, the top-level ``CMakeLists.txt`` notices the file is missing
and copies ``cmake/BuildConfig.cmake.template`` into place. After that,
edits to ``BuildConfig.cmake`` are yours to keep.

The four user-facing options are:

``BC_BUILD_VIEWER_ONLY`` (default ``OFF``)
   Build only ``blackchirp-viewer``: skip the hardware library, skip the
   main GUI library, and skip the main application executable. Useful on
   analysis machines without lab hardware attached. The main GUI library
   pulls in ``Qt6::SerialPort`` transitively, which is the reason for
   the hard split.

``BC_BUILD_TESTS`` (default ``ON``)
   Build the unit-test executables and the ``tests`` aggregate custom
   target. Implies ``enable_testing()``, so ``ctest`` works in the build
   directory.

``BC_BUILD_DOCUMENTATION`` (default ``OFF``)
   Wire the ``docs`` and ``doxygen`` custom targets. Requires
   ``sphinx-build`` (and Doxygen for the API reference) to be
   discoverable; otherwise the targets are silently omitted.

``BC_ENABLE_CUDA`` (default ``OFF``)
   Enable the CUDA module under ``src/modules/cuda/`` for GPU-side FID
   averaging. Requires the NVIDIA CUDA Toolkit. Turning this on enables
   the ``CUDA`` language and adds a small set of ``.cu`` sources to the
   main app target.

   .. warning::

      The CUDA module is unmaintained and unlikely to build or run
      successfully without significant work. Developers may experiment
      with it at their own risk. Contributions are welcome for
      modernizing the code, configuring the build (PTX, fixed-architecture
      builds), and runtime discovery / integration with
      :cpp:class:`ApplicationConfigManager`.

There is no compile-time switch for hardware availability. If you add a
new driver under ``src/hardware/core/<type>/`` or
``src/hardware/optional/<type>/`` matching one of the recognized name
patterns described below, it is picked up automatically on the next
cmake configure.

Building targets
----------------

Top-level targets you will use most often:

.. code-block:: bash

   cmake --build build/Desktop-Debug/ --target blackchirp -j$(nproc)
   cmake --build build/Desktop-Debug/ --target blackchirp-viewer -j$(nproc)
   cmake --build build/Desktop-Debug/ --target tests -j$(nproc)
   cmake --build build/Desktop-Debug/ --target docs -j$(nproc)
   cmake --build build/Desktop-Debug/ --target doxygen -j$(nproc)

The ``tests`` aggregate target depends on every individual test
executable and is the convenient way to build the whole suite.
``docs`` depends transitively on ``doxygen`` when both Sphinx and
Doxygen were located, so a single ``--target docs`` invocation
produces both the prose pages and the API reference.

Hardware aggregator headers
---------------------------

``BlackchirpHardware.cmake`` writes three headers into
``src/hardware/core/`` at configure time:

* ``hw_base.h`` — every hardware *base type*: ``clock.h``,
  ``ftmwscope.h``, ``awg.h``, ``pulsegenerator.h``,
  ``flowcontroller.h``, ``gpibcontroller.h``, ``ioboard.h``,
  ``pressurecontroller.h``, ``temperaturecontroller.h``,
  ``lifscope.h``, ``liflaser.h``.
* ``hw_impl.h`` — every concrete implementation header that the
  configure-time ``file(GLOB)`` calls find under
  ``src/hardware/core/<type>/`` and ``src/hardware/optional/<type>/``,
  plus every Python trampoline header under ``src/hardware/python/``.
* ``hw_h.h`` — a one-line wrapper that ``#include``-s both of the
  above. This is the header consumers refer to when they want "all
  hardware types and implementations."

These are not just convenience headers. They exist because of how
``CMAKE_AUTOMOC`` interacts with static libraries and Qt's static
registration model.

Every concrete driver registers itself with
:cpp:class:`HardwareRegistry` at static-initialization time via
``REGISTER_HARDWARE_META`` and friends (see
:doc:`/classes/hardwareregistry`). The registration code lives in the
driver's ``.cpp`` translation unit at file scope. In a static-library
build, the linker is allowed to drop any object file whose symbols are
not referenced from the final executable — and a static initializer
counts as "unreferenced" for that purpose. Without an explicit symbol
reference into each implementation, the registrations would silently
disappear at link time and the registry would come up empty.

The fix is to feed the implementation headers to ``AUTOMOC``. AUTOMOC
generates ``moc_<class>.cpp`` for every ``Q_OBJECT`` it finds, and the
generated ``meta_object_offsets`` references pull the corresponding
object file out of the static library at link time. Listing every
implementation header in ``hw_impl.h`` (which is itself part of the
``blackchirp-hardware`` source set) is what gives AUTOMOC the
visibility it needs.

This is also why the Python trampoline headers
(``src/hardware/python/python*.h``) get appended to
``hw_impl.h`` even though they live outside the standard
``hardware/<type>/`` glob: the trampolines are ``Q_OBJECT`` subclasses
that register themselves the same way, and their meta-object code has
to be generated alongside the rest.

Glob-based source discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BlackchirpHardware.cmake`` discovers driver source files via
``file(GLOB)`` against fixed name patterns under each hardware-type
directory. Dropping a new ``xxxx.cpp``/``xxxx.h`` pair into one of the
recognized directories under one of the recognized patterns is enough;
no edit to ``CMakeLists.txt`` or ``BlackchirpHardware.cmake`` is
needed, but you do need to re-run ``cmake`` (not just ``cmake --build``)
so the glob is re-evaluated.

The recognized patterns, by hardware type:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Directory
     - Patterns
   * - ``hardware/core/ftmwdigitizer/``
     - ``virtual*``, ``dsa*``, ``m4i*``, ``dso*``, ``dpo*``, ``mso*``
   * - ``hardware/core/clock/``
     - ``fixedclock``, ``valon*``, ``hp*``
   * - ``hardware/core/lifdigitizer/``
     - ``virtual*``, ``m4i*``, ``rigol*``
   * - ``hardware/core/liflaser/``
     - ``virtual*``, ``opolette``, ``sirah*``
   * - ``hardware/optional/chirpsource/``
     - ``virtual*``, ``awg*``, ``ad*``, ``m8*``
   * - ``hardware/optional/pulsegenerator/``
     - ``virtual*``, ``qc*``, ``bnc*``, ``srs*``
   * - ``hardware/optional/flowcontroller/``
     - ``virtual*``, ``mks*``
   * - ``hardware/optional/gpibcontroller/``
     - ``virtual*``, ``prologix*``
   * - ``hardware/optional/ioboard/``
     - ``virtual*``, ``labjack*``, ``u3.cpp`` (UNIX only;
       removed on Windows)
   * - ``hardware/optional/pressurecontroller/``
     - ``virtual*``, ``intellisys*``
   * - ``hardware/optional/tempcontroller/``
     - ``virtual*``, ``lakeshore*``

If a new driver does not match any existing pattern (for example, a
new vendor prefix), add the prefix to both the implementations glob and
the headers glob in ``BlackchirpHardware.cmake``. The two lists are
parallel; keep them in sync.

Python hardware files are handled by a separate pair of globs against
``src/hardware/python/``:

* ``src/hardware/python/*.cpp`` — the C++ trampolines (one per
  hardware type) compile straight into ``blackchirp-hardware``.
* ``src/hardware/python/python_hw_host.py`` and
  ``src/hardware/python/python_*_template.py`` — the runtime files are
  ``configure_file``-d into the build directory (so a
  source-tree-relative dev run finds them) and ``install``-ed under
  ``${CMAKE_INSTALL_DATADIR}/blackchirp/`` (so a packaged install does
  too). The host script is the only Python file Blackchirp loads at
  runtime; the templates are seeds copied into the user's data
  directory when they create a new Python driver profile.

See :doc:`adding_a_driver` for the contributor-side recipe and
:doc:`python_hardware` for the Python hardware architecture.

Test infrastructure
-------------------

When ``BC_BUILD_TESTS=ON`` (the default), ``CMakeLists.txt`` calls
``enable_testing()``, requires ``Qt6::Test``, defines a series of
``add_executable``/``target_link_libraries``/``add_test`` triples, and
collects them into the ``tests`` aggregate target.

Test executables and what each covers:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Executable
     - Coverage
   * - ``tst_settingsstoragetest``
     - :cpp:class:`SettingsStorage` round-trip and key-namespace rules.
   * - ``tst_headerstoragetest``
     - :cpp:class:`HeaderStorage` write/read round-trip.
   * - ``tst_blackchirpcsvtest``
     - The semicolon-delimited CSV reader/writer.
   * - ``tst_ftworkertest``
     - :cpp:class:`FtWorker` Fourier-transform path (uses GSL).
   * - ``tst_genericxyparser``
     - The generic XY parser plus its test data set under ``tests/testdata/``.
   * - ``tst_operation_capability_only``
     - Overlay-operation capability flags in isolation from the rest of the GUI layer.
   * - ``tst_overlayoperations_simple``
     - Concrete overlay operations against synthetic input.
   * - ``tst_spcatparser``, ``tst_xiamparser``
     - Catalog/spectrum parsers against fixture files.
   * - ``tst_hardwareregistrytest``
     - Registry behavior: registration macros, lookups, factory invocation.
   * - ``tst_runtimehardwareconfigtest``
     - :cpp:class:`RuntimeHardwareConfig` profile resolution and selection.
   * - ``tst_hardwareprofilemanagertest``
     - :cpp:class:`HardwareProfileManager` add/remove/activate flows.
   * - ``tst_hardwarekeys``
     - Static hardware-key declarations in ``hardwarekeys.h``.
   * - ``tst_experimentloading``
     - Round-tripping a saved experiment through the data layer.
   * - ``tst_scientificspinboxtest``
     - The ``ScientificSpinBox`` widget; runs against ``QT_QPA_PLATFORM=offscreen``.
   * - ``tst_zoompanplotthreadsafety``
     - Concurrent access patterns on the plot layer; also offscreen.
   * - ``tst_waveformbuffertest``
     - The :cpp:class:`WaveformBuffer` ring buffer.
   * - ``tst_loadoutmanagertest``
     - :cpp:class:`LoadoutManager` save/load flows.

Run the full suite with:

.. code-block:: bash

   ctest --test-dir build/Desktop-Debug

Or build and run a single test directly:

.. code-block:: bash

   cmake --build build/Desktop-Debug --target tst_settingsstoragetest -j$(nproc)
   build/Desktop-Debug/tst_settingsstoragetest

The ``blackchirp-test-hardware`` library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A handful of tests need the hardware base classes plus the
``virtual*`` implementations, but cannot link against the main
``blackchirp-hardware`` library because that pulls in
:cpp:class:`HardwareManager`, which in turn references symbols that
only the main application provides. ``CMakeLists.txt`` defines a
parallel static library, ``blackchirp-test-hardware``, that includes
the hardware base classes, the communication protocols, and just the
``virtual*`` implementations. Tests like ``tst_experimentloading``
link against this instead so they can exercise the registration
machinery without dragging in the manager.

Adding a new test follows a four-step recipe:

1. Add ``add_executable(tst_yourthing tests/tst_yourthing.cpp)``.
2. ``target_link_libraries(tst_yourthing blackchirp-data Qt6::Test
   Qt6::Core)`` — substitute ``blackchirp-test-hardware`` for
   ``blackchirp-data`` if the test instantiates virtual hardware.
3. ``add_blackchirp_definitions(tst_yourthing)`` and (if applicable)
   ``add_test(NAME YourThingTest COMMAND tst_yourthing)``.
4. Append ``tst_yourthing`` to the ``add_custom_target(tests
   DEPENDS …)`` list near the bottom of the test block so the
   aggregate target picks it up.

If your test needs widgets or runs headless, set
``QT_QPA_PLATFORM=offscreen`` via ``set_tests_properties(... PROPERTIES
ENVIRONMENT ...)`` — the existing widget-touching tests do this.

Documentation build
-------------------

With ``BC_BUILD_DOCUMENTATION=ON``, ``BlackchirpDocumentation.cmake``
finds ``sphinx-build`` and Doxygen and registers the ``docs``,
``doxygen``, ``docs-pdf`` (if LaTeX is found), ``docs-linkcheck``, and
``docs-clean`` custom targets. The Sphinx config and the Doxyfile
template (``Doxyfile.in``) live in ``doc/source/``.

The Sphinx setup uses Breathe to ingest the Doxygen XML, which means a
stale Doxygen tree can leave the HTML build referencing classes that no
longer exist. Editing an RST page and rebuilding sometimes does not
trigger a Doxygen regeneration on its own. The reliable recipe is:

.. code-block:: bash

   touch doc/source/index.rst
   conda run -n breathe cmake --build build --target docs

The ``touch`` forces Sphinx to re-read the toctree, and running inside
the ``breathe`` conda environment ensures the right Sphinx, Breathe,
and theme versions are on the path.

Output locations:

* ``build/docs/html/`` — Sphinx HTML output (the ``index.html``
  CMake actually depends on lives at ``build/docs/html/index.html``).
* ``build/docs/doxygen/html/`` — Doxygen HTML browser, when the
  ``doxygen`` target has run.
* ``build/docs/doxygen/xml/`` — Doxygen XML, the input to Breathe.

Packaging
---------

Blackchirp's binary distribution is driven by **CPack** for the
per-platform package formats and by **linuxdeploy** for the Linux
AppImage. The configuration is small enough to fit on this page; for
the contributor-level "what does each install rule do" view, read
``cmake/Packaging.cmake`` alongside this section.

Per-platform CPack generators are selected in ``Packaging.cmake``:

* **Linux** — ``DEB``, ``RPM``, ``TGZ``. DEB dependencies are
  auto-derived via ``CPACK_DEBIAN_PACKAGE_SHLIBDEPS=ON``; RPM
  dependencies via ``CPACK_RPM_PACKAGE_AUTOREQ=ON``. Both read the
  shared-library references in the linked binary and emit the right
  Qt6/GSL/Qwt requirement lines without hard-coding distro-specific
  package names. Snap and Flatpak are intentionally omitted because
  their sandboxes restrict the serial-port and USB access that
  Blackchirp's hardware drivers depend on.
* **macOS** — ``DragNDrop`` (DMG) and ``TGZ``. The DMG generator picks
  up bundle metadata from the ``MACOSX_BUNDLE_*`` properties set on
  the executable targets in ``BlackchirpApplication.cmake`` /
  ``BlackchirpViewerApplication.cmake``; do not use ``CPACK_BUNDLE_*``
  variables, which apply to a different generator.
* **Windows** — ``NSIS`` (installer ``.exe``) and ``ZIP``. NSIS adds
  Start-menu and desktop shortcuts and writes an uninstaller.

Both apps install with ``BUNDLE DESTINATION .`` so the ``.app`` lands
at the install-prefix root — that matches the DragNDrop layout (drag
the app onto the ``Applications`` shortcut) and is the path
``blackchirp_deploy_qt`` looks for at install time when it runs
``macdeployqt`` against the bundle.

``CPACK_COMPONENTS_ALL`` is restricted to a single ``Applications``
component. The ``Libraries`` and ``Development`` install rules — the
static ``.a`` archives, headers, and CMake export files — exist for
``cmake --install`` workflows in source trees but are not shipped in
binary packages. Every ``blackchirp-*`` library is STATIC and linked
into the executables, so shipping the archives or headers would only
inflate the package size.

The Qt redistribution path differs by platform:

* **Linux** — Qt comes from the system package manager; the
  shlibdeps/AUTOREQ machinery records the dependency.
* **Windows / macOS** — ``cmake/QtDeployment.cmake`` provides
  ``blackchirp_deploy_qt(<target>)``, an ``install(CODE)`` hook that
  locates ``windeployqt`` or ``macdeployqt`` from
  ``Qt6::qmake``'s ``IMPORTED_LOCATION`` and runs it against the
  staged binary at install time. Each application module calls this
  function once after its ``install(TARGETS ...)`` rule.
* **AppImage** — built outside CPack with ``linuxdeploy`` plus the
  ``linuxdeploy-plugin-qt`` plugin, which walks the executable's
  shared-library closure and bundles everything needed.

CPack convenience targets exposed by ``Packaging.cmake``:

.. code-block:: bash

   cmake --build build --target package-deb     # Linux
   cmake --build build --target package-rpm     # Linux
   cmake --build build --target package-dmg     # macOS
   cmake --build build --target package-nsis    # Windows
   cmake --build build --target package-all     # Binary + source

Continuous integration
~~~~~~~~~~~~~~~~~~~~~~

``.github/workflows/release.yml`` drives all five binary platforms from
a single workflow file. The five jobs are:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Job
     - Runner
     - Output
   * - ``linux-deb``
     - ``ubuntu-latest``
     - ``.deb``
   * - ``linux-rpm``
     - ``opensuse/leap:16.0`` container
     - ``.rpm``
   * - ``linux-appimage``
     - ``ubuntu-latest`` + ``linuxdeploy``
     - ``.AppImage``
   * - ``macos-dmg``
     - ``macos-latest``
     - ``.dmg`` and ``.tar.gz``
   * - ``windows-nsis``
     - ``windows-latest`` (MSVC)
     - NSIS ``.exe`` and ``.zip``

Triggers are ``workflow_dispatch`` (with per-platform boolean inputs so
a single platform can be re-run in isolation while iterating on a CI
issue) and ``release: published`` (every job builds and uploads its
artifact, then ``gh release upload --clobber`` attaches it to the
release). Each job follows the same shape: install system dependencies,
install Qt 6.9.1 via ``jurplel/install-qt-action@v4``, restore-or-build
Qwt 6.3.0 from the SourceForge tarball (cached per-OS by Qt and Qwt
version), then ``cmake → cmake --build → ctest → cpack`` (or
``linuxdeploy`` for the AppImage), then upload.
