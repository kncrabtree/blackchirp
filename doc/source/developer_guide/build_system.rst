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
   single: documentation; pipeline
   single: documentation; Sphinx
   single: documentation; Doxygen
   single: documentation; Breathe
   single: documentation; nbsphinx
   single: documentation; Read the Docs
   single: Python module; build
   single: Python module; pyproject.toml
   single: Python module; PyPI
   single: pyproject.toml

Build System and Project Layout
===============================

This page is the contributor-facing tour of Blackchirp's build
plumbing. The bulk is CMake — the ``cmake/*.cmake`` modules, the
user-facing build options, the auto-generated hardware aggregator
headers that hold the runtime registration model together, the test
infrastructure, the documentation targets, and the CPack-based
packaging path — but the documentation pipeline and the Python module
have their own sections below for the pieces CMake does not own.
:doc:`/user_guide/installation` covers the binary-package install
path; this page covers the from-source build that the installation
page links to.

Blackchirp produces three independently-buildable deliverables. The
**C++ application** (``blackchirp`` and ``blackchirp-viewer``) is
driven by CMake and packaged by CPack. The **documentation** under
``doc/source/`` is also a CMake target (``docs``), but its pipeline
chains four tools — Doxygen, Breathe, Sphinx, and nbsphinx — described
in *Documentation build* below. The **Python module** under
``python/blackchirp/`` is built outside CMake using ``pyproject.toml``
and is versioned and released to PyPI on its own cadence; *Python
module build* below covers it.

Hardware *selection* is not part of the build. Every hardware
driver in ``src/hardware/`` is always compiled into the
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
   All hardware base classes, every concrete driver,
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

Prerequisites
-------------

Install the following before configuring a build:

* A C++23-capable compiler (GCC 13+, Clang 16+, MSVC 19.35+).
* `CMake <https://cmake.org/>`_ 3.16 or later.
* `Qt 6 <https://www.qt.io/download-qt-installer-oss>`_ 6.4 or later
  with the Core, Gui, Widgets, Network, SerialPort, Concurrent, and
  Test modules. The Linux DEB release is built against the Qt that
  Ubuntu 24.04 (Noble) ships in ``qt6-base-dev`` (6.4.2); the other
  jobs build against newer Qt releases. The C++ code is kept
  compatible with the 6.4 baseline.
* `Qwt <https://qwt.sourceforge.io/>`_ 6.2 or later (Qt6 build).
  Distributions without a Qt6-compatible Qwt package require an
  in-tree build; see ``BC_BUNDLE_QWT`` in
  :doc:`packaging`.
* `GNU Scientific Library (GSL) <https://www.gnu.org/software/gsl/>`_
  2.1 or later.
* `Eigen3 <https://eigen.tuxfamily.org/>`_ 3.3 or later (header-only).

Optional:

* NVIDIA CUDA Toolkit — required only when ``BC_ENABLE_CUDA=ON``.
  See the warning under ``BC_ENABLE_CUDA`` below before turning this
  on.
* `Doxygen <https://www.doxygen.nl/>`_ plus a Python environment that
  satisfies ``doc/source/requirements.txt`` — required only when
  ``BC_BUILD_DOCUMENTATION=ON``. See *Documentation build* below.

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
  ``ftmwdigitizer.h``, ``awg.h``, ``pulsegenerator.h``,
  ``flowcontroller.h``, ``gpibcontroller.h``, ``ioboard.h``,
  ``pressurecontroller.h``, ``temperaturecontroller.h``,
  ``lifdigitizer.h``, ``liflaser.h``.
* ``hw_impl.h`` — every concrete driver header that the
  configure-time ``file(GLOB)`` calls find under
  ``src/hardware/core/<type>/`` and ``src/hardware/optional/<type>/``,
  plus every Python trampoline header under ``src/hardware/python/``.
* ``hw_h.h`` — a one-line wrapper that ``#include``-s both of the
  above. This is the header consumers refer to when they want "all
  hardware types and drivers."

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
reference into each driver, the registrations would silently
disappear at link time and the registry would come up empty.

The fix is to feed the driver headers to ``AUTOMOC``. AUTOMOC
generates ``moc_<class>.cpp`` for every ``Q_OBJECT`` it finds, and the
generated ``meta_object_offsets`` references pull the corresponding
object file out of the static library at link time. Listing every
driver header in ``hw_impl.h`` (which is itself part of the
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
new vendor prefix), add the prefix to both the drivers glob and
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
``virtual*`` drivers, but cannot link against the main
``blackchirp-hardware`` library because that pulls in
:cpp:class:`HardwareManager`, which in turn references symbols that
only the main application provides. ``CMakeLists.txt`` defines a
parallel static library, ``blackchirp-test-hardware``, that includes
the hardware base classes, the communication protocols, and just the
``virtual*`` drivers. Tests like ``tst_experimentloading``
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

The documentation pipeline weaves four tools together: Doxygen reads
the C++ headers under ``src/`` and emits an XML representation of every
class, struct, enum, and free function; Breathe (a Sphinx extension)
turns that XML into RST entities at Sphinx build time so the project's
hand-written RST pages can pull C++ symbols in via
``.. doxygenclass::`` and friends; Sphinx renders all the RST sources
under ``doc/source/`` into HTML (or PDF); and nbsphinx — together with
nbsphinx-link, which wires standalone notebook files into the doctree
— renders the example Jupyter notebooks under ``python/`` into pages
alongside the rest of the documentation.

For Python entities, Sphinx's built-in ``autodoc`` extension introspects
the ``blackchirp`` module on disk (``conf.py`` puts
``python/blackchirp/src`` on ``sys.path``) and renders Google-style
docstrings via the ``napoleon`` extension. The contract between source
code, the generators, and the rendered API pages is documented in
:ref:`api-reference-style`.

CMake-side wiring
~~~~~~~~~~~~~~~~~

With ``BC_BUILD_DOCUMENTATION=ON``, ``BlackchirpDocumentation.cmake``
finds ``sphinx-build`` and Doxygen and registers the ``docs``,
``doxygen``, ``docs-pdf`` (if LaTeX is found), ``docs-linkcheck``, and
``docs-clean`` custom targets. The Sphinx config (``conf.py``) and the
Doxyfile template (``Doxyfile.in``) live in ``doc/source/``.

The ``docs`` target depends transitively on ``doxygen`` when both
Sphinx and Doxygen were located, so a single
``cmake --build build --target docs`` invocation produces both the
prose pages and the API reference. ``conf.py`` also calls
``doxygen Doxyfile`` directly at import time so that Read the Docs
(which does not use the CMake build) still gets fresh XML before
Sphinx parses anything; the call is idempotent in either environment.

Build environment
~~~~~~~~~~~~~~~~~

The full set of Python dependencies is in ``doc/source/requirements.txt``:

* ``sphinx`` — the documentation generator.
* ``sphinx_rtd_theme`` — the Read the Docs HTML theme.
* ``breathe`` — Doxygen-XML adapter for Sphinx.
* ``nbsphinx`` and ``nbsphinx-link`` — notebook rendering.
* ``ipython`` — required by nbsphinx for syntax highlighting.
* ``sphinxcontrib-lightbox2`` — image lightbox in rendered HTML.

System-level dependencies: ``doxygen`` itself, plus the LaTeX toolchain
(``pdflatex``, ``latexmk``) if you want PDF output.

Activate an environment that satisfies these requirements before
invoking the docs target. The build target itself is environment-
agnostic; how a particular dev box satisfies the requirements (conda,
pip, virtualenv) is a per-checkout convention recorded in the local
``AGENTS.local.md`` file at the project root.

Running the build
~~~~~~~~~~~~~~~~~

The reliable recipe is:

.. code-block:: bash

   touch doc/source/index.rst
   cmake --build build --target docs

The ``touch`` forces Sphinx to re-evaluate the toctree so that pages
added or removed since the last build are picked up; without it,
Sphinx may skip regeneration on the assumption that the toctree has
not changed.

A stale Doxygen tree is the other common gotcha: Breathe references
the XML that the previous Doxygen run produced, so editing a header
and rebuilding only the Sphinx side leaves the API pages out of sync.
``cmake --build build --target doxygen`` followed by
``cmake --build build --target docs`` refreshes both halves; the
``docs`` target's dependency on ``doxygen`` makes a single
``--target docs`` invocation safe in most cases.

Read the Docs
~~~~~~~~~~~~~

The published documentation at https://blackchirp.readthedocs.io/ is
built from ``master`` (and PR previews from feature branches) by Read
the Docs, configured by ``.readthedocs.yaml`` at the project root. That
config pins Python 3.11 and Ubuntu 22.04, points at
``doc/source/conf.py``, and installs ``doc/source/requirements.txt``
into the build environment. Doxygen runs via the
``subprocess.call('doxygen Doxyfile')`` line in ``conf.py`` so that
Read the Docs does not need its own Doxygen install step. Changes to
the requirements file or to the ``conf.py`` Doxygen invocation flow
through to Read the Docs on the next push.

Output locations
~~~~~~~~~~~~~~~~

* ``build/docs/html/`` — Sphinx HTML output (the ``index.html`` CMake
  actually depends on lives at ``build/docs/html/index.html``).
* ``build/docs/doxygen/html/`` — Doxygen HTML browser, when the
  ``doxygen`` target has run.
* ``build/docs/doxygen/xml/`` — Doxygen XML, the input to Breathe.

Python module build
-------------------

The standalone ``blackchirp`` Python package under
``python/blackchirp/`` is built independently of CMake. It uses the
standard PEP 517 / 518 toolchain driven by ``pyproject.toml``, has its
own dependency list (numpy, scipy, pandas at runtime; pytest as a dev
extra), and is versioned and released to PyPI on its own cadence. CMake
does not touch the Python module: the C++ application's build, test,
and packaging pipelines do not depend on it being installed, and
breaking the Python module does not break the CMake build.

Layout and pyproject.toml
~~~~~~~~~~~~~~~~~~~~~~~~~

The package follows the ``src/`` layout:

::

   python/
   ├── blackchirp/
   │   ├── pyproject.toml      # PEP 621 metadata, dependencies, version
   │   ├── README.md           # PyPI listing description
   │   ├── LICENSE
   │   ├── src/blackchirp/     # importable package
   │   │   ├── __init__.py     # public API surface (re-exports)
   │   │   ├── bcfid.py
   │   │   ├── bcftmw.py
   │   │   ├── bclif.py
   │   │   ├── blackchirpexperiment.py
   │   │   └── coaverage.py
   │   └── tests/              # pytest suite
   ├── single-fid.ipynb        # example notebooks (not part of the package)
   ├── single-lif.ipynb
   ├── example-data/           # fixtures shared with the C++ test suite
   ├── environment.yml         # conda env recipe (notebook-friendly)
   └── requirements.txt        # pip equivalents

The ``src/`` layout means the package cannot be imported from
``python/blackchirp/`` itself; it must be installed (or installed in
editable mode with ``pip install -e .``) to be importable. This is
deliberate — it prevents accidental imports from a half-built source
tree and matches how downstream users get the package from PyPI.

``pyproject.toml`` declares the runtime dependencies (numpy, scipy,
pandas), the dev extra (pytest), the supported Python versions
(``>=3.9``), and the version string. The dependency list is
load-bearing; see :doc:`/developer_guide/conventions` for the
minimal-dependency policy that constrains additions.

Build, test, install
~~~~~~~~~~~~~~~~~~~~

Build the wheel and source distribution:

.. code-block:: bash

   python -m build python/blackchirp

Output lands in ``python/blackchirp/dist/`` as
``blackchirp-<version>-py3-none-any.whl`` and
``blackchirp-<version>.tar.gz``.

For development, install in editable mode with the ``dev`` extra:

.. code-block:: bash

   pip install -e "python/blackchirp[dev]"

This puts ``pytest`` on the path along with the package itself,
imported directly from the source tree so edits take effect without a
rebuild.

Run the tests:

.. code-block:: bash

   pytest --rootdir python/blackchirp python/blackchirp/tests

The test fixtures live under ``python/example-data/``. They are shared
with the C++ test suite (specifically ``tst_experimentloading``), which
keeps the on-disk schema definitions consistent across the two
languages: a fixture-format change that breaks the C++ loader will
also break the Python loader.

Versioning
~~~~~~~~~~

The Python module's version is declared in
``python/blackchirp/pyproject.toml`` under ``[project] version``. It is
**independent** of the C++ application's version (set in the top-level
``CMakeLists.txt``). The Python module ships its first release as
``0.1.0rc1`` while the C++ application is at ``2.0.0`` — the two
numbers do not track each other. Bumping the version is a one-line
edit to ``pyproject.toml`` and should be done in the same PR that
ships a user-visible Python change, with the rationale recorded in the
PR's changelog entry under ``doc/source/changelog/``.

Distribution
~~~~~~~~~~~~

The package is published to PyPI as
`blackchirp <https://pypi.org/project/blackchirp/>`_. There is **no
automation** in this repository for PyPI uploads: ``twine upload`` runs
from the maintainer's machine after a manual review of the wheel and
sdist. Publishing requires PyPI credentials that are not in the
repository or in CI, and there is no GitHub Actions workflow that
pushes to PyPI on release.

Contributors and agents must not run ``twine upload``, ``python -m
build`` followed by ``twine upload``, or any other PyPI-publication
command without explicit user consent. Building the wheel locally for
inspection is fine; pushing to a public index is not.

Packaging
---------

Binary distribution — the CPack per-platform generators, the Linux
AppImage built with ``linuxdeploy``, Qt/Qwt redistribution, GPG
signing and build-provenance attestation, and the GitHub Actions
release workflow that drives all five platforms — is the topic of
:doc:`packaging`. On the CMake side it is the ``Packaging.cmake`` and
``QtDeployment.cmake`` entries in the *CMake module map* above:
``Packaging.cmake`` owns the CPack configuration and the
``BC_BUNDLE_QWT`` option, and ``QtDeployment.cmake`` provides the
``blackchirp_deploy_qt(<target>)`` install hook that each application
module calls after its ``install(TARGETS ...)`` rule.
