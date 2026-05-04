Installation
============

Blackchirp is a cross-platform data acquisition application for
CP-FTMW spectrometers. It runs on Windows, macOS, and Linux. Two
installation paths are available:

* **Binary packages** — pre-built installers and packages distributed
  through GitHub Releases. This is the recommended path for most
  users.
* **Source build** — compile Blackchirp from source using CMake. This
  path is appropriate when you need to enable a hardware driver
  that the binary distribution does not include, or when you want to
  contribute to development.

.. _installation-binary:

Binary Distribution
-------------------

Pre-built packages for all supported platforms are published on the
`Blackchirp Releases page <https://github.com/kncrabtree/blackchirp/releases>`_.

Each release provides the following artifacts:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Platform
     - Artifact
     - Notes
   * - Linux (Debian/Ubuntu)
     - ``.deb``
     - Dependencies resolved automatically by ``apt`` / ``dpkg``
   * - Linux (openSUSE/Fedora/RHEL)
     - ``.rpm``
     - Dependencies resolved automatically by ``zypper`` / ``dnf`` / ``yum``
   * - Linux (universal)
     - ``.AppImage``
     - Self-contained; runs on Arch, NixOS, and any other Linux distribution
   * - Linux (generic)
     - ``.tar.gz``
     - Binary tarball; extract and run
   * - macOS
     - ``.dmg``
     - Drag-and-drop application bundle
   * - Windows
     - ``.exe`` (NSIS installer)
     - Standard Windows installer with shortcuts and uninstall entry

.. note::
   Snap and Flatpak packages are intentionally not provided. Their
   sandbox models restrict the serial-port and USB access that
   Blackchirp requires for hardware communication.

Per-Platform Install Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux — DEB package**

Download the ``.deb`` file and install it with your package manager::

    sudo apt install ./blackchirp-<version>-Linux.deb

Qt, Qwt, and GSL are listed as dependencies and will be pulled in
automatically from your distribution's repositories.

**Linux — RPM package**

Download the ``.rpm`` file and install it::

    sudo zypper install ./blackchirp-<version>-Linux.rpm   # openSUSE
    sudo dnf install ./blackchirp-<version>-Linux.rpm      # Fedora / RHEL

Required libraries (Qt6, Qwt, GSL) are derived automatically from the
binary's shared-library references and resolved from your distribution's
repositories.

**Linux — AppImage**

Download the ``.AppImage`` file, make it executable, and run it::

    chmod +x blackchirp-<version>-x86_64.AppImage
    ./blackchirp-<version>-x86_64.AppImage

The AppImage bundles Qt, Qwt, and all other required libraries; no
system-level dependencies need to be installed.

**macOS — DMG**

Download the ``.dmg`` file, open it, and drag the ``blackchirp.app``
bundle to your ``Applications`` folder. Qt and Qwt frameworks are
bundled inside the application.

**Windows — NSIS installer**

Download the ``.exe`` installer and run it. The installer copies
Blackchirp and all required Qt DLLs and platform plugins to the chosen
directory, adds Start Menu shortcuts, and registers an uninstaller.

.. _installation-source:

Building from Source
--------------------

Prerequisites
~~~~~~~~~~~~~

Install the following before configuring the build:

* A C++ compiler with C++23 support (GCC 13+, Clang 16+, MSVC 19.35+)
* `CMake <https://cmake.org/>`_ 3.25 or later
* `Qt 6 <https://www.qt.io/download-qt-installer-oss>`_ — Core, GUI,
  Widgets, Network, SerialPort, Concurrent, Test
* `Qwt <https://qwt.sourceforge.io/>`_ 6.2 or later (Qt6 build)
* `GNU Scientific Library (GSL) <https://www.gnu.org/software/gsl/>`_
  2.1 or later
* `Eigen3 <https://eigen.tuxfamily.org/>`_ 3.3 or later (header-only)

Optional:

* NVIDIA CUDA Toolkit — required only when ``BC_ENABLE_CUDA=ON``
* `Doxygen <https://www.doxygen.nl/>`_ and a Python environment with the
  packages listed in ``doc/source/requirements.txt`` — required only
  when ``BC_BUILD_DOCUMENTATION=ON``. See
  :ref:`installation-source-docs` below for setup details.

Configuring and Building
~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository and build in the ``build/`` subdirectory. Two
configurations are supported:

**Debug** (includes debugging symbols; ``QDEBUG`` output enabled)::

    cmake . -B build/Desktop-Debug/
    cmake --build build/Desktop-Debug/ -j$(nproc)

**Release** (optimized; ``QDEBUG`` output suppressed)::

    cmake . -B build/Desktop-Release/ -DCMAKE_BUILD_TYPE=Release
    cmake --build build/Desktop-Release/ -j$(nproc)

After a successful build, the ``blackchirp`` and ``blackchirp-viewer``
executables are located inside the build directory.

To build a specific target only::

    cmake --build build/Desktop-Debug/ --target blackchirp -j$(nproc)
    cmake --build build/Desktop-Debug/ --target blackchirp-viewer -j$(nproc)

Tunable Build Options
~~~~~~~~~~~~~~~~~~~~~

On the first CMake configure, a file named ``cmake/BuildConfig.cmake``
is created from the template ``cmake/BuildConfig.cmake.template``. Edit
this file to change the options below. Re-run the configure step after
saving changes.

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Default
     - Description
   * - ``BC_BUILD_DOCUMENTATION``
     - ``OFF``
     - Build HTML documentation and C++ API reference. Requires
       Doxygen and the Python packages listed in
       ``doc/source/requirements.txt`` (see
       :ref:`installation-source-docs` below).
   * - ``BC_ENABLE_CUDA``
     - ``OFF``
     - Enable NVIDIA CUDA GPU acceleration for FID averaging and other
       compute-intensive operations. Requires the CUDA Toolkit.
   * - ``BC_BUILD_VIEWER_ONLY``
     - ``OFF``
     - Build only the ``blackchirp-viewer`` data-analysis application,
       omitting all hardware dependencies. Useful on systems without
       laboratory hardware attached.
   * - ``BC_BUILD_TESTS``
     - ``ON``
     - Build unit-test executables.

**LIF module** — Laser-Induced Fluorescence is a runtime toggle
controlled through the application's experiment-configuration interface.
There is no separate build flag for LIF; it is always compiled in.

.. _installation-source-docs:

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Generating the HTML documentation and C++ API reference requires two
external tools:

* **Doxygen** (system package) — used to extract API reference data
  from the C++ source. Install through your distribution's package
  manager (``apt install doxygen``, ``dnf install doxygen``,
  ``zypper install doxygen``, ``brew install doxygen``) or download
  from `<https://www.doxygen.nl/>`_.
* **Sphinx and supporting Python packages** — pinned in
  ``doc/source/requirements.txt`` (Sphinx, ``sphinx_rtd_theme``,
  ``breathe``, ``nbsphinx``, ``nbsphinx-link``, ``ipython``).

Install the Python packages directly into your system Python or, if
you prefer to keep them isolated, into a dedicated environment.

**Using ``pip`` and a virtual environment**::

    python -m venv .venv-docs
    source .venv-docs/bin/activate          # Windows: .venv-docs\Scripts\activate
    pip install -r doc/source/requirements.txt

**Using Conda**::

    conda create -n blackchirp-docs python pip
    conda activate blackchirp-docs
    pip install -r doc/source/requirements.txt

With Doxygen on your ``PATH`` and the Python dependencies available
(either from your active environment or system Python), configure and
build the docs::

    cmake . -B build -DBC_BUILD_DOCUMENTATION=ON
    cmake --build build --target docs       # HTML + API
    cmake --build build --target doxygen    # API only

If you used an isolated Python environment, activate it before each
``cmake`` invocation so the Sphinx executables on your ``PATH`` come
from that environment.

Output is written to ``build/docs/html/`` (HTML) and
``build/docs/doxygen/`` (API). Open ``build/docs/html/index.html`` in
a browser to view the rendered documentation.

Running Tests
~~~~~~~~~~~~~

::

    cmake . -B build/tests/
    cmake --build build/tests/ --target tests -j$(nproc)
    ctest --test-dir build/tests/

.. _installation-hardware:

Hardware Configuration
----------------------

Hardware selection in Blackchirp is performed at runtime through the
application's hardware-configuration interface — no hardware-specific
recompilation is required for the standard set of supported devices.
After installing or building Blackchirp, open the application and use
the Hardware menu to select and configure the devices connected to your
system.

For a full walkthrough of the hardware-configuration workflow, see
:doc:`hardware_config`.
