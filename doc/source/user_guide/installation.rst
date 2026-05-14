.. index::
   single: installation
   single: binary packages
   single: AppImage
   single: DMG
   single: NSIS installer
   single: signing; verifying downloads
   single: attestation; verifying downloads
   single: Gatekeeper; quarantine attribute

Installation
============

Blackchirp ships pre-built packages for Windows, macOS, and Linux. The
recommended path is to download a binary release for your platform
from the project's
`Releases page <https://github.com/kncrabtree/blackchirp/releases>`_;
no compiler, CMake, or Qt installation is required.

Source builds are documented separately for contributors and for
users who need a hardware module that is not bundled into the
published binaries — see :ref:`installation-source` below.

.. _installation-binary:

Choose a package
----------------

Pick the artifact that matches your operating system from the latest
release.

Linux — Debian / Ubuntu
   ``Blackchirp-<version>-Linux.deb``. Installs on Ubuntu 24.04
   (Noble) or newer, Debian 13 (Trixie) or newer, and downstream
   derivatives.

Linux — openSUSE / Fedora / RHEL
   ``Blackchirp-<version>-Linux.rpm``. Installs on openSUSE Leap
   16.0 or newer, openSUSE Tumbleweed, Fedora 41 or newer, RHEL 9
   (incl. AlmaLinux 9, Rocky Linux 9), and similar.

Linux — any other distribution
   ``Blackchirp-x86_64.AppImage`` (main acquisition app) and
   ``Blackchirp-Viewer-x86_64.AppImage`` (standalone viewer).
   Self-contained — runs on Arch, NixOS, and any glibc 2.35-or-newer
   Linux system without installing system dependencies.

macOS
   ``Blackchirp-<version>-Darwin-arm64.dmg`` for Apple Silicon
   (M1/M2/M3/M4) and
   ``Blackchirp-<version>-Darwin-x86_64.dmg`` for pre-2022 Intel
   Macs. Requires macOS 13.3 (Ventura) or newer.

Windows
   ``Blackchirp-<version>-Windows-AMD64.exe`` (NSIS installer).
   Requires Windows 10 version 1809 or newer, or Windows 11.

.. note::
   Snap and Flatpak packages are not provided: their sandbox models
   restrict the serial-port and USB access that Blackchirp needs to
   communicate with instruments.

.. _installation-installing:

Installing
----------

Linux DEB
~~~~~~~~~

.. code-block:: console

    $ sudo apt install ./Blackchirp-<version>-Linux.deb

The package declares its Qt6 and GSL dependencies and ``apt`` pulls
them from your distribution's repositories.

Linux RPM
~~~~~~~~~

.. code-block:: console

    $ sudo zypper install ./Blackchirp-<version>-Linux.rpm   # openSUSE
    $ sudo dnf install ./Blackchirp-<version>-Linux.rpm      # Fedora / RHEL

Qt6 and GSL dependencies are resolved automatically from your
distribution's repositories.

Linux AppImage
~~~~~~~~~~~~~~

Mark the file executable and run it directly:

.. code-block:: console

    $ chmod +x Blackchirp-x86_64.AppImage
    $ ./Blackchirp-x86_64.AppImage

Each AppImage is self-contained — Qt, Qwt, GSL, and every other
runtime dependency is bundled inside.

The main ``Blackchirp-x86_64.AppImage`` also bundles
``blackchirp-viewer`` internally, so the separate
``Blackchirp-Viewer-x86_64.AppImage`` is needed only when the viewer
is the *only* application you want to keep on disk. To run the bundled
viewer from the main AppImage, see
:ref:`installation-appimage-viewer-from-main`.

macOS DMG
~~~~~~~~~

Open the ``.dmg`` that matches your hardware and drag
``blackchirp.app`` (and, optionally, ``blackchirp-viewer.app``) into
``/Applications``. Qt and Qwt frameworks are bundled inside each
application package.

The first launch may fail with the message
*"blackchirp.app is damaged and can't be opened. You should move it
to the Trash."* The bundle is not damaged. The message appears
because Blackchirp is not signed with a paid Apple Developer ID, so
macOS's Gatekeeper rejects the ``com.apple.quarantine`` extended
attribute the browser attached to the downloaded ``.dmg``. Clear the
attribute once after installing:

.. code-block:: console

    $ xattr -d com.apple.quarantine /Applications/blackchirp.app
    $ xattr -d com.apple.quarantine /Applications/blackchirp-viewer.app

After running these commands the applications launch normally.
Verifying the build-provenance attestation
(:ref:`installation-verify-attestation`) before clearing the
quarantine is the recommended way to confirm the download came from
the project's CI pipeline.

Windows
~~~~~~~

Run the downloaded ``.exe`` installer. It copies Blackchirp, the
Qt DLLs, and the platform plugins to a directory of your choosing,
adds Start-menu shortcuts, and registers an uninstall entry.

.. _installation-appimage-viewer-from-main:

Launching the viewer from the main AppImage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main AppImage bundles both ``blackchirp`` and ``blackchirp-viewer``
internally. The AppImage launcher runs the main application by default,
but the viewer can be invoked directly through one of two recipes.

**One-off launch.** Open two terminals. In the first, mount the
AppImage:

.. code-block:: console

    $ ./Blackchirp-x86_64.AppImage --appimage-mount

The AppImage prints a mount path (similar to
``/tmp/.mount_BlackcXXXXXX``) and stays in the foreground to keep the
mount alive. In the second terminal:

.. code-block:: console

    $ /tmp/.mount_BlackcXXXXXX/usr/bin/blackchirp-viewer

When the viewer exits, return to the first terminal and press
:kbd:`Ctrl+C` to unmount.

**Repeated use.** A one-time extraction makes both binaries directly
launchable:

.. code-block:: console

    $ ./Blackchirp-x86_64.AppImage --appimage-extract
    $ ./squashfs-root/usr/bin/blackchirp-viewer

.. _installation-verify:

Verifying downloads
-------------------

Verifying a download confirms the file was produced by the project's
CI pipeline rather than tampered with in transit or substituted by a
third party. Two independent mechanisms are available:

* **GPG signatures** cover the Linux artifacts (``.deb``, ``.rpm``,
  AppImage).
* **GitHub build-provenance attestations** cover every artifact on
  every platform — including macOS and Windows, which are not
  OS-level signed.

Verification is optional but recommended on first install, especially
on macOS and Windows where the attestation is the only authentication
signal available.

GPG signatures (Linux)
~~~~~~~~~~~~~~~~~~~~~~

The release key ID is ``898734DF7EDBDE45``. It is published in three
identical channels — pick whichever is convenient:

* the ``keys.openpgp.org`` keyserver,
* the ``blackchirp-release.asc`` file attached to every GitHub
  release, or
* ``packaging/blackchirp-release.asc`` in the source repository.

Import the key once per system. ``rpm`` and ``gpg`` keep separate
keyrings, so import into both if you plan to verify both formats:

.. code-block:: console

    $ gpg --keyserver keys.openpgp.org --recv-keys 898734DF7EDBDE45
    $ gpg --export --armor 898734DF7EDBDE45 | sudo rpm --import /dev/stdin

For an RPM, the signature is embedded — ``zypper install`` and
``dnf install`` verify it automatically once the key is imported, and
``rpm --checksig <file>`` verifies without installing.

For a DEB or AppImage, download the matching ``.asc`` next to the
artifact and verify the pair together:

.. code-block:: console

    $ gpg --verify Blackchirp-<version>-Linux.deb.asc \
                   Blackchirp-<version>-Linux.deb

A successful check starts with ``Good signature from``.

.. _installation-verify-attestation:

Build-provenance attestation (all platforms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every release artifact carries a SLSA build-provenance attestation
signed by Sigstore through GitHub's OIDC identity and recorded in the
public Rekor transparency log. Verifying it proves the artifact was
produced from a specific commit by a specific CI workflow run.

Install the `GitHub CLI <https://cli.github.com/>`_ and run:

.. code-block:: console

    $ gh attestation verify <artifact> --owner kncrabtree

A verified artifact prints ``Loaded digest sha256:…`` followed by the
workflow identity and ``verified successfully``.

.. _installation-source:

Building from source
--------------------

The published binaries cover the standard set of supported devices.
Building from source is only necessary when a vendor SDK module
needs to be enabled at compile time, or when you are working on
Blackchirp itself.

The full build procedure — prerequisites, CMake invocation, the
tunable ``BC_*`` options, the test target, and the documentation
target — is documented in :doc:`/developer_guide/build_system`.

.. _installation-hardware:

Next: configure hardware
------------------------

Hardware selection happens at runtime through the application's
hardware-configuration interface; no driver-specific recompilation is
required for any device in the standard set. After installing,
launch Blackchirp and follow :doc:`first_run` to complete the initial
setup, then see :doc:`hardware_config` for the full
hardware-configuration workflow.
