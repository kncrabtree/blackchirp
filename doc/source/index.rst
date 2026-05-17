.. Blackchirp documentation master file.
   The toctrees below define the sidebar navigation. Captioned
   toctrees become sidebar section headers.


.. toctree::
   :hidden:
   :caption: Getting Started

   user_guide/installation
   user_guide/first_run
   user_guide/application_config
   user_guide/ui_overview

.. toctree::
   :hidden:
   :caption: Hardware Setup

   user_guide/hardware_config
   user_guide/python_hardware
   user_guide/hardware_menu
   user_guide/hardware_details

.. toctree::
   :hidden:
   :caption: Running Experiments

   user_guide/experiment_setup
   user_guide/ftmw_configuration
   user_guide/lif/configuration

.. toctree::
   :hidden:
   :caption: Inspecting Data

   user_guide/cp-ftmw
   user_guide/lif/lif_tab
   user_guide/rolling-aux-data
   user_guide/log_tab
   user_guide/overlays
   user_guide/plot_controls

.. toctree::
   :hidden:
   :caption: Offline Analysis

   python
   user_guide/viewer

.. toctree::
   :hidden:
   :caption: Data Format and Diagnostics

   user_guide/data_storage
   user_guide/crash_reports

.. toctree::
   :hidden:
   :caption: Contributing

   developer_guide
   developer_guide/conventions
   developer_guide/build_system
   developer_guide/packaging
   developer_guide/python_module

.. toctree::
   :hidden:
   :caption: Architecture

   developer_guide/architecture
   developer_guide/experiment_lifecycle
   developer_guide/persistence
   developer_guide/crash_handling
   classes

.. toctree::
   :hidden:
   :caption: Hardware Subsystem

   developer_guide/hardware_configuration
   developer_guide/hardware_runtime
   developer_guide/python_hardware
   developer_guide/vendor_libraries

.. toctree::
   :hidden:
   :caption: Acquisition Pipelines

   developer_guide/ftmw_acquisition
   developer_guide/lif_acquisition

.. toctree::
   :hidden:
   :caption: Extending Blackchirp

   developer_guide/adding_a_driver
   developer_guide/adding_a_hardware_type
   developer_guide/adding_an_experiment_mode

.. toctree::
   :hidden:
   :caption: Version History

   migration
   changelog

Blackchirp Documentation
========================

.. important::

   Blackchirp 2.0 is in pre-release. Documentation for the current
   v1.1.0 release is at https://blackchirp.readthedocs.io/1.1.x/.
   For information about accessing the 2.0 pre-release, join the
   `Discord server`_.

Blackchirp is a cross-platform (Windows, macOS, Linux) data acquisition
program for chirped-pulse Fourier transform microwave (CP-FTMW)
spectrometers. It accommodates a wide range of hardware combinations —
digitizers, arbitrary waveform generators, delay generators, mass flow
controllers, analog/digital IO boards, pressure controllers, and
temperature sensors — and supports several acquisition modes including
segmented acquisitions and double-resonance experiments. Acquired data
can be inspected in real time during a run, and is stored on disk in a
plain-text semicolon-delimited CSV format that is easy to parse from
Python or any other analysis environment.

Where to start
==============

The user guide walks through installing Blackchirp, connecting and
configuring hardware, running experiments, and inspecting data, in
roughly the order a new user encounters when setting up a spectrometer
for the first time:

* :doc:`Getting Started <user_guide/installation>` — install Blackchirp,
  complete the first-run wizard, and learn the main UI.
* :doc:`Hardware Setup <user_guide/hardware_config>` — configure
  profiles, loadouts, and per-device drivers.
* :doc:`Running Experiments <user_guide/experiment_setup>` — set up
  FTMW, LIF, and combined acquisitions.
* :doc:`Inspecting Data <user_guide/cp-ftmw>` — view FTMW and LIF
  results during and after acquisition, with overlays and plot
  customization.
* :doc:`Data Format and Diagnostics <user_guide/data_storage>` — on-disk
  file formats and crash-report locations.
* :doc:`Blackchirp Viewer <user_guide/viewer>` — the standalone viewer
  application.

Other resources:

* :doc:`migration` — upgrade notes for users coming from Blackchirp 1.x.
* :doc:`changelog` — version history.
* :doc:`developer_guide` — architecture, build system, and contribution
  conventions.
* :doc:`classes` — C++ API reference generated from the source.
* :doc:`python` — companion Python module for offline analysis.

If your hardware is not yet supported, please open an issue on
GitHub_ or join the `Discord server`_ to discuss it with other
users and the maintainers.

.. _GitHub: https://github.com/kncrabtree/blackchirp
.. _Discord server: https://discord.gg/88CkbAKUZY


Indices and tables
==================

* :ref:`genindex`

.. * :ref:`modindex`
.. * :ref:`search`
