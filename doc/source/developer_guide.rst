Developer Guide
===============

This chapter is for contributors working on Blackchirp's source tree.
It covers the build system, the conventions that hold the C++ code,
Python code, and documentation prose together, the C++ application
architecture and threading layout, the cross-manager experiment
lifecycle, the data-flow pipelines for FTMW and LIF acquisition, the
persistence model, the standalone Python analysis module, and the
recipes that walk through adding a new driver, a new hardware type, or
a new experiment mode.

The chapter has three audiences, served by different sets of pages:

* **C++ application contributors** are the primary audience. Strong
  C++ and Qt6 skills (``QObject``, signal/slot, ``QThread``,
  ``QtConcurrent``, ``QSettings``, the metaobject system) are
  assumed; Blackchirp-specific knowledge is not. The
  :doc:`developer_guide/architecture` chapter and everything after
  it are C++-application-specific.
* **Python module contributors** working on ``python/blackchirp/``
  are served by :doc:`developer_guide/python_module` — the module's
  architecture, schema-versioning model, public API surface, and
  test layout. C++ knowledge is not required.
* **Documentation contributors** working on ``doc/source/`` are
  served by :doc:`developer_guide/conventions` (prose style and
  the API reference contract) and :doc:`developer_guide/build_system`
  (the Sphinx + Doxygen + Breathe + nbsphinx pipeline).

Topics that require coordination across multiple files or subsystems
are explained here. Topics that are confined to a single class belong
in the :doc:`API reference </classes>` (or, for Python, on the
per-class page under :doc:`/python`); topics about operating the
program belong in the :doc:`User Guide </user_guide>`.

Pages in this chapter assume the API reference is available alongside
them. Where a topic touches a class with its own API page, the
developer guide provides a brief orientation and cross-links the API
page; the API page carries the per-method contract while this chapter
carries the cross-system flow. The contract between source code, the
generators that read it (Doxygen for C++, autodoc + napoleon for
Python), and the Sphinx pages that surface them is documented in the
:ref:`api-reference-style` section of
:doc:`/developer_guide/conventions`.

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/build_system
   developer_guide/packaging
   developer_guide/conventions
   developer_guide/python_module
   developer_guide/architecture
   developer_guide/hardware_configuration
   developer_guide/hardware_runtime
   developer_guide/experiment_lifecycle
   developer_guide/ftmw_acquisition
   developer_guide/lif_acquisition
   developer_guide/persistence
   developer_guide/python_hardware
   developer_guide/vendor_libraries
   developer_guide/crash_handling
   developer_guide/adding_a_driver
   developer_guide/adding_a_hardware_type
   developer_guide/adding_an_experiment_mode
