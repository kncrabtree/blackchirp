Developer Guide
===============

This chapter is for contributors working on the Blackchirp source tree.
It covers the build system, the conventions that hold the C++ and
Python code together, the architecture and threading layout, the
cross-manager experiment lifecycle, the data-flow pipelines for FTMW
and LIF acquisition, the persistence model, and the recipes that walk
through adding a new driver, a new hardware type, or a new experiment
mode.

The target reader has strong C++ and Qt6 skills (``QObject``,
signal/slot, ``QThread``, ``QtConcurrent``, ``QSettings``, the
metaobject system) but no prior knowledge of Blackchirp. Topics that
require coordination across multiple files or subsystems are explained
here. Topics that are confined to a single class belong in the
:doc:`API reference </classes>`; topics about operating the program
belong in the :doc:`User Guide </user_guide>`.

Pages in this chapter assume the API reference is available alongside
them. Where a topic touches a class with its own API page, the
developer guide provides a brief orientation and cross-links the API
page; the API page carries the per-method contract while this chapter
carries the cross-system flow. The
:doc:`API Reference Style <developer_guide/api_style>` page at the end
of this chapter documents the contract between the headers, the
Doxygen XML they produce, and the Sphinx pages that surface them.

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/build_system
   developer_guide/conventions
   developer_guide/architecture
   developer_guide/hardware_configuration
   developer_guide/hardware_runtime
   developer_guide/experiment_lifecycle
   developer_guide/ftmw_acquisition
   developer_guide/lif_acquisition
   developer_guide/persistence
   developer_guide/python_hardware
   developer_guide/vendor_libraries
   developer_guide/adding_a_driver
   developer_guide/adding_a_hardware_type
   developer_guide/adding_an_experiment_mode
   developer_guide/api_style
