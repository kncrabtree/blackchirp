# Documentation Revision

Update the Sphinx/Readthedocs documentation for Blackchirp. The documentation
was last updated in the `master` branch; `devel` and `cmakemigration` have since
moved on with undocumented changes and features. One of the most recent doc
commits was `8bc115aeba017986786a1d70e1346be9cd08aaf9`.

- Documentaton root is `doc/`
- Main Sphinx configuration file: `doc/source/conf.py`
- Main landing page: `doc/source/index.rst`
- User guide: `doc/source/user_guide.rst` and `doc/source/user_guide/`
- Class reference: `doc/source/clases.rst` and `doc/source/classes/`
- Python module documentation for data analysis (not main blackchirp application):

  - Documentation files: `doc/source/python.rst` and `doc/source/python/`
  - Python module: `python/blackchirp/src/`
  - Python example: `python/single-fid.ipynb` and `python/example-data/mtbe/`

## Objectives

### Primary Objective: User Guide Revisions

1. Ensure the User Guide informs blackchirp users about how to install, configure,
   and use blackchirp.
2. Reorganize the User Guide to improve user navigation experience.
3. Improve cross-referencing and indexing of key terms in the user guide.
4. Add a changelog/release notes section which is maintained regularly.

Important features to document or update:

- Installation (from binary distributed from GitHub) - Windows, Mac, Linux (rpm/deb)
- First-run: Application configuration and data storage.
- Hardware Management: Hardware profile creation, settings, and activation/deactivation
- Python Hardware (dev-docs/python-hardware.md):
  - IPC communication overview
  - Selecting a Python Driver
  - Configuring a custom Python environment
  - Making a Python Driver: Template classes / example
  - Injected proxies
  - Hot script reloading (dev-docs/python-script-reload.md)
- Hardware Loadouts (dev-docs/loadout-system.md)
- FTMW configuration:
  - Rf configuration and clock management
  - Chirp setup and markers
  - FTMW Digitizer Configuration
- Hardware control and custmization (status panel and HwDialog)
- Communication protocols and connection testing
- LIF configuration (LIF and Ref channel concepts, setting gates, averages)
- Experiment setup:
  - FTMW Acquitision types and initialization from loadout or previous acquisition
    - Explanation of LO scan setup and operation
    - Explanation of DR scan setup and operation
  - LIF Acquisition options and initialization from previous acquisition
  - Optional hardware configuration (initialized to live settings)
  - Validation settings/auto abort
  - Warnings and errors
  - Quick Experiment: Loading settings from previous experiment.
  - Batch Acquisitions.
- FTMW Data Visualization and Storage
  - Overview of FtmwViewWidget (Live Plot, Plot 1, Plot 2, and Main Plot)
  - FT Processing toolbar functionality
  - Plot display functionality
    - Segments, Frames, Backups, and Differential Backup option
    - Main Plot display modes
    - Sideband deconvolution algorithms
  - Overlays
    - Overlay Manager and Adding Overlays
    - Overlay types:
      - BCExperimentOverlay: Creating and customizing FT
      - CatalogOverlay: SPFIT/XIAM integration, convolution options
      - GenericXYOverlay
    - Overlay Data Storage
  - FTMW data storage files and format details
- LIF Data Visualization and Storage
  - Overview of LIF Display tab: Time trace, delay plot, laser plot, 2D plot
  - Controlling data display
  - LIF Processing options
  - LIF data storage files and format details
- Rolling and Aux data: Viewing, customizing, data storage
- Log Tab: Explanation, log file location, debugging mode
- Viewing previous experiments and the blackchirp-viewer program.

### Secondary Objective: Class Documentation/API Reference

The class reference should include important classes that a developer should know
and use when implementing new Blackchirp features. Currently it contains
`SettingsStorage`, `HeaderStorage`, `HardwareObject`, `CustomInstrument`, and
`CommunicationProtocol`. Review the header comments and method documentation
to ensure its accuracy. Other candidate classes:

- Hardware module: `RuntimeHardwareConfig`, `HardwareRegistry`,
  `HardwareProfileManager`, `PythonProcess`, `PythonHardwareBase`,
  `VendorLibrary`
- Gui module: `ExperimentConfigPage`, `OverlayBaseOptionsWidget`,
  `OverlayTypeSpecificWidget`, `ZoomPanPlot`,`CurveFactory`,
   `BlackChirpPlotCurve`, `ThemeColors`, `EnumComboBox`, `ScientificSpinBox`
- Data module: All `HeaderStorage` subclasses, `Fid`, `Ft`, `OverlayBase`,
  `CurveAppearance`, `LoadoutManager`, `OverlayOperation`, `FileParser`,
  `BlackchirpCSV`, `DataStorageBase`, `FidStorageBase`, `OverlayStorage`,
  `WaveformBuffer`, `LogHandler`

### Tertiary Objective: Developer Guide

Explain overall program architecture, conventions (including source code style),
and build system. Guides for common development tasks.

- Requirements for compiling from source
- Variable naming, string usage, code formatting, etc
- Architecture: GUI, HardwareManager, AcquisitionManager
- Hardware registration and runtime configuration
- SettingsStorage concepts
- Experiment initialization and lifecycle
- Experiment types, data flow, and data storage
  - Note in particular FID performance (dev-docs/digitizer-data-flow.md)
- Adding a new hardware implementation of an existing type (C++ driver)
- Adding a new hardware type:
  - HardwareObject inheritance and virtual functions
  - Virtual instance, Python instance, python template
  - Hardware Settings Registry
  - HeaderStorage, Rolling/Aux Data, Validation keys
  - Status Box, control widget, and usage patterns
  - Config object (if needed)
- Adding a new batch experiment
- Adding a new experiment objective
