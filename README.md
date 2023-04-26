# Blackchirp

![Blackchirp Logo](src/resources/icons/bc_logo_med.png)

Data acquisition software for CP-FTMW spectrometers. Blackchirp is written to control a variety of different CP-FTMW spectrometers with versatile and configurable hardware combinations. At minimum, Blackchirp can function simply by connecting to a high-speed digitizer, but it also supports tunable local oscillators, delay generators, mass flow controllers, analog/digital IO boards, pressure controllers, temperature sensors, and it features a versatile chirp editor which can write chirps or chirp sequences to arbitrary waveform generators. FIDs and FTs are displayed for the user in real time with customizable post-processing settings, allowing a user to monitor the progress during an acquisition. All of Blackchirp's data is written in plain-text semicolon-delimited CSV format so that it can be readily imported into external analysis software.

For more information about installation and usage, please see the [User Guide](https://blackchirp.readthedocs.io/en/latest/index.html).

## v1.0 Beta Release

The beta version of Blackchirp v1.0 is now avaialble and recommended for general use! We have been using this version in my lab for about a year now without any major issues.

Major changes:
- Hardware control updates: All hardware controls have moved to windows that are accessible under the Hardware main menu. In addition, "soft" settings that are stored in the config file are editable in the same window.
- Quick experiment updates: Settings from any previous experiment may be loaded from disk and applied to a new experiment, as long as the hardware configuration remains the same.
- Data format updates: All data is now stored in a consistent semicolon-delimited plain text CSV format. FIDs are stored as integers with base-36 encoding instead of the previous binary format.
- User guide: A user guide that documents the main features and usage of Blackchirp is available online, but can also be built locally. To do so, run `make html` from the doc directory. Building the documentation requires doxygen to be installed, and your python environment must have the sphinx, sphinx_rtd_theme, and breathe packages. Other output formats are possible; see the [Sphinx documentation](https://www.sphinx-doc.org/en/master/).
- Plotting: Unified handling of curves/display items on plots so that all plot items are customizable and exportable.

Known issues and to-do items:
- Some pages in the user guide are incomplete. Contributions are welcome!
- The Quantum Composers 9528 delay generator has an error when switching between external trigger mode and internal mode.
- Python module needs implementation and development.

## What's New

**June 1, 2021** - I am implementing a feature freeze for the current version of blackchirp, which is being tagged as version 0.9.

As the program has added new features over the years, it's grown cumbersome to improve and add new things without breaking existing functionality. To address this, I need to do some major refactoring of the existing codebase, but it will also set the stage for new features that I've wanted to add for a long time. While most of the work I'll be doing won't be user-facing, here are some things I hope to work in to the next major version:

- User Guide - A walkthrough of the main program features aimed at users of blackchirp
- Developer Guide - Partially annotated API mostly aimed at user/developers who wish to add new hardware to the program
- Settings Editor - A UI that more easily allows you to configure values in the config file
- Quick Experiment Improvements - Easily run experiments using settings from older ones
- Improved Python Module - A more comprehensive set of classes and utilities for working with Blackchirp data in Python

The new version of the code will be developed in a separate branch, and I will continue to fix bugs in the current version at least until the new version is released.






