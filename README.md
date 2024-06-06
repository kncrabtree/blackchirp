# Blackchirp

![Blackchirp Logo](src/resources/icons/bc_logo_med.png)

Blackchirp is open-source data acquisition software for CP-FTMW spectroscopy. It is designed to control a variety of different CP-FTMW spectrometers with versatile and configurable hardware combinations. At minimum, Blackchirp can function simply by connecting to a high-speed digitizer, but it also supports tunable local oscillators, delay generators, mass flow controllers, analog/digital IO boards, pressure controllers, temperature sensors, and it features a versatile chirp editor which can write chirps or chirp sequences to arbitrary waveform generators. FIDs and FTs are displayed for the user in real time with customizable post-processing settings, allowing a user to monitor the progress during an acquisition. All of Blackchirp's data is written in plain-text semicolon-delimited CSV format, and a python module is available for importing the data and performing common processing tasks.

Join the [Discord Server](https://discord.gg/88CkbAKUZY) for news, to request help from other users, or to discuss future improvements.

## Documentation

- [Documentation Home](https://blackchirp.readthedocs.io/en/dev-1.0/index.html)
- [Installation](https://blackchirp.readthedocs.io/en/dev-1.0/user_guide/installation.html)
- [User Guide](https://blackchirp.readthedocs.io/en/dev-1.0/user_guide.html)

## Python Module

The Blackchirp python module depends only on numpy, scipy, and pandas. It can be installed with

```
pip install blackchirp
```

- [PyPI Listing](https://pypi.org/project/blackchirp/)
- [Documentation Home](https://blackchirp.readthedocs.io/en/dev-1.0/python.html)
- [Example Notebooks](https://blackchirp.readthedocs.io/en/dev-1.0/python/example.html)


## What's New

### June 5 2024

General updates:

- Blackchirp has now been tested with Qt 6.7, and currently compiles against either Qt 5.12+ or Qt 6.7. **Users are strongly recommended to switch to Qt6 as soon as possible; Qt5 support will be dropped in a future release.**
- Blackchirp now compiles with the ``c++_latest`` QMAKE flag by default, meaning that it is built with more modern C++ language support. At present, only C++20 is required, so users can add ``CONFIG -= c++latest`` and ``CONFIG += c++20`` to their ``config.pri`` file if their compiler does not support the most recent C++ standard (currently c++2b).
- Currently no new features are planned before the v1.0.0 release, which is scheduled for next week. Only bugfixes and documentation updates will be implemented between now and then.

New features:

- Added Berkeley Nucleonics BNC 577 series pulse generators.
- Plot items may now have transparency.
- Enum names are now written to csv files as strings instead of integers.
- All output files are set to UTF-8 encoding.
- LIF display widget now uses timed updates similar to the CP-FTMW display.
- When loading experiments from disk, the experiment number is no longer required when providing the full path.
- Updates to developer documentation for some core classes.

Bugfixes:

- Fixed a bug parsing the response to the sync channel query for the Quantum Composers 9214 pulse generator, which included an extra character.
- Fixed the display of the number of shots in a double sideband deconvolution.
- Fixed a bug in the LIF control widget where the rolling average algorithm lost precision once the target number of shots was reached.
- Prevented pandas from interpreting the string "None" as NaN for enum types.

### May 17 2024

General updates:

- A new [Discord Server](https://discord.gg/88CkbAKUZY) has been launched for Blackchirp users and developers.
- A brand-new python module is available and listed on the Python Package Index, along with an initial example notebook showing simple usage. Links to the documentation, examples, and the PyPI listing are above.

New features:

- The backup interval may be set in units of minutes instead of hours.
- Implementation for SRS DG645 pulse generator.
- New FID processing option added: Exponential filter. Multiplies the FID with an exponential decay with adjustable time constant.
- FID processing settings are now saved in a ``fid/processing.csv`` file with each experiment. The settings stored in the file are loaded when viewing the experiment with Blackchirp and by the new Blackchirp python module for default FT processing settings. See [the documentation](https://blackchirp.readthedocs.io/en/dev-1.0/user_guide/cp-ftmw.html) for more details.
- The sideband deconvolution algorithms for LO Scan mode have been completely rewritten, and the default averaging algorithm is now a weighted harmonic mean.
- The [User Guide](https://blackchirp.readthedocs.io/en/dev-1.0/user_guide.html) has received major updates; most pages are now written.
- The documentation now requires the following packages to build locally (see ``doc/source/requirements.txt``):
  - sphinx
  - sphinx_rtd_theme
  - breathe
  - nbsphinx
  - nbsphinx-link
  - ipython

Bugfixes:

- The FT Start/FT End processing settings for experiments acquired with older versions of Blackchirp could not be adjusted when viewing a previous experiment.
- The autoscale range for FID plots was incorrect when a window function was used.
- Updating processing settings during a live acquisition would cause the Live and Main plots to go blank until the next refresh timer tick.
- Spinbox ranges for viewing backups were not set correctly.
- The sideband deconvolution algorithm did not weight the average of overlapping segments by the number of shots if segments had different shot numbers.
- The double sideband deconvolution algorithm did not correctly use the minimum offset parameter.

### May 3 2024

- The "Start Experiment" wizard has been completely rewritten to minimize the amount of clicking required to initiate an experiment. All settings are organized into pages which are accessible in any order using the new navigation menu on the left. The dialog attempts to detect incorrect/invalid settings and issues an error or warning if any are identified, as shown in the screenshot below. 

![New Experiment Setup Dialog](doc/source/_static/user_guide/experiment/expsetup.png)

- Blackchirp now supports having multiple pieces of hardware of the same "type" for most hardware types. For example, you can now have two pulse generators, etc. This has always been the case for Clocks, but now most other types support this as well. The exceptions are the FtmwScope, AWG, and GpibController types (and the LifScope and LifLaser for the lif module). Because of this change, the hardware keys in the settings file have been changed. If you have been using a previous version of Blackchirp, you can preserve your existing settings by manually editing the config file (~/.config/CrabtreeLab/Blackchirp.conf on Unix, in the Registry on Windows). Simply add ".0" to all hardware keys (e.g., \[AWG\] becomes \[AWG.0\]) with the exception of Clock entries, where you should instead add a dot between "Clock" and the integer attached to it (e.g., \[Clock1\] becomes \[Clock.1\]). For convenience, the hardware keys are: AWG, ClockN (N=0,1,2,...), FlowController, FtmwDigitizer, GpibController, IOBoard, PressureController, PulseGenerator, and TemperatureController. Some may not be present in your config file.

- **Hardware selection at compile time has been changed.** Instead of using numbers, now each piece of hardware is identified by its model (case insensitive). See the config.pri.template file for examples.



### April 2023: v1.0 Beta Release

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







