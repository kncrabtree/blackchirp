# Blackchirp

![Blackchirp Logo](src/resources/icons/bc_logo_large.png)

Blackchirp is open-source data acquisition software for CP-FTMW
spectroscopy. It controls a wide variety of spectrometer hardware —
high-speed digitizers, arbitrary waveform generators, tunable local
oscillators, delay generators, mass flow controllers, analog/digital
IO boards, pressure controllers, temperature sensors, and more — and
ships with a versatile chirp editor capable of writing chirps and
chirp sequences to supported AWGs. FIDs and FTs are displayed in real
time with configurable post-processing, and all acquired data is
written to disk in a plain-text semicolon-delimited CSV format that is
easy to parse from Python or any other analysis environment. A
companion `blackchirp` Python module is available for offline analysis.

Join the [Discord Server](https://discord.gg/88CkbAKUZY) for news,
help from other users, or to discuss future improvements.

## What's New in 2.0.0-alpha

The 2.0.0 line is a substantial overhaul of Blackchirp:

- **Binary distribution.** Cross-platform installer packages for
  Windows, macOS, and Linux are produced via CPack; building from
  source is no longer required for most users.
- **CMake build system.** The qmake project files have been replaced
  with a modern CMake configuration, with optional components (LIF,
  CUDA acceleration, documentation) controlled by build options.
- **Runtime hardware configuration.** Hardware loadouts and FTMW
  presets can be created, edited, and switched without rebuilding;
  per-instance settings are isolated from one another by stable
  identifiers.
- **Python hardware drivers.** Devices not supported by the built-in
  C++ drivers can be controlled by user-supplied Python scripts
  that conform to a documented interface.
- **Blackchirp-viewer.** A standalone viewer for browsing and
  comparing experiments without launching the full acquisition
  application.

See the [migration
guide](https://blackchirp.readthedocs.io/en/latest/migration.html)
for upgrade notes from Blackchirp 1.x.

## Documentation

- [Documentation Home](https://blackchirp.readthedocs.io/en/latest/index.html)
- [Installation](https://blackchirp.readthedocs.io/en/latest/user_guide/installation.html)
- [User Guide](https://blackchirp.readthedocs.io/en/latest/user_guide.html)
- [Migration Guide](https://blackchirp.readthedocs.io/en/latest/migration.html)
- [Changelog](https://blackchirp.readthedocs.io/en/latest/changelog.html)
- [Developer Guide](https://blackchirp.readthedocs.io/en/latest/developer_guide.html)
- [API Reference](https://blackchirp.readthedocs.io/en/latest/classes.html)

## Python Module

The Blackchirp Python module depends only on numpy, scipy, and pandas.
Install with:

```bash
pip install blackchirp
```

- [PyPI Listing](https://pypi.org/project/blackchirp/)
- [Documentation Home](https://blackchirp.readthedocs.io/en/latest/python.html)
- [Example Notebooks](https://blackchirp.readthedocs.io/en/latest/python/example.html)
