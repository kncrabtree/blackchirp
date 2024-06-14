# Blackchirp

![Blackchirp Logo](src/resources/icons/bc_logo_med.png)

Blackchirp is open-source data acquisition software for CP-FTMW spectroscopy. It is designed to control a variety of different CP-FTMW spectrometers with versatile and configurable hardware combinations. At minimum, Blackchirp can function simply by connecting to a high-speed digitizer, but it also supports tunable local oscillators, delay generators, mass flow controllers, analog/digital IO boards, pressure controllers, temperature sensors, and it features a versatile chirp editor which can write chirps or chirp sequences to arbitrary waveform generators. FIDs and FTs are displayed for the user in real time with customizable post-processing settings, allowing a user to monitor the progress during an acquisition. All of Blackchirp's data is written in plain-text semicolon-delimited CSV format, and a python module is available for importing the data and performing common processing tasks.

Join the [Discord Server](https://discord.gg/88CkbAKUZY) for news, to request help from other users, or to discuss future improvements.

## Documentation

- [Documentation Home](https://blackchirp.readthedocs.io/en/latest/index.html)
- [Installation](https://blackchirp.readthedocs.io/en/latest/user_guide/installation.html)
- [User Guide](https://blackchirp.readthedocs.io/en/latest/user_guide.html)

## Python Module

The Blackchirp python module depends only on numpy, scipy, and pandas. It can be installed with

```
pip install blackchirp
```

- [PyPI Listing](https://pypi.org/project/blackchirp/)
- [Documentation Home](https://blackchirp.readthedocs.io/en/latest/python.html)
- [Example Notebooks](https://blackchirp.readthedocs.io/en/latest/python/example.html)


## What's New

### June 14 2024 - v1.0.0 Release

Version 1.0.0 of Blackchirp is now available and recommended for general use! New users can install immediately; if you are upgrading from an earlier version, please see the changelog.md file for a summary of what has changed and what you may need to do to migrate to the new version.
