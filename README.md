# Blackchirp

![Blackchirp Logo](icons/bc_logo_med.png)

Data acquisition code for CP-FTMW spectrometers.

Check out the [Project Wiki](https://github.com/kncrabtree/blackchirp/wiki) for details about installation, features, and usage.

## What's New

**June 1, 2021** - I am implementing a feature freeze for the current version of blackchirp, which is being tagged as version 0.9.

As the program has added new features over the years, it's grown cumbersome to improve and add new things without breaking existing functionality. To address this, I need to do some major refactoring of the existing codebase, but it will also set the stage for new features that I've wanted to add for a long time. While most of the work I'll be doing won't be user-facing, here are some things I hope to work in to the next major version:

- User Guide - A walkthrough of the main program features aimed at users of blackchirp
- Developer Guide - Partially annotated API mostly aimed at user/developers who wish to add new hardware to the program
- Settings Editor - A UI that more easily allows you to configure values in the config file
- Quick Experiment Improvements - Easily run experiments using settings from older ones
- Improved Python Module - A more comprehensive set of classes and utilities for working with Blackchirp data in Python

The new version of the code will be developed in a separate branch, and I will continue to fix bugs in the current version at least until the new version is released.






