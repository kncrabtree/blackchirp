.. Blackchirp documentation master file, created by
   sphinx-quickstart on Tue Jun  8 20:09:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Blackchirp Documentation
========================

Blackchirp is a cross-platform (Windows, Mac, Linux) data acquisition program for chirped-pulse Fourier transform microwave spectrometers.
It accomodates a variety of hardware combinations (digitizers, waveform generators, delay generators, flow controllers, etc) and supports multiple acquisition modes, including segmented acquisitions and double resonance experiments.
Data acquired by Blackchirp can be viewed and interacted with in real time during an ongoing acquisition, and is written to disk in a plain-text CSV-like format for easy processing with Python scripts or other analysis packages.

Getting Started
===============

* Download the source code from Github_
* Read the `installation requirements`_
* Check out the supported hardware.
* See the `user guide`_ for details on running Blackchirp

If your hardware is not supported, you can submit a request by filing an issue on Github_.

.. _Github: https://github.com/kncrabtree/blackchirp
.. _installation requirements: user_guide/installation.html#requirements
.. _user guide: user_guide/user_guide.html


.. toctree::
   :maxdepth: 3
   :hidden:

   user_guide/user_guide.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Docs
====
.. highlight:: cpp
.. doxygenclass:: SettingsStorage
