.. Blackchirp documentation master file.
   The toctree below is the canonical entry point for the rendered docs.


.. toctree::
   :hidden:

   user_guide
   migration
   changelog
   developer_guide
   classes
   python

Blackchirp Documentation
========================

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

* :doc:`user_guide` — install Blackchirp, configure hardware, run
  experiments, and view data.
* :doc:`migration` — upgrade notes for users coming from Blackchirp
  1.x.
* :doc:`changelog` — version history.
* :doc:`developer_guide` — architecture, build system, and
  contribution conventions.
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
