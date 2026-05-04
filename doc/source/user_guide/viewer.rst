.. index::
   single: blackchirp-viewer
   single: Viewer; standalone application
   single: Experiments; viewing saved
   single: Data; offline analysis

.. _viewer:

Blackchirp Viewer
=================

``blackchirp-viewer`` is a standalone application for opening and inspecting
saved experiments. It ships in the same binary package as ``blackchirp``:
installing from a release package places both executables side by side.
Building from source produces both via the ``blackchirp`` and
``blackchirp-viewer`` CMake targets.

.. figure:: /_static/user_guide/viewer/main_window.png
   :width: 800
   :target: ../_images/main_window.png
   :align: center
   :alt: Blackchirp Viewer launcher window with a saved experiment opened in a separate experiment view window

   The Blackchirp Viewer launcher (right) lists previously opened experiments;
   loading one opens it in its own experiment view window (left), with the FID
   and FT plots, processing controls, overlays, and peak-find tools.


Purpose
-------

The viewer is designed for situations where you need to inspect data without
access to the configured hardware:

- **Reviewing data on a laptop** or any machine that does not have the
  spectrometer hardware attached.
- **Working alongside a live acquisition**: ``blackchirp`` holds hardware
  resources while running; ``blackchirp-viewer`` can open saved experiments
  at the same time without interfering.
- **Sharing data** with collaborators who do not have Blackchirp's hardware
  configured on their machines.


Launching the Viewer
--------------------

Run ``blackchirp-viewer`` from the command line, the application launcher, or
the Start menu, depending on your platform. No special arguments are needed.
The viewer opens to a small launcher window from which you can browse to and
open a saved experiment; each opened experiment appears in its own experiment
view window, so several experiments can be compared side by side.


Relationship to the Main Program
---------------------------------

The main ``blackchirp`` program already provides a **View Experiment** workflow
that loads a saved experiment in a viewer window. The view presented by that
in-application window and the view presented by the standalone
``blackchirp-viewer`` are the same: the same FT and FID plots, the same overlay
tools, the same peak finder, and the same per-experiment summary panel. The
standalone viewer is simply that view running as an independent process, which
means it does not require the main program to be open and does not compete for
hardware resources.


What You Can Do in the Viewer
------------------------------

- View the FT and FID plots for a saved experiment.
- Apply processing options to recompute the Fourier transform from the saved
  FIDs (window functions, zero-padding, etc.) and save the updated processing
  settings back to the experiment so they are applied automatically the next
  time the experiment is opened.
- Browse aux data recorded during the experiment.
- Open spectral overlays (catalog lines, custom data files) and compare them
  against the measured spectrum, and save the configured overlays to the
  experiment for later reuse.
- Export plotted data in plain XY (text) format for use in external analysis
  tools.
- Run peak finding and export the resulting peak list.


Limitations
-----------

The viewer is intentionally hardware-free and does not modify acquired data:

- **Acquired data is read-only**: the FIDs and aux-data records captured at
  acquisition time cannot be altered or re-acquired from the viewer.
  (Processing settings and overlay configurations associated with an
  experiment can be saved and updated; only the underlying measurements are
  fixed.)
- **No hardware**: the viewer does not communicate with any instrument. There
  is no Hardware menu, no communication setup, and no status panel.
- **No acquisition-time tabs**: the Log tab and Rolling/Aux Data tab are absent
  because they reflect live session activity. Only the saved experiment view
  is available.
- **No experiment wizard**: new experiments cannot be started from the viewer.
