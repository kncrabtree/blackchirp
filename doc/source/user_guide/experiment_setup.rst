Experiment Setup
================

.. toctree::
   :hidden:

   experiment/acquisition_types
   experiment/optional_hardware
   experiment/validation
   experiment/quick_experiment
   experiment/sequence_mode
   lif/experiment_setup

An ``Experiment`` is the basic acquisition event in Blackchirp; it may
consist of CP-FTMW measurements and/or a laser scan (see the
:doc:`LIF experiment setup <lif/experiment_setup>` page for laser-scan
acquisitions). FID records from the FTMW Digitizer are averaged in the
time domain; the average FID and its Fourier transform are viewed on
the :doc:`CP-FTMW tab <cp-ftmw>`. :doc:`Auxiliary Data <rolling-aux-data>`
is also recorded throughout the run.

Depending on the :doc:`acquisition type <experiment/acquisition_types>`,
an experiment ends when one of the following conditions is met:

1. The objective of the experiment is reached (e.g., the requested
   number of shots has been collected).
2. The user clicks the ``Abort`` button.
3. One of the :doc:`validation conditions <experiment/validation>`
   falls outside the designated range.
4. A communication failure occurs with a critical piece of hardware.

To start an experiment, select an option from the ``Acquire`` menu in
the :ref:`user_guide/ui_overview:Main Toolbar`. ``Start Experiment``
opens a wizard that walks through the FTMW acquisition type, the
:doc:`FTMW Configuration <ftmw_configuration>` (Rf Configuration,
:doc:`Chirp <ftmw_configuration/chirp_setup>`, and
:doc:`Digitizer <ftmw_configuration/digitizer_setup>`), settings for
any present :doc:`optional hardware <experiment/optional_hardware>`,
and the desired :doc:`validation conditions <experiment/validation>`.
A past experiment can be repeated through the
:doc:`Quick Experiment <experiment/quick_experiment>` action, which
pre-populates the wizard from a saved experiment.

After the wizard closes, Blackchirp initializes each device in turn;
errors during initialization cancel the experiment and surface on the
``Log`` tab. Once initialization succeeds, a new data folder is
created and acquisition begins.
