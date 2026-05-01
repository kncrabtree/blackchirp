.. index::
   single: LIF Module
   single: Laser-Induced Fluorescence
   single: REMPI
   single: Laser Scan

LIF Module
==========

.. toctree::
   :hidden:

   lif/experiment_setup
   lif/configuration
   lif/lif_tab
   lif/data_storage

The LIF module adds time-gated laser-scan acquisition to Blackchirp,
enabling laser-induced fluorescence (LIF), REMPI, and related
laser-frequency or delay-time scanning experiments. During a LIF
acquisition, Blackchirp steps a laser through a sequence of wavelength
and/or delay positions, acquires a digitizer waveform at each point, and
integrates the signal over a user-defined time gate. A second reference
channel can be enabled for shot-by-shot laser-power normalization.

The LIF module is built into every Blackchirp binary. Enabling it is an
application-wide toggle in the Application Configuration dialog (see
:ref:`application-config`). When LIF is enabled, a dedicated tab appears
in the main window and a LIF configuration page is added to the experiment
wizard.

LIF acquisition can run alongside CP-FTMW or as a standalone experiment;
the two subsystems share the same experiment number and folder on disk.

.. rubric:: In this chapter

- :doc:`lif/experiment_setup` — configuring a LIF scan in the experiment wizard
- :doc:`lif/configuration` — channel and gate setup in the LIF tab
- :doc:`lif/lif_tab` — the LIF Display tab during and after acquisition
- :doc:`lif/data_storage` — on-disk file layout for LIF experiments
