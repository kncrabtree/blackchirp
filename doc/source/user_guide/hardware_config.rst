.. index::
   single: Hardware Configuration
   single: Library Status
   single: Profiles
   single: Loadouts
   single: FTMW Presets

.. _hardware-config:

Hardware and Library Configuration
==================================

The Hardware Configuration dialog (**Hardware → Hardware Selection**)
is where you map Blackchirp to the physical or virtual instruments on
this system and where you check that any vendor-supplied driver
libraries are installed correctly. It is the same dialog that opens
during :ref:`first-run-hardware-onboarding` on a fresh system.

The dialog has two tabs:

**Hardware Configuration**
   Manage hardware profiles, loadouts, and FTMW presets.

**Library Status**
   View the load state of optional vendor libraries (LabJack,
   Spectrum) and configure custom search paths.

.. toctree::
   :hidden:

   hardware_config/profiles
   hardware_config/loadouts
   hardware_config/ftmw_presets
   hardware_config/library_status

Profiles, loadouts, and presets
-------------------------------

Blackchirp identifies and manages hardware through three layered
concepts:

**Profiles**
   A single physical or virtual instrument. Each profile binds a
   hardware type (for example, FTMW Digitizer or AWG) to a
   human-readable label and a driver. Per-profile settings — number
   of digitizer channels, AWG sample-rate tables, and so on —
   persist across application restarts.

**Loadouts**
   A named collection of profiles that together describe a complete
   instrument: one or more profiles per hardware type, covering
   everything Blackchirp needs to run an experiment. Loadouts make
   it convenient to switch between, say, an 8–18 GHz instrument and
   a 26–40 GHz instrument. A profile may appear in any number of
   loadouts.

**FTMW presets**
   Named operating points inside a loadout. A preset captures the RF
   chain configuration, clock frequencies, chirp waveform, and
   digitizer settings that define one experimental operating point.
   Switching presets within a loadout changes the FTMW configuration
   without touching the hardware map — useful for, e.g., keeping a
   "10-shot co-average" preset alongside a "10-shot retain-each-FID"
   preset, or for stepping through chirp durations tuned to different
   target species.

A running experiment uses one *active loadout* and, within it, one
*current FTMW preset* that drives the FTMW configuration widgets.

.. rubric:: In this chapter

:doc:`hardware_config/profiles`
   How profiles are created, labeled, and managed. Covers the
   Type / Label / Driver triple, the settings priority sections
   (Required, Important, Optional / Advanced), and enabling or
   disabling profiles within a loadout.

:doc:`hardware_config/loadouts`
   What a loadout is, how to create and switch between loadouts, and
   the preview-state semantics that protect you from accidentally
   discarding work. Covers the drift-detection prompt that appears
   when hardware changes would invalidate existing FTMW presets.

:doc:`hardware_config/ftmw_presets`
   What an FTMW preset captures and how to create, apply, save,
   rename, and delete presets from the preset bar in the FTMW
   Configuration dialog and from the Hardware menu.

:doc:`hardware_config/library_status`
   The Library Status tab: vendor-library load state, search-path
   configuration, and per-library installation guidance.

.. seealso::

   :doc:`/user_guide/hardware_menu`
