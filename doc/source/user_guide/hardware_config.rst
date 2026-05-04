.. index::
   single: Hardware Configuration
   single: Profiles
   single: Loadouts
   single: FTMW Presets

.. _hardware-config:

Hardware Configuration
======================

Blackchirp identifies and manages hardware through three layered concepts:
*profiles*, *loadouts*, and *FTMW presets*.

**Profiles** identify a single physical (or virtual) instrument.
Each profile binds a hardware type (for example, FTMW Scope or AWG), a
human-readable label, and a driver implementation. Profile settings — such
as the number of digitizer channels or the AWG output sample rate table —
are stored persistently and survive application restarts.

**Loadouts** group profiles into a named, complete hardware map: one or more
profiles per hardware type, covering everything Blackchirp needs to run an
experiment. Loadouts are convenient for switching between, e.g., an 8-18
GHz instrument and a 26-40 GHz instrument. Profiles exist independently of loadouts;
a profile may appear in any number of loadouts.

**FTMW presets** are named operating points that live inside a loadout.
A preset captures the RF chain configuration, clock frequencies, chirp
waveform, and digitizer settings that together define one experimental
operating point. Switching presets within a loadout swaps the FTMW
configuration without touching the hardware map. For example, you might
create one preset for that co-averages 10 full-bandwidth FIDs, and a second
that retains all 10 FIDs separately so you can see how the FID quality
varies with time relative to the gas pulse. Or you might set several
presets which vary the chirp duration based on estimated dipole moments of
different target species for easy switching without needing to edit the
chirp settings directly.

The three concepts compose as follows: a running experiment uses one
*active loadout*, and within that loadout one *current FTMW preset* drives
the FTMW configuration widgets.

.. toctree::
   :hidden:

   hardware_config/profiles
   hardware_config/loadouts
   hardware_config/ftmw_presets

.. rubric:: In this chapter

:doc:`hardware_config/profiles`
   How profiles are created, labeled, and managed. Covers the
   Type / Label / Implementation triple, the settings priority sections
   (Required, Important, Optional / Advanced), and enabling or disabling
   profiles within a loadout.

:doc:`hardware_config/loadouts`
   What a loadout is, how to create and switch between loadouts, and
   the preview-state semantics that protect you from accidentally
   discarding work. Covers the drift-detection prompt that appears when
   hardware changes would invalidate existing FTMW presets.

:doc:`hardware_config/ftmw_presets`
   What an FTMW preset captures and how to create, apply, save, rename,
   and delete presets from the preset bar in the FTMW Configuration
   dialog and from the Hardware menu.

.. rubric:: Opening the Hardware Configuration dialog

All profile and loadout management takes place in the **Hardware
Configuration** dialog. Open it from the menu bar:

**Hardware → Hardware Selection**

The same dialog appears during the :ref:`first-run-hardware-onboarding`
sequence.

.. seealso::

   :doc:`/user_guide/hardware_config/profiles`,
   :doc:`/user_guide/hardware_config/loadouts`,
   :doc:`/user_guide/hardware_config/ftmw_presets`

   :doc:`/user_guide/hardware_menu`

   .. todo:: Cross-reference target ``/user_guide/hardware_menu`` is pending bundle 04.
