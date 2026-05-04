.. index::
   single: FTMW Presets
   single: Preset Bar
   single: FTMW Configuration

.. _hardware-config-ftmw-presets:

FTMW Presets
============

An **FTMW preset** is a named snapshot of the complete FTMW operating
configuration for a given loadout. It captures everything Blackchirp needs
to reproduce a particular set of experimental conditions:

- **RF chain** — upconversion and downconversion frequency settings,
  clock role assignments, and scalar RF parameters.
- **Clock frequencies** — the configured frequency for each clock role
  (reference, upconversion local oscillator, etc.).
- **Chirp waveform** — the chirp segment table (start frequency, end
  frequency, duration, amplitude) for each segment in the waveform.
- **Digitizer configuration** — record length, sample rate, trigger
  settings, and channel assignments, tied to the specific digitizer profile
  active in the loadout.

The AWG hardware itself determines the available sample rates; sample rate
is a hardware property and is **not stored inside a preset**. When you apply
a preset, Blackchirp restores all captured parameters but leaves the AWG
sample rate at whatever the hardware currently reports.

A preset cannot exist outside a loadout. Creating a preset always creates
it within the active loadout.

.. _hardware-config-ftmw-presets-current:

Restoring Your Last Configuration
----------------------------------

Blackchirp remembers the FTMW configuration every time you accept the FTMW
Configuration dialog, regardless of whether you saved a named preset. When
you reopen the dialog, this remembered configuration is restored
automatically — even if you never saved it to a named preset.

Each loadout also tracks the most recently applied or saved named preset.
This drives the initial selection in the preset bar when you open the FTMW
Configuration dialog. If you have not applied a named preset, the dialog
opens with the configuration you used last — your previous settings are
restored, but no named preset is selected.

.. _hardware-config-ftmw-presets-presetbar:

The Preset Bar
--------------

The preset bar at the top of the FTMW Configuration dialog is the primary
interface for working with presets. See
:doc:`/user_guide/ftmw_configuration` for the buttons and their behavior.

.. figure:: /_static/user_guide/ftmw_configuration/ftmw_configuration.png
   :width: 800
   :target: ../../_images/ftmw_configuration.png
   :alt: FTMW Configuration dialog with the FTMW Preset bar at the top
   :align: center

   The FTMW Configuration dialog. The **FTMW Preset** bar at the top
   contains the preset selector and the action buttons described in
   :doc:`/user_guide/ftmw_configuration`.

.. _hardware-config-ftmw-presets-menu:

Hardware Menu Preset Switching
-------------------------------

Named presets for the active loadout also appear in the main menu bar:

**Hardware → FTMW Preset → [preset name]**

.. figure:: /_static/user_guide/hardware_config/ftmw_presets_menu.png
   :alt: Hardware menu with the FTMW Preset submenu open showing the named presets in the active loadout
   :align: center

   The **Hardware → FTMW Preset** submenu lists the named presets in the
   active loadout. The currently selected preset is marked.

Selecting a preset from this submenu applies it immediately without opening
the FTMW Configuration dialog. This menu is available only when Blackchirp
is in the **Idle** state (hardware connected and not acquiring). It is
disabled when the active loadout has no named presets.

.. _hardware-config-ftmw-presets-delete:

Deleting a Preset
-----------------

The current preset cannot be deleted while it is active. To delete the
current preset, first apply a different preset, then delete the one you
no longer need. Use the **Delete** button in the preset bar (see
:doc:`/user_guide/ftmw_configuration`) to remove a preset.

.. seealso::

   :doc:`/user_guide/hardware_config` — chapter overview

   :doc:`/user_guide/hardware_config/loadouts`

   :doc:`/user_guide/hardware_config/profiles`
