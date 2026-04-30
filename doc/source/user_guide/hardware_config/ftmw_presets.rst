.. index::
   single: FTMW Presets
   single: Preset Bar
   single: __LastUsed__
   single: currentFtmwPreset
   single: FtmwConfigDialog
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

.. _hardware-config-ftmw-presets-sentinel:

The ``__LastUsed__`` Sentinel
------------------------------

Each loadout maintains a hidden internal preset named ``__LastUsed__``.
This sentinel is updated automatically every time you accept the FTMW
Configuration dialog, regardless of whether you saved a named preset.
It records the exact FTMW configuration that was in effect when the dialog
was last accepted.

The ``__LastUsed__`` sentinel is never shown in preset dropdowns or in the
**Hardware → FTMW Preset** menu. You cannot select, rename, copy, or delete
it through the user interface. It exists solely so that Blackchirp can
restore the last accepted configuration when you reopen the FTMW
Configuration dialog.

.. _hardware-config-ftmw-presets-current:

The Current FTMW Preset
------------------------

Each loadout also stores a *current FTMW preset name* — a pointer to the
most recently applied, saved, or accepted named preset. This pointer drives
the initial selection in the preset bar when you open the FTMW
Configuration dialog. When no named preset has been applied, the pointer
refers to ``__LastUsed__``, meaning the dialog opens with the last accepted
configuration restored but no named preset selected.

.. _hardware-config-ftmw-presets-presetbar:

The Preset Bar
--------------

The preset bar at the top of the **FTMW Configuration** dialog
(**Hardware → FTMW Configuration**) is the primary interface for working
with presets. It contains a dropdown showing all named presets for the
active loadout and a row of action buttons.

.. figure:: /_static/user_guide/hardware_config/preset_bar.png
   :alt: FTMW Configuration dialog with the FTMW Preset bar at the top
   :align: center

   The FTMW Configuration dialog. The **FTMW Preset** bar at the top
   contains the preset selector and the action buttons described below;
   the rest of the dialog hosts the RF, Chirp, and Digitizer configuration
   tabs (covered in :doc:`/user_guide/rf_configuration`).

.. seealso::

   :doc:`/user_guide/rf_configuration`

   .. todo:: Cross-reference target ``/user_guide/rf_configuration`` is pending bundle 07.

The action buttons in the preset bar behave as follows:

**Apply / Reset**
    When the dropdown selection differs from the current preset, this
    button reads **Apply**: clicking it loads the selected preset into the
    FTMW configuration widgets and makes it the current preset. When the
    dropdown matches the current preset, this button reads **Reset**:
    clicking it discards any unsaved changes and restores the widget state
    to the saved preset. **Reset** is enabled only when the configuration
    has been modified since the last apply or save (the *dirty* state).

**Save**
    Overwrites the current named preset with the configuration shown in
    the widgets. Enabled only when the preset is a real named preset
    (not ``__LastUsed__``) and the configuration is dirty.

**Save As**
    Saves the current widget configuration to a new preset with a name
    you supply. If the name already exists, Blackchirp asks whether to
    overwrite. After saving, the new preset becomes the current preset.

**Rename**
    Renames the current named preset. Not available when the current
    preset is ``__LastUsed__``.

**Delete**
    Deletes the preset selected in the dropdown. See
    :ref:`hardware-config-ftmw-presets-delete` for the constraint on
    which presets can be deleted.

.. _hardware-config-ftmw-presets-accept:

Accepting the FTMW Configuration Dialog
-----------------------------------------

When you click **OK** (or the equivalent accept button) in the FTMW
Configuration dialog and the widget configuration has unsaved changes, a
prompt offers the following options:

**Overwrite "[preset name]"**
    Saves the modified configuration over the current named preset and
    accepts the dialog. This option is enabled only when there is a current
    named preset (not ``__LastUsed__``).

**Save as new preset...**
    Prompts you for a name, saves the configuration as a new preset, makes
    that preset current, and accepts the dialog.

**Proceed without saving**
    Accepts the dialog without saving a named preset. The configuration is
    recorded to ``__LastUsed__`` so it is restored the next time the dialog
    opens, but no named preset is created or modified.

**Cancel**
    Returns to the dialog without accepting. No changes are written.

In all non-cancel paths, ``__LastUsed__`` is updated with the accepted
configuration.

If the configuration has no unsaved changes when you click OK, the dialog
accepts immediately and ``__LastUsed__`` is updated silently.

.. _hardware-config-ftmw-presets-menu:

Hardware Menu Preset Switching
-------------------------------

Named presets for the active loadout also appear in the main menu bar:

**Hardware → FTMW Preset → [preset name]**

.. figure:: /_static/user_guide/hardware_config/ftmw_presets_menu.png
   :alt: Hardware menu with the FTMW Preset submenu open showing the named presets in the active loadout
   :align: center

   The **Hardware → FTMW Preset** submenu lists the named presets in the
   active loadout. The currently selected preset is marked. The hidden
   ``__LastUsed__`` sentinel never appears in this menu.

Selecting a preset from this submenu applies it immediately without opening
the FTMW Configuration dialog. This menu is available only when Blackchirp
is in the **Idle** state (hardware connected and not acquiring). It is
disabled when the active loadout has no named presets.

.. _hardware-config-ftmw-presets-delete:

Deleting a Preset
-----------------

To delete a preset, select it in the preset-bar dropdown and click
**Delete**. The Delete button is enabled only when the dropdown selection
is *different* from the current preset — the current preset cannot be
deleted while it is active. To delete the current preset, first apply
a different preset to make it current, then delete the one you no longer
need.

.. seealso::

   :doc:`/user_guide/hardware_config` — chapter overview

   :doc:`/user_guide/hardware_config/loadouts`

   :doc:`/user_guide/hardware_config/profiles`
