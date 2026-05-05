.. index::
   single: FTMW Configuration
   single: FTMW Preset
   single: Preset Bar

.. _ftmw-configuration:

FTMW Configuration
==================

The **FTMW Configuration** dialog is the primary interface for setting up
the RF chain, chirp waveform, and digitizer parameters for a CP-FTMW
experiment. Open it from the menu bar:

    **Hardware → FTMW Configuration**

The dialog contains a :ref:`preset bar <ftmw-preset-bar>` at
the top and three tabs below it: **RF Config**, **Chirp Config**, and
**Digitizer Config**. The RF Config tab is described in
:doc:`rf_configuration`; the Chirp Config tab is described in
:doc:`ftmw_configuration/chirp_setup`; the Digitizer Config tab is described in
:doc:`ftmw_configuration/digitizer_setup`.

The FTMW Configuration dialog is available only while Blackchirp is in
the **Idle** state (hardware connected, no experiment running). The dialog
is opened in read-only mode when Blackchirp is disconnected.

.. figure:: /_static/user_guide/ftmw_configuration/ftmw_configuration.png
   :width: 800
   :alt: FTMW Configuration dialog showing the preset bar at the top and
         the RF, Chirp, and Digitizer tabs below it.

   The FTMW Configuration dialog. The **FTMW Preset** bar at the top
   provides quick access to named operating points. The three tabs give
   access to the RF chain, chirp waveform, and digitizer settings.

.. _rf-configuration-preset-bar:
.. _ftmw-preset-bar:


An **FTMW preset** is a named snapshot of the complete FTMW operating
configuration — RF chain parameters, clock frequencies, chirp waveform,
and digitizer settings — saved within the active loadout. Switching
presets restores all captured parameters at once. See
:doc:`hardware_config/ftmw_presets` for a full description of preset
semantics and the preset lifecycle. Presets can be created, switched, and
deleted using the controls at the top of the FTMW Configuration dialog. The same controls are also available when initializing an experiment.

**Preset selector (combo box)**
    Lists all named presets belonging to the active loadout. If you have
    not applied a named preset, the dialog opens with the configuration you
    used last — your previous settings are restored, but no named preset is
    selected.

**Apply / Reset**
    When the combo selection differs from the current preset, this button
    reads **Apply**. Clicking it loads the selected preset into the RF,
    Chirp, and Digitizer widgets and makes it the current preset. When
    the combo matches the current preset, this button reads **Reset**.
    Clicking it discards any unsaved edits and restores the widgets to
    the saved preset state. **Reset** is enabled only when the
    configuration has been modified since the last apply or save (the
    *dirty* state).

**Save**
    Overwrites the current named preset with the widget contents. Enabled
    only when a named preset is currently selected and the configuration
    has unsaved changes.

**Save As**
    Saves the widget contents to a new preset with a name you supply. If
    the name already exists, Blackchirp asks whether to overwrite. After
    saving, the new preset becomes the current preset.

**Rename**
    Renames the current named preset. Available only when a named preset
    is currently selected.

**Delete**
    Deletes the preset selected in the combo. The Delete button is enabled
    only when the selected preset is different from the current preset.
    To delete the current preset, first apply a different preset, then
    delete the one you no longer need.

Accepting the FTMW Configuration dialog with unsaved changes opens a
prompt offering to overwrite the current named preset, save as a new
preset, proceed without saving, or cancel. In every non-cancel path,
your accepted configuration is remembered so the dialog reopens with it.

.. seealso::

   :ref:`hardware-config-ftmw-presets` — full preset lifecycle reference.

   :doc:`hardware_config/ftmw_presets` — preset creation, switching, and
   deletion from the Hardware Configuration dialog and the Hardware menu.

The three tabs of the FTMW Configuration dialog are documented on the
following pages.

.. toctree::
   :maxdepth: 2

   rf_configuration
   ftmw_configuration/chirp_setup
   ftmw_configuration/digitizer_setup
