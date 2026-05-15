.. index::
   single: Loadouts
   single: Hardware Map
   single: Default Loadout
   single: Loadout Switching
   single: Drift Detection
   single: Preview State

.. _hardware-config-loadouts:

Loadouts
========

A **loadout** is a named hardware map: a complete assignment of driver
profiles to hardware-type slots. Switching loadouts tells Blackchirp to
use a different set of physical (or virtual) instruments. A typical use
case is a lab that shares one computer between two spectrometers; each
spectrometer has its own loadout, and switching reconfigures the
application for the instrument currently in use.

.. figure:: /_static/user_guide/hardware_config-loadouts_menu.png
   :alt: Hardware menu with the Loadout submenu open showing several saved loadouts
   :align: center

   The **Hardware → Loadout** submenu. The currently active loadout is
   marked. Selecting any other loadout switches the active loadout
   immediately, prompting first if there are unsaved changes.

.. _hardware-config-loadouts-default:

The Default Loadout
-------------------

Blackchirp creates one loadout named **Default** on first run. This loadout
uses the system (virtual) profiles for any required hardware types that have
no real driver yet. You can rename, edit, or delete the Default
loadout once you have created at least one other loadout to replace it.

.. _hardware-config-loadouts-dialog:

Managing Loadouts
-----------------

All loadout management takes place in the **Hardware Configuration** dialog
(**Hardware → Hardware Selection**). The **Loadout** panel on the far
left of the dialog lists all saved loadouts and provides the following
operations via the buttons below the list:

**Activate**
    Switch the active loadout to the one selected in the list. If the
    current loadout has unsaved changes, Blackchirp prompts you before
    proceeding (see :ref:`hardware-config-loadouts-preview`).

**Save**
    Persist the preview configuration to the currently active loadout.
    If the hardware map changes would invalidate existing FTMW presets,
    Blackchirp shows the drift-detection prompt first
    (see :ref:`hardware-config-loadouts-drift`).

**Save As**
    Save the current preview configuration as a new loadout with a name
    you supply. If the source loadout has named FTMW presets and the
    FTMW-relevant hardware (AWG, FTMW Digitizer, Clocks) is unchanged,
    Blackchirp asks whether to copy those presets into the new loadout.
    Otherwise the new loadout starts empty.

**Copy**
    Duplicate an existing loadout, including its FTMW presets.

**Delete**
    Remove the selected loadout and all its FTMW presets. At least one
    loadout must exist at all times; the active loadout cannot be deleted.

Loadouts can also be switched from the main menu bar without opening
the Hardware Configuration dialog (**Hardware → Loadout → [loadout
name]**); see :ref:`hardware-menu-loadouts` for the submenu's
state-gating behavior.

.. _hardware-config-loadouts-preview:

Preview-State Semantics
------------------------

The Hardware Configuration dialog operates on a *preview* of the
configuration. Changes you make in the dialog — adding profiles, enabling
or disabling profiles, adjusting settings — are held in memory until you
either persist them to a loadout or apply them to the running application.
The **Configuration Overview** panel reflects the preview, not the live
state of Blackchirp.

Two buttons at the bottom of the dialog govern the preview:

**Apply Configuration**
    Applies the preview to the running application and closes the dialog.
    If the preview has unsaved changes relative to the active loadout,
    Blackchirp first prompts you to choose how to handle those changes
    (see below).

**Cancel**
    Discards all pending changes and closes the dialog. The live hardware
    map and the saved loadouts are unchanged.

The **Save** button under the Loadout panel writes the preview to the active
loadout *without* closing the dialog. Use it when you want to commit
changes to persistent storage and continue working in the dialog.

If you click **Apply Configuration** while the preview has unsaved changes,
Blackchirp prompts you with four choices:

**Save and apply**
    Persists the preview to the active loadout, then applies it. This is
    equivalent to clicking **Save** followed by **Apply Configuration**.

**Apply without saving**
    Applies the preview to the running application but leaves the saved
    loadout unchanged. The next time the dialog opens, the loadout's
    saved configuration is restored.

**Save to new loadout...**
    Opens the Save As dialog so you can persist the preview as a new
    loadout, then applies it. The original loadout is left intact.

**Cancel**
    Returns to the dialog without applying anything.

If you attempt to switch loadouts while the current loadout has unsaved
changes (using the **Activate** button or the Hardware → Loadout menu),
Blackchirp prompts you with a separate three-choice dialog:

- **Save and activate** — saves the current loadout first, then switches.
- **Discard and activate** — discards unsaved changes and switches immediately.
- **Cancel** — returns to the dialog without switching.

.. _hardware-config-loadouts-drift:

Drift-Detection Prompt
-----------------------

A loadout owns the
:doc:`FTMW presets </user_guide/ftmw_configuration/presets>` saved
against its hardware map. When the active AWG, FTMW Digitizer, or
Clock profiles in the preview differ from those of the last saved
hardware map, the existing presets may no longer be compatible with
the new hardware. This is called *hardware drift*.

If the loadout has any named FTMW presets when drift is detected, Blackchirp
shows a warning dialog with three choices:

.. figure:: /_static/user_guide/hardware_config-drift_prompt.png
   :alt: Hardware Configuration Changed dialog with Discard, Save As instead, and Cancel buttons
   :align: center

   The drift-detection prompt. The dialog title is **Hardware
   Configuration Changed**; its body identifies the affected loadout and
   the three available outcomes.

**Discard FTMW presets and save**
    All named FTMW presets in the loadout are deleted and the updated
    hardware map is saved. Choose this option when the existing presets
    are no longer valid and you are prepared to reconfigure the FTMW
    settings from scratch after saving.

**Save As instead**
    Blackchirp opens the Save As dialog so you can save the updated
    hardware map as a *new* loadout, leaving the original loadout and all
    its presets intact. Choose this option when you want to keep the
    existing loadout unchanged and create a new one for the revised hardware
    configuration.

**Cancel**
    No changes are made. The dialog returns to the preview state with the
    hardware-map changes still pending. Choose this option if you selected
    the wrong profiles by mistake and want to correct the configuration
    before saving.

If the loadout has no named presets when drift is detected, Blackchirp
saves without prompting and resets the remembered FTMW configuration for
that loadout.

.. seealso::

   :doc:`/user_guide/hardware_config` — chapter overview

   :doc:`/user_guide/hardware_config/profiles`
