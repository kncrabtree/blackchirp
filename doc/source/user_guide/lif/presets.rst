.. index::
   single: LIF Presets
   single: Preset Bar; LIF
   single: LIF Preset

.. _lif-presets:

LIF Presets
============

A **LIF preset** is a named, saved configuration of the
:doc:`frequency-conversion chain <conversion>` — analogous to an
:doc:`FTMW preset </user_guide/ftmw_configuration/presets>`, but
currently narrower in scope: a LIF preset captures only the conversion
wiring (which input feeds each stage, and which stage is marked as the
excitation beam), not the LIF digitizer, gate, or manual laser-control
settings, which persist separately. The conversion operation and the
harmonic order of each NHG stage are hardware identity, re-read from
each stage's own profile when the chain is assembled, and are not part
of the preset. Like FTMW presets, LIF presets are saved inside a loadout
and cannot exist outside one; see :doc:`/user_guide/hardware_config/loadouts`.

.. _lif-preset-bar:

The Preset Bar
----------------

The **LIF Preset** group box at the top of the
:doc:`Conversion tab <conversion>` provides:

**Preset selector (combo box)**
   Lists the named LIF presets belonging to the active loadout.

**Apply / Reset**
   When the combo selection differs from the currently applied preset,
   this button reads **Apply**; clicking it loads the selected
   preset's wiring into the table, prompting to discard unsaved
   changes first if there are any. When the combo selection matches
   the currently applied preset, the button reads **Reset**, and is
   enabled only while there are unsaved changes to discard.

**Save**
   Overwrites the currently applied preset with the table's current
   wiring. Enabled only when a named (non-last-used) preset is applied
   and the table has unsaved changes.

**Save As...**
   Saves the table's current wiring to a new preset with a name you
   supply. If the name already exists, Blackchirp asks whether to
   overwrite it. After saving, the new preset becomes the applied
   preset.

**Rename...**
   Renames the currently applied preset. Disabled when no named preset
   is applied.

**Delete**
   Removes the preset selected in the combo box from the loadout, after
   a confirmation prompt. Enabled only when the combo selects a named
   preset other than the one currently applied. This button appears in
   the standalone **Hardware → LIF Configuration** dialog; it is hidden
   in the Experiment Setup wizard's Conversion tab.

.. _lif-presets-current:

Restoring the Last Configuration
-----------------------------------

Blackchirp remembers your last-used conversion wiring even if you
don't save it as a named preset: the Conversion tab reopens with the
wiring you left it in, whether or not a named preset is selected in
the combo box. Each loadout also tracks its most recently applied or
saved named LIF preset, which drives the initial selection in the
preset bar.

.. _lif-presets-accept:

Saving Changes When the Wizard Is Accepted
----------------------------------------------

If the Conversion tab has unsaved changes when the Experiment Setup
wizard is accepted, Blackchirp shows a **Save LIF changes?** prompt
before the experiment starts, offering three choices:

- **Overwrite "<preset name>"** — saves the current wiring over the
  applied preset (disabled if no preset was applied to overwrite).
- **Save as new preset...** — prompts for a name and saves the wiring
  as a new preset, as with **Save As...** in the preset bar.
- **Proceed without saving** — starts the experiment with the current
  wiring without creating or updating a named preset. Blackchirp still
  remembers the wiring as the last-used configuration (see
  :ref:`lif-presets-current`).

You will see this prompt any time you tweak the conversion wiring in
the wizard and accept it, so it is worth knowing the three outcomes
before you click through it.

.. seealso::

   :doc:`conversion` — the Conversion tab and the wiring a LIF preset
   captures

   :doc:`/user_guide/hardware_config/loadouts` — loadouts own the
   presets
