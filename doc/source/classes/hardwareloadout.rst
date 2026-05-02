.. index::
   single: HardwareLoadout
   single: FtmwPreset
   single: loadouts; data model
   single: BC::Loadout

HardwareLoadout
===============

``HardwareLoadout`` is the value-typed record that represents one named
hardware map and the FTMW operating points associated with it. A
loadout binds each ``"<Type>.<label>"`` slot — for example,
``"FtmwScope.default"`` or ``"AWG.frontPanel"`` — to an implementation
key, and owns a collection of named :cpp:struct:`FtmwPreset` records
that capture full FTMW configurations against that hardware map.
Loadouts are created, stored, and switched by
:cpp:class:`LoadoutManager`; the user-facing model is described in the
:doc:`/user_guide/hardware_config/loadouts` chapter.

A loadout's ``hardwareMap`` field has the same shape as the active
selection table held by :cpp:class:`RuntimeHardwareConfig`: switching
loadouts replaces the active map with the loadout's stored map (after
the drift-detection prompt in
``RuntimeHardwareConfigDialog``). The ``ftmwPresets`` collection is
keyed by user-visible preset name; the reserved name ``__LastUsed__``
identifies the per-loadout sentinel preset described under
:cpp:class:`LoadoutManager`. The ``currentFtmwPresetName`` field points
at whichever preset most recently drove the FTMW configuration widget;
the active preset cannot be deleted without first switching away from
it.

FtmwPreset
----------

:cpp:struct:`FtmwPreset` is the named FTMW operating point owned by a
loadout. It aggregates the four pieces of state that fully describe an
FTMW measurement:

- an :cpp:struct:`RfConfigSnapshot` for the RF chain and clock
  frequencies,
- a ``ChirpConfig`` for the chirp waveform,
- an ``FtmwDigitizerConfig`` for the digitizer settings, and
- the digitizer's hardware key (``digiHwKey``), which is used to
  validate that a preset is being applied against the digitizer it was
  captured from.

Presets cannot exist outside a loadout; their lifetime is managed
entirely by :cpp:class:`LoadoutManager`. AWG sample rate is a
hardware-derived value and is reconstructed from the active AWG profile
on read rather than being stored in the preset. The user-facing model
for naming, switching, and editing presets is described in the
:doc:`/user_guide/hardware_config/ftmw_presets` chapter.

Persistence helpers
-------------------

The ``BC::Loadout`` namespace declares the free-function helpers that
``LoadoutManager`` uses to flatten loadout and preset structs into
``SettingsStorage::SettingsMap`` records and to reconstruct them on
read. Two groups are exposed:

- *RfConfigSnapshot conversions* — ``rfConfigScalarsMap``,
  ``rfConfigClocksArray``, and ``rfConfigSnapshotFromMaps`` translate
  between an :cpp:struct:`RfConfigSnapshot` and the scalar/array
  records persisted under each preset's ``rfScalars`` and ``rfClocks``
  sub-groups.
- *Hardware-map conversions* — ``hardwareMapArray`` and
  ``hardwareMapFromArray`` translate between a loadout's ``hardwareMap``
  and the array record persisted under its ``hardwareMap`` sub-group.

``copyClocksMatching`` and ``copyRfScalars`` support the per-component
copy operations exposed by the FTMW configuration dialog tabs: they
move a subset of fields between two snapshots without touching the
remaining state. Chirp- and digitizer-side conversions are declared in
``chirpconfigloadout.h`` and ``ftmwdigitizerloadout.h`` respectively;
they follow the same pattern but are omitted from this reference page
because their surface area is purely persistence plumbing.

The keys used by these helpers are declared in two namespaces:
extensions to ``BC::Store::RFC`` for the RF-side fields appear in the
header below, and the loadout-level key vocabulary lives in
``BC::Store::LM`` (documented on :doc:`loadoutmanager`).

.. highlight:: cpp

API Reference
-------------

.. doxygenstruct:: HardwareLoadout
   :members:
   :undoc-members:

.. doxygenstruct:: FtmwPreset
   :members:
   :undoc-members:

.. doxygennamespace:: BC::Loadout
   :members:
