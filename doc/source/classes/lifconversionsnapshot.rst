.. index::
   single: LifConversionSnapshot
   single: LIF Presets; conversion snapshot
   single: loadouts; LIF serialization

LifConversionSnapshot
======================

``LifConversionSnapshot`` is the serializable wiring-only helper used
by ``LifPreset``. It mirrors :cpp:struct:`RfConfigSnapshot`:
it captures the subset of a LIF conversion topology that belongs to a
stored preset — the per-stage input wiring and FINAL marker — without
the hardware-owned state a live topology also carries. Deliberately
excluded are each stage's conversion operation and harmonic order:
those are device-identity settings, always read from the owning
:cpp:class:`LaserFreqConversionStage`'s own :cpp:class:`SettingsStorage`
snapshot, never persisted inside a preset. See *Key invariants* on
:doc:`/developer_guide/lif_acquisition` for why that split is load-bearing.
The snapshot is value-typed and freely copied between the conversion
table model, the preset bar, and the persistence layer.

``laserKey`` records which :cpp:class:`LifLaser` the wiring was
captured against, but it is provenance only — applying a preset
substitutes the *current* active laser key rather than trusting the
stored one, so a saved preset still applies cleanly after the laser
hardware is swapped. ``fromNodes`` builds a snapshot from the
``inputs``/``isFinal`` fields of an already-joined
``BC::LifConv::Node`` list, discarding op/harmonic; ``toNodes`` is the
inverse, rejoining a snapshot's wiring with op/harmonic values supplied
through caller callbacks (in practice backed by per-stage
:cpp:class:`SettingsStorage` hardware snapshots — see
``lifConversionNodesFromSnapshot`` on :doc:`laserfreqconversionstage` for
the non-callback equivalent used outside the ``data/`` → ``hardware/``
layering boundary).

Serialization is performed by the free functions in the
``BC::Loadout`` namespace declared alongside :cpp:struct:`HardwareLoadout`:
``lifConversionScalarsMap`` flattens the snapshot's scalar
(laser-key) field, and ``lifConversionWiringArray`` flattens its
per-stage wiring table. ``LoadoutManager`` writes these maps under each
LIF preset's ``lifConversionScalars`` and ``lifConversionWiring``
QSettings sub-groups; see :doc:`loadoutmanager` for the storage layout
and :doc:`/developer_guide/lif_acquisition` (*Frequency conversion*)
for the user-facing preset model.

.. highlight:: cpp

API Reference
-------------

.. doxygenstruct:: LifConversionSnapshot
   :members:
   :undoc-members:

.. doxygenstruct:: BC::LifConv::StageWiring
   :members:
   :undoc-members:
