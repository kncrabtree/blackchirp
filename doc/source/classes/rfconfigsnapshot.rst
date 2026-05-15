.. index::
   single: RfConfigSnapshot
   single: FTMW Presets; RF snapshot
   single: loadouts; serialization

RfConfigSnapshot
================

``RfConfigSnapshot`` is the serializable RF-state helper used by
:cpp:class:`FtmwPreset`. It captures the subset of an
:cpp:class:`RfConfig` that belongs to a stored preset — the up- and
down-conversion mixer settings, the AWG and chirp multipliers, and the
desired clock frequencies for each clock role — without touching the
runtime, hardware-derived state that an :cpp:class:`RfConfig` accumulates
while an experiment is configured. The snapshot is value-typed: it holds plain
data members and is freely copied between presets, dialog tabs, and the
persistence layer.

The snapshot deliberately does not record which physical clock
implements each role. That information lives in the owning
:cpp:class:`HardwareLoadout`'s hardware map. When a snapshot is applied
back to an :cpp:class:`RfConfig` via ``applyTo``, the receiving config is
responsible for translating the snapshot's role-keyed clock entries into
calls against whatever clock hardware is currently active.

Serialization is performed by the free functions in the ``BC::Loadout``
namespace declared alongside :cpp:struct:`HardwareLoadout`:
``rfConfigScalarsMap`` flattens the scalar fields, and
``rfConfigClocksArray`` flattens the clock table. ``LoadoutManager``
writes these maps under each preset's ``rfScalars`` and ``rfClocks``
QSettings sub-groups; see :doc:`loadoutmanager` for the storage layout
and the :doc:`/user_guide/ftmw_configuration/presets` chapter for the
user-facing model.

.. highlight:: cpp

API Reference
-------------

.. doxygenstruct:: RfConfigSnapshot
   :members:
   :undoc-members:
