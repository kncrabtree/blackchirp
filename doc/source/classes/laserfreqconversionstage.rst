.. index::
   single: LaserFreqConversionStage
   single: Laser Frequency Conversion Stage; hardware type
   single: SirahFcu

LaserFreqConversionStage
========================

``LaserFreqConversionStage`` is the hardware-type base class for one
crystal or compensator node in the LIF frequency-conversion topology —
a doubler, a sum-frequency mixer, a difference-frequency mixer. It is a
direct :cpp:class:`HardwareObject` child, a sibling of
:cpp:class:`LifLaser` rather than a subtype of it, registered like any
other hardware type (see :doc:`/developer_guide/adding_a_hardware_type`,
which uses this class as a worked example of a type whose
device-identity settings are pinned by concrete drivers). The base
class owns only the generic contract every conversion node shares: the
registered conversion operation (:cpp:func:`conversionOp`) and harmonic
order (:cpp:func:`harmonicOrder`) — both snapshot-visible via
:cpp:class:`SettingsStorage` so a GUI or data-layer caller can read them
without touching a live threaded device — a verify flag/tolerance pair,
and the :cpp:func:`setPosition`/:cpp:func:`readPosition` dispatch slots
that move to and confirm a local input-beam wavenumber (cm⁻¹). The DAG
wiring itself (input references, the FINAL marker) is per-experiment
state owned by :cpp:struct:`LifConversionSnapshot` and
:cpp:class:`LifConfig`, not by the stage — see *Key invariants* on
:doc:`/developer_guide/lif_acquisition` for the full op/harmonic-versus-wiring
split.

A concrete driver such as ``SirahFcu`` pins :cpp:func:`conversionOp` to
a hardware-frozen constant (a doubler *is* ``NHG`` by device identity,
not a free choice) while leaving the registered ``op`` setting in
place, and overrides :cpp:func:`setHarmonicOrder` to route a harmonic
change through the device before persisting it — the canonical example
of the *gated setting* pattern described on
:doc:`/developer_guide/hardware_configuration`.
``VirtualLaserFreqConversionStage`` and ``FixedLaserFreqConversionStage``
are the uncontrolled/CI implementations.

Three free functions declared alongside the class join a
:cpp:struct:`LifConversionSnapshot`'s wiring with each stage's
snapshot-read op/harmonic into an assembled
:cpp:class:`LifConversion`, without ever touching a live threaded
device: ``lifConversionNodesFromSnapshot`` builds the joined node list,
``assembleLifConversion`` assembles it, and
``assembleCurrentLifConversion`` is the convenience wrapper that
resolves the current loadout's current ``LifPreset``,
falling back to the identity conversion when none is selected. These
are the building blocks behind the prep-time and connection-complete
assembly paths described on :doc:`/developer_guide/lif_acquisition`
(*Frequency conversion* → *Data flow by moment*).

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: LaserFreqConversionStage
   :members:
   :undoc-members:

.. doxygenfunction:: lifConversionNodesFromSnapshot

.. doxygenfunction:: assembleLifConversion

.. doxygenfunction:: assembleCurrentLifConversion
