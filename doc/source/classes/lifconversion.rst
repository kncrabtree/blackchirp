.. index::
   single: LifConversion
   single: LIF; frequency-conversion topology
   single: conversion DAG; assembly

LifConversion
=============

``LifConversion`` is the assembled, validated form of a LIF
frequency-conversion topology: an affine model, resolved in vacuum
wavenumber (cm⁻¹) throughout, that maps the tunable laser's fundamental
through zero or more conversion nodes to the FINAL excitation (output)
beam. It is a pure value type — no :cpp:class:`HardwareObject` or
:cpp:class:`SettingsStorage` dependency — so it can be constructed,
copied, and evaluated from any thread. :cpp:func:`LifConversion::assemble`
is the sole validating construction path; every other member reads a
successfully-assembled (or default-identity) instance. The
architecture this class sits in — where node lists come from, how they
are joined with hardware-owned op/harmonic state, and how the result
reaches the laser and each conversion stage during acquisition — is
covered in full on the *Frequency conversion* section of
:doc:`/developer_guide/lif_acquisition`.

``assemble`` takes a ``std::vector<BC::LifConv::Node>`` — one node per
conversion stage, each carrying its operation (``BC::LifConv::Op::NHG``
/ ``SFG`` / ``DFG``), a harmonic order for NHG, one or two
``BC::LifConv::InputRef`` inputs (each a reference to the tunable laser,
another node's output, or a fixed mixing beam), and an ``isFinal``
marker. An empty node list assembles to the identity conversion
(output equals fundamental, no FINAL marker required). A non-empty list
is rejected — returning ``AssemblyResult{false, errorString, {}}`` —
when an input reference does not resolve, a node's input count does
not match its operation, the graph does not have exactly one FINAL
node, the graph contains a cycle, or the FINAL beam has no net
dependence on the tunable source. Nothing outside ``assemble`` performs
this validation; callers that need a joined node list build one via
``lifConversionNodesFromSnapshot``, :cpp:func:`LifConversionSnapshot::toNodes`,
or their own caller-supplied list, documented on
:doc:`liffreqconversionstage` and :doc:`lifconversionsnapshot`.

Once assembled, ``laserToOutput``/``outputToLaser`` convert between the
fundamental and the FINAL beam (exact, since the topology is affine —
no numerics involved); ``stageInput``/``stageOutput`` return a named
node's local input and output wavenumbers for a given fundamental, used
both to drive each conversion stage's setpoint during acquisition
(:cpp:func:`HardwareManager::setLifConversionStages`) and to record the
resolved per-node coefficients when writing ``liftopology.csv``
(:cpp:func:`LifConfig::writeTopologyFile`); ``outputRange`` maps a
laser's native cm⁻¹ range through the topology, used by
``ExperimentTypePage::updateLifLaserBounds`` to compute the scan-axis
bounds shown in the wizard; and ``isIdentity`` reports the
no-conversion-stages case that several call sites treat as a sentinel
(see *Key invariants* on :doc:`/developer_guide/lif_acquisition`).

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: LifConversion
   :members:
   :undoc-members:
