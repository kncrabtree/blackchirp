.. index::
   single: LifConfig
   single: LIF; configuration data model
   single: LifScanOrder
   single: LifCompleteMode

LifConfig
=========

``LifConfig`` defines the scan parameters, runtime state, and FID storage for
a Laser-Induced Fluorescence (LIF) acquisition.  It inherits
:cpp:class:`ExperimentObjective` for the acquisition lifecycle interface and
``HeaderStorage`` for configuration persistence.  The class owns a
``LifDigitizerConfig`` (accessible through ``digitizerConfig()``) and a
``LifStorage`` instance that persists raw LIF traces and processing-gate
settings alongside the experiment.

A LIF acquisition sweeps a two-dimensional grid of delay times and laser
positions.  The traversal order is controlled by the ``LifScanOrder``
enumerator: ``DelayFirst`` cycles through all delay points before advancing
the laser, and ``LaserFirst`` cycles through all laser positions before
advancing the delay.  The ``d_delayRandom`` flag causes the delay axis to be
randomly permuted at the start of each sweep.  After the grid has been fully
covered once, behavior is governed by the ``LifCompleteMode`` enumerator:
``StopWhenComplete`` ends acquisition, while ``ContinueAveraging`` allows
further sweeps to accumulate.

The dialog that drives this configuration is described in
:doc:`/user_guide/lif/configuration`; the experiment-wizard page in
:doc:`/user_guide/lif/experiment_setup`; the on-disk layout in
:doc:`/user_guide/lif/data_storage`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: LifConfig
   :members:
   :protected-members:
   :undoc-members:
