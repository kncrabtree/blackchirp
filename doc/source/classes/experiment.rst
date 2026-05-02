.. index::
   single: Experiment
   single: experiment; data model
   single: HeaderStorage; Experiment root
   single: FtmwConfig; enabling
   single: LifConfig; enabling

Experiment
==========

``Experiment`` is the root of Blackchirp's data model for a single
acquisition. It owns an optional :cpp:class:`FtmwConfig`, an optional
:cpp:class:`LifConfig`, a set of optional hardware sub-configurations
(pulse generator, flow controller, IO board, pressure controller,
temperature sensor), an :cpp:class:`AuxDataStorage` for time-series
auxiliary readings, and an :cpp:class:`OverlayStorage` (see
:doc:`overlaybase`) for overlay annotations. All of these objects derive from :cpp:class:`HeaderStorage`
and contribute their fields to the experiment's shared CSV header file;
``Experiment`` is the root of that tree.

Two construction paths are provided. The default constructor creates
an empty experiment ready for setup via :cpp:func:`Experiment::enableFtmw`
and the other configuration accessors; this path is used by the
acquisition wizard. The three-argument constructor reads a saved experiment
from disk by number, populating every sub-configuration
from the header file, chirp file, clock file, FID data, and overlay
storage; passing ``headerOnly = true`` skips FID, auxiliary, and overlay
loading for lightweight summary access (for example, in the experiment
browser). The on-disk format is described in
:doc:`/user_guide/data_storage`.

Experiment objectives
---------------------

Active acquisition objectives — :cpp:class:`FtmwConfig` and
:cpp:class:`LifConfig` — are tracked in the ``d_objectives`` set as
non-owning pointers alongside the owning shared pointers. The
:cpp:func:`Experiment::isComplete` predicate delegates to each
objective's own completion check. Calling
:cpp:func:`Experiment::abort` propagates to all objectives.

Peak Up mode creates a *dummy* experiment: ``d_isDummy`` is set to
true, the experiment number is set to -1, and no files are written to
disk. The dummy flag is checked at every save site so that no disk I/O
is attempted.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: Experiment
   :members:
   :protected-members:
   :undoc-members:
