.. index::
   single: AuxDataStorage
   single: AuxDataMap
   single: TimePointData
   single: auxiliary data; time series
   single: BC::Aux; keyTemplate
   single: auxdata.csv

AuxDataStorage
==============

``AuxDataStorage`` collects auxiliary time-series data during an experiment.
It accumulates scalar readings from pressure sensors, flow controllers,
temperature monitors, and any other hardware that contributes periodic
measurements, and serializes them to ``BC::CSV::auxFile`` at each time step.
:cpp:class:`Experiment` owns an ``AuxDataStorage`` instance; unlike the
:cpp:class:`DataStorageBase` hierarchy, ``AuxDataStorage`` is not a
``DataStorageBase`` subclass and its lifecycle is managed directly by the
acquisition system.

Keys are declared before acquisition begins by calling ``registerKey`` for
each hardware object and value combination. The compound key format is
``"ObjKey.ValueKey"``, produced by the ``makeKey`` static helper using
``BC::Aux::keyTemplate``. During acquisition, hardware objects call
``addDataPoints`` to merge fresh readings into the current time point;
``startNewPoint`` is called periodically to seal the current point, append it
to ``BC::CSV::auxFile``, and advance the timestamp. The ``savedData``
accessor returns all sealed points in chronological order.

Two construction paths are provided. The default constructor creates a
detached instance with ``d_number`` set to ``-1``; no file I/O is performed
until a valid experiment number is assigned. The ``(BlackchirpCSV*, number,
path)`` constructor loads the full time series from an existing
``BC::CSV::auxFile``, making it available immediately through ``savedData``.
``AuxDataMap`` is registered with the Qt meta-object system via
``Q_DECLARE_METATYPE`` so that instances can be transported through
``QVariant`` and signal-slot connections.

For the file layout and delimiter conventions used when reading and writing
the auxiliary data file, see :cpp:class:`BlackchirpCSV`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: AuxDataStorage
   :members:
   :undoc-members:
