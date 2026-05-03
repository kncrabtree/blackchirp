.. index::
   single: FtmwConfig
   single: FtmwConfigSingle
   single: FtmwConfigPeakUp
   single: FtmwConfigDuration
   single: FtmwConfigForever
   single: FtmwConfigLOScan
   single: FtmwConfigDRScan
   single: FTMW; configuration data model
   single: FtmwType
   single: WaveformBuffer; non-owning pointer

FtmwConfig
==========

``FtmwConfig`` is the abstract base class that unifies the digitizer
configuration, RF/chirp settings, and FID storage for a CP-FTMW acquisition.
It inherits both :cpp:class:`ExperimentObjective` — which gives it the
lifecycle interface (``initialize``, ``advance``, ``hwReady``, ``cleanupAndSave``)
— and ``HeaderStorage``, which serializes the configuration alongside the
experiment data.  A concrete subclass is selected at setup time based on the
``FtmwType`` enumerator; the six concrete types are defined in
``ftmwconfigtypes.h`` and documented below.

RF and chirp settings are held by :cpp:class:`RfConfig` (see
:doc:`rfconfig`), accessible through the public member ``d_rfConfig``.
Digitizer settings are encapsulated in an
``FtmwDigitizerConfig`` shared instance (accessible via ``scopeConfig()``).
The FTMW configuration dialog and the user-facing setup workflow are covered in
:doc:`/user_guide/ftmw_configuration`; digitizer setup specifics appear in
:doc:`/user_guide/experiment/digitizer_setup`.

WaveformBuffer integration
--------------------------

``FtmwConfig`` holds a *non-owning* pointer to a :cpp:class:`WaveformBuffer` object.
The buffer is created and owned by the ``FtmwScope`` hardware object, which
calls ``setWaveformBuffer()`` during acquisition setup; ``FtmwConfig`` must
not free or outlive this pointer.  ``AcquisitionManager`` retrieves the buffer
through ``waveformBuffer()`` and drains it in a worker thread, calling
``addBatchFids()`` with each batch of :cpp:class:`Fid` objects (see
:doc:`fid`).

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: FtmwConfig
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: FtmwConfigSingle
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: FtmwConfigPeakUp
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: FtmwConfigDuration
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: FtmwConfigForever
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: FtmwConfigLOScan
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: FtmwConfigDRScan
   :members:
   :protected-members:
   :undoc-members:
