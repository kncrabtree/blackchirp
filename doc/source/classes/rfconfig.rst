.. index::
   single: RfConfig
   single: ClockFreq
   single: ClockType
   single: RF chain; configuration
   single: LO scan
   single: DR scan
   single: upconversion
   single: downconversion

RfConfig
========

``RfConfig`` bridges :cpp:class:`FtmwConfig` and :cpp:class:`ChirpConfig`
by holding the full RF frequency chain for an FTMW acquisition: the
upconversion path from AWG output to the sample, the downconversion path
from the received signal to the digitizer, all clock assignments, sweep
parameters, and the embedded :cpp:class:`ChirpConfig`. ``RfConfig`` is
stored as ``FtmwConfig::d_rfConfig`` and is a child
:cpp:class:`HeaderStorage` node in the experiment header tree. The
user-facing RF configuration controls are described in
:doc:`/user_guide/ftmw_configuration/rf_configuration`.

Clock semantics
---------------

Clocks are assigned by logical role (:cpp:enum:`RfConfig::ClockType`):
``UpLO`` and ``DownLO`` for the up- and downconversion mixers, ``AwgRef``
and ``DigRef`` for instrument reference inputs, ``ComRef`` for a shared
10 MHz (or other) distribution, and ``DRClock`` for double-resonance pump
sources. When ``d_commonUpDownLO`` is true a single hardware source serves
both ``UpLO`` and ``DownLO``; setting either role via
:cpp:func:`RfConfig::setClockFreqInfo` automatically mirrors the
assignment to the other.

Each :cpp:struct:`RfConfig::ClockFreq` record holds a desired output
frequency, a multiplication or division factor, and the hardware key of
the clock source. :cpp:func:`RfConfig::clockFrequency` returns the
desired output frequency; :cpp:func:`RfConfig::rawClockFrequency` inverts
the factor to recover the hardware oscillator frequency that the driver
programs.

For non-scan acquisitions the clock configuration list is empty until
:cpp:func:`RfConfig::prepareForAcquisition` is called, at which point the
template is copied to form a single-step list. For LO and DR scans,
:cpp:func:`RfConfig::addLoScanClockStep` and
:cpp:func:`RfConfig::addDrScanClockStep` append one entry per frequency
point before acquisition begins. Calling
:cpp:func:`RfConfig::advanceClockStep` increments the active step index
and, when the step list wraps to zero, increments
``d_completedSweeps``.

Frequency chain calculations
-----------------------------

The chirp frequency reaching the sample for a given AWG output frequency
is:

.. code-block:: text

   chirpFreq = (awgFreq × awgMult ± upLO) × chirpMult

where the sign is positive for the upper sideband and negative for the
lower sideband. :cpp:func:`RfConfig::calculateChirpFreq` and
:cpp:func:`RfConfig::calculateAwgFreq` perform these conversions in both
directions. :cpp:func:`RfConfig::calculateChirpAbsOffset` returns the
absolute difference between the chirp frequency and the down-LO, which is
the nominal IF center frequency at the digitizer input. The full FTMW
digitizer configuration is documented on :doc:`ftmwconfig`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: RfConfig
   :members:
   :protected-members:
   :undoc-members:
