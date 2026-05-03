.. index::
   single: ChirpConfig
   single: MarkerChannel
   single: MarkerRole
   single: ChirpSegment
   single: chirp waveform; configuration
   single: AWG markers

ChirpConfig
===========

``ChirpConfig`` stores the complete AWG output waveform definition: the
ordered list of frequency-sweep and gap segments for each chirp in a
multi-chirp sequence, the inter-chirp interval, the AWG sample rate, and
the marker channel definitions. All time quantities are in microseconds;
all frequency quantities are in MHz. ``ChirpConfig`` is owned by
:cpp:class:`RfConfig` as the ``d_chirpConfig`` member and is a child
:cpp:class:`HeaderStorage` node in the experiment header tree. The
user-facing controls for building chirp waveforms are described in
:doc:`/user_guide/experiment/chirp_setup`.

The segment list is a two-dimensional structure: a vector of chirps,
each containing an ordered vector of :cpp:struct:`ChirpConfig::ChirpSegment`
records. A segment is either a linear frequency sweep (``empty = false``)
or a silent gap (``empty = true``). If all chirps share the same segment
list, :cpp:func:`ChirpConfig::allChirpsIdentical` returns true, which
allows :cpp:class:`RfConfig` to simplify sweep accounting.

Marker channels
---------------

:cpp:struct:`MarkerChannel` descriptors define the timing and role of
each AWG marker output. Each AWG reports how many physical marker outputs
it supports via the ``BC::Key::AWG::markerCount`` setting. The waveform
generation methods produce output indexed in logical channel order; each
AWG driver maps logical bit positions to hardware bit positions as
required by its data format.

The :cpp:enum:`MarkerRole` classification drives safety validation in the
experiment wizard:

- ``Protection`` — the marker pulse that prevents high-power chirp energy
  from reaching sensitive amplifiers. The wizard warns when no enabled
  protection marker fully encloses the gate and chirp windows.
- ``Gate`` — the amplifier-enable pulse. Enclosed by the protection marker.
- ``Trigger`` — a general digitizer or instrument trigger.
- ``Custom`` — any other use; no safety constraints are enforced.

All marker channels use ``ChirpRelative`` timing:
``startTime`` is the start offset in μs relative to each chirp's start
(negative = before the chirp begins) and ``endTime`` is the end offset
in μs relative to each chirp's end (positive = after the chirp ends). The
waveform lead time and tail time are computed as the maximum required
offsets across all enabled channels.

For guidance on integrating ``ChirpConfig`` into a new experiment
mode, see :doc:`/developer_guide/adding_an_experiment_mode`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: ChirpConfig
   :members:
   :protected-members:
   :undoc-members:

.. doxygenstruct:: ChirpConfig::ChirpSegment
   :members:
   :undoc-members:

.. doxygenstruct:: MarkerChannel
   :members:
   :undoc-members:

.. doxygenenum:: MarkerRole
