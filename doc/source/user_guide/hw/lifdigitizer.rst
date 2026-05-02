LIF Digitizer
=============

* Overview_
* Settings_
* Implementations_

Overview
--------

The LIF Digitizer reads in laser-induced fluorescence data. It is
designed around two channels: one for the LIF signal and a second for
a reference detector used to normalize by laser power. Each channel
has a user-adjustable integration gate. When the reference channel is
enabled, the integral of the LIF signal is divided by the integral of
the reference signal during processing. The full time-domain LIF and
reference traces are saved to disk during acquisition so they can be
reprocessed later.

.. note::
   The LIF data channel is hardcoded to channel 1 and the reference
   channel to channel 2.

Settings
--------

The LIF digitizer shares its underlying configuration with the
:doc:`/user_guide/hw/ftmwdigitizer`, and most settings are exposed in
the :doc:`hardware dialog </user_guide/hwdialog>` with inline labels
and tooltips. Two differences are worth keeping in mind:

* Multi-record and block-averaging modes are not used for LIF, even
  if the underlying device advertises them. Each shot is transferred
  to the host individually.
* Because LIF traces are typically short and acquired at modest
  repetition rates, transfer-rate tuning is rarely necessary; pick a
  sample rate that resolves the gate region cleanly and leave the
  rest at their defaults.

Implementations
---------------

Virtual
.................

A dummy implementation used for development and for installs that
only view archived data.

Spectrum Instrumentation M4i2211x8
..............................................

The Spectrum Instrumentation M4i.2211x8 is a 2-channel high-speed
digitizer with 500 MHz of analog bandwidth.

This implementation requires the Spectrum Instrumentation ``spcm``
driver to be installed and linked at compile time. See
:doc:`/user_guide/library_status` for installation details and to
verify that the library is detected by the running Blackchirp build.

Rigol DS2302A
.......................

The `Rigol DS2302A <https://www.testequity.com/product/31591-1-DS2302A>`_
is a 2-channel, 300 MHz oscilloscope with a maximum sampling rate of
2 GSa/s. Unlike most scopes Blackchirp supports, the sample rate
cannot be set directly; the instrument chooses a rate based on the
total record duration. The implementation takes the user's requested
sample rate and record length, computes the corresponding time
window, sets the horizontal scale to the nearest "nice" coarse value
(1, 2, or 5 × 10\ :sup:`N` seconds), then queries the resulting
sample rate and updates the record length to match.

The scope also offers no way to detect individual trigger events, so
the implementation polls for screen data on a timer. The polling
interval is controlled by the ``queryInterval_ms`` setting, in
milliseconds. Set it slightly longer than the interval between
trigger events (for example, 101 ms for a 10 Hz trigger) and update
it whenever the repetition rate changes.
