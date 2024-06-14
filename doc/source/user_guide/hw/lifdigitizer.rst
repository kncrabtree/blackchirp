LIF Digitizer
=============

* Overview_
* Settings_
* Implementations_

Overview
--------

The LIF Digitizer reads in LIF data. It is designed to support 2 channels: one for the LIF signal, and a second from a reference detector for normalizing by laser power. Each channel has a user-adjustable gate that will be used to integrate the digitized signal. If the reference channel is enabled, the integral of the LIF signal will be divided by the integral of the reference signal when processed. The full time-domain LIF and Reference signals are both saved to disk during an acquisition so they can be reprocessed later.

.. note::
   At present, the LIF data channel is hardcoded to channel 1 and the reference channel to 2. This can be made more versatile in a future version if needed.

Settings
--------

Same as FTMW Digitizer, but multiple record and block averaging modes are not supported.


Implementations
---------------

Virtual (virtual)
.................

A dummy implementation.

Spectrum Instrumentation M4i2211x8 (m4i2211x8)
..............................................

The Spectrum Instrumentation M4i.2211x8 is a 2-channel high-speed digitizer
with a bandwidth of 500 MHz. Using this digitizer requires that the spcm
library be installed and linked to Blackchirp at compile time.

Rigol DS2302A (ds2302a)
.......................

The `Rigol DS2302A <https://www.testequity.com/product/31591-1-DS2302A>`_ is a
2-channel, 300 MHz oscilloscope with a maximum sampling rate of 2 GSa/s. Unlike
many other oscilloscopes, the sample rate cannot be set manually for this
scope; the scope chooses a sampling rate automatically based on the total
record duration. This implementation takes the user's sampling rate and record
length, calculates the time requested, and sets the horizontal scale of the
scope to the nearest "nice" value obtainable by coarse tunings, which are 1, 2,
and 5eN seconds, where N is an integer. Then the implementation queries the
sampling rate at that value, and updates the record length accordingly.

Another feature of this scope is that there seems to be no way to detect
individual trigger events. The current implementation uses a timer to transfer
the data shown on the screen to the computer. The timer interval is set by the
`queryInterval_ms` setting, which as units of ms. It is recommended to set this
interval to a value just slightly greater than the interval between trigger
events (e.g., a value of 101 ms for a trigger rate of 10 Hz). This setting
should be changed anytime the repetition rate is changed.
