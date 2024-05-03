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

The Spectrum Instrumentation M4i.2211x8 is a 2-channel high-speed digitizer with a bandwidth of 500 MHz. Using this digitizer requires that the spcm library be installed and linked to Blackchirp at compile time.

