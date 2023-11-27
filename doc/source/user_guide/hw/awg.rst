AWG
===

* Overview_
* Settings_
* Implementations_

Overview
--------

An AWG represents the source that generates chirps in the FTMW spectrometer, whether or not the device is truly an Arbirary Waveform Generator. If enabled, Blackchirp can be used to `create and configure chirps <experiment/chirp_setup.html>`_ that are associated with an experiment. It is optional; however, even if you do not intend to use Blackchirp to create the chirps, it is recommended that you use the virtual implementation, which will allow you to record the details of the chirp you are using with an experiment.

The basic concept for an AWG is modeled after the Tektronix AWG7122B, and uses one analog output channel along with two (optional) digital markers: one of which is used to control a protection switch, and the other of which is used to control the gate of a TWT amplifier. For AWGs that do not have markers, Blackchirp can use a `pulse generator <pulsegenerator.html>`_ to generate analogous signals. Some AWGs have multiple outputs, however Blackchirp only supports a single analog output at present.

Settings
--------

* ``hasAmpEnablePulse`` (true/false): If true, a digital output is used to control a gate signal delivered to the high-power amplifier. If false, a Pulse Generator channel with the "Amp" role may be used to generate a similar pulse.
* ``hasProtectionPulse`` (true/false): If true, a digital output is used to control a gate signal delivered to a switch protecting the low-noise amplifier. If false, a Pulse Generator channel with the "Prot" role may be used to generate a similar pulse.
* ``maxFreqMHz`` (float): Maximum frequency, in MHz. By default this will be set to the maximum frequency supported by the device; however setting this to a smaller number is possible if only a limited range of the device is useful.
* ``minFreqMHz`` (float): Minimum frequency, in MHz. By default this will be set to the minimum frequency supported by the device; however setting this to a larger number is possible if only a limited range of the device is useful.
* ``maxSamples`` (int): Maximum number of samples in an AWG record. This is limited by the memory of the device, but may be set to a smaller number if desired.
* ``rampOnly`` (true/false): Set to true for devices that are ramp generators (e.g., AD9914) rather than AWGs. For these devices, only one chirp per trigger is used, and the chirp may consist of only a single segment which may not be empty.
* ``sampleRateHz`` (int): Sample rate of the AWG, in Hz. Currently Blackchirp does not support controlling the sample rate of the AWG. By default this value is set to the maximum sample rate of the device; however, if this is incorrect (or if you manually change the operating sample rate), the value can be changed to reflect that value.
* ``triggered`` (true/false): If true, the AWG is triggered externally. If set to false, the chirp will be played repeatedly on supported AWGs. *Note: This is not yet implemented for all AWGs!*


.. warning::

  Setting the sample rate and maxSamples to incorrect values may lead to incorrect chirps or potentially device errors.

Implementations
---------------

Virtual (0)
...........

The virtual implementation can be thought of as a "read-only" device useful for recording the chirp settings in use. This implementation is recommended if you use another program to produce your chirps. In addition, setting the ``hasAmpEnablePulse`` and/or ``hasProtectionPulse`` options to false will allow Blackchirp to configure appropriate pulse generator channels automatically if desired.

Tektronix AWG70002A (1)
.......................

The `AWG70002A <https://www.tek.com/en/signal-generator/awg70000-arbitrary-waveform-generator-manual>`_ is a 16 GSa/s AWG with a maximum output frequency of 6.25 GHz, and this implementation communicates over a TCP socket. The default TekVisa software running on the AWG communicates on port 4000. The implementation is currently hardcoded to be externally triggered with a rising edge on trigger A with the trigger mode set to synchronous to ensure phase coherence (assuming the trigger source is locked to the same external reference as the AWG). The chirp output is on channel 1, and both markers are always used (marker A = Protection, marker B = Amplifier Gate).

When an experiment begins, the chirp data and markers are written to the device over and Blackchirp maintains a list of waveforms that have been sent to the AWG. As long as the AWG hasn't been reset, if you reuse a chirp, it is likely that Blackchirp will find it in the list and reload it rather than re-transferring the data. If this behavior causes problems, you can always restart the AWG or delete the offending records to force Blackchirp to reload the chirp. If successful, the outputs are enabled and the device is set to run mode, which will then play waveforms upon receiving a trigger. This will generate an audible click due to the relays in the AWG. At the end of the experiment, the outputs are disabled (producing another audible click) and the device placed into standby.

**Known Issues**

 * A bug in earlier versions of the TekVISA firmware would cause an error when transmitting chirp data to the AWG if the binary data sequence contained a 0x1d character. This has likely been fixed in newer versions. However, if you encounter this problem, Tektronix produced a "Socket Server Plus" program that communicates on port 4001. Contact Tektronix for further details.

Tektronix AWG7122B (2)
......................

The `AWG7122B <https://www.tek.com/en/datasheet/arbitrary-waveform-generators-7>`_ is a 24 GSa/s AWG with a maximum output frequency of 12 GHz, and this implementation communicates over a TCP socket. The default TekVisa software running on the scope communicates on port 4000. This AWG may either be externally triggered or it may play waveforms continuously. In the case that it is externally triggered, the trigger is set to a rising edge on trigger A, and the device otherwise works identically to the AWG70002B described above. Otherwise, when the experiment begins, the AWG will be placed into Run mode (if it was not already).

.. note::
  At the end of the experiment, the AWG7122B will be left in Run mode if it is **not** externally triggered. Usually one of the marker signals is used to trigger gas pulses, etc, and so these are left running in order to not interfere with any PID loops, etc.

Analog Devices AD9914 (3)
.........................

The `AD9914 <https://www.analog.com/en/products/ad9914.html>`_ is a direct digital synthesis chip that generates waveforms based on samples from an external clock. According to specs, the maximum external clock frequency is something like 3 GHz (but it seems to work still even at 4 GHz), and the maximum output frequency is half of the clock frequency. The AD9914 contains a built-in ramp generator that can be used to generate linear chirps.

.. warning::

  Support for this device should be considered experimental at best. At present, control of the AD9914 goes through an Arduino that uses the parallel interface to set register values on the AD9914 through a modified evaluation board. The performance is inconsistent and there are a number of register combinations that just do not seem to work as described in the documentation. You should strongly consider any other option than this! For more information, raise an issue on Github.

Keysight M8195A (4)
...................

The `M8195A <https://www.keysight.com/us/en/product/M8195A/65-gsa-s-arbitrary-waveform-generator.html>`_ is a 4-channel 65 GSa/s AWG with a maximum frequency of 25 GHz. Currently, it is hardcoded to lock to an external 10 MHz reference, and its chirp is output on channel 1. The protection pulse and the amplifier gate pulse is output on channel 3 and channel 4, respecivetly. It may be optionally externally triggered by a rising edge on the trigger input. Unlike the AWG7122B, at the end of the experiment, the outputs are disabled whether or not the scope is triggered.

**Known Issues**

 * There have been reports that errors may occur when Blackchirp writes waveform data to the device, but currently there has not been enough information to debug the issue, so the current status is unknown.

