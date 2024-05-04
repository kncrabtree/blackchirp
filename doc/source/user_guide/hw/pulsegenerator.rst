Pulse Generator
===============

* Overview_
* Settings_
* Implementations_

Overview
--------

A PulseGenerator is a source of timing pulses whose delays and/or widths can be varied with respect to a master timing source (either an internal rate generator or an external trigger). Each channel is assumed to have an individually controllable width and delay. If supported by the device, it may be externally triggered and channels may be set to operate in a lower duty cycle mode or synced to another specified channel.

.. image:: /_static/hardware/pulsegenerator_menu.png
   :width: 800
   :alt: Pulse Generator Control Screenshot

The image above shows an overview of the pulse generator control menu giving an idea of the main features. In addition to what is shown above, individual channels may be further configured using the button with the wrench icon (see the image below). There, the channel name and role may be set, as well as settings related to the active level, duty cycle mode, and step sizes for the numeric controls.

.. image:: /_static/hardware/pulsegenerator_submenu.png
   :width: 300
   :alt: Pulse Generator Control Channel Configuration

Blackchirp has some predefined channel roles, some of which have special functionality within the program:
* ``None``: No role configured
* ``Gas``: Channel controls a gas pulse. If this role is assigned, the experiment dialog will give a warning if the channel is disabled when starting an experiment.
* ``AWG``: Channel triggers an AWG. If assigned, a warning will be issued if this channel is disabled during an FTMW acquisition.
* ``DC``: Channel controls an electric discharge. No special behavior.
* ``Prot``: Channel controls a protection switch. If assigned, an error will be thrown if this channel is disabled for an FTMW experiment.
* ``Amp``: Channel triggers an amplifier gate. If assigned, a warning will be issued if this channel is disabled for an FTMW experiment.
* ``Laser``: Channel triggers a generic laser. No special behavior
* ``XMer``: Channel triggers an excimer laser. No special behavior.
* ``LIF``: Only available in the LIF module. This channel's delay is adjusted during an LIF scan to control the timing of the LIF laser pulse.


Settings
--------

* ``canDutyCycle`` (bool): If true, the delay generator supports duty cycle mode in which a channel generates a pulse for N trigger events followed by no pulse for M subsequent trigger events, where N and M are set by the user.
* ``canSyncToChannel`` (bool): If true, one channel's delay may be referenced to the delay of another channel rather than the master clock.
* ``canDisableChannels`` (bool): If true, individual channel outputs may be enabled/disabled independently of the overall system status.
* ``canTrigger`` (bool): If true, the pulse sequence can be initiated from an external trigger instead of an internal rate generator.
* ``dutyMaxPulses`` (int): Maximum number of pulses that can be set in duty cycle mode for on/off sequences.
* ``lockExternal`` (bool): If true, timing clock will be set to lock to an external reference.
* ``maxDelay`` (double): Maximum allowed delay, in microseconds.
* ``maxRepRateHz`` (double): Maximum repetition rate for internal rate generator, in Hz.
* ``maxWidth`` (double): Maximum allowed pulse width, in microseconds.
* ``minDelay`` (double): Minimum allowed delay, in microseconds.
* ``minRepRateHz`` (double): Minimum allowed repetition rate, in Hz.
* ``minWidth`` (double): Minimum allowed pulse width, in microseconds.


Implementations
---------------

Virtual (virtual)
.................

A dummy implementation with 8 channels and all options enabled.

Quantum Composers 9528 (qc9528)
...............................

The `Quantum Composers 9528 <https://www.quantumcomposers.com/pulse-delay-generator-9520>`_ is an 8-channel pulse generator which supports all features implemented in Blackchirp. The communication is set to RS232.

.. warning::
   There is a known bug with certain versions of the QC 9528 firmware which causes an intermittent error when trying to toggle between Continuous and Triggered modes. Do not change this setting while configuring an experiment, as it is likely to cause an initialization failure. The issue can be avoided by making this setting using the pulse generator's front panel and then opening the pulse generator control menu in Blackchirp, which triggers reading all of the device settings.

Quantum Composers 9518 (qc9518)
...............................

The Quantum Composers 9518 is an 8-channel pulse generator which supports all features implemented in Blackchirp except locking to an external reference. The communication is set to RS232. This model is no longer sold by Quantum Composers; the QC 9528 is recommended instead.

Quantum Composers 9214 (qc9214)
...............................

The `Quantum Composers 9214 <https://www.quantumcomposers.com/pulse-delay-generator-sapphire>`_ is a low-cost, 4-channel pulse generator that supports most Blackchirp features and communicates via RS232.

Stanford Research Systems DG645 (dg645)
.......................................

The `SRS DG645 <https://www.thinksrs.com/products/dg645.html>`_ is a 4-channel pulse generator which is configured to communicate over its RS232 output. This device supports external triggering and synchronizing channels to one another, but does not support disabling individual channels or duty cycle mode for individual channels.

