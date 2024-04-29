Clock
=====

* Overview_
* Settings_
* Implementations_

Overview
--------

A clock is any type of single-frequency oscillator. A single clock may have multiple independent output channels, each of which may be at a different frequency, and it may or may not be tunable. Clocks may be assigned to different logical roles within the program (see `Rf Configuration <../hardware_menu.html#rf-configuration>`_ for details), and Blackchirp supports up to 6 different clock devices.

.. note::
  Currently there is no means for controlling the power output of a clock from Blackchirp. This is planned for the future; if you require this functionality, please make a feature request on Github.

Settings
--------

 * ``lockExternal`` (true/false): Whether the clock should be locked to an external reference. Some implementations may use this to force the clock to lock (or raise an error if unsuccessful); however, this is not used by all implementations.
 * ``manualTune`` (bool): If true, Blackchirp prompts the user to manually tune the clock if needed (e.g., for LO or DR scans when the clock is not controlled by Blackchirp).
 * ``maxFreqMHz`` (float): Maximum frequency, in MHz.
 * ``minFreqMHz`` (float): Minimum frequency, in MHz.
 * ``outputs`` (menu): This contains the settings made for each output in the `Rf Configuration <../hardware_menu.html#rf-configuration>`_ menu and should not be edited manually.
 * ``tunable`` (bool): Whether the frequency of the clock is adjustable. Currently this setting is not used anywhere in Blackchirp.

Implementations
---------------

FixedClock (fixed)
..................

A fixed clock is a virtual implementation of a clock. This is the implementation that should be selected in the event you use a fixed-frequency oscillator (e.g., a PLDRO) in your setup. Up to 6 different "outputs" are available for each fixed clock.

Valon Technology 5009 (valon5009)
.................................

The `Valon 5009 <https://www.valonrf.com/frequency-synthesizer-6ghz.html>`_ is a two-channel synthesizer; each channel is independent and has a maxiumum frequency of 6 GHz. It can be locked to an external reference. If ``lockExternal`` is set to true, the device is set to use a 10 MHz external reference at the start of an experiment, and otherwise it is set to use its internal 20 MHz reference. The device has an internal USB-RS232 converter, so connecting it to a computer via USB generates a virtual serial port on the computer.

Valon Technology 5015 (valon5015)
.................................

The `Valon 5015 <https://www.valonrf.com/5015-frequency-synthesizer-15ghz.html>`_ is a single-channel synthesizer with a maximum output frequency of 15 GHz. Its implementation is otherwise identical to the 5009.

Hewlett-Packard 83712B (hp83712b)
.................................

The HP 83712B is a single-channel synthsizer that is tunable between 1-20,000 MHz.
It is configured to connect through GPIB. The implementation has not been thoroughly tested.
