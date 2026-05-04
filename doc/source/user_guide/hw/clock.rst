Clock
=====

* Overview_
* Settings_
* Drivers_

Overview
--------

A clock is any single-frequency oscillator. A single clock may have multiple independent output channels, each of which may be at a different frequency, and it may or may not be tunable. Clocks are assigned to logical roles within the program (see :doc:`/user_guide/rf_configuration`), and Blackchirp supports up to six clock devices.

.. note::
  There is no means for controlling the power output of a clock from Blackchirp. If you require this functionality, please make a feature request on Github.

Settings
--------

Most clock settings are exposed in the :doc:`hardware dialog </user_guide/hwdialog>` with inline labels and tooltips, so they need no additional explanation here. A few items are worth highlighting:

* ``minFreqMHz`` / ``maxFreqMHz`` set the allowable frequency range for the clock in MHz. Blackchirp refuses to drive an output outside this window, so the values must reflect the actual hardware capability rather than a user-imposed limit.
* ``lockExternal`` requests that the clock lock to an external 10 MHz reference at the start of an experiment. Drivers that cannot read back the lock state ignore the setting.
* ``manualTune`` indicates that Blackchirp cannot drive this clock programmatically. When true, scans that step the clock frequency (LO or DR scans) pause at each point and prompt the user to set the new frequency on the instrument by hand before the scan continues.
* The ``outputs`` array stores the role and multiplication-factor assignments configured on the **Clocks** tab of the RF configuration widget. It should not be edited manually.

Clock frequencies are pushed to hardware whenever the active loadout's FTMW preset changes (loadout switch or :ref:`Hardware Configuration <hardware-menu-loadouts>` accept) and at the start of each experiment. The **Clocks** tab of the RF configuration widget also provides an **Apply Clock Settings Now** button, which sends the current clock table to hardware immediately without starting an experiment.

Drivers
-------

FixedClock
..................

A virtual clock with up to six independently configurable outputs. Use this driver for any fixed-frequency oscillator (e.g., a PLDRO) whose frequency Blackchirp does not tune.

Valon Technology 5009
.................................

The `Valon 5009 <https://www.valonrf.com/frequency-synthesizer-6ghz.html>`_ is a two-channel synthesizer; each channel is independent with a maximum frequency of 6 GHz. When ``lockExternal`` is true, the device is configured to use a 10 MHz external reference at the start of an experiment; otherwise it uses its internal 20 MHz reference. The device has an internal USB-RS232 converter, so connecting it to a computer via USB generates a virtual serial port.

Valon Technology 5015
.................................

The `Valon 5015 <https://www.valonrf.com/5015-frequency-synthesizer-15ghz.html>`_ is a single-channel synthesizer with a maximum output frequency of 15 GHz. Its driver is otherwise identical to the 5009.

Hewlett-Packard 83712B
.................................

The HP 83712B is a single-channel synthesizer tunable between 1 and 20,000 MHz. It connects through GPIB. The driver has not been thoroughly tested.
