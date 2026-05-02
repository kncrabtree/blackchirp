Flow Controller
===============

* Overview_
* Settings_
* Implementations_

Overview
--------

A FlowController oversees a bank of mass flow controllers with an optional pressure sensor. Implementations typically support two operating modes: holding each channel at a fixed flow rate, or maintaining a constant ratio of flow rates while regulating the total flow to sustain a target pressure read by an integrated sensor. Blackchirp records the names and setpoints of all enabled gases at the start of an experiment, and reads flow rates and pressure as auxiliary data throughout the run.

The flow channel count is a runtime setting (``numChannels``) rather than a compile-time constant, so a single build supports devices with different numbers of channels. Changing ``numChannels`` from the device dialog rebuilds the per-channel array immediately on accept; the data of channels that still exist is preserved.

Settings
--------

Most settings are exposed in the :doc:`hardware dialog </user_guide/hwdialog>` with inline labels and tooltips supplied by the settings registry. A few items are worth highlighting:

* ``numChannels`` sets the number of mass flow channels the controller manages. It defaults to 4 and is registered as an Important setting, so the per-channel ``channels`` array is rebuilt as soon as the new value is saved; data for channels that still exist is preserved.
* ``pUnits`` (the pressure-units string used for display and file output) is registered at the FlowController base class and defaults to ``kTorr``. Implementations that need a different default (for example, the Python flow-controller bridge) override only this key; users still see the same setting in the dialog.
* ``intervalMs`` is the timer period between successive readback queries. Each tick advances by one channel through the channel list, so the effective per-channel update rate scales with the configured channel count.
* The per-channel ``channels`` array (``chUnits``, ``chMax``, ``chDecimals``) controls display formatting only. Maximum values are advisory: Blackchirp does not clamp setpoints to ``chMax`` before sending them to the device.

The rolling and auxiliary display for flow data is rendered in the gas-flow status panel of the main window. Channels with no name and a zero setpoint are hidden so that small experiments do not waste vertical space; channels become visible automatically as soon as either a name or a non-zero setpoint is assigned. Each row carries the live flow value, a status LED indicating an active setpoint, and a tooltip summarizing the current setpoint with units. The pressure row at the top of the panel uses the registry-supplied pressure units and decimal places for formatting.

Implementations
---------------

Virtual
.................

A dummy 4-channel implementation that echoes setpoints back as flow readings. Useful for offline UI testing.

MKS 647C
..................

A 4-channel device with a built-in PID controller for maintaining a target pressure. The communication protocol is RS232. The implementation maintains an internal table of supported gas-correction factor ranges and pressure ranges; both are selected automatically based on the configured maxima.

MKS 946
................

A modular platform from MKS supporting different combinations of mass flow controllers and pressure sensors. The Blackchirp implementation assumes 4 flow channels and 1 pressure channel and adds three implementation-specific settings (``address``, ``offset``, and ``pressureChannel``) for matching the controller's RS-232 address and the physical slot numbers of the MFC and pressure inputs. The communication protocol is RS232.
