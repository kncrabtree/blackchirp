Pulse Generator
===============

* Overview_
* Settings_
* Drivers_

Overview
--------

A pulse generator is a source of timing pulses whose delays and widths can be varied with respect to a master timing source (either an internal rate generator or an external trigger). Each channel has an individually controllable width and delay. If supported by the device, the pulse sequence may be triggered externally and channels may be set to operate in a lower duty-cycle mode or synchronized to another channel.

.. image:: /_static/user_guide/hw-pulsegenerator-menu.png
   :width: 800
   :alt: Pulse Generator Control Screenshot

The image above shows an overview of the pulse generator control widget. Individual channels can be further configured with the wrench-icon button, which opens a per-channel dialog (shown below) for setting the channel name and role, the active level, the duty-cycle behavior, and the step sizes for the numeric controls.

.. image:: /_static/user_guide/hw-pulsegenerator-submenu.png
   :alt: Pulse Generator Control Channel Configuration

Blackchirp recognizes a fixed list of channel roles, several of which trigger validation when an experiment is started:

* ``None``: No role configured.
* ``Gas``: Channel controls a gas pulse. The experiment dialog warns if the channel is disabled when starting an experiment.
* ``AWG``: Channel triggers an AWG. A warning is issued if this channel is disabled during an FTMW acquisition.
* ``DC``: Channel controls an electric discharge. No special behavior.
* ``Prot``: Channel controls a protection switch. An error is raised if this channel is disabled for an FTMW experiment.
* ``Amp``: Channel triggers an amplifier gate. A warning is issued if this channel is disabled for an FTMW experiment.
* ``Laser``: Channel triggers a generic laser. No special behavior.
* ``XMer``: Channel triggers an excimer laser. No special behavior.
* ``LIF``: Available only with the LIF module. The channel's delay is adjusted during an LIF scan to control the timing of the LIF laser pulse.

Settings
--------

The pulse generator's per-channel controls (name, role, width, delay, level, duty-cycle parameters, and sync target) are presented inline by the device dialog, with labels and tooltips supplied by the settings registry. The device-level settings exposed in the :doc:`hardware dialog </user_guide/hwdialog>` carry inline descriptions as well. A few items are worth highlighting:

* ``numChannels`` is required and defines the number of channels the device exposes; for fixed-channel hardware the driver sets it to the device's actual count.
* ``minWidth`` / ``maxWidth``, ``minDelay`` / ``maxDelay``, and ``minRepRate`` / ``maxRepRate`` set the allowed ranges (microseconds for widths and delays, Hz for the repetition rate). Blackchirp refuses values outside these ranges, so they must reflect the hardware capability rather than a user-imposed limit.
* The capability flags ``canDutyCycle``, ``canTrigger``, ``canSyncToChannel``, and ``canDisableChannels`` advertise optional features. When a flag is false, the corresponding controls are hidden or disabled in the channel dialog and the matching set operations are rejected.
* ``lockExternal`` requests that the timing clock lock to an external 10 MHz reference; drivers that lack this capability ignore the setting.

Drivers
-------

Drivers are organized by product family; devices within a series typically differ only in channel count and share a command protocol. The Quantum Composers and Berkeley Nucleonics drivers communicate over RS232, TCP, or GPIB; the SRS DG645 driver is RS232-only.

Virtual
...............................

An 8-channel stub driver with all capability flags enabled. Used for development and for installs that drive their pulse-train hardware out-of-band from Blackchirp.

Quantum Composers 9210 Series
............................................

The `9210 / Sapphire family <https://www.quantumcomposers.com/pulse-delay-generator-sapphire>`_ is the 4-channel low-cost line, including the QC 9214. The driver supports the full feature set Blackchirp exposes for QC pulse generators: external triggering, channel synchronization, individual channel disable, and duty-cycle mode.

Quantum Composers 9510 Series
............................................

The 9510 family is the 8-channel predecessor to the 9520 series, including the QC 9518. Feature support matches the 9210 series above. The 9510 hardware does not support locking to an external reference clock; ``lockExternal`` is ignored on this driver.

Quantum Composers 9520 Series
............................................

The `9520 family <https://www.quantumcomposers.com/pulse-delay-generator-9520>`_ is the 8-channel current Quantum Composers offering, including the QC 9528. It supports every feature Blackchirp exposes for pulse generators, including locking to an external 10 MHz reference.

.. warning::
   Some 9520-series firmware revisions have a known bug that causes intermittent errors when toggling between Continuous and Triggered modes. Avoid changing this setting while configuring an experiment, as it is likely to cause an initialization failure. The issue can be sidestepped by making the change on the device's front panel and then opening the pulse generator control widget in Blackchirp, which forces a re-read of all device settings.

Berkeley Nucleonics 577
................................

The `BNC 577 family <https://www.berkeleynucleonics.com/model-577>`_ is offered in 4-channel and 8-channel variants that share a single Blackchirp driver; the channel count is selected from the device dialog when the hardware is added to a profile. The 577 supports the full QC-style feature set.

Stanford Research Systems DG645
..........................................

The `SRS DG645 <https://www.thinksrs.com/products/dg645.html>`_ is a 4-channel delay generator. It supports external triggering and channel-to-channel synchronization, but its hardware design does not allow per-channel disable or per-channel duty-cycle mode; the matching capability flags are forced false and the corresponding controls are hidden in the channel dialog.
