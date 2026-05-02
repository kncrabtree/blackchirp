IO Board
========

* Overview_
* Settings_
* Implementations_

Overview
--------

An IOBoard is a device with one or more analog and/or digital input/output channels. In Blackchirp, an IOBoard takes a single measurement at designated time intervals to monitor analog or digital signals. Analog channels can be read as rolling data for continuous monitoring as well as auxiliary data during a scan. Digital channels are read as auxiliary data during a scan and can be used to automatically abort a scan when a validation condition is violated.

Blackchirp does not drive analog or digital output signals, even on devices that support it.

Settings
--------

The IOBoard reuses many of the same settings as the :doc:`/user_guide/hw/ftmwdigitizer`, and the :doc:`hardware dialog </user_guide/hwdialog>` presents them with inline labels and tooltips. Most digitizer-only settings (``maxRecordLength``, ``canBlockAverage``, multi-record options, and so on) are ignored because the IO board only reads one sample on demand.

Two settings are worth highlighting:

* ``numAnalogChannels`` selects how many channels are configured as analog inputs on devices where channels can be reassigned (such as the LabJack U3). It is a Required setting and is read-only after profile creation.
* ``numDigitalChannels`` sets the number of digital input/output channels and is similarly read-only after profile creation.

Per-channel role and naming options are exposed in the IO Board configuration page of the hardware dialog.

Implementations
---------------

Virtual
.................

A dummy implementation that returns a random value for each enabled channel (8 analog channels, 8 digital channels).

LabJack U3
......................

The `LabJack U3 <https://labjack.com/products/u3>`_ is a multichannel, configurable IO board with a variable number of analog/digital channels. The implementation defaults to 8 analog inputs (pins 0-7, corresponding to the 4 analog inputs and the first 4 FIO pins) and 8 digital input/outputs corresponding to FIO4-11. Setting ``numAnalogChannels`` to 4 frees pins for up to 12 digital channels; values below 4 are not supported and may produce errors at runtime.

The LabJack U3 is supported on Linux, macOS, and Windows. Linux and macOS builds talk to the device through the ``liblabjackusb`` Exodriver; Windows builds load the LabJackUD driver instead. In every case the driver must be installed on the host computer and discoverable by Blackchirp's runtime library loader. See :doc:`/user_guide/library_status` for the current driver load state and platform-specific installation guidance, including the Windows UD-driver install hint.
