IO Board
========

* Overview_
* Settings_
* Implementations_

Overview
--------

An IOBoard is a device with one or more analog and/or digital input/output channels. In Blackchirp, an IOBoard is used to monitor analog or digital signals by taking a single measurement at designated time intervals. Analog channels can be read as rolling data for continous monitoring in addition to auxiliary data during a scan. Digital channels are read as auxiliary data during a scan and used to automatically abort a scan.

At present, Blackchirp does not support analog or digital output signals, even for devices which support it. Control of digital output signals is planned for future support. The "Role" field available on the IO Board configuration page

Settings
--------

The settings are similar to the `FTMW Digitizer <hw/ftmwdigitizer.html>`_. However, some settings such as ``maxRecordLength``, ``canBlockAverage``, etc are not used and will be ignored if changed, as the IO board is configured to only read 1 sample on demand. In addition, there are the following settings:

* ``numAnalogChannels`` (int): Number of analog input channels. For some IO boards (e.g., LabJack U3), some channels may by configured as analog or digital, and this allows a user to change how many channels are configured as each.
* ``numDigitalChannels`` (int): Number of digital input/output channels.


Implementations
---------------

Virtual (virtual)
.................

A dummy implementation which returns a random value for each enabled channel (8 analog channels, 8 digital channels).

LabJack U3 (labjacku3)
......................

The `LabJack U3 <https://labjack.com/products/u3>`_ is a multichannel, configurable IO board with a variable number of analog/digital channels. The implementation defaults to 8 analog inputs (pins 0-7 which correspond to the 4 analog inputs and the first 4 FIO pins) and 8 digital input/outputs corresponding to FIO4-11. If ``numAnalogChannels`` is set to 4, then up to 12 digital channels can be used. Errors may occur if ``numAnalogChannels`` is less than 4.

The LabJack U3 requires the LabJackUSB driver to be installed on the system and linked to the applcication executable.
