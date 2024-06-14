Temperature Controller
======================

* Overview_
* Settings_
* Implementations_

Overview
--------

A TemperatureController in Blackchirp is a device which simply monitors the temperature of one or more sources, originally designed to work with buffer gas cells. The temperature of each active channel is monitored and used as both Rolling data and Auxiliary data, and can therefore be used as a validation setting to terminate an acquisition.

.. note::
   In the future, Blackchirp may support controlling setpoints for devices which have PID regulation.

Settings
--------

* ``channels`` (menu): Settings for each channel.
   - ``decimal`` (int): Number of decimal places to display.
   - ``units`` (string): Units to display.
* ``pollIntervalMs`` (int): Time between polling sequential enabled channels, in ms.


Implementations
---------------

Virtual (virtual)
.................

A dummy implementation which returns a random value near 4 for each enabled channel.

Lakeshore 218 (lakeshore218)
............................

The `Lakeshore 218 Cryogenic Temperature Monitor <https://www.lakeshore.com/products/categories/overview/temperature-products/cryogenic-temperature-monitors/model-218-temperature-monitor>`_ is an 8-channel device designed for cryogenic applications.

.. warning::
   This implementation has not been thoroughly tested, as the UC Davis spectrometer does not have this device.



