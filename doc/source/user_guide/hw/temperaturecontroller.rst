Temperature Controller
======================

* Overview_
* Settings_
* Implementations_

Overview
--------

A TemperatureController in Blackchirp is a device that monitors the temperature of one or more sources. The original use case is buffer-gas cells, but any multi-channel temperature monitor fits the same role. The temperature of each active channel is recorded as both rolling data and auxiliary data, so it can be used as a validation condition to terminate an acquisition when a channel drifts outside an acceptable range.

.. note::
   Blackchirp does not control setpoints for PID-regulated devices; temperature controllers are read-only here.

Settings
--------

Most settings are exposed in the :doc:`hardware dialog </user_guide/hwdialog>` with inline labels and tooltips. The poll interval defaults to 500 ms; raise it for instruments whose temperature update rate is the bottleneck during an acquisition. Per-channel settings (display name, enabled flag, decimal places, and display units) are edited in the channels table; units default to ``K`` and the decimal count defaults to 4.

Implementations
---------------

Virtual
.................

A dummy implementation that returns a random value near 4 K for each enabled channel.

Lakeshore 218
............................

The `Lakeshore 218 Cryogenic Temperature Monitor <https://www.lakeshore.com/products/categories/overview/temperature-products/cryogenic-temperature-monitors/model-218-temperature-monitor>`_ is an 8-channel device designed for cryogenic applications. It connects via RS232 with a 500 ms timeout and ``\r\n`` termination.

.. warning::
   This implementation has not been thoroughly tested, as the UC Davis spectrometer does not have this device.
