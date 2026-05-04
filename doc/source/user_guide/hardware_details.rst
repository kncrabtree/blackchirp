Hardware Details
================

Blackchirp interfaces with a variety of different pieces of hardware. The only items strictly required to run an experiment are a :doc:`clock </user_guide/hw/clock>` and an :doc:`FTMW digitizer </user_guide/hw/ftmwdigitizer>`; all other devices are optional. The set of devices in use for a given instrument is selected through a hardware loadout (see :ref:`hardware-menu-loadouts`).

Two settings are common to every hardware object and worth highlighting here:

* ``critical`` (true/false): When true, an error reported by this device aborts the running experiment, and new experiments cannot be started until the connection is retested from the :ref:`hardware-menu-communication` dialog. Set this to false for auxiliary devices whose failure should not stop acquisition.
* ``rollingDataIntervalSec`` (int): Polling interval, in seconds, for :doc:`rolling auxiliary data </user_guide/rolling-aux-data>`. Setting the value to 0 disables rolling data for that device. Not every hardware object produces rolling data; see the per-device pages for details.

Most other settings are surfaced inline by the hardware settings registry: each setting carries a human-readable label and tooltip that appear directly in the device dialog. See the :doc:`hardware dialog </user_guide/hwdialog>` page for the conventions used in that UI. The per-device pages below document only the behavior the inline tooltips do not already convey, along with driver-specific caveats.

.. toctree::
   :caption: Detailed Documentation
   :glob:

   hw/*
