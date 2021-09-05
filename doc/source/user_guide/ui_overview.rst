.. index::
   single: FixedClock
   single: Clock
   single: TemperatureController
   single: PulseGenerator
   single: FlowController

User Interface Overview
=======================

.. image:: /_static/user_guide/ui_overview/ui.png
   :width: 800
   :alt: User interface screenshot

Instrument Status
.................

The left side of the user interface shows the current instrument status, including the most recent readings from hardware that is periodically polled.
Depending on your hardware configuration, some of these boxes may not be present.
The first two sections

- ``Expt`` displays the last experiment number in the current save directory, or 0 if no experiments have been performed. The number increments upon successful initialization of an experiment.
- ``Clocks`` shows the current frequencies of the oscillators that have been configured. Upon program startup, all physical clocks that are assigned to clock roles are read and the frequencies updated. For FixedClocks, the frequency will be recalled from the previous Blackchirp instance. See the `Hardware Menu`_ page for more details on the clock roles.
- ``Hardware Status`` boxes show the most recent reading(s) for the respective hardware, and/or LEDs that indicate whether a particular channel is active. For instance, if a TemperatureController is enabled, the Temperature Status box will show the readings of all enabled channels, while if a PulseGenerator is enabled, then the Pulse Status box shows which channels are currently enabled. A FlowController, on the other hand, always shows the readings of all channels, and LEDs are used to indicate whether a channel is enabled.
- ``FTMW Progress`` shows the completion percentage of an ongoing FTMW acquisition.

.. _Hardware Menu: hardware_menu.html







