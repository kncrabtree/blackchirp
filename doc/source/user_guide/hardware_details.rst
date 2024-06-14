Hardware Details
================

Blackchirp is capable of interfacing with a variety of different pieces of hardware; see the `installation page <installation.html#hardware-implementations>`_ for details on how to select which hardware are in use for your instrument. The only pieces of hardware that are strictly required to run Blackchirp are a `clock <hw/clock/html>`_ and an `FTMW digitizer <hw/ftmwdigitizer.rst>`_; all other pieces of hardware are optional and may be omitted from the program entirely by commenting them out in ``config.pri``.

All pieces of hardware have some settings in common, and these may be edited in the `hardware settings menu <hardware_menu.html#hardware-control-settings>`_ associated with each item. These are:

  * **critical** (true/false): If true, an experiment will be aborted if an error occurs with this hardware, and experiments cannot be started until the `connection is retested <hardware_menu.html#communication>`_.
  * **rollingDataIntervalSec** (int): Time between `rolling data samples <rolling-aux-data.html>`_, in seconds. If set to 0, rolling data is disabled. Not all pieces of hardware generate rolling data; see the documentation for a particular piece of hardware to see what is available.

Further details about each hardware item, their user-controllable settings, and implementation-specific details/known issues are available on the pages below.

.. toctree::
   :caption: Detailed Documentation
   :glob:

   hw/*

