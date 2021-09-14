Rolling/Aux Data
================

.. image:: /_static/user_guide/rolling_aux_data/rollingdata.png
   :width: 800
   :align: center
   :alt: Rolling data screenshot

**Rolling data** and **Aux data** both refer to signals that are periodically recorded as a function of time, and they are plotted against the date and time the measurement was taken.
Rolling data is taken as long as Blackchirp is running, while aux data is associated with a particular experiment, and is only recorded during the time that the experiment is active.
An example is shown in the screenshot above; a virtual `TemperatureController <hw/temperaturecontroller.html>`_ is configured to monitor 4 different temperature channels at an interval of 5 seconds.
Only some pieces of hardware support rolling/aux data; details are available on each item's entry on the `Hardware Details <hardware_details.html>`_ page.

Entries on the `main toolbar <ui_overview.html#main-toolbar>`_ for both rolling and aux data allow customization of the number of plots and offer a control to autoscale all plots.
A curve can be moved from one plot to another by right-clicking the plot where the curve is presently located, selecting it from the **Curves** menu, and choosing the **Move to plot** option.
Plots are numbered from left to right, top to bottom.

Each particular curve that appears on these plots is associated with an identifier of the form::

   HardwareObject.Implementation[.DisplayName].SignalID

The DisplayName, if present, is the name that is displayed in the legend of the plot where the curve appears.
For example, in the TemperatureController object, the DisplayName is set to the name entered in the `TemperatureController <hw/temperaturecontroller.html>`_ `Hardware Control menu <hardware_menu.html#hardware-control-settings>`_, and likewise for the `FlowController <hw/flowcontroller.html>`_ and `IOBoard <hw/ioboard.html>`_.
The identifier is used to refer to the particular trace when data are logged to disk, as discussed further below.
Examples of identifiers are::

   TemperatureController.virtual.Buffer Gas Cell.Temperature1
   PressureController.intellisys.chamberPressure


Configuring Rolling Data
------------------------

By default, rolling data is disabled.
Enabling rolling data is accomplished by selecting the appropriate hardware entry in the `Hardware Menu <hardware_menu.rst>`_, enabling channels if applicable, and setting the ``rollingDataIntervalSec`` setting to a positive integer.
The value entered for that setting is the time between data points, in seconds, and it can be set to different values for each piece of hardware if desired.
Keep in mind that rolling data is continually logged to disk, so setting a short interval between readings will increase the disk usage.
The interval can be changed at any time as needed, except during an experiment.

In addition, in the Rolling Data menu on the `main toolbar <ui_overview.html#main-toolbar>`_, the minimum amount of history displayed in the program can be changed.
Blackchirp will record data until the amount of displayed history reaches 150% of the selected history duration.
At that time, all data points older than the requested history duration are discarded.
For example, if the history is set to 12 hours, then once 18 hours worth of data have been recorded, the oldest 6 hours' worth of points are discarded.
This only applies to the data shown on screen; the discarded data points are still available for later viewing on disk.

Aux data is configured by setting the **Aux Data Interval** option to the desired sample interval when `starting a new experiment <experiment_setup.html>`_.
The aux data interval need not be the same as the rolling data interval, but all enabled aux data sources will be recorded during the experiment regardless of whether they are currently configured for rolling data.
A few extra data sources are available as aux data, such as the total number of FTMW shots acquired over time, and information related to phase correction algorithms (if enabled).
Aux data channels are used as the inputs for `Experiment Validation <experiment/validation.html>`_ conditions that can automatically abort an experiment if data falls outside a desired range.

Rolling Data Storage
--------------------

Blackchirp logs rolling data in the ``rollingdata`` subdirectory of the current `Data Storage Location <first_run.html#data-storage-location>`_.
Inside that folder, data are organized into one folder per year that contains folders for each month.
Within the folder for a particular month, Blackchirp writes a csv file per rolling data source.
If the identifier contains an optional display name, data will be appended to the existing CSV file if the same display name has been previously used.
Otherwise, a new file is created when the display name changes.

The data format of the rolling data CSV file is shown below for a file titled ``TemperatureController.virtual.BG Cell Temperature.Temperature1.csv``::

   timestamp;epochtime;TemperatureController.virtual.BG Cell Temperature.Temperature1
   Wed Sep 1 22:21:02 2021;1630560062;4.8975830078125
   Wed Sep 1 22:21:12 2021;1630560072;5.18426513671875
   Wed Sep 1 22:21:22 2021;1630560082;4.06927490234375
   Wed Sep 1 22:21:32 2021;1630560092;5.293487548828125
   Wed Sep 1 22:21:42 2021;1630560102;5.805755615234375
   Wed Sep 1 22:21:52 2021;1630560112;4.14141845703125
   Sat Sep 4 10:45:25 2021;1630777525;5.842254638671875
   Sat Sep 4 10:45:35 2021;1630777535;4.129913330078125
   Sat Sep 4 10:45:45 2021;1630777545;5.730865478515625
   Sat Sep 4 10:45:55 2021;1630777555;5.56634521484375
   Sat Sep 4 10:46:05 2021;1630777565;5.209564208984375
   Sat Sep 4 10:46:15 2021;1630777575;4.839202880859375
   Sat Sep 4 10:46:25 2021;1630777585;4.56243896484375

The first column contains the date and time of the data point in string format, while the second is the `Unix epoch time <https://en.wikipedia.org/wiki/Unix_time>`_.
The final column contains the data value.
As can be seen in the snippet above, there may be gaps in time within the files if Blackchirp was shut down or if the name assigned to this channel of the temperature controller was changed and later reset to the same value (``BG Cell Temperature``).
Aux data has a similar format, and it is discussed on the `Data Format <experiment/data_format.html>`_ page.

