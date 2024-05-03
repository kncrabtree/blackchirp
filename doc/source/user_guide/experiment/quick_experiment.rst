.. index::
   single: Experiment
   single: Quick Experiment

Quick Experiment
================

.. image:: /_static/user_guide/experiment/quickexpt_1.png
   :align: center
   :width: 400
   :alt: Quick Experiment Dialog

The Quick Experiment action initializes an experiment using some or all of the settings from a previous experiment. Select the desired experiment to repeat using the number field.

.. note::
   At present, Blackchirp only supports repeating experiments that are in the current Data Storage Location. In the future, support will be added for loading experiments from other locations as well.

On the right, the "Use Current Settings" box allows the user to override settings for certain devices with the currently enabled settings instead of those saved with the experiment. For example, the width of gas and/or discharge pulses may be optimized day-to-day, and it would be preferred to use today's optimized values instead of whatever values were used the previous day. Checking the box corresponding to the pulse generator would ensure that the current settings are used. By default, all "optional" hardware (i.e., Pulse Generators, Flow Controllers, Pressure Controllers, IO Boards, and Temperature Controllers) will use current settings rather than saved values.

The details of the experiment can be viewed in the lower menu. If only a couple of settings need to be changed, then the "Configure Experiment" button will launch a new experiment setup dialog whose values are all initialized to the selected experiment (including any "current settings" boxes which are checked). The "New Experiment" button discards any settings from the selected experiment and launches the same dialog that would have been launched with the "Start Experiment" button on the main user interface. Finally, the "Start Experiment" button will immediately begin the selected experiment.

Blackchirp will only allow a quick experiment to be launched if the current hardware configuration is identical to the configuration used in the previous experiment and the major version of Blackchirp is the same. Otherwise, an error will be displayed as shown in the image below.

.. image:: /_static/user_guide/experiment/quickexpt_2.png
   :align: center
   :width: 400
   :alt: Quick Experiment Dialog Error
