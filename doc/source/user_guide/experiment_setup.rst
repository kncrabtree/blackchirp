Experiment Setup
================

.. toctree::
   :hidden:
   :glob:

   experiment/*

An ``Experiment`` is the basic acquisition event in Blackchirp, and it may consist of CP-FTMW measurements and/or a laser scan experiment (discussed more in the LIF module section of the user guide).
During an experiment, FID records are collected from the `FTMW Digitizer <hw/ftmwdigitizer.html>`_ and averaged in the time domain.
The aveage FID and its Fourier transform can be viewed and processed in a number of ways on the `CP-FTMW tab <cp-ftmw.html>`_ on the main user interface.
`Auxiliary Data <rolling-aux-data.html>`_ is also recorded during the experiment.
Depending on the `acquisition type <experiment/acquisition_types.html>`_, the experiment will finish when one of the following conditions is met:

1. The objective of the experiment is reached (e.g., the requested number of shots has been collected).
2. The user clicks the ``Abort`` button on the user interface.
3. One of the `validation conditions <experiment/validation.html>`_ is outside the designated range.
4. A communication failure occurs with a critical piece of hardware.

To start an experiment, select one of the options under the ``Acquire`` menu in the `main toolbar <ui_overview.html#main-toolbar>`_.
Initially, you will most likely want to choose the ``Start Experiment`` option, which will bring up a dialog that walks you through configuring the important details of the acquisition.
This includes not only acquisition parameters but also many hardware settings (e.g., digitizer sample rate, FID record length, etc.).


The key points for an FTMW acquisition are:

1. Select the desired `acquisition type <experiment/acquisition_types.html>`_ and verify the `Rf Configuration <hardware_menu.html#rf-configuration>`_.
2. Set up the `FTMW chirp <experiment/chirp_setup.html>`_ and configure the `Pulse Generator <hw/pulsegenerator.html>`_ (if enabled).
3. Configure the `FTMW Digitizer settings <experiment/digitizer_setup.html>`_ and, if present, the IO Board settings.
4. Set up any desired `validation conditions <experiment/validation.html>`_.

Alternatively, a past experiment can be repeated using the `Quick Experiment <experiment/quick_experiment.html>`_ option.

Once you have finished with the wizard, Blackchirp reads the settings for each piece of hardware and attempts to initialize each item in turn.
If any errors occur during this process, the experiment is cancelled and error messages will be displayed on the ``Log`` tab.
In the event that you encounter errors, check the `Hardware Details <hardware_details.html>`_ section of this manual to see if there are known issues or workarounds, and consider `reporting a bug <https://github.com/kncrabtree/blackchirp/issues>`_ if the problem cannot be solved.
Once all hardware has been initialized successfully, Blackchirp will ensure that it can acquire system resources (including GPU initialization if applicable), create a new folder to store data, and begin the acquisition.
