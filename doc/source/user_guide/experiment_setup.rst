Experiment Setup
================

.. toctree::
   :hidden:

   experiment/acquisition_types
   experiment/optional_hardware
   experiment/validation
   experiment/quick_experiment
   experiment/sequence_mode

An ``Experiment`` is the basic acquisition event in Blackchirp, and it may consist of CP-FTMW measurements and/or a laser scan experiment (discussed more in the LIF module section of the user guide).
During an experiment, FID records are collected from the FTMW Digitizer and averaged in the time domain.
The average FID and its Fourier transform can be viewed and processed in a number of ways on the :doc:`CP-FTMW tab <cp-ftmw>` on the main user interface.
:doc:`Auxiliary Data <rolling-aux-data>` is also recorded during the experiment.
Depending on the :doc:`acquisition type <experiment/acquisition_types>`, the experiment will finish when one of the following conditions is met:

1. The objective of the experiment is reached (e.g., the requested number of shots has been collected).
2. The user clicks the ``Abort`` button on the user interface.
3. One of the :doc:`validation conditions <experiment/validation>` is outside the designated range.
4. A communication failure occurs with a critical piece of hardware.

To start an experiment, select one of the options under the ``Acquire`` menu in the :ref:`user_guide/ui_overview:Main Toolbar`.
The ``Start Experiment`` option opens a wizard that walks through configuring acquisition parameters and hardware settings (e.g., digitizer sample rate, FID record length, and optional hardware).

The key points for an FTMW acquisition are:

1. On the first wizard page, select the desired :doc:`acquisition type <experiment/acquisition_types>` and enter the acquisition objective.
   For LO Scan and DR Scan types, the scan parameters appear on the same page immediately below the type selector.
2. Verify and adjust the :doc:`FTMW Configuration <ftmw_configuration>` (Rf Configuration, :doc:`Chirp Configuration <ftmw_configuration/chirp_setup>`, and :doc:`FTMW Digitizer Configuration <ftmw_configuration/digitizer_setup>`).
3. Review the settings for any present :doc:`optional hardware <experiment/optional_hardware>`.
4. Set up any desired :doc:`validation conditions <experiment/validation>`.

A past experiment can be repeated using the :doc:`Quick Experiment <experiment/quick_experiment>` option, which pre-populates the wizard from a saved experiment.

Once you have finished with the wizard, Blackchirp reads the settings for each piece of hardware and attempts to initialize each item in turn.
If any errors occur during this process, the experiment is canceled and error messages will be displayed on the ``Log`` tab.
In the event that you encounter errors, check the :doc:`Hardware Details <hardware_details>` section of this manual to see if there are known issues or workarounds, and consider `reporting a bug <https://github.com/kncrabtree/blackchirp/issues>`_ if the problem cannot be solved.
Once all hardware has been initialized successfully, Blackchirp will ensure that it can acquire system resources (including GPU initialization if applicable), create a new folder to store data, and begin the acquisition.
