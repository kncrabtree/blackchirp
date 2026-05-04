.. index::
   single: Target Shots
   single: Target Duration
   single: Forever
   single: Peak Up
   single: LO Scan
   single: DR Scan
   single: Phase Correction
   single: Chirp Scoring
   single: Aux Data

Acquisition Types
=================

.. image:: /_static/user_guide/experiment/startpage.png
   :align: center
   :width: 800
   :target: ../../_images/startpage.png
   :alt: Experiment start page

.. todo:: Capture screenshot at ``doc/source/_static/user_guide/experiment/startpage.png``.

When starting a new experiment, the first page of the wizard allows for selecting the desired type of the FTMW acquisition along with other options related to real-time FID processing.
The type of the acquisition determines how many FIDs will be recorded, and in some cases, how the entire spectrum will be partitioned into multiple **segments**.
For the LO Scan and DR Scan types, the scan configuration controls appear inline on this same first wizard page immediately below the type selector; no separate page is shown for those settings.

The page is divided into a ``Common Settings`` group at the top, which holds the ``Aux Data Interval`` and ``Backup Interval`` boxes that apply to every acquisition type, and an ``FTMW`` group below it containing the type selector and the type-specific controls.
When LIF is enabled, an ``LIF`` group is shown alongside the ``FTMW`` group with its own delay and laser scan controls.

Available acquisition types are:

- :ref:`user_guide/experiment/acquisition_types:Target Shots`
- :ref:`user_guide/experiment/acquisition_types:Target Duration`
- :ref:`user_guide/experiment/acquisition_types:Forever`
- :ref:`user_guide/experiment/acquisition_types:Peak Up`
- :ref:`user_guide/experiment/acquisition_types:LO Scan`
- :ref:`user_guide/experiment/acquisition_types:DR Scan`

Real-time processing includes phase correction (adjusting for slow temporal phase drift during long averages) and/or chirp scoring, which attempts to reject "weak" chirps.

Target Shots
------------

**Target Shots** is the default and most straightforward type of acquisition.
In this mode, FIDs are recorded and averaged until the desired number of shots (as indicated in the ``Shots`` box) is reached.
This is considered a **single segment** acquisition: only a single Rf/Clock configuration is used during the experiment.

Target Duration
---------------

**Target Duration** mode records and averages FIDs until a period of time has elapsed.
The desired acquisition duration is specified in the ``Duration`` box, and an estimate of the experiment completion time is provided in the text below the box.

.. note::
   The estimated completion time is computed simply by adding the duration to the current time, and does not account for the time spent configuring the remainder of the experiment and initializing.
   The actual experiment timer begins after the experiment has been successfully initialized.

This mode is also a **single segment** acquisition.

Forever
-------

In **Forever** mode, FIDs are recorded and averaged until the user hits the abort button or a critical hardware failure occurs.
No acquisition-objective field is shown when this mode is selected.
Like Target Shots and Target Duration, this is a **single segment** acquisition.

.. note::
   Blackchirp supports averaging up to :math:`2^{64} - 1 = 18446744073709551615` shots. However, each FID data point is stored as the sum of raw digitizer counts and therefore overflow of an FID data point may occur before this limit is reached. See :ref:`user_guide/data_storage:FID CSV Files` for more details.

Peak Up
-------

**Peak Up** mode acquires and averages FIDs until a target number of shots is reached, but does not stop at that point.
Processing continues as a `modified moving average <https://en.wikipedia.org/wiki/Moving_average#Modified_moving_average>`__, which is a special case of an exponentially-weighted moving average.
The average can be reset and the number of shots included in the average can be changed during the acquisition with the controls on the :ref:`user_guide/cp-ftmw:CP-FTMW Tab`.

.. warning::
   Unlike the other acquisition modes, **Peak Up mode experiments are not saved to disk.** Once a new experiment has been started, the results from the Peak Up mode experiment are discarded. Be sure to export any data you wish to save manually!

Another key difference between Peak Up mode and the other acquisition modes is that most program controls (hardware, settings, etc) remain unlocked and can be modified without stopping the acquisition.
For example, gas flow rates or pulse timings may be modified and the results monitored in real time.

.. warning::
   In certain situations (e.g., when a delay generator produces a protection pulse for a switch), it may be possible for the user to enter settings that are inappropriate. Use caution when changing settings during a Peak Up mode acquisition.

.. note::
   Under the hood, Blackchirp applies an 8-bit `left shift <https://en.wikipedia.org/wiki/Bitwise_operation#Bit_shifts>`__ to all digitizer values during peak-up mode, equivalent to multiplying each ADC value by 256. This creates extra bits that are needed for the rolling average operation.

LO Scan
-------

The **LO Scan** mode implements a version of `segmented CP-FTMW spectroscopy <https://doi.org/10.1364/OE.21.019743>`__ in which a certain number of FIDs are acquired and averaged, then the upconversion and/or downconversion local oscillators are stepped.
By sampling a variety of LO frequencies, the spectral range covered by the instrument can far exceed the instantaneous bandwidth of the FTMW digitizer, allowing CP-FTMW spectroscopy to be performed with (comparatively) inexpensive hardware.
On the :ref:`user_guide/cp-ftmw:CP-FTMW Tab`, each frequency segment can be viewed individually, and algorithms are available for stitching together the entire spectrum and deconvolving dual sidebands (see the linked page for more details about these algorithms).
As a **multi-segment** acquisition type, Blackchirp writes a backup at each segment boundary, so the ``Backup Interval`` setting in Common Settings has no effect.
The number of shots collected at each LO step is set in the ``Shots/Point`` box within the LO scan configuration below the type selector.

.. note::
   Before the LO Scan configuration controls become accessible, both an ``UpLO`` and a ``DownLO`` clock source must be assigned in the :ref:`user_guide/hardware_menu:Rf Configuration` (or, if the ``Common LO`` box is checked, only the ``UpLO`` needs to be set). The exact frequencies entered for those clocks are unimportant, as they are overridden by the scan parameters.

.. image:: /_static/user_guide/experiment/loscan.png
   :align: center
   :alt: LO Scan configuration

.. todo:: Capture screenshot at ``doc/source/_static/user_guide/experiment/loscan.png``.

The LO scan widget is laid out as three stacked groups: ``Scan Settings``, ``Upconversion LO``, and ``Downconversion LO``.
The ``Scan Settings`` group contains the ``Shots/Point`` and ``Target Sweeps`` boxes.
Blackchirp acquires the indicated number of ``Shots/Point`` at each LO frequency specified.
The complete set of LO frequencies is considered one **sweep**, and Blackchirp returns to the beginning LO configuration and resumes integrating until the desired number of ``Target Sweeps`` is reached.
When Blackchirp repeats a segment, the new FIDs are automatically averaged together with the ones from the previous sweep(s).

If the ``UpLO`` and ``DownLO`` correspond to the same hardware output, the entire ``Downconversion LO`` group is disabled.

Blackchirp divides the scan range into major and minor steps.
Essentially, at each major step, a series of measurements can be made with slightly shifted LO positions to assist with sideband deconvolution.
The LO is incremented by the minor step size for each minor step associated with the indicated major step.
In the screenshot above, the Upconversion LO covers the total range of 1008 MHz in 5 major steps plus 3 minor steps per major step.
Given the minor step size of 4 MHz, the major steps work out to 11520, 11770, 12020, 12270, and 12520 MHz.
At each of these major step values, 3 minor steps are recorded, separated by 4 MHz: 11520, 11524, 11528, 11770, 11774, 11778, ..., 12520, 12524, 12528 MHz, for a total of 15 (3x5) steps.
As the ``Start``, ``End``, and major/minor steps boxes are adjusted, some of the other boxes' values are updated to ensure the range is consistent.

The Downconversion LO is configured similarly.
As the number of major or minor steps is changed, the Downconversion LO step counts and step sizes are also adjusted to ensure the number of steps matches the Upconversion LO and that the step sizes are consistent with the indicated range.
Two checkboxes at the bottom of the ``Downconversion LO`` group modify this behavior.
If the ``Fixed Frequency`` box is checked, the Downconversion LO is not changed during the acquisition; its frequency remains at the ``Start`` value, and all other Downconversion LO boxes are disabled.
If the ``Constant Offset`` box is instead checked, the Downconversion LO frequency at each step is set to maintain the same difference between the Downconversion LO and Upconversion LO start frequencies.
For example, in the screenshot above the Downconversion LO is offset from the Upconversion LO by :math:`40960-11520 = 29440` MHz, and that difference is held constant as the Upconversion LO is stepped through its values.

DR Scan
-------

In a **DR Scan** (double resonance), FIDs are recorded while a second ``DR Clock`` source is scanned across a desired frequency range in a series of steps.
The purpose of this scan mode is to monitor the intensity of one or more transitions as a function of the DR Clock frequency: when two transitions share a common state, the intensity of a line may be depleted or enhanced, depending on the pulse powers and timings.
Like the LO Scan mode, this is a **multi-segment** acquisition mode.
Blackchirp writes a backup at each segment boundary, so the ``Backup Interval`` setting in Common Settings has no effect.
The number of shots at each DR step is set in the ``Shots Per Step`` box within the DR scan configuration below the type selector.

.. note::
   A physical clock source must be assigned to the ``DR Clock`` role in the :ref:`user_guide/hardware_menu:Rf Configuration` before DR Scan parameters can be configured.

.. image:: /_static/user_guide/experiment/drscan.png
   :align: center
   :alt: DR Scan configuration

.. todo:: Capture screenshot at ``doc/source/_static/user_guide/experiment/drscan.png``.

To set up a DR scan, enter the desired ``Start`` frequency, ``Step Size``, ``Num Steps``, and ``Shots Per Step``.
The ``End`` box updates automatically to show the final DR frequency.
When viewing the DR Scan on the :ref:`user_guide/cp-ftmw:CP-FTMW Tab`, peak intensity as a function of DR frequency is not directly displayed.
Instead, it is recommended to use the ``FT1 - FT2`` plot mode, as discussed in more detail on that page.

Other Experiment Options
------------------------

In addition to the acquisition type, other options accessible on the first wizard page involve real-time FID processing (phase correction and chirp scoring) and the Common Settings group (auxiliary data and backups; backups are also discussed on the :ref:`user_guide/data_storage:Data Storage` page).

Phase Correction
................

Time-domain averaging requires that FIDs are mutually in phase with one another.
This is typically achieved by locking all oscillators to a common reference frequency generated by a stable source such as a rubidium clock.
However, in some cases, over the course of a long acquisition, some phase drift may occur (we discovered this one winter day at Harvard when a lab door to the outside was open for 30 minutes for moving equipment out, and the intensity of an OCS signal steadily decreased).
To mitigate this potential case, Blackchirp has a "phase correction" algorithm that attempts to correct for slow phase drift.
This algorithm can be enabled by checking the ``Phase Correction`` box.

In order to work, the FID record from the digitizer must contain the chirp, and the chirp should not be saturated on the scale of the digitizer.
Often the leakage through a switch is sufficient, but we have also used directional couplers and/or SPDT switches with some success to allow an attenuated version of the chirp to bypass a diode limiter/protection switch combination.
Blackchirp attempts to determine where the chirp is located within the wavefunction (more on this below), and then computes the cross correlation between the chirp in the new record with the current average chirp after at least 20 FIDs have been recorded.
This calculation is performed with trial shifts of up to a few points in time to determine the maximum.
Usually the shift is 0, but over time a nonzero shift may accumulate due to, e.g., temperature fluctuations.
Blackchirp attempts to be conservative, only changing the current shift if there is a significant improvement in the computed figure of merit.
If the shift exceeds 50 points, an error is thrown.

If the ``Chirp Start`` box is set to ``Automatic``, Blackchirp assumes that the digitizer is triggered at the start of the protection pulse.
The chirp start is then set to the sum of the pre-chirp protection and pre-chirp delay settings entered on the :ref:`user_guide/ftmw_configuration/chirp_setup:Chirp Setup` page, and the chirp end is determined from the chirp duration entered on that page.
These computed values will be displayed on the FID plots displayed on the :ref:`user_guide/cp-ftmw:CP-FTMW Tab`.
If they are not correct, you can override the chirp starting time by entering the actual chirp start you observe into the ``Chirp Start`` box.
The duration still comes from the :ref:`user_guide/ftmw_configuration/chirp_setup:Chirp Setup` value.

Chirp Scoring
.............

Occasionally, amplifiers may show significant shot-to-shot jitter in the chirp amplitude.
By enabling the ``Chirp Scoring`` feature, Blackchirp will compute the squared sum of the chirp embedded in the FID record (as described above) and compare it to the squared sum of the averaged chirp.
If the new value falls below the fraction of the averaged chirp value indicated in the ``Chirp Threshold`` box, the FID is rejected.
Higher values of the threshold result in more strict acceptance criteria.
Similar to the Phase Correction algorithm, the chirp scoring routine requires that the chirp be visible and unsaturated within the digitizer record, and its position may be manually adjusted with the ``Chirp Start`` box.

.. warning::
   Setting the chirp threshold too high will result in a large fraction of chirps being rejected, and an acquisition may therefore appear to stall.

Common Settings
...............

The ``Common Settings`` group at the top of the wizard page contains two boxes that apply to every acquisition type.

The ``Aux Data Interval`` box sets the period between :doc:`Aux Data readings <../rolling-aux-data>`.
More frequent readings increase data storage requirements but provide more regular opportunities to automatically abort an acquisition using one of the :doc:`validation conditions <validation>`.

The ``Backup Interval`` box sets how often Blackchirp writes a backup copy of the experiment to disk during a single-segment acquisition.
Setting the box to its minimum value displays ``Disabled`` and turns backups off.
Multi-segment acquisition types (LO Scan and DR Scan) write a backup at each segment boundary regardless of this setting.
