.. index::
   single: FTMW
   single: Digitizer
   single: FID Processing
   single: Segment
   single: Frame
   single: Backup
   single: Window Function
   single: Zero Padding
   single: Peak Finding
   single: Peak-up Mode
   single: Sideband Deconvolution

CP-FTMW Tab
===========

.. image:: /_static/user_guide/ui_overview/cp_ftmw.png
   :align: center
   :width: 800
   :alt: FTMW Digitizer setup


The CP-FTMW tab allows for visualization and processing of FID and Fourier Transform data. An example of the tab is shown above. During an acquisition, 7 plots are visible. The uppermost pair labeled "FID Live" and "FT Live" always display the most current data, and these plots are removed when the acquisition completes. The pairs labeled "FID 1"/"FT 1" and "FID 2"/"FT 2" display user-controllable views of the spectrum (discussed firther below). Most user interaction takes place with the large "Main FT" plot. By default, the Main FT plot is configured to display the "Live FT" which shows the data currently being collected. When an experiment ends, the plot shows the contents of the "FT 1" plot. Interaction with the plot (zooming, panning, peak finding, etc) can be done during and after an acquisition. More details are available on the `Plot Controls <plot_controls.html>`_ page.

The main toolbar at the top allows access to various data processing settings which are discussed in the sections of the page below. Finally, the "Refresh Interval" box controls how frequently Blackchirp updates the plots on the screen during an acquisition. The processing is queued and occurs in a separate thread from the real-time averaging. In addition, plots are updated anytime a setting affecting the data shown on the plot is adjusted by the user. If multiple update requests are made while Blackchirp is processing an FT, Blackchirp will discard all but the most recent request.

FID Processing Settings
.......................

By clicking on the "FID Processing Settings" button, a toolbar will appear with various settings that can alter processing and/or display of the CP-FTMW data. These settings are applied to the data displayed on all plots. These are:

* ``FT Start``: The starting time for the data to be Fourier transformed. Points before this time are zeroed out. This setting is useful when the digitizer is triggered prior to the excitation chirp. In this case, the FT Start can be set to 0 to view the chirp and monitor phase coherence. The FT start can also be adjusted to exclude signals from ringing of the excitation pulse of switch bounce.
* ``FT End``: The ending time for the FT. Points after this time are zeroed out. Can be useful in conjunction with FT start to assess the T2 relaxation time of the FID.
* ``Exp Filter``: Time constant for an exponential filter applied to the FID prior to the FT. Matching the filter time constant to the natural T2 relaxation time provides noise suppression.
* ``VScale Ignore``: A frequency range near the LO to ignore when computing the autoscale range. Often CP-FTMW data have large uninteresting signals near DC, and this setting prevents those from overwhelming the default vertical scale.
* ``Zero Pad``: Appends zeros to the FID to artificially increase the digital resolution of the FT. A setting of 1 appends zeros until the length of the array is double the next power of two. For example, for a record length of 750,000 points, the next power of two is 2^20, or 1,048,576 points. With zero pad = 1, zeros are appended until the data length is 2^21, or 2,097,152 points. Each subsequent increase of the zero pad setting increases the length by another factor of 2.
* ``Remove DC``: Subtracts the average value of the FID prior to the Fourier transform. Removes large-envelope DC artifacts.
* ``Window Function``: Applies a window function prior to Fourier transformation. Window functions are useful for suppressing spectral leakage from strong signals, which tend to obscure nearby weaker transitions. A window function cuts down on these sidelobes at the expense of reducing the signal-to-noise ratio slightly and decreasing the spectral resolution. See the `Data Storage <data_storage.html#processing-csv>`_ page for the definitions of the window functions implemented in Blackchirp.
* ``FT units``: Changes the vertical scaling of the FT.
* ``Reset``: Restores processing settings to the most recently-saved values.
* ``Save``: Writes current processing settings to a processing.csv file. By default, processing settings are written when an experiment first starts, but may be overwritten at any time.

The FID plot shows the post-processed FID data.

Plot Settings
.............

Clicking on the "Plot Settings" button opens a menu which controls what data are displayed on the various plots on the CP-FTMW tab. The menu is organized into three sections: one for the main plot, then one each for plots FT1 and FT2.

For the main plot, the primary control is the Mode selection box, which controls the data displayed on the main plot. The available options are:

* ``Live``: Main plot shows the data on the "Live" set of plots. For acquisition modes that involve changing the clock settings (LO Scan, DR Scan), the main plot will follow the current acquisition settings as they change. At the end of an acquisition, this option is disabled and the setting is changed to FT1 if Live was selected.
* ``FT1``: Main plot shows the data selected for display on the FT1 plot, which includes its segment, frame, and backup options.
* ``FT2``: Main plot shows the data selected for display on the FT2 plot, which includes its segment, frame, and backup options.
* ``FT1_minus_FT2``: Main plot shows the result of subtracting FT2 from FT1.
* ``FT2_minus_FT1``: Main plot shows the result of subtracting FT1 from FT2.
* ``Upper Sideband``: Only available in LO Scan mode. Performs sideband deconvolution using only the higher-frequency sideband.
* ``Lower Sideband``: Only available in LO Scan mode. Performs sideband deconvolution using only the lwoer-frequency sideband.
* ``Both Sidebands``: Only available in LO Scan mode. Performs sideband deconvolution using both sidebands.

In addition to the mode selection box, in LO scan mode an additional "Sideband Processing" menu is available. These settings are discussed in the `Sideband Deconvolution`_ section below.

For Plot 1 and Plot 2, the segment, frame, and backup boxes allow for selection of different data to be shown in the FT1 and FT2 plots, respectively. The meanings are:

* ``Segment``: For acqusition modes which involve multiple different hardware settings in a single experiment (e.g., LO scan, DR scan), each individual hardware setting is associated with a "Segment." The nomenclature comes from segmented CP-FTMW spectroscopy, which is implemented as an LO Scan in Blackchirp. By changing the segment box, the indicated plot would show the data associated with each individual LO tuning in such a scan.
* ``Frame``: For "Multiple Record" acquisitions (see the `Digitizer Setup <experiment/digitizer_setup.html>`_ page for more detail), this box controls which individual record is displayed, indexed starting from 1. With a value of 0 (default), the box will display the word "Average" and Blackchirp will coaverage the individual records.
* ``Backup``: For long acquisitions in which backups are enabled, the backup box will display the FID and FT associated with each backup checkpoint.

Peak Up Options
...............

During a peak-up mode acquisition, the number of averages can be changed on-the-fly, and the current average can be reset using the options in this menu.

Peak Find
.........

.. image:: /_static/user_guide/ui_overview/peakfind.png
   :align: center
   :width: 800
   :alt: FTMW Digitizer setup

The Peak Find menu has an implementation of a rough peak finding algorithm. In the peak finding routine, the FT data is run through a `Savitsky-Golay filter <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`_ which returns the second derivative of a smoothed version of the FT, determined by the window size (which must be odd) and a polynomial order which is used to fit the points within the window (must be less than the window size). A peak is identified when a 5-point local minimum in the second derivative is located and the corresponding point in the FT is at least SNR times an estimate of the local noise level.

.. note::
   This peak finding algorithm works reasonably well for windowed data, but often finds many false positives in the absence of a window function in the vicinity of strong signals with significant spectral leakage.


.. note::
   Significant improvements to analysis algorithms are envisioned in the future.

The export menu allows for the peak find list to be exported to a CSV file or an FTB file, the latter of which is an input for the cavity FTMW software `QtFTM <https://github.com/kncrabtree/qtftm>`_.


Sideband Deconvolution
......................

.. image:: /_static/user_guide/ui_overview/sideband_processing.png
   :align: center
   :width: 800
   :alt: FTMW Digitizer setup

The sideband deconvolution algorithm employed by Blackchirp is designed to suppress image frequencies in an LO scan. Most segmented LO scanning spectrometers employ a low-frequency chirp which is mixed up to the target frequency via a tunable LO. This leads to two simultaneous chirps: one at the LO frequency + chirp frequency and the other at the LO frequency - chirp frequency. If both of these are within the bandwidth of the amplifier, then the sample experiences both chirps simultaneously, yielding molecular FID signals in both windows. However, upon downconversion with a second mixer, both of these sidebands are downconverted to the same range of frequencies, so each downconverted frequency in the FT may correspond to either of the two sidebands. This uncertainty is eliminated by tuning the LO frequency slightly and observing which "direction" the signal moves relative to the LO.

In Blackchirp, the sideband deconvolution algorithms are based on overlapping frequency-shifted versions of the FT onto a common frequency grid. Becasue the spectra are acquired at different LO tunings, the frequency bins for each FT may not perfectly align. Blackchirp computes a global frequency grid spanning all sidebands and uses linear interpolation to resample all FTs onto that grid.

Consider the simple case of an LO frequency of 10 GHz and a signal observed at 500 MHz in the FT (with a digitizer and chirp bandwidth of 1 GHz). This may correspond to a molecular frequency of either 9.5 or 10.5 GHz. Next, increase the LO frequency by 100 MHz to 10.1 GHz. If the molecular frequency is 10.5 GHz, the new frequency observed by the digitizer is 400 MHz, while if it is 9.5 GHz, then the new digitizer frequency is 600 MHz. In the "Upper Sideband" deconvolution algorithm, it is assumed that all molecular emission occurs in the higher-frequency sideband. In this case, Blackchirp would compute 2 FTs for the two LO tunings: one spanning 10-11 GHz, and the other spanning 10.1-11.1 GHz. Blackchirp aligns these two tunings and coaverages the spectra where they overlap. In both cases, the signal appears at an apparent frequency of 10.5 GHz, so the signal adds.

However, in the "Lower Sideband" algorithm, Blackchirp would assign the frequency axes as 10.0-0.0 and 10.1-9.1 GHz, respectively. Because the true molecular frequency was 10.5 GHz, the signal which appeared at a 9.5 GHz apparent frequency appears with an apparent frequency of 9.7 GHz (10.1 GHz - 0.4 GHz) in the second LO tuning. Coaveraging these two spectra attenuates the signal.

In "Both Sidebands" mode, both sideband deconvolutions are computed and a composite spectrum is created by concatenating their respective frequency axes. This mode has the additional benefit of providing additional averages when the same frequency is covered in both sidebands as the LO is tuned over a broad range.

.. warning::
   If the effective sensitivity of the two sidebands is very different (which could be caused by variable mixer efficiency or by choosing LO tunings too close to the limits of the amplifier bandwidth), then "Both Sidebands" mode could result in artificial signal suppression.

Various processing options are available for controlling details of the deconvolution algorithm:

* ``Frame``: If multiple FIDs are acquired per pulse, this box controls which frame is shown. Setting this to 0 will cause Blackchirp to average all frames.
* ``Min Offset``: Minimum offset frequency (relative to the LO frequency) of the FT to include when processing each sideband. By default, this should be set to the minimum chirp frequency relative to the LO. Blackchirp calculates this value automatically from the Rf Configuration and Chirp Configuration.
* ``Max Offset``: Maximum offset frequency (relative to the LO frequency) of the FT to include when processing each sideband.
* ``Avg Algorithm``: The coaveraging algorithm to use when "overlapping" the frequency-shifted spectra. Blackchirp does not calculate the arithmetic mean of the spectra; this would provide very poor image suppression. The available options are:

   - ``Harmonic Mean``: A shots-weighted harmonic mean, which is a measure of central tendency which  is strongly biased toward the lowest value in the set. This is desirable for sideband suppression, as we wish for the spectrum to average strongly toward 0 if a line is present in only one of the shifted spectra. For this reason, it is the default algorithm. Let s\ :sub:`1` and s\ :sub:`2` be the numbers of shots for the two data points y\ :sub:`1` and y\ :sub:`2`, respectively. Assuming all samples and shots are positive and nonzero, the weighted harmonic mean is:

   .. math::
      y_{\text{avg}} = \frac{s_1 + s_2}{\frac{s_1}{y_1} + \frac{s_2}{y_2}}

   - ``Geometric Mean``: A shots-weighted geometric mean, which falls between the harmonic and arithmetic means. The weighted geometric mean is:

   .. math::
      y_{\text{avg}} = \exp\left(\frac{s_1\ln y_1 + s_2\ln y_2}{s_1 + s_2}\right)
