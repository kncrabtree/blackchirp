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

The CP-FTMW tab allows for visualization and processing of FID and Fourier Transform data. An example of the tab is shown above. In this image, there are a total of 5 plots: one pair called "FID 1" and "FT 1", another called "FID 2" and "FT 2" and a larger "Main FT" plot. During an active acquisition, an additional pair of plots labeled "FID Live" and "FT Live" are displayed at the top of the tab.

Most user interaction takes place with the "Main FT" plot. By default, the Main FT plot is configured to display the "Live FT" which shows the data currently being collected. When an experiment ends, the plot shows the contents of the "FT 1" plot. Interaction with the plot (zooming, panning, peak finding, etc) can be done during and after an acquisition. More details are available on the `Plot Controls <plot_controls.html>`_ page.

FID Processing Settings
.......................

By clicking on the "FID Processing Settings" button, a toolbar will appear with various settings that can alter processing and/or display of the CP-FTMW data. These settings are applied to the data displayed on all plots. These are:

* ``FT Start``: The starting time for the data to be Fourier transformed. Points before this time are zeroed out. This setting is useful when the digitizer is triggered prior to the excitation chirp. In this case, the FT Start can be set to 0 to view the chirp and monitor phase coherence. The FT start can also be adjusted to exclude signals from ringing of the excitation pulse of switch bounce.
* ``FT End``: The ending time for the FT. Points after this time are zeroed out. Can be useful in conjunction with FT start to assess the T2 relaxation time of the FID.
* ``Exp Filter``: Time constant for an exponential filter applied to the FID prior to the FT. Matching the filter time constant to the natural T2 relaxation time provides noise suppression.
* ``VScale Ignore``: A frequency range near the LO to ignore when computing the autoscale range. Often CP-FTMW data have large uninteresting signals near DC, and this setting prevents those from overwhelming the default vertical scale.
* ``Zero Pad``: Appends zeros to the FID to artificially increase the digital resolution of the FT. A setting of 1 appends zeros until the length of the array is double the next power of two. For example, for a record length of 750,000 points, the next power of two is 2^20, or 1,048,576 points. With zero pad = 1, zeros are appended until the data length is 2^21, or 2,097,152 points. Each subsequent increase of the zero pad setting increases the length by another factor of 2.
* ``Remove DC``: Subtracts the average value of the FID prior to the Fourier transform. Removes large-envelope DC artifacts.
* ``Window Function``: Applies a window function prior to Fourier transformation. Window functions are useful for suppressing spectral leakage from strong signals, which tend to obscure nearby weaker transitions. A window function cuts down on these sidelobes at the expense of reducing the signal-to-noise ratio slightly and decreasing the spectral resolution.
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

In addition to the mode selection box, in LO scan mode, the other three boxes may become available. The "Follow" box is enabled when one of the sideband deconvolution modes is selected. The indicated plot is used to retrieve the frame and backup (is applicable) for the sideband deconvolution routine. The min and max offset boxes control the range of frequencies from the individual segments which are sent into the sideband deconvolution process. It is advisable to set the minimum offset high enough to ignore undesired signals near the LO frequency, and set the maximum offset less than or equal to the bandwidth of the digitizer.

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

The sideband deconvolution algorithm employed by Blackchirp is designed to suppress image frequencies in an LO scan. Most segmented LO scanning spectrometers employ a low-frequency chirp which is mixed up to the target frequency via a tunable LO. This leads to two simultaneous chirps: one at the LO frequency + chirp frequency and the other at the LO frequency - chirp frequency. If both of these are within the bandwidth of the amplifier, then the sample experiences both chirps simultaneously, yielding molecular FID signals in both windows. However, upon downconversion with a second mixer, both of these sidebands are downconverted to the same range of frequencies, so each downconverted frequency in the FT may correspond to either of the two sidebands. This uncertainty is eliminated by tuning the LO frequency slightly and observing which "direction" the signal moves relative to the LO.

In Blackchirp, the sideband deconvolution algorithms are based on computing the geometric mean of frequency-shifted versions of the FT. Consider the simple case of an LO frequency of 10 GHz and a signal observed at 500 MHz in the FT (with a digitizer and chirp bandwidth of 1 GHz). This may correspond to a molecular frequency of either 9.5 or 10.5 GHz. Next, increase the LO frequency by 100 MHz to 10.1 GHz. If the molecular frequency is 10.5 GHz, the new frequency observed by the digitizer is 400 MHz, while if it is 9.5 GHz, then the new digitizer frequency is 600 MHz. In the "Upper Sideband" deconvolution algorithm, it is assumed that all molecular emission occurs in the higher-frequency sideband. In this case, Blackchirp would compute 2 FTs for the two LO tunings: one spanning 10-11 GHz, and the other spanning 10.1-11.1 GHz. Blackchirp aligns these two tunings and coaverages the spectra where they overlap. In both cases, the signal appears at an apparent frequency of 10.5 GHz, so the signal adds.

However, in the "Lower Sideband" algorithm, Blackchirp would assign the frequency axes as 10.0-0.0 and 10.1-9.1 GHz, respectively. Because the true molecular frequency was 10.5 GHz, the signal which appeared at a 9.5 GHz apparent frequency appears with an apparent frequency of 9.7 GHz (10.1 GHz - 0.4 GHz) in the second LO tuning. Coaveraging these two spectra attenuates the signal.

Importantly, Blackchirp employs a geometric mean algorithm rather than an arithmetic mean. The geometric mean is the Nth root of the product of N samples. In the simplified Lower Sideband case above, when coaveraging, at 9.5 GHz one spectrum would have a positive signal and the other would be 0, yielding a geometric mean of 0, thereby suppressing the signal in the undesired sideband. In reality, the signal is never truly 0 and there is always the chance of a coincidental overlap of molecular signals (especially for rich spectra), and it is therefore desirable to use more than 2 LO tunings to ensure good suppression of undesired signals.

.. note::
   While currently Blackchirp employs a geometric mean for image suppression, a case can be made that a harmonic mean may provide more effective suppression at the potential expense of true signal attenuation. In the future, Blackchirp may provide both as an option.

Finally, in "Both Sidebands" mode, both sideband deconvolutions are computed and a composite spectrum is created by concatenating their respective frequency axes. This mode has the additional benefit of providing additional averages when the same frequency is covered in both sidebands as the LO is tuned over a broad range.

.. warning::
   If the effective sensitivity of the two sidebands is very different (which could be caused by variable mixer efficiency or by choosing LO tunings too close to the limits of the amplifier bandwidth), then "Both Sidebands" mode could result in artificial signal suppression.
