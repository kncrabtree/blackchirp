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

* **FT Start**: The starting time for the data to be Fourier transformed. Points before this time are zeroed out. This setting is useful when the digitizer is triggered prior to the excitation chirp. In this case, the FT Start can be set to 0 to view the chirp and monitor phase coherence. The FT start can also be adjusted to exclude signals from ringing of the excitation pulse of switch bounce.
* **FT End**: The ending time for the FT. Points after this time are zeroed out. Can be useful in conjunction with FT start to assess the T2 relaxation time of the FID.
* **VScale Ignore**: A frequency range near the LO to ignore when computing the autoscale range. Often CP-FTMW data have large uninteresting signals near DC, and this setting prevents those from overwhelming the default vertical scale.
* **Zero Pad**: Appends zeros to the FID to artificially increase the digital resolution of the FT. A setting of 1 appends zeros until the length of the array is double the next power of two. For example, for a record length of 750,000 points, the next power of two is 2^20, or 1,048,576 points. With zero pad = 1, zeros are appended until the data length is 2^21, or 2,097,152 points. Each subsequent increase of the zero pad setting increases the length by another factor of 2.
* **Remove DC**: Subtracts the average value of the FID prior to the Fourier transform. Removes large-envelope DC artifacts.
* **Window Function**: Applies a window function prior to Fourier transformation. Window functions are useful for suppressing spectral leakage from strong signals, which tend to obscure nearby weaker transitions. A window function cuts down on these sidelobes at the expense of reducing the signal-to-noise ratio slightly and decreasing the spectral resolution.
* **FT units**: Changes the vertical scaling of the FT.

The FID plot shows the post-processed FID data.

Plot Settings
.............


