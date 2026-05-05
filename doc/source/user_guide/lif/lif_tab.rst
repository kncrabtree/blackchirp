.. index::
   single: LIF Tab
   single: LIF Display
   single: LIF Time Trace
   single: LIF Slice Plot; delay
   single: LIF Slice Plot; laser
   single: LIF 2D Spectrogram
   single: LIF Processing Panel
   single: Refresh Interval; LIF
   single: Savitzky-Golay Filter; LIF
   single: Low Pass Filter; LIF
   single: Reprocess; LIF
   single: processing.csv

.. _lif-tab:

LIF Display Tab
===============

The **LIF** tab provides real-time and post-acquisition visualization of LIF
data. It is visible in the main window whenever the LIF module is enabled
(see :ref:`application-config`).

.. image:: /_static/user_guide/lif/lif_tab.png
   :align: center
   :width: 800
   :alt: LIF Display tab showing time trace, slice plots, 2D spectrogram, and processing panel

Plot areas
----------

The tab contains four plot areas:

**Time trace (upper left)**
   Displays the digitizer waveform for the scan point selected on the 2D
   spectrogram. During an acquisition it updates at the rate set by the
   :ref:`refresh interval <lif-tab-refresh>` control, showing the accumulated
   average waveform at the current scan position. Colored regions mark the
   LIF and reference integration gates. Both the LIF channel waveform and
   the reference channel waveform (when enabled) are drawn on the same plot.

**Delay slice (upper right)**
   Plots integrated LIF signal as a function of delay time, evaluated at the
   laser-frequency column currently selected on the 2D spectrogram.

**Laser slice (upper right, second panel)**
   Plots integrated LIF signal as a function of laser position, evaluated at
   the delay row currently selected on the 2D spectrogram.

**2D spectrogram (lower right)**
   Renders the full two-dimensional dataset as a false-color map with delay
   on one axis and laser frequency on the other. Each cell represents one
   scan point; its color encodes the integrated LIF signal at that point.
   The color scale updates automatically as new data arrive.

Selecting a scan point on the 2D spectrogram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The 2D spectrogram carries two cursors — a horizontal line at the selected
delay and a vertical line at the selected laser position — that drive both
slice plots and the time trace. The selected point can be moved in two ways:

- **Drag a cursor.** Click and drag the horizontal cursor to change the
  delay selection (updating the laser slice and the time trace), or the
  vertical cursor to change the laser-frequency selection (updating the
  delay slice and the time trace).
- **Right-click for the context menu.** Right-click anywhere on the
  spectrogram and choose **Move delay cursor here**, **Move frequency
  cursor here**, or **Move both cursors here** to jump the corresponding
  cursor(s) to the click location. The same menu has a **Follow live data**
  entry that re-locks both cursors to the most recent live acquisition
  point.

Dragging a cursor or invoking either of the move actions detaches the
display from live-following; choose **Follow live data** to re-attach.

Processing panel
----------------

The processing panel sits on the left side of the tab and exposes the
integration gate positions, optional smoothing filters, and the
post-acquisition workflow buttons. Changes to gate or filter values take
effect immediately on the displayed plots.

**LIF Gate Start / LIF Gate End**
   Sample-point indices defining the integration window for the LIF channel.
   The gate is applied to the running accumulated waveform before computing
   the integrated value. Units are digitizer sample points; hold Ctrl while
   scrolling to adjust in steps of 10. See :ref:`lif-configuration` for the
   relationship between sample points and time.

**Reference Gate Start / Reference Gate End**
   Sample-point indices for the reference channel gate. Active only when the
   reference channel is enabled in the :doc:`configuration`.

**Low Pass Filter Alpha**
   Applies a single-pole IIR low-pass filter to each waveform before
   integration:

   .. math::

      x_n = \alpha \, x_{n-1} + (1 - \alpha) \, x_n

   Setting alpha to 0 (the special value displayed as **Disabled**) bypasses
   the filter. Higher values increase smoothing at the cost of temporal
   resolution within the waveform.

**Savitzky-Golay Filter Enabled**
   When checked, applies a `Savitzky-Golay
   <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`_
   polynomial smoothing filter to each waveform before integration.

**Savitzky-Golay Window**
   Window size for the Savitzky-Golay filter. Must be an odd number and at
   least 3.

**Savitzky-Golay Polynomial Order**
   Polynomial order for the Savitzky-Golay filter. Must be at least 2 and
   strictly less than the window size.

The three buttons at the bottom of the processing panel operate on the
stored data and on ``processing.csv``:

- **Reprocess All** — re-integrates every stored waveform using the current
  gate and filter settings, then redraws all plots. Available after the
  acquisition completes.
- **Reset** — restores the gate and filter settings to the values most
  recently saved to ``processing.csv``.
- **Save** — writes the current gate and filter settings to
  ``lif/processing.csv`` in the experiment folder. These become the default
  values applied when the experiment is opened in the future.

.. note::
   The recorded time-trace data on disk are never modified. Each scan
   point's averaged waveform is written during acquisition and remains
   untouched thereafter; **Save** only updates ``processing.csv`` with the
   new default gate and filter values for that experiment. **Reprocess
   All** likewise re-integrates the stored waveforms in memory and updates
   the plots, without rewriting the trace files.

.. _lif-tab-refresh:

Refresh interval
----------------

The **Refresh Interval** spin box sets how frequently the plots are updated
during an acquisition, in milliseconds. Shorter intervals give a more
responsive display at the cost of additional processing overhead. The plots
are also redrawn whenever a processing setting changes. Multiple refresh
requests that arrive while a previous update is still being processed are
coalesced; only the most recent request is acted upon.
