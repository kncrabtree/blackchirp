.. index::
   single: LIF Module; configuration
   single: LIF Channel
   single: Reference Channel; LIF
   single: Integration Gate; LIF
   single: Laser Power Normalization
   single: LIF Digitizer
   single: LIF Configuration
   single: Laser Control; LIF
   single: Flashlamp; LIF

.. _lif-configuration:

LIF Configuration
=================

The **LIF Configuration** dialog collects the LIF digitizer settings,
laser controls, and processing parameters that define an LIF
measurement. The same controls surface in two places:

- As the **LIF Configuration** page of the :doc:`Experiment Setup
  dialog </user_guide/experiment_setup>`, where they are reviewed and
  validated before an experiment starts. This is the path most users
  take during routine acquisition.
- As a standalone dialog opened from **Hardware → LIF Configuration**.
  Open it this way to tune the digitizer, run a live preview, or
  exercise the laser between experiments. Values set in one context
  are persisted and visible in the other.

.. image:: /_static/user_guide/lif-lif_config.png
   :align: center
   :width: 800
   :alt: LIF Configuration dialog

The dialog is divided into three areas:

- A **time-trace plot** spans the top. It displays the live digitizer
  waveform for the LIF channel (and the reference channel if enabled),
  with the current integrated value and the running average count shown
  in the top-right corner. Colored overlay regions mark the active
  integration gates so the gate positions can be verified visually.
- The **LIF Digitizer** group occupies the bottom-left and holds the
  digitizer configuration, an **Averages** spin box, a **Reset**
  button, and the **Start Acquisition** / **Stop Acquisition** buttons
  that drive the live preview.
- The bottom-right column stacks two groups: **Laser**, with manual
  laser position and flashlamp controls, and **Processing**, with the
  integration gates and optional smoothing filters.

LIF channel and reference channel
---------------------------------

The LIF digitizer records up to two analog channels. The role
assignment is fixed and stated in the caption above the channel table:
Channel 1 is the LIF signal and Channel 2, if enabled, is the
reference.

**LIF channel** (Ch 1)
   The primary signal channel. Blackchirp acquires this channel at
   every scan point and integrates its waveform over the LIF gate to
   produce the measurement value for that point.

**Reference channel** (Ch 2)
   An optional channel used to record a signal proportional to the
   laser pulse energy (for example, from a photodiode or a pick-off
   power meter). The reference channel is disabled by default; enable
   it by checking the Ch 2 row in the **Analog Channels** table.

When the reference channel is enabled, Blackchirp integrates both channels
over their respective gates. The normalized LIF signal for each shot is::

   signal_normalized = signal_lif / signal_ref

Normalization is applied before co-averaging. If the reference gate
integration returns a value of zero for a shot, that shot is not included
in the normalized average to avoid division by zero.

Digitizer settings
------------------

The **LIF Digitizer** group embeds the same widget the FTMW digitizer
tab uses — an **Analog Channels** table on top with the **Data
Transfer** and **Trigger** sub-tables below it. The Acquisition Setup
group is omitted because LIF always acquires single records. For the
field-by-field walkthrough, see
:doc:`/user_guide/ftmw_configuration/digitizer_setup`. For per-device
wiring, see :doc:`/user_guide/hw/lifdigitizer` and
:doc:`/user_guide/hw/liflaser`.

Below the digitizer fields, three controls drive the live preview:

**Averages**
   Sets the number of waveforms co-averaged in the time-trace plot
   during a live preview. The same value is used as the
   **shots-per-point** target when the dialog is reached through the
   Experiment Setup dialog: each scan point in the resulting
   acquisition accumulates this many shots before advancing.

**Reset**
   Discards the running average displayed in the time-trace plot, restarting
   the average from the next acquired waveform. Use this after changing
   acquisition parameters or after the laser has been retuned, to avoid
   contaminating the displayed average with shots taken under different
   conditions.

**Start Acquisition / Stop Acquisition**
   Begin or end a live preview acquisition. Start Acquisition commits
   the current digitizer settings, locks them, and tells the LIF
   digitizer to begin streaming waveforms; the time-trace plot updates
   as shots arrive and the **Processing** group becomes active. Stop
   Acquisition halts the stream and re-enables digitizer editing. The
   same buttons drive the preview in both the standalone and Experiment
   Setup contexts; the preview does not record data to disk.

Laser control
-------------

The **Laser** group hosts the manual laser controls. The laser
hardware driver determines the displayed units (e.g. nanometers,
wavenumbers, or hertz) and the allowed range.

**Position**
   Spin box for the laser setpoint. Edit the value, then click **Set**
   to send the new position to the laser. The spin box is disabled
   until the hardware reports the new position back, at which point
   the read-back value is loaded and the box is re-enabled.

**Flashlamp**
   A checkable toggle button. The button reads **Enable** when the
   flashlamp is off and **Disable** when it is on; clicking it sends
   the change to the hardware and the button is briefly disabled
   until the hardware confirms. The Flashlamp control is omitted on
   lasers whose driver does not expose flashlamp gating. Disabling
   the flashlamp between acquisitions preserves its finite firing
   lifetime; see the **Auto Disable Flashlamp** option in
   :doc:`experiment_setup` for the automatic equivalent at the end of
   a scan.

Manual tuning from this group is independent of any acquisition. It
is the recommended way to verify laser tuning, exercise the
flashlamp, or reposition the laser between experiments.

Integration gates
-----------------

Gates define the time window over which the digitizer waveform is
integrated to produce a scalar measurement value at each scan point.
The **Gates** sub-group inside Processing is a 2×2 grid: the rows
**LIF** and **Reference** select the channel, and the **Start** and
**End** columns hold the gate positions in **sample points** relative
to the beginning of the digitizer record. The relationship between
sample points and real time is:

.. math::

   t = \text{point} \times \Delta t

where :math:`\Delta t` is the reciprocal of the digitizer sample
rate. The integration is performed by summing the raw digitizer
values within ``[start, end)``; each gate must span at least two
points (End > Start), and Blackchirp nudges the other column
automatically when an edit would violate that.

The Reference row is editable only when the reference channel is
enabled in the Analog Channels table. The reference gate does not
need to coincide with the LIF gate; it should be positioned over the
portion of the waveform that best represents the laser power for
each shot.

Hold **Ctrl** while clicking a spin box arrow (or pressing Up/Down)
to adjust the gate position in steps of 10 sample points.

.. note::
   Gate positions are saved with the experiment and can be
   re-adjusted after the acquisition completes; the LIF tab
   re-integrates the stored waveforms with the new gates and updates
   the displayed plots. See :doc:`lif_tab` for the post-acquisition
   workflow.

Processing filters
------------------

Two optional filters smooth each waveform before integration:

**Low pass filter**
   First-order exponential filter, :math:`x_n = \alpha x_{n-1} +
   (1 - \alpha) x_n`. The **α** spin box sets the coefficient;
   setting α to 0 displays ``Disabled`` and bypasses the filter.

**Savitzky-Golay smoothing**
   `Savitzky-Golay
   <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`__
   smoothing. The group box is checkable — uncheck the title to
   bypass the filter. **Window** sets the (odd) window size and
   **Order** sets the polynomial order; Blackchirp clamps the order
   to ``Window − 1`` and snaps even window values down to the next
   odd value.

The same filter controls appear on the LIF tab during and after
acquisition; see :ref:`lif-tab` for the full per-control reference
and the same-named **Reprocess All**, **Reset**, and **Save** buttons
at the bottom of the Processing group. Those buttons are present in
this dialog but inert — they activate only on the LIF tab after an
acquisition has completed and there are stored waveforms to
reprocess.

In this dialog the filters affect only the live preview trace; their
values are persisted along with the gate positions and applied at the
start of the next acquisition. The Processing group is disabled until
**Start Acquisition** has been clicked, since the filters operate on
a known record length.
