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

LIF channel, gate, and laser control are accessed through the **LIF
Configuration** dialog. Two paths open it:

- **Hardware → LIF Configuration** in the main menu. This standalone view
  is useful for tuning the digitizer and laser before starting an experiment.
- The **LIF Configuration** page of the experiment wizard, which embeds the
  same widget so the configuration can be reviewed and adjusted as part of
  setting up an acquisition.

The two contexts share identical controls; values set in one are persisted
and visible in the other.

.. image:: /_static/user_guide/lif/lif_config.png
   :align: center
   :width: 800
   :target: ../../_images/lif_config.png
   :alt: LIF Configuration dialog

The dialog is divided into three areas:

- A **time-trace plot** spans the top. It displays the live digitizer
  waveform for the LIF channel (and the reference channel if enabled).
  Colored overlay regions mark the active integration gates so the gate
  positions can be verified visually.
- The **LIF Digitizer** group, in the bottom-left, contains the digitizer
  configuration along with the **Averages** spin box, **Reset** button, and
  **Start Acquisition** / **Stop Acquisition** buttons that drive the live
  preview.
- The bottom-right column stacks two groups: **Laser**, with manual laser
  position and flashlamp controls, and **Processing**, with the integration
  gate positions and optional smoothing filters.

LIF channel and reference channel
---------------------------------

The LIF digitizer records up to two analog channels.

**LIF channel**
   The primary signal channel. Blackchirp acquires this channel at every scan
   point and integrates its waveform over the LIF gate to produce the
   measurement value for that point. Channel 1 is the LIF channel by
   convention.

**Reference channel**
   An optional second channel used to record a signal proportional to the
   laser pulse energy (for example, from a photodiode or a pick-off power
   meter). Channel 2 is the reference channel by convention. The reference
   channel is disabled by default and must be explicitly enabled in the
   digitizer configuration.

When the reference channel is enabled, Blackchirp integrates both channels
over their respective gates. The normalized LIF signal for each shot is::

   signal_normalized = signal_lif / signal_ref

Normalization is applied before co-averaging. If the reference gate
integration returns a value of zero for a shot, that shot is not included
in the normalized average to avoid division by zero.

Digitizer settings
------------------

The **LIF Digitizer** group exposes the standard digitizer controls (sample
rate, record length, input coupling, voltage range, trigger settings, and
per-channel enable/range). These follow the same conventions as the FTMW
digitizer widget; see :doc:`/user_guide/experiment/digitizer_setup` for the
field-by-field walkthrough. For per-device wiring, see
:doc:`/user_guide/hw/lifdigitizer` and :doc:`/user_guide/hw/liflaser`.

Below the digitizer fields, three controls drive the live preview:

**Averages**
   Sets the number of waveforms co-averaged in the time-trace plot during a
   live preview. The same value is used as the **shots-per-point** target
   when the dialog is part of an experiment-wizard configuration: each scan
   point in the resulting acquisition will accumulate this many shots before
   advancing.

**Reset**
   Discards the running average displayed in the time-trace plot, restarting
   the average from the next acquired waveform. Use this after changing
   acquisition parameters or after the laser has been retuned, to avoid
   contaminating the displayed average with shots taken under different
   conditions.

**Start Acquisition / Stop Acquisition**
   Begin or end a live preview acquisition. Start Acquisition commits the
   current digitizer settings, locks them, and tells the LIF digitizer to
   begin streaming waveforms; the time-trace plot updates as shots arrive
   and the **Processing** group becomes active. Stop Acquisition halts the
   stream and re-enables digitizer editing. The same buttons drive the
   preview in both standalone and experiment-wizard contexts; the preview
   does not record data to disk.

Laser control
-------------

The **Laser** group hosts the manual laser controls. The laser hardware
driver determines the displayed units (e.g. nanometers, wavenumbers, or
hertz) and the allowed range.

**Position**
   Spin box showing the current laser setpoint. Edit the value, then click
   **Set** to send the new position to the laser. The position read-back is
   updated as the hardware reports the actual position.

**Flashlamp**
   Checkbox that enables or disables the laser flashlamp. Disabling the
   flashlamp between acquisitions preserves its finite firing lifetime; see
   the **Auto Disable Flashlamp** option in :doc:`experiment_setup` for the
   automatic equivalent at the end of a scan.

Manual tuning from this group is independent of any acquisition. It is the
recommended way to verify laser tuning, exercise the flashlamp, or reposition
the laser between experiments.

Integration gates
-----------------

Gates define the time window over which the digitizer waveform is integrated
to produce a scalar measurement value at each scan point. Gate positions are
specified in **sample points** relative to the beginning of the digitizer
record. The relationship between sample points and real time is:

.. math::

   t = \text{point} \times \Delta t

where :math:`\Delta t` is the reciprocal of the digitizer sample rate.

**LIF Gate Start** and **LIF Gate End**
   Define the integration window for the primary LIF signal. The gate must
   span at least two points (end > start). The integration is performed by
   summing the raw digitizer values within ``[start, end)``.

**Reference Gate Start** and **Reference Gate End**
   Define the integration window for the reference channel. These controls are
   active only when the reference channel is enabled. The reference gate does
   not need to coincide with the LIF gate; it should be positioned over the
   portion of the waveform that best represents the laser power for each shot.

.. note::
   Gate positions are saved with the experiment and can be re-adjusted after
   the acquisition completes; the LIF tab re-integrates the stored waveforms
   with the new gates and updates the displayed plots. See :doc:`lif_tab`
   for the post-acquisition workflow.

Processing filters
------------------

The remaining controls in the **Processing** group — the Low Pass Filter
alpha and the Savitzky-Golay filter window and polynomial order — apply
optional smoothing to each waveform before integration. The same controls
appear in the LIF tab during and after acquisition; see
:ref:`lif-tab` for the full per-control reference.

In this dialog the filters affect only the live preview trace; their values
are persisted along with the gate positions and applied at the start of the
next acquisition. The Processing group is disabled until **Start
Acquisition** has been clicked, since the filters operate on a known record
length.
