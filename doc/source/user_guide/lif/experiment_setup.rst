.. index::
   single: LIF Module; experiment setup
   single: LIF Module; wizard page
   single: Delay Axis; LIF
   single: Laser Axis; LIF
   single: Shots Per Point; LIF
   single: Randomize Delay Order
   single: Scan Order; LIF
   single: Auto Disable Flashlamp

.. _lif-experiment-setup:

LIF Experiment Setup
====================

When the LIF module is enabled in the Application Configuration (see
:ref:`application-config`), the first page of the experiment wizard gains
an **LIF** group alongside the **FTMW** group described in
:doc:`/user_guide/experiment/acquisition_types`. The LIF group defines the
scan-axis parameters and options for the laser scan; channel and gate
configuration live separately in the :doc:`configuration` (opened from
**Hardware → LIF Configuration**), where shots-per-point and digitizer
settings are also configured.

.. image:: /_static/user_guide/lif/lif_exp_setup.png
   :align: center
   :alt: LIF group on the first page of the experiment wizard

The LIF group is divided into three panels: **Delay**, **Laser**, and
**Options**. The **LIF** checkbox at the top of the group enables or disables
LIF acquisition for this experiment. When the checkbox is unchecked, all
controls in the group are inactive and no LIF data are recorded.

Delay axis
----------

The **Delay** panel configures the timing axis of the scan. The delay refers
to the time between the laser pulse and the detection gate, controlled via
a dedicated channel on the pulse generator.

- **Start** — the delay at the first point, in microseconds.
- **Step** — the step size between consecutive delay points, in microseconds.
- **Points** — the number of delay points.
- **End** — the delay at the last point, computed automatically from
  ``Start + (Points - 1) * Step``. This field is read-only and updates as
  the other three values change.

To perform a fixed-delay acquisition (scanning only the laser frequency),
set Points to 1; the End value tracks Start automatically.

Laser axis
----------

The **Laser** panel configures the wavelength axis of the scan. The units
displayed (here **nm**) are determined by the connected laser hardware driver.

- **Start** — the starting laser position.
- **Step** — the step size between consecutive laser points.
- **Points** — the number of laser points.
- **End** — the ending laser position, computed automatically from
  ``Start + (Points - 1) * Step``. This field is read-only.

To perform a fixed-wavelength acquisition (scanning only the delay), set
Points to 1.

Both axes can be active simultaneously. In a two-dimensional scan,
Blackchirp steps through all combinations of delay and laser positions,
collecting the configured number of shots at each combination. The
shots-per-point value is set on the :doc:`configuration`, not in the wizard.

Options
-------

The **Options** panel controls the order and behavior of the scan.

**Scan Order**
   Controls which axis is the outer loop during acquisition.

   - ``DelayFirst`` — the delay axis is the outer loop; for each delay value,
     Blackchirp steps through all laser positions before advancing to the next
     delay.
   - ``LaserFirst`` — the laser axis is the outer loop; for each laser
     position, Blackchirp steps through all delay values.

   Scan Order affects only the order in which points are visited; it does
   not change the on-disk data layout, which is always indexed by delay and
   laser axes independently. See :doc:`data_storage` for the storage layout.

**Complete Mode**
   Controls what happens when all grid points have been acquired once.

   - ``StopWhenComplete`` — the acquisition ends after a single sweep through
     all points.
   - ``ContinueAveraging`` — the acquisition continues, co-averaging additional
     shots at each point indefinitely. Use this mode when higher signal-to-noise
     is needed; the experiment ends when the user clicks **Abort**.

**Auto Disable Flashlamp**
   When checked, Blackchirp automatically disables the laser flashlamp when
   the acquisition ends. The purpose is to preserve the flashlamp itself,
   which has a finite firing lifetime; leaving it running between
   acquisitions consumes lifetime that could otherwise be spent collecting
   data.

**Randomize Delay Order**
   When checked, the delay points are visited in a randomized order within
   each sweep rather than sequentially from start to end. Randomizing the
   delay order is useful when the sample or background signal drifts slowly
   with time, because it decorrelates systematic drift from the delay
   coordinate. The laser axis is always stepped sequentially; only the delay
   ordering is randomized.

Initializing from a previous experiment
---------------------------------------

When a past experiment is opened and the **Quick Experiment** option is used,
the LIF group is pre-populated with the scan parameters from that experiment.
All axis ranges, step sizes, and options are restored, providing a
convenient starting point for a repeat acquisition.
