.. index::
   single: LIF Module; data storage
   single: lif/; LIF data directory
   single: lifparams.csv
   single: LIF Trace; on-disk format
   single: Processing Settings; LIF on-disk
   single: header.csv; LIF sections

.. _lif-data-storage:

LIF Data Storage
================

LIF data are stored inside the same experiment folder as CP-FTMW data. For
the experiment-folder naming convention and the files that every experiment
writes (``header.csv``, ``hardware.csv``, ``auxdata.csv``, etc.), see
:doc:`/user_guide/data_storage`.

LIF-specific files are placed in a ``lif/`` subdirectory of the experiment
folder::

   experiments/Z/Y/X/
   ├── header.csv
   ├── hardware.csv
   ├── auxdata.csv
   ├── ...
   └── lif/
       ├── lifparams.csv
       ├── processing.csv
       ├── 0.csv
       ├── 1.csv
       └── ...

LIF parameters in header.csv
----------------------------

Before any LIF-specific files are read, the scan-axis parameters and
digitizer configuration can be recovered from the experiment's
``header.csv``. Two sections are written:

- **LifConfig** — the scan parameters set on the wizard's LIF group (see
  :doc:`/user_guide/lif/experiment_setup`).
- **LifDigitizer.<key>** — the digitizer settings set on the
  :doc:`/user_guide/lif/configuration` (the section name embeds the configured digitizer
  hardware key, e.g. ``LifDigitizer.Default``).

For example, an experiment with a 6×6 (delay × laser) grid, 10 shots per
point, randomized delay order, and a single-channel LIF acquisition writes
the following ``LifConfig`` section to ``header.csv``::

   LifConfig;;;CompleteMode;StopWhenComplete;
   LifConfig;;;DelayPoints;6;
   LifConfig;;;DelayRandom;true;
   LifConfig;;;DelayStart;200;μs
   LifConfig;;;DelayStep;10;μs
   LifConfig;;;LaserPoints;6;
   LifConfig;;;LaserStart;250;nm
   LifConfig;;;LaserStep;5;nm
   LifConfig;;;ScanOrder;DelayFirst;
   LifConfig;;;ShotsPerPoint;10;

The companion ``LifDigitizer.Default`` section captures the digitizer
configuration that produced the trace files::

   LifDigitizer.Default;;;BlockAverageEnabled;false;
   LifDigitizer.Default;;;ByteOrder;LittleEndian;
   LifDigitizer.Default;;;BytesPerPoint;1;
   LifDigitizer.Default;;;LifChannel;1;
   LifDigitizer.Default;;;LifRefChannel;2;
   LifDigitizer.Default;;;LifRefEnabled;false;
   LifDigitizer.Default;;;RecordLength;10000;
   LifDigitizer.Default;;;SampleRate;1.25e+09;Hz
   LifDigitizer.Default;;;TriggerChannel;0;
   LifDigitizer.Default;;;TriggerDelay;0;μs
   LifDigitizer.Default;;;TriggerEdge;RisingEdge;
   LifDigitizer.Default;;;TriggerLevel;0.3;V
   LifDigitizer.Default;AnalogChannel;0;Enabled;true;
   LifDigitizer.Default;AnalogChannel;0;FullScale;0.05;V
   LifDigitizer.Default;AnalogChannel;0;Index;1;
   LifDigitizer.Default;AnalogChannel;0;VerticalOffset;0;V
   LifDigitizer.Default;AnalogChannel;1;Enabled;false;
   LifDigitizer.Default;AnalogChannel;1;FullScale;0.05;V
   LifDigitizer.Default;AnalogChannel;1;Index;2;
   LifDigitizer.Default;AnalogChannel;1;VerticalOffset;0;V

Together these sections fully describe how the trace files in ``lif/``
were acquired. ``ShotsPerPoint`` is needed to convert the on-disk
accumulated values to per-shot voltages (see below).

lif/lifparams.csv
-----------------

This file is the index of all acquired trace files. One row is written per
scan point that has been acquired. The columns are::

   lIndex;dIndex;shots;lifsize;refsize;spacing;lifymult;refymult

- ``lIndex`` — laser position index (0-based).
- ``dIndex`` — delay index (0-based).
- ``shots`` — number of waveforms accumulated at this point.
- ``lifsize`` — number of samples per LIF channel waveform.
- ``refsize`` — number of samples per reference channel waveform (0 when no
  reference channel is enabled).
- ``spacing`` — time between samples, in seconds.
- ``lifymult`` — scale factor to convert one LIF integer sample to volts.
- ``refymult`` — scale factor to convert one reference integer sample to
  volts (0 when no reference channel is enabled).

For a single-channel acquisition with ``ShotsPerPoint = 10``, a 10000-sample
record at 1.25 GS/s, and a 50 mV / 128-step LIF range, ``lifparams.csv``
begins::

   lIndex;dIndex;shots;lifsize;refsize;spacing;lifymult;refymult
   0;0;10;10000;0;8e-10;0.000390625;0
   1;0;10;10000;0;8e-10;0.000390625;0
   2;0;10;10000;0;8e-10;0.000390625;0
   ...

lif/N.csv — trace files
-----------------------

Each acquired scan point is stored as a separate CSV file. The file name is
an integer index ``N`` computed as::

   N = dIndex * laserPoints + lIndex

This ordering matches the row order in ``lifparams.csv`` and allows direct
file lookup given a delay and laser index pair. The same encoding scheme is
used for FTMW data (see :doc:`/user_guide/data_storage`): values are written
as **signed base-36 integers**, one row per sample point. When only the LIF
channel is recorded the file has a single ``lif`` column; when a reference
channel is enabled, a second ``ref`` column is added.

A single-channel trace file from the example above begins::

   lif
   4b
   -1b
   -4i
   -94
   c0
   3r
   58
   3n
   3g
   ...

With the reference channel enabled, the file is::

   lif;ref
   <lifvalue0>;<refvalue0>
   <lifvalue1>;<refvalue1>
   ...

.. important::
   The on-disk values are the **sum of raw digitizer samples across all
   shots**, not the co-averaged waveform. To recover the per-shot voltage
   for sample ``i`` of a given scan point, divide by ``shots`` first and
   then apply the y-multiplier:

   .. math::

      V_i = \frac{v_i}{N_\text{shots}} \times y_\text{mult}

   where ``shots`` and ``ymult`` come from the matching row of
   ``lifparams.csv``. This convention matches the FTMW FID files and lets
   Blackchirp resume averaging when an experiment is reopened.

The ``blackchirp`` Python package (``pip install blackchirp``) provides
loader functions that perform the base-36 decoding and per-shot voltage
conversion for both LIF and FTMW data, returning numpy arrays.

lif/processing.csv
------------------

This file records the integration gate positions and waveform-filter
settings associated with the experiment. It is **always present**: when an
experiment begins, Blackchirp writes the gate and filter values active on
the :doc:`/user_guide/lif/configuration` at that moment, so the file captures the
processing state used during acquisition. Clicking **Save** in the LIF
tab's processing panel (see :ref:`lif-tab`) overwrites the file with the
current values, making them the new defaults when the experiment is
reopened.

The file uses the standard ``ObjKey;Value`` metadata layout::

   ObjKey;Value
   LifGateEndPoint;1421
   LifGateStartPoint;240
   LowPassAlpha;0
   RefGateEndPoint;1
   RefGateStartPoint;0
   SavGolEnabled;false
   SavGolPoly;3
   SavGolWindow;11

The keys correspond to the controls described on the :ref:`LIF tab
processing panel <lif-tab>`:

- ``LifGateStartPoint``, ``LifGateEndPoint`` — LIF integration gate bounds,
  in sample points.
- ``RefGateStartPoint``, ``RefGateEndPoint`` — reference integration gate
  bounds, in sample points.
- ``LowPassAlpha`` — IIR low-pass filter coefficient (0 = disabled).
- ``SavGolEnabled`` — whether the Savitzky-Golay filter is active.
- ``SavGolWindow``, ``SavGolPoly`` — Savitzky-Golay window size and
  polynomial order.
