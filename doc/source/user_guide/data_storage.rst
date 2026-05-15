.. index::
   single: FTMW
   single: FID
   single: FID Processing
   single: Segment
   single: Frame
   single: Backup
   single: Window Function
   single: Markers

Data Storage
============

Blackchirp stores its data in your selected :ref:`Data Storage Location <first-run-data-path>`. The location may be changed by selecting the Settings > Data Storage menu item.

.. image:: /_static/user_guide/first_run-savepathdialog.png
   :alt: Data storage location

At this location, Blackchirp creates 4 subfolders:

  * ``experiments``: Storage for experiment data files. A more detailed description is below.
  * ``log``: Location of program log files. All messages that were displayed on the log tab are written in CSV format, and a new CSV file is generated each calendar month. The ``log/crashes`` subdirectory holds diagnostic reports when Blackchirp terminates unexpectedly; see :doc:`crash_reports`.
  * ``rollingdata``: CSV files containing monitoring data (see `Rolling and Aux Data <rolling-aux-data.html>`_)
  * ``textexports``: Default location for XY export files for graph data.

All of Blackchirp's data files are written in plain-text CSV format using the separator character ``;``.

Experiments
-----------

Each experiment that Blackchirp performs is associated with an experiment number, and the location of its data files within the experiments folder is derived from that number. To avoid having an excessive number of directories at a single level, Blackchirp collects experiments in groups of 1000 and stores each group under a single directory. For a given scan number X, its data will be stored in the directory ``experiments/Z/Y/X``, where ``Z = X//1000000`` and ``Y = X//1000`` (integer division, discarding remainder). For example, experiment 123456789 would be stored in directory::

  experiments/123/123456/123456789

and experiment 480 would be stored in::

  experiments/0/0/480

Each experiment is associated with several CSV files that contain information about aspects of the experiment.

auxdata.csv
...........

This file contains all auxiliary data collected during the experiment, which is plotted on the :doc:`Aux Data <rolling-aux-data>` tab. Aux data is collected at fixed time intervals throughout the experiment as determined by the "Aux Data Interval" option on the :doc:`Experiment Setup <experiment/acquisition_types>` page. Example::

  timestamp;epochtime;elapsedsecs;FlowController.Main.Pressure;Ftmw.Shots;TemperatureController.default.Temperature Ch2.Temperature2
  Thu Apr 30 19:50:51 2026;1777603851;0;0;0;4.932009643731726

The format is similar to the Rolling Data format, with two exceptions: first, there is an additional ``elapsedsecs`` column that tells the number of seconds since the start of the experiment, and second, there may be many columns of data.

chirps.csv
..........

This file contains information about the CP-FTMW chirps. A single chirp may be built from multiple segments (any of which may be empty), and each segment has a starting frequency, an ending frequency, and a duration. Example::

  Chirp;Segment;StartMHz;EndMHz;DurationUs;Alpha;Empty
  0;0;4895;1520;2;-1687.5;false

The first column (``Chirp``) is an index identifying the chirp, and the second (``Segment``) identifies which segment of the chirp is being described. In this experiment, a single chirp consists of one segment starting at 4895 MHz and ending at 1520 MHz, with a duration of 2 microseconds. The ``Alpha`` column is the sweep rate in MHz/μs. If the ``Empty`` column is true, then the start and end values are ignored, and the segment contains just 0 over the indicated duration.

An entire experiment may consist of many different chirps. The example below shows an LO scan in which each of 20 identical chirps is recorded at each LO tuning::

  Chirp;Segment;StartMHz;EndMHz;DurationUs;Alpha;Empty
  0;0;4895;1520;1;-3375;false
  1;0;4895;1520;1;-3375;false
  ...
  19;0;4895;1520;1;-3375;false

.. note::
   The frequencies and sweep rate contained in the chirps.csv file refer to AWG frequencies. The actual chirp range depends on the `Rf Configuration <hardware_menu.html#rf-configuration>`_.

clocks.csv
..........

This file contains the configuration of the clocks (upconversion LO, downconversion LO, etc) as discussed on the `Rf Configuration <hardware_menu.html#rf-configuration>`_ page. In a typical CP-FTMW experiment, each clock is set to a single value. Example::

  Index;ClockType;FreqMHz;Operation;Factor;HwKey;OutputNum
  0;UpLO;11520;Multiply;2;Clock.virtual;0
  0;DownLO;40960;Multiply;8;Clock.virtual;1

In some cases (e.g., an `LO Scan <experiment/acquisition_types.html#lo-scan>`_ or a `DR Scan <experiment/acquisition_types.html#dr-scan>`_), one or more of the clocks may be tuned to different values throughout the experiment. The following is an excerpt from an LO scan in which the upconversion and downconversion LOs were each stepped by 250 MHz::

  Index;ClockType;FreqMHz;Operation;Factor;HwKey;OutputNum
  0;DownLO;40960;Multiply;8;Clock.virtual;1
  0;UpLO;11520;Multiply;2;Clock.virtual;0
  0;DRClock;7000;Multiply;1;Clock.virtual;2
  1;DownLO;41210;Multiply;8;Clock.virtual;1
  1;UpLO;11770;Multiply;2;Clock.virtual;0
  1;DRClock;7000;Multiply;1;Clock.virtual;2
  2;DownLO;41460;Multiply;8;Clock.virtual;1
  2;UpLO;12020;Multiply;2;Clock.virtual;0
  2;DRClock;7000;Multiply;1;Clock.virtual;2

The ``Index`` column refers to each step of the experiment. ``ClockType`` identifies the particular clock role (UpLO, DownLO, AwgRef, DRClock, DigRef, or ComRef). ``FreqMHz`` is the logical clock frequency in MHz. ``Operation`` (Multiply or Divide) and ``Factor`` account for any frequency divider or multiplier on the clock output, and these values are used by Blackchirp to determine how to convert the logical frequencies into hardware frequency. ``HwKey`` and ``OutputNum`` tell which piece of hardware was used and which output (in the event that the clock has multiple outputs).

hardware.csv
............

This file contains the list of hardware compiled into Blackchirp when the experiment was performed. It is used in the program to determine whether it is possible to perform a Quick Experiment, which can be done only if the same hardware configuration is available. Example::

  key;driver
  LifDigitizer.Default;PythonLifDigitizer
  PulseGenerator.Default;VirtualPulseGenerator
  LifLaser.virtual;VirtualLifLaser
  TemperatureController.default;VirtualTemperatureController
  FlowController.Main;VirtualFlowController
  Clock.virtual;FixedClock
  AWG.Ka;VirtualAwg
  FtmwDigitizer.virtual;VirtualFtmwDigitizer

The ``key`` field uses the format ``HardwareClass.Label``, where the label is the user-assigned name configured at setup time (for example, ``FtmwDigitizer.virtual``, ``PulseGenerator.Default``, ``FlowController.Main``). The ``driver`` field records which specific hardware driver was used. The number of items in this list may vary depending on your configuration. Older experiments may carry the historical ``subKey`` header label in place of ``driver``, and a third ``hardwareType`` column carrying the integer enum value; the loader accepts either header label and silently ignores the redundant third column when present.

header.csv
..........

This file contains the vast majority of the program, acquisition, and hardware settings for the experiment. Essentially, any setting in effect at the beginning of the experiment that cannot change during the experiment is stored here. Example::

  ObjKey;ArrayKey;ArrayIndex;ValueKey;Value;Units
  ChirpConfig;;;ChirpInterval;20;μs
  ChirpConfig;;;SampleInterval;6.25e-05;μs
  ChirpConfig;;;SampleRate;16000;MHz
  Experiment;;;BCBuildVersion;"508a6973c274ae9fcf24f0949ba70970b7c51d39";
  Experiment;;;BCMajorVersion;2;
  Experiment;;;BCMinorVersion;0;
  Experiment;;;BCPatchVersion;0;
  Experiment;;;BCReleaseVersion;devel;
  Experiment;;;BackupInterval;0;min
  Experiment;;;Number;49;
  Experiment;;;TimeDataInterval;5;s
  FtmwConfig;;;ChirpScoringEnabled;false;
  FtmwConfig;;;Objective;100;
  FtmwConfig;;;PhaseCorrectionEnabled;false;
  FtmwConfig;;;TargetShots;100;
  FtmwConfig;;;Type;Target_Shots;
  FtmwDigitizer.virtual;;;RecordLength;750000;
  FtmwDigitizer.virtual;;;SampleRate;5e+10;Hz

There are 6 columns in total. ``ObjKey`` identifies the "object" associated with an entry. This is a concept internal to Blackchirp, but the object names are usually sufficiently descriptive. Hardware objects use the label-based key format (e.g. ``FtmwDigitizer.virtual``, ``PulseGenerator.Default``, ``FlowController.Main``). In the example above, the ``FtmwConfig`` entries are associated with the settings pertaining to the CP-FTMW acquisition. ``ValueKey``, ``Value``, and ``Units`` together record the value and any associated units for the particular entry being made.

``ArrayKey`` and ``ArrayIndex`` are used when there are multiple instances of data that would otherwise have the same ``ValueKey``. For example, a PulseGenerator object may have several channels, each one of which has an associated delay, width, etc. An example of such a situation is::

  PulseGenerator.Default;;;RepRate;1;Hz
  PulseGenerator.Default;Channel;0;ActiveLevel;ActiveHigh;
  PulseGenerator.Default;Channel;0;Delay;0;μs
  PulseGenerator.Default;Channel;0;Enabled;false;
  PulseGenerator.Default;Channel;0;Name;Gas;
  PulseGenerator.Default;Channel;0;Role;Gas;
  PulseGenerator.Default;Channel;0;Width;1;μs
  PulseGenerator.Default;Channel;1;ActiveLevel;ActiveHigh;
  PulseGenerator.Default;Channel;1;Delay;0;μs
  PulseGenerator.Default;Channel;1;Enabled;false;
  PulseGenerator.Default;Channel;1;Name;AWG;
  PulseGenerator.Default;Channel;1;Role;AWG;
  PulseGenerator.Default;Channel;1;Width;1;μs

Here there is an ``ArrayKey`` named "Channel" and the ``ArrayIndex`` identifies which particular channel is referred to. That channel is associated with multiple different ``ValueKey`` entries, so the ``ArrayKey``, ``ArrayIndex``, and ``ValueKey`` would be used together to identify any desired value.

log.csv
.......

This file contains a record of all messages sent to the Log tab during the experiment. Example::

  Timestamp;Epoch_msecs;Code;Message
  Wed Jul 13 14:36:46 2022;1657748206527;Highlight;Starting experiment 38.
  Wed Jul 13 14:37:06 2022;1657748226794;Highlight;Experiment 38 complete.

The ``Timestamp`` and ``Message`` columns are self-explanatory. ``Epoch_msecs`` is the `number of milliseconds since the Unix epoch <https://currentmillis.com/>`_. ``Code`` contains the level of the message: Normal, Highlight, Warning, Error, or Debug.

markers.csv
...........

.. index::
   single: Markers
   single: AWG markers

This file contains the AWG marker channel definitions for the experiment. Marker channels are output signals generated by the AWG in synchrony with the chirp waveform, used for purposes such as protecting the receiver amplifier (Protection role), enabling an amplifier gate (Gate role), triggering other instruments (Trigger role), or arbitrary user-defined purposes (Custom role). The number of available marker channels depends on the AWG model. For details on configuring marker channels in the user interface, see :ref:`chirp-setup-markers`.

The columns are:

* ``Channel``: Zero-based marker channel index.
* ``Name``: User-defined label for the channel.
* ``Role``: One of ``Protection``, ``Gate``, ``Trigger``, or ``Custom``.
* ``TimingMode``: ``ChirpRelative`` (timing is repeated relative to each chirp start/end) or ``Absolute`` (timing is relative to the first chirp start and fires once per waveform).
* ``StartUs``: Start time in microseconds. For ``ChirpRelative`` mode, this is relative to the chirp start (negative values begin before the chirp starts). For ``Absolute`` mode, this is relative to the first chirp start.
* ``EndUs``: End time in microseconds. For ``ChirpRelative`` mode, this is relative to the chirp end (positive values extend past the chirp end).
* ``Enabled``: Whether the marker channel is active.

Example for a two-channel AWG with a protection pulse on channel 0 and an amplifier gate on channel 1::

  Channel;Name;Role;TimingMode;StartUs;EndUs;Enabled
  0;Protection;Protection;ChirpRelative;-0.5;0.5;true
  1;Gate;Gate;ChirpRelative;-0.5;0.5;true

If no AWG is configured with marker support (``markerCount == 0``), this file is not written.

objectives.csv
..............

This file is used internally by Blackchirp to configure data structures when the experiment is opened with the "View Experiment" dialog.

.. warning::
  Modifying this file and then trying to load the experiment with Blackchirp may cause an error or crash.

version.csv
...........

This file stores information about the Blackchirp version used with the experiment. The purpose is to enable the possibility of future backward compatibility. An example::

  ;
  key;value
  BCMajorVersion;2
  BCMinorVersion;0
  BCPatchVersion;0
  BCReleaseVersion;devel
  BCBuildVersion;"508a6973c274ae9fcf24f0949ba70970b7c51d39"

The first line contains the separator character used for all of the CSV files associated with this experiment. The second line tells the titles of the columns (``key`` and ``value``, respectively). The subsequent lines contain the detailed Blackchirp version information. The ``BCBuildVersion`` field contains the full git commit hash, quoted because it is a bare hex string.

FIDs
----

Like other files, FIDs are stored in plain-text CSV format. The FIDs for an experiment are located in a ``fid`` subfolder within the experiment folder. FIDs themselves are in a set of numbered CSV files starting from 0. In addition, there is a ``fidparams.csv`` file that contains useful information.

fidparams.csv
.............

This file contains information needed to convert raw FID data into numerical values, as well as the information needed to determine the appropriate frequency values following a Fourier transform. Here is an example ``fidparams.csv`` file::

  index;spacing;probefreq;vmult;shots;sideband;size
  0;2e-11;40960;0.000390625;100;LowerSideband;750000

In this example, 100 shots were recorded at a single clock configuration. For LO scan experiments with multiple clock configurations, there will be one row per configuration, and the ``index`` column identifies the corresponding FID CSV file.

The ``index`` column identifies a particular FID and the number of its corresponding CSV file (e.g. ``0.csv``, ``1.csv``, etc.). The ``size`` column tells the number of points in the FID.

In its FID files, Blackchirp does not store the averaged digitizer voltage. Instead, Blackchirp stores *the sum of the raw digitizer readings*. To convert the FID values to average voltage, the numbers in the FID file need to be multiplied by ``vmult`` and divided by ``shots``. The ``vmult`` column contains the conversion between digitization levels and voltage, while ``shots`` contains the number of digitizer readings that have been summed.

Finally, for calculating the frequency axis of the FT, the ``spacing`` tells the time between samples in seconds; the ``probefreq`` tells the downconversion LO frequency in MHz, and ``sideband`` tells whether the FT frequency should be added (UpperSideband or 0) or subtracted (LowerSideband or 1) from the ``probefreq``.

FID CSV Files
.............

In an effort to balance plaintext readability, ease of integration with other analysis software, and file size, the summed digitizer values are written as **base-36 signed integers**. A simple example may begin with::

  fid0
  -7n
  -k
  10
  -p
  -21
  6j
  -8o
  -2v
  4c
  -2x
  -1s
  -11

The first row is a column label, and each subsequent row contains a single FID point (e.g., -7n = -275). In some configurations, the FID file may contain data from multiple frames, as shown in the example below which has 20 FIDs (only the first 9 points for each FID are shown)::

  fid0;fid1;fid2;fid3;fid4;fid5;fid6;fid7;fid8;fid9;fid10;fid11;fid12;fid13;fid14;fid15;fid16;fid17;fid18;fid19
  -33;-1u;-22;7z;-4r;-4r;36;-4t;-r;2m;-as;-bk;1g;-8j;-3u;-50;-73;-b1;1u;-5s
  -w;-5v;-4p;7u;-br;-2j;-2n;-7h;-3v;-8z;-5t;-89;-5p;-be;23;-4e;q;-2l;-4a;-ck
  4g;-5f;2t;h;-i9;-a3;-d1;-r;-n;-hg;6c;-4p;-k1;-99;-31;-2z;-6i;-a0;-3w;-bw
  32;-cr;i;-9q;-b4;-bi;-2w;4c;5s;-iv;72;-7m;-7a;-2j;-6s;-cj;-77;-hj;2z;-e5
  -22;-l9;-72;-af;-82;4;-3j;-6a;-8e;-9l;-59;-2g;3n;m;-ch;-el;-l;-f7;-e;-gc
  -4o;-fi;-2e;-c0;-bk;58;-8w;-dj;-bo;-2z;-7v;6d;-6p;-6f;-i2;-8p;-8l;-au;-49;-68
  -5o;-3p;-9;-bv;-cu;-3e;2v;-6c;-1y;1j;-6v;5b;-2x;-9f;-dl;-4y;-ex;-2f;-3o;-8
  -az;-33;-99;-4r;-ee;-9p;-8e;-2l;-dk;56;-fq;-3t;38;3a;-7f;-4a;-2b;3m;-e;-4t
  -bg;-82;-6s;-7r;-8k;-3o;-id;-2j;-i9;3f;-gw;-7c;-6b;-r;-57;-4v;-2o;-h;-3r;-20


processing.csv
..............

This file contains the default `processing settings <cp-ftmw.html#fid-processing-settings>`_ associated with the FID data. An example is shown below.::

  ObjKey;Value
  AutoscaleIgnoreMHz;0
  FidEndUs;15
  FidExpfUs;0
  FidRemoveDC;false
  FidStartUs;0
  FidWindowFunction;None
  FidZeroPadFactor;0
  FtUnits;6

The ``FtUnits`` value refers to a scaling factor of 10\ :sup:`N` (i.e., a setting of 6 would convert the FT from units of V to μV.). The window functions are shown below, where ``N`` is the number of samples and ``n`` ranges from 0 to ``N-1``:

* None/Boxcar (0)

.. math::
   f(n) = 1

* Bartlett (1)

.. math::
   f(n) = 1-\left|\frac{2n}{N-1}-1\right|

* Blackman (2)

.. math::
   f(n) = 0.42 - 0.5\cos\frac{2\pi n}{N} + 0.08\cos\frac{4\pi n}{N}

* BlackmanHarris (3)

.. math::
   f(n) = 0.35875 - 0.48829\cos\frac{2\pi n}{N} + 0.14128\cos\frac{4\pi n}{N} - 0.1168\cos\frac{6\pi n}{N}

* Hamming (4)

.. math::
   f(n) = 0.54 - 0.46\cos\frac{2\pi n}{N}

* Hanning (5)

.. math::
   f(n) = 0.5 - 0.5\cos\frac{2\pi n}{N}

* KaiserBessel (6). I\ :sub:`0` = regular modified cylindrical Bessel function, β=14.0

.. math::
   f(n;\beta) = \frac{I_0\left(\beta\sqrt{1-\left[\frac{2x}{N-1}\right]^2}\right)}{I_0(\beta)},\quad x = n-\frac{N-1}{2}

Overlays
--------

When :doc:`overlays` are created for an experiment, their data and settings are automatically saved in an ``overlays`` subfolder within the experiment directory. This allows overlays to be restored when the experiment is reopened. The overlay storage system uses a combination of CSV files to preserve both the overlay data and all configuration settings.

overlays.csv
............

This master file contains the list of all overlays associated with the experiment. Each overlay is identified by its label, and the value indicates the overlay type: 0 = Blackchirp Experiment, 1 = Catalog, 2 = Generic XY Data. The file also stores version information for compatibility tracking. Example::

  ObjKey;Value
  BCBuildVersion;"508a6973c274ae9fcf24f0949ba70970b7c51d39"
  BCMajorVersion;2
  BCMinorVersion;0
  BCPatchVersion;0
  BCReleaseVersion;devel
  Exp17;0
  c047527_full;1

Individual Overlay Files
........................

For each overlay listed in ``overlays.csv``, two additional files are created:

**[label].settings.csv**
  Contains all configuration parameters for the overlay, including:

  - Source file path and overlay type-specific settings
  - Curve appearance properties (color, line style, thickness, visibility)
  - Data processing parameters (scaling, offsets, frequency filtering)
  - Plot assignment and display preferences
  - Version information for compatibility

  Example for a catalog overlay::

    ObjKey;Value
    catalogConvolutionEnabled;false
    catalogLineshapeType;Gaussian
    catalogLinewidthKHz;100
    catalogTransitionCount;87
    curve_color;#40963a
    curve_thickness;2
    enabled;true
    label;c047527_full
    sourceFile;/path/to/catalog.cat
    yScale;-25056766.9100865

**[label].data.csv**
  Contains the processed overlay data ready for display, with columns for frequency (X) and intensity (Y) values. The data format varies by overlay type:

  - **Catalog overlays**: Contains frequencies and intensities of peaks, along with other transition metadata (quantum numbers, etc).
  - **Generic XY overlays**: Contains the parsed and filtered XY data from the source file
  - **Blackchirp Experiment overlays**: Contains the Fourier transform data from the referenced experiment

  The data files use standard CSV format with semicolon separators, maintaining consistency with other Blackchirp data files.
