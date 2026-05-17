.. index::
   single: header.csv
   single: hardware.csv
   single: version.csv
   single: objectives.csv
   single: log.csv
   single: auxdata.csv

.. _data-storage-experiment:

General Experiment Files
========================

Every experiment writes a set of CSV files describing the program,
acquisition, and hardware state, regardless of acquisition type. They sit
directly in the experiment folder (see :doc:`/user_guide/data_storage` for
the folder-numbering scheme). CP-FTMW-specific files are described on
:doc:`ftmw`, and LIF-specific files on :doc:`lif`.

header.csv
----------

This file contains the majority of the program, acquisition, and hardware
settings for the experiment. Any setting in effect at the start of the
experiment that cannot change during the experiment is stored here.
Example::

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

There are 6 columns. ``ObjKey`` identifies the "object" associated with an
entry. This is a concept internal to Blackchirp, but the object names are
usually sufficiently descriptive. Hardware objects use the label-based key
format (e.g. ``FtmwDigitizer.virtual``, ``PulseGenerator.Default``,
``FlowController.Main``). In the example above, the ``FtmwConfig`` entries
are associated with the settings pertaining to the CP-FTMW acquisition.
``ValueKey``, ``Value``, and ``Units`` together record the value and any
associated units for the particular entry.

``ArrayKey`` and ``ArrayIndex`` are used when there are multiple instances
of data that would otherwise have the same ``ValueKey``. For example, a
PulseGenerator object may have several channels, each one of which has an
associated delay, width, etc. An example of such a situation is::

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

Here there is an ``ArrayKey`` named "Channel" and the ``ArrayIndex``
identifies which particular channel is referred to. That channel is
associated with multiple different ``ValueKey`` entries, so the
``ArrayKey``, ``ArrayIndex``, and ``ValueKey`` are used together to
identify any desired value.

hardware.csv
------------

This file contains the list of hardware compiled into Blackchirp when the
experiment was performed. It is used to determine whether it is possible
to perform a Quick Experiment, which can be done only if the same hardware
configuration is available. Example::

  key;driver
  LifDigitizer.Default;PythonLifDigitizer
  PulseGenerator.Default;VirtualPulseGenerator
  LifLaser.virtual;VirtualLifLaser
  TemperatureController.default;VirtualTemperatureController
  FlowController.Main;VirtualFlowController
  Clock.virtual;FixedClock
  AWG.Ka;VirtualAwg
  FtmwDigitizer.virtual;VirtualFtmwDigitizer

The ``key`` field uses the format ``HardwareClass.Label``, where the label
is the user-assigned name configured at setup time (for example,
``FtmwDigitizer.virtual``, ``PulseGenerator.Default``,
``FlowController.Main``). The ``driver`` field records which specific
hardware driver was used. The number of items in this list varies with the
configuration. Older experiments may carry the historical ``subKey``
header label in place of ``driver``, and a third ``hardwareType`` column
carrying the integer enum value; the loader accepts either header label
and silently ignores the redundant third column when present.

version.csv
-----------

This file stores information about the Blackchirp version used with the
experiment, enabling future backward compatibility. An example::

  ;
  key;value
  BCMajorVersion;2
  BCMinorVersion;0
  BCPatchVersion;0
  BCReleaseVersion;devel
  BCBuildVersion;"508a6973c274ae9fcf24f0949ba70970b7c51d39"

The first line contains the separator character used for all of the CSV
files associated with this experiment. The second line gives the column
titles (``key`` and ``value``). The subsequent lines contain the detailed
Blackchirp version information. The ``BCBuildVersion`` field contains the
full git commit hash, quoted because it is a bare hex string.

objectives.csv
--------------

This file is used internally by Blackchirp to configure data structures
when the experiment is opened with the "View Experiment" dialog.

.. warning::
  Modifying this file and then trying to load the experiment with
  Blackchirp may cause an error or crash.

log.csv
-------

This file contains a record of all messages sent to the Log tab during the
experiment. Example::

  Timestamp;Epoch_msecs;Code;Message
  Wed Jul 13 14:36:46 2022;1657748206527;Highlight;Starting experiment 38.
  Wed Jul 13 14:37:06 2022;1657748226794;Highlight;Experiment 38 complete.

The ``Timestamp`` and ``Message`` columns are self-explanatory.
``Epoch_msecs`` is the number of milliseconds since the Unix epoch.
``Code`` contains the severity of the message: Normal, Highlight, Warning,
Error, or Debug. The same format is used for the application-wide log
files described on :doc:`other`; see :doc:`/user_guide/log_tab` for the
severity meanings.

auxdata.csv
-----------

This file contains all auxiliary data collected during the experiment,
which is plotted on the :doc:`Aux Data </user_guide/rolling-aux-data>`
tab. Aux data is collected at fixed time intervals throughout the
experiment as determined by the ``Aux Data Interval`` option in
:ref:`user_guide/experiment_setup:Common Settings`. Example::

  timestamp;epochtime;elapsedsecs;FlowController.Main.Pressure;Ftmw.Shots;TemperatureController.default.Temperature Ch2.Temperature2
  Thu Apr 30 19:50:51 2026;1777603851;0;0;0;4.932009643731726

The format is similar to the rolling-data format (see :doc:`other`), with
two exceptions: there is an additional ``elapsedsecs`` column giving the
number of seconds since the start of the experiment, and there may be many
columns of data.
