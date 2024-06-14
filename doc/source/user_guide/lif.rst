LIF Module
==========

By enabling the ``lif`` configuration, Blackchirp can perform laser scans with time-gated integration, such as LIF or REMPI experiments, in addition to (or simultaneously with) CP-FTMW spectroscopy.
During an LIF scan, Blackchirp can vary the laser wavelength/frequency and/or the delay time of the laser using dedicated pulse generator channel.
An oscilloscope or digitizer then records a signal as a function of time, and the program integrates the signal over a user-defined range.
A second reference channel is supported for laser power normalization.

At present, the LIF module is considered to be experimental and is under active development and testing.
For more information about this module and its usage, please visit the `Discord server <https://discord.gg/88CkbAKUZY>`_.
