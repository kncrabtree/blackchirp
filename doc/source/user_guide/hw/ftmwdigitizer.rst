FTMW Digitizer
==============

* Overview_
* Settings_
* Drivers_

Overview
--------

The FTMW Digitizer is the fast digitizer used to record FIDs during a
CP-FTMW experiment. At its simplest, on each trigger event the digitizer
records a fixed number of points at the configured sample rate and sends
the raw samples to Blackchirp for averaging. Some devices support fast
retriggering (e.g., Tektronix's "FastFrame" mode), which lets a series
of *frames* be acquired in a single shot. For pulsed gas sources this
allows each gas pulse to be probed by a series of chirps, with the
resulting frames transmitted individually.

Many digitizers also support internal averaging (sometimes called
*block averaging*), where some number of records are co-averaged on the
device before transfer. On some scopes block averaging produces a single
averaged frame; others can co-average multiple records in parallel and
transfer a record that still contains several (averaged) frames.

Regardless of how many analog channels the digitizer has, Blackchirp
records data from a single input channel.

To keep up with high shot rates, Blackchirp owns a small fixed-size
buffer between the digitizer thread and the acquisition thread. If
processing falls behind, the digitizer automatically accumulates shots
locally so that no triggers are lost; the accumulated data is handed off
as soon as the buffer drains. This is transparent to the user, but it
means that brief stalls in the GUI or storage path do not corrupt the
shot count.

.. note::
   Some errors on Keysight and Tektronix scopes can leave the
   instrument unresponsive. For Keysight scopes, closing and reopening
   the scope software clears the condition. For Tektronix scopes, a
   full instrument restart is usually required.

Settings
--------

Most digitizer settings are exposed in the :doc:`hardware dialog
</user_guide/hwdialog>` with inline labels and tooltips, so they do not
need to be re-documented here. A few behaviors are worth highlighting:

* **Capability flags** (``canBlockAverage``, ``canMultiRecord``,
  ``canBlockAndMultiRecord``, ``hasAuxTriggerChannel``) describe what
  the device supports. Drivers set these to match the
  hardware; do not enable a capability the device cannot actually
  provide.
* **Memory-derived limits** (``maxRecordLength``, ``maxRecords``,
  ``maxAverages``) reflect device memory. Enabling multi-record mode
  divides the available memory among records, and on some devices
  enabling block averaging further reduces the usable record length.
* **Sample rates** are presented as a fixed menu of supported values
  rather than a free-form numeric entry, because most scopes only
  accept a discrete set.
* ``bandwidthMHz`` is informational only; it does not change the
  configuration sent to the device.

Transfer rate is often the practical bottleneck. Two recommendations
that still apply:

* Prefer a link-local Ethernet connection of at least 1 Gbps to
  network-attached scopes; lower bandwidth quickly becomes the limit
  for long records or fast retriggering.
* Where the digitizer supports it, enable block averaging to push
  co-averaging onto the device. The record sent to Blackchirp is then
  the sum of many shots, which dramatically reduces network traffic
  and host-side processing per FID.

Drivers
-------

Virtual
.................

The virtual driver synthesizes a fresh FID on every shot by
summing 10–100 sinusoidal components at randomly chosen frequencies,
amplitudes, and phases, then adds Gaussian noise at the byte width
configured for the run. It honors the current vertical scale, sample
rate, and byte order, so changes made in the dialog take effect
immediately. The driver is intended for development and for
installs that only view archived data; it should not be enabled on a
real acquisition machine.

Tektronix DSA71604C
...............................

A 4-channel, 100 GSa/s oscilloscope with 16 GHz analog bandwidth. It
supports FastFrame acquisition, so a variable number of waveforms can
be captured with a low retrigger interval (about 4 μs); the frame
count is bounded by scope memory. The scope can co-average frames and
send only the average; Blackchirp uses this mode when block averaging
is enabled. Communication is over TCP, and a link-local connection of
at least 1 Gbps is recommended. In practice the scope's internal
processing, not network bandwidth, often sets the throughput limit.

Tektronix MSO72004C
...............................

Functionally identical to the DSA71604C, but with 20 GHz analog
bandwidth.

Keysight DSOV204A
............................

The `DSOV204A <https://www.keysight.com/us/en/product/DSOV204A/infiniium-v-series-oscilloscope-20-ghz-4-analog-channels.html>`_
is an 80 GSa/s oscilloscope with up to 20 GHz of bandwidth.
Communication is over a TCP socket on port 5025, requiring a static
IP address configured in the Windows OS running on the scope. The
scope can be triggered on any of its four analog channels, but
triggering on the AUX channel is recommended.

Keysight DSOX92004A
................................

The `DSOX92004A <https://www.keysight.com/us/en/product/DSOX92004A/infiniium-high-performance-oscilloscope-20-ghz.html>`_
is an 80 GSa/s oscilloscope with 20 GHz of bandwidth, upgradable to
33 GHz. Communication is over a TCP socket on port 5025, requiring a
static IP address configured in the Windows OS running on the scope.
Treat this driver as untested.

Tektronix MSO64B
.........................

A 4-channel scope with 2.5 GHz bandwidth, suitable for segmented LO
scanning. Tektronix's FastFrame backend on this model breaks the
CURVESTREAM mode that Blackchirp relies on for fast real-time data
transfer, so the data transfer rate of this scope is severely
limited.

Spectrum Instrumentation M4i2220x8
..............................................

A high-speed digitizer with a 2.5 GSa/s sampling rate and 1.25 GHz
analog bandwidth, suitable for segmented LO scanning. The
driver requires the device to have the block-averaging
firmware module enabled; with that module, sustained acquisition
rates of 50,000 FIDs/s have been achieved.

This driver requires the Spectrum Instrumentation ``spcm`` library
to be installed on the host. See
:doc:`/user_guide/hardware_config/library_status` for installation
details and to confirm that Blackchirp has located the library.

Tektronix DPO71254B
...............................

The `DPO71254B <https://www.tek.com/en/oscilloscope/dpo70000-mso70000-manual-18>`_
is a 50 GSa/s oscilloscope with up to 12.5 GHz of bandwidth.
Communication is over a TCP socket on port 4000, the default for
TekVisa, requiring a static IP address configured in the Windows OS
running on the scope. The scope can be triggered on any of its four
analog channels, but triggering on the AUX channel is recommended.
Treat this driver as experimental.

