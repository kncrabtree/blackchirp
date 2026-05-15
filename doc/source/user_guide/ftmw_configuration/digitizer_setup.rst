.. index::
   single: FTMW
   single: Digitizer
   single: Sample Rate
   single: Record Length
   single: Byte Order
   single: FID
   single: Block Average
   single: Multiple Records
   single: Segmented Memory
   single: FastFrame
   single: FTMW Preset; digitizer settings

Digitizer Setup
===============

The **Digitizer Config** tab of the :doc:`/user_guide/ftmw_configuration`
dialog holds the settings sent to the FTMW digitizer at the start of a
CP-FTMW acquisition.

.. figure:: /_static/user_guide/ftmw_configuration-digitizer.png
   :width: 800
   :alt: Digitizer Config tab of the FTMW Configuration dialog showing
         the analog channel list on the left, data-transfer settings,
         the trigger group, and the acquisition-mode selector.

   The **Digitizer Config** tab. Analog channels are listed on the left;
   the data-transfer, trigger, and acquisition-mode controls are grouped
   on the right.

Analog Channels
...............

Each analog channel on the digitizer is listed on the left side of the
tab. Use the checkboxes to select which channel carries the FID signal.
Exactly one channel must be selected; the Experiment Setup validator
reports an error if zero or more than one channel is checked. For the
selected channel, configure the full vertical scale (the range from
minimum to maximum, not the per-division scale) and the vertical offset.
For a DC-coupled input the signal range is offset ± full scale.

.. note::
   In most CP-FTMW setups the digitizer should be AC-coupled, which
   makes the vertical offset effectively zero. Some digitizer
   drivers may not expose a coupling control through Blackchirp;
   in those cases the coupling must be set on the instrument directly.

Data Transfer
.............

These settings control how the waveform is encoded during transfer from
the digitizer to Blackchirp.

**Record Length**
    Number of samples per FID. Dividing the record length by the sample
    rate gives the total time spanned by the FID, which determines the
    frequency resolution of the Fourier transform.

**Sample Rate**
    The digitizer's sampling rate in samples per second. Each supported
    digitizer has a default rate; the available rates are configured in
    the per-device settings opened from the Hardware menu (see
    :doc:`/user_guide/hwdialog`).

**Bytes per Point**
    Number of bytes encoding each digitizer sample. Determined by the
    ADC bit depth (1 byte for 8-bit records, 2 bytes for 12- or 16-bit
    records) or by the accumulator width for on-board averaging. A
    digitizer driver may override this value automatically.

**Byte Order**
    Whether multi-byte samples are transmitted most-significant byte
    first (Big Endian) or least-significant byte first (Little Endian).
    A digitizer driver may override this value automatically.

Trigger
.......

Blackchirp assumes the digitizer is triggered by an edge signal. Set the
trigger channel, slope (rising or falling edge), delay, and level to
match the signal available in your experimental setup.

.. note::
   When a TTL-level marker output from an AWG is used to trigger a
   Tektronix oscilloscope, a trigger level of 0.35 V has been found to
   improve stability and phase coherence empirically. The origin of this
   improvement is not fully understood, but it is worth trying if
   coherence problems are observed.

Acquisition Setup
.................

This section controls how the digitizer encodes one or more FIDs into a
single transfer to Blackchirp.

**Block Average**
    Select this mode when the digitizer pre-averages multiple FIDs
    internally before transferring a single record to Blackchirp. For
    Tektronix oscilloscopes this typically corresponds to FastFrame
    acquisition with summary-frame transfer; for Agilent and similar
    instruments it corresponds to segmented memory averaging. The
    **# Averages** field sets how many FIDs are averaged per transfer.
    When multiple chirps are configured, the number of averages should
    match the number of chirps per AWG record.

    When Block Average is active, the digitizer accumulates the
    configured number of shots before generating a transfer event.
    Blackchirp treats each transfer as one independent record and
    co-averages it with all previous records.

**Multiple Records**
    Select this mode when a single transfer consists of several
    independent FIDs concatenated end-to-end. The **# Records** field
    sets how many FIDs are in each transfer; the total transfer length
    is N × Record Length samples. This mode is commonly used with an
    AWG configured for a sequence of chirps, letting each digitizer
    record hold the response to one chirp. Individual records can be
    scrolled through in the CP-FTMW tab, and Blackchirp can co-average
    them in post-processing. When multiple chirps are configured, the
    number of records should match the number of chirps per AWG record.

Most digitizers support only one of these two modes at a time, making
Block Average and Multiple Records mutually exclusive. In hardware that
supports both simultaneously, ensure the device has sufficient onboard
memory for the total data volume; Blackchirp does not verify memory
capacity at configuration time.

Maximizing Transfer Efficiency
..............................

Blackchirp treats each digitizer transfer as one independent record and
co-averages it with the records collected so far. Polling a
continuously-accumulating math waveform (as some oscilloscopes support)
works only with a short accumulation window and incurs the associated
overhead.

Two factors limit the acquisition rate:

1. **Processing time on the digitizer.** If the instrument spends time
   generating an internal average (e.g., a math waveform on a Tektronix
   scope), triggers are not accepted during that computation, which caps
   the effective FID rate regardless of transfer speed. FPGA-based
   instruments with hardware accumulators avoid this bottleneck entirely
   and can sustain much higher acquisition rates.

2. **Transfer time over the connection.** Blackchirp communicates with
   most digitizers over a 1 Gbps LAN interface. During a transfer,
   triggers arriving at the digitizer are ignored. For short records or
   moderate repetition rates this is rarely a bottleneck, but for very
   long records at high repetition rates increasing the number of
   averages per transfer reduces the fraction of time spent transferring
   and can significantly improve the overall FID rate.
