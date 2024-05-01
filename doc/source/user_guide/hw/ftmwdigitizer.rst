FTMW Digitizer
==============

* Overview_
* Settings_
* Implementations_

Overview
--------

The FTMW Digitizer is the fast digitizer used to record FIDs during a CP-FTMW experiment. At is simplest, on each trigger event, the digitizer simply records a given number of data points at the desired sample rate, then transmits the raw digitizer values to Blackchirp for averaging. However, some devices support fast retriggering (e.g., Tektronix's "FastFrame" mode), and it is possible to acquire a series of *frames* in sequence. For applications involving pulsed gas sources, this allows each gas pulse to be probed by a series of chirps. The resultant frames may be transmitted to Blackchirp individually. Some digitizers also support internal averaging (sometimes called *block averaging*), in which some number of records may be averaged by the digitizer prior to being transmitted to Blackchirp. For some scopes, this block averaging is only possible among frames captured in a single acquisition event, thereby resulting in a single frame. Others though can block average multiple records in parallel prior to transmitting to Blackchirp, and in this case the final record would still contain multiple (averaged) frames.

No matter how many analog channels the digitizer possesses, Blackchirp only records data from a single input channel.

Settings
--------

 * ``bandwidthMHz`` (float): The analog bandwidth of the digitizer, in MHz. This value has no effect on the program at present.
 * ``canBlockAndMultiRecord`` (true/false): Indicates whether the digitizer can simultaneously average multiple frames separately. In this case, the record transmitted to Blackchirp would contain several FIDs, each of which has been averaged for a designated number of shots.
 * ``canBlockAverage`` (true/false): Indicates whether the digitizer can perform averaging prior to transmitting data to Blackchirp.
 * ``canMultiRecord`` (true/false): Indicates whether the digitizer supports transmitting a single record containing multiple frames.
 * ``hasAuxTriggerChannel`` (true/false): Indicates whether the digitizer has a separate external trigger channel in addition to the possibility of triggering on an input channel.
 * ``maxAverages`` (int): Maximum number of averages that can be accommodated in block averaging mode. This may be limited by the number of bytes used to store averaged data on the device.
 * ``maxBytesPerPoint`` (int): Maximum number of bytes that may be used to encode data. The value entered here affects the range of allowed bytes per point values on the digitizer configuration page, but most implementations will override this value, setting the appropriate number for the data requested.
 * ``maxFullScale`` (float): Maximum full scale voltage for the digitizer, in V.
 * ``maxRecordLength`` (int): Maximum length of an FID record, typically limited by digitizer memory. Note that enabling multi record mode will decrease the maximum length, as the memory is then divided into multiple records. For some devices, enabling block averaging also limits the memory available.
 * ``maxRecords`` (int): Maximum number of records that can be requested in multi record mode. The actual number of records that is possible may be limited by scope memory for long records.
 * ``maxTrigDelayUs`` (float): Maximum delay between the trigger event and the start of the record, in μs. A positive delay means that the start of the record begins after the trigger, while a negative delay means that the start of the record begins before the trigger.
 * ``maxTrigLevel`` (float): Maximum edge trigger level, in V.
 * ``maxVOffset`` (float): Maximum vertical offset on an input channel.
 * ``minFullScale`` (float): Minimum full scale voltage for the digitizer, in V.
 * ``minTrigDelayUs`` (float): Minimum delay between the trigger event and the start of the record, in μs. A positive delay means that the start of the record begins after the trigger, while a negative delay means that the start of the record begins before the trigger.
 * ``minTrigLevel`` (float): Minimum edge trigger level, in V.
 * ``maxVOffset`` (float): Minimum vertical offset on an input channel.
 * ``sampleRates`` (menu): Allowed sample rates. Most scopes support only a few discrete values. For each entry, there are 2 subitems:
   - ``text`` (string): The text to be displayed in a drop-down options box when configuring the digitizer.
   - ``value`` (float): The sample rate in Sa/s (or Hz)

Implementations
---------------

Virtual (virtual)
.................

The virtual implementation comes with a resource file that contains a sample chirp-FID waveform that is 750k points in length. It adjusts the vertical scaling, sample rate, byte order, etc, to change the encoding of the data dynamically as settings are adjusted. Regular users should only enable this device if running Blackchirp on a machine for viewing data.

Tektronix DSA71604C (dsa71604c)
...............................

This is a 4-channel, 100 GSa/sec oscilloscope with 16 GHz analog bandwidth. It supports FastFrame acquisition which allows a variable number of waveforms to be collected with a low retrigger interval (about 4 &mu;s); the number of these frames is limited by the scope's memory which is variable. The scope is capable of coaveraging the frames collected and sending only a record containing the average; Blackchirp uses this mode when "Block Averaging" is enabled. Communication takes place over TCP, and it is recommended that the scope be connected via link-local networking with a minimum bandwidth of 1 Gbps. That said, often the data rate of the scope is limited by its own internal processing, not the bandwidth of the connection.

Tektronix MSO72004C (mso72004c)
...............................

Virtually identical to the DSA71604C, except that its bandwidth is 20 GHz.

Keysight DSOV204A (dsov204a)
............................

This Keysight scope is more or less equivalent to the MSO72004C.

Keysight DSOx92004a (dsox92004a)
................................

Also equivalent to the DSOV204A and MSO72004C.

Tektronix MSO64B (mso64b)
.........................

A 4-channel scope with 2.5 GHz bandwidth, appropriate for segmented LO scanning setups. However, when Tektronix switched to a new FastFrame backend within their scopes, they broke the operation of "CURVESTREAM" mode which allows for fast, real-time data transfer. As a result, the data transfer rate of this scope is extremely limited.

Spectrum Instrumentation M4i2220x8 (m4i2220x8)
..............................................

A high-speed digitizer with an acquisition rate of 2.5 GSa/s and an analog bandwidth of 1.25 GHz, appropriate for segmented LO scanning setups. The implementation here requires that the device have the "block averaging" firmware module enabled, and as a result the acquisition rate can be extremely fast (50,000 FIDs/sec has been possible). This digitizer requires that the spcm drivers from Spectrum Instrumentation are installed and linked to the application at compile time.

