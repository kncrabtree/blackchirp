FTMW Digitizer
==============

* Overview_
* Settings_
* Implementations_

Overview
--------

The FTMW Digitizer is the fast digitizer used to record FIDs during a CP-FTMW experiment. At is simplest, on each trigger event, the digitizer simply records a given number of data points at the desired sample rate, then transmits the raw digitizer values to Blackchirp for averaging. However, some devices support fast retriggering (e.g., Tektronix's "FastFrame" mode), and it is possible to acquire a series of *frames* in sequence. For applications involving pulsed gas sources, this allows each gas pulse to be probed by a series of chirps. The resultant frames may be transmitted to Blackchirp individually. Some digitizers also support internal averaging (sometimes called *block averaging*), in which some number of records may be averaged by the digitizer prior to being transmitted to Blackchirp. For some scopes, this block averaging is only possible among frames captured in a single acquisition event, thereby resulting in a single frame. Others though can block average multiple records in parallel prior to transmitting to Blackchirp, and in this case the final record would still contain multiple (averaged) frames.

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

Virtual (0)
...........

The virtual implementation can be thought of as a "read-only" device useful during a first build of Blackchirp and/or for a machine that is not connected to any hardware at all (e.g. office or shared group computer) for data analysis purposes.

Tektronix DSA71604C (1)
.......................

TO DO

Tektronix MSO72004C (2)
.......................

TO DO

Spectrum M4i2220x8 (3)
.......................

TO DO

Keysight DSOX92004A (4)
.......................

The `DSOX92004A <https://www.keysight.com/us/en/product/DSOX92004A/infiniium-high-performance-oscilloscope-20-ghz.html>`_ is an 80 GSa/s oscilloscope with a bandwidth of 20 GHz, upgradable to 33 GHz, and this implementation communicates over a TCP socket. A static IP address has to be set in the Windows OS running on the scope and usual remote communication is on port 5025. Caution: This implementation has not been tested since Blackchirp's update to v1.0.

Tektronix MSO64B (5)
.......................

TO DO

Keysight DSOV204A (6)
.......................

The `DSOV204A <https://www.keysight.com/us/en/product/DSOV204A/infiniium-v-series-oscilloscope-20-ghz-4-analog-channels.html>`_ is an 80 GSa/s oscilloscope with a maximum bandwidth of 20 GHz, and this implementation communicates over a TCP socket. A static IP address has to be set in the Windows OS running on the scope and usual remote communication is on port 5025. This implementation is currently coded so that the scope can be triggered on any of its 4 analog channels but triggering on the AUX channel is recommended.

Tektronix DPO71254B (7)
.......................

The `DPO71254B <https://www.tek.com/en/oscilloscope/dpo70000-mso70000-manual-18>`_ is a 50 GSa/s oscilloscope with a maximum bandwidth of 12.5 GHz, and this implementation communicates over a TCP socket. A static IP address has to be set in the Windows OS running on the scope and the default TekVisa software running on the scope communicates on port 4000. This implementation is currently coded so that the scope can be triggered on any of its 4 analog channels but triggering on the AUX channel is recommended. This implementation is currently being tested.

Tektronix DPO72004 (8)
.......................

The `DPO72004 <https://www.tek.com/en/oscilloscope/dpo70000-mso70000-manual-18>`_ is a 50 GSa/s oscilloscope with a maximum bandwidth of 20 GHz, and this implementation communicates over a TCP socket. A static IP address has to be set in the Windows OS running on the scope and the default TekVisa software running on the scope communicates on port 4000. This implementation is currently coded so that the scope can be triggered on any of its 4 analog channels but triggering on the AUX channel is recommended. This implementation is currently being tested.


**Known Issues**

 * There have been reports that once a specific error occurs, the software on the Keysight scope crashes. Closing and reopening the scope software will resolve this issue.
 * There have been reports that once a specific error occurs, the Tektronix scope will not allow TCP connection and the socket will time out. Restarting the scope resolves this issue.
