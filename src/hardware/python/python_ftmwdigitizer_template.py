"""
Blackchirp Python FTMW Digitizer Driver Template

This script is loaded by the PythonFtmwDigitizer C++ trampoline class. It provides
a complete virtual FTMW digitizer implementation that you can customize for
your hardware.

The FTMW digitizer in Blackchirp captures free-induction decay (FID) waveforms
following each chirped-pulse excitation. The C++ FtmwDigitizer base class handles:
  - Waveform buffering and pre-accumulation
  - Shot dispatching to AcquisitionManager
  - Experiment configuration (applies digitizer config at start via
    prepareForExperiment)

Acquisition is push-driven: your script runs its own acquisition loop in a
background thread and calls self.digi.emit_shot(raw_bytes) when data is ready.
The main thread must remain free to receive C++ IPC messages (e.g.,
end_acquisition, sleep, read_settings).

Your Python script implements configure() to set up the digitizer hardware,
begin_acquisition() to start the acquisition loop, and end_acquisition() to
stop it.

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "FtmwDigitizerDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel
    self.digi    -- push waveform data to C++ (call self.digi.emit_shot())

Configuration lifecycle:
    After a successful connection and before each experiment, the C++ side
    calls ``configure()`` with the full digitizer configuration. Your script
    should apply settings to the hardware, validate them, and return a dict
    with ``success`` (bool) and ``config`` (the validated configuration).
    The ``config`` dict in the return value will be deserialized back into
    the C++ FtmwDigitizerConfig, so any adjustments (e.g., clamped record
    length) are reflected in the application.

Waveform data format:
    ``self.digi.emit_shot(raw_bytes)`` must be called with raw waveform
    bytes. The byte layout must match the configured digitizer parameters:

    - Size = record_length × bytes_per_point × num_records (multi-record)
             or record_length × bytes_per_point (single record)
    - Each sample is a signed integer:
        - 1 byte per point (bytes_per_point=1): int8  (range -128 to 127)
        - 2 bytes per point (bytes_per_point=2): int16 (range -32768 to 32767)
    - Multi-record data is laid out as num_records contiguous records, each
      of length record_length × bytes_per_point bytes.
    - Byte order (byte_order=0 LittleEndian, byte_order=1 BigEndian) must
      be respected for 2-byte samples.

    The FID channel is specified by fid_channel in the config dict; other
    enabled analog channels may be ignored for the purposes of generating
    the waveform bytes, but real hardware may return interleaved data for
    all channels depending on the digitizer model.
"""

import random
import struct
import threading
import time


# ---------------------------------------------------------------------------
# Connection settings
# ---------------------------------------------------------------------------
# When this profile uses the **Custom** communication protocol in Blackchirp,
# self.comm is not connected to anything — the driver is expected to talk to
# its hardware on its own (vendor SDK, USB-HID library, memory-mapped device,
# etc.). Declare any required parameters here so they live in one obvious
# place at the top of the file. Edit these values for your installation.
#
# The example below targets the Spectrum Instrumentation `spcm` Python
# package (https://github.com/SpectrumInstrumentation/spcm),
# which addresses cards by a string identifier such as "/dev/spcm0" on Linux
# or "TCPIP::192.168.1.10::INSTR" on a remote networked card. Replace the
# value with whatever your acquisition card needs.
SPCM_DEVICE = "/dev/spcm0"

# Optional: timeout (ms) applied to vendor-library calls.
SPCM_TIMEOUT_MS = 5000


class FtmwDigitizerDriver:
    """Python FTMW Digitizer hardware driver.

    You must implement:
        configure(...)      -- apply/validate config, return success + config
        begin_acquisition() -- start the acquisition loop
        end_acquisition()   -- stop the acquisition loop

    Optional lifecycle methods:
        initialize()        -- one-time setup
        test_connection()   -- verify hardware communication
        sleep(sleeping)     -- enter/exit standby mode
        read_settings()     -- reload settings without restart
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        Use this to set up internal state. The comm proxy is available
        but the connection has not been tested yet.
        """
        self.log.log("FTMW Digitizer driver initialized")

        # Internal config state (populated by configure())
        self._record_length = 1000
        self._bytes_per_point = 1
        self._byte_order = 0          # 0 = LittleEndian, 1 = BigEndian
        self._multi_record = False
        self._num_records = 1
        self._fid_channel = 0
        self._block_average = False
        self._num_averages = 1
        self._configured = False

        # Acquisition thread state
        self._acquiring = False
        self._acq_thread = None

    def test_connection(self):
        """Verify communication with the digitizer.

        Called from PythonFtmwDigitizer::testConnection(). If this returns True,
        the C++ side will send a configure() call with the current config.

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity over the C++ comm transport:
            # response = self.comm.query("*IDN?\\n")
            # return "ACQIRIS" in response or "KEYSIGHT" in response

            # Or, when the Custom protocol is selected and the driver owns
            # the connection (e.g., Spectrum spcm Python package):
            # import spcm
            # try:
            #     self._card = spcm.Card(SPCM_DEVICE)
            #     self._card.timeout(SPCM_TIMEOUT_MS)
            #     return True
            # except spcm.SpcmException as exc:
            #     self.log.error(f"Spectrum card open failed: {exc}")
            #     return False
        """
        self.log.log("Testing FTMW Digitizer connection")
        return True

    def configure(self, analog_channels=None, digital_channels=None,
                  trigger=None, sample_rate=0.0, record_length=1000,
                  bytes_per_point=1, byte_order=0, block_average=False,
                  num_averages=1, multi_record=False, num_records=1,
                  fid_channel=0, **kwargs):
        """Apply and validate the digitizer configuration.

        Called after a successful connection and before each experiment.
        The C++ side sends the desired configuration; this method should:

        1. Apply settings to the hardware (sample rate, record length, etc.)
        2. Read back actual values from hardware if they may differ
        3. Return a dict with 'success' (bool) and 'config' (the validated
           configuration dict). The config dict will be deserialized back
           into the C++ FtmwDigitizerConfig, so any adjustments are
           reflected in the application.

        Config dict structure:
            analog_channels (dict): {
                "0": {"enabled": True, "full_scale": 1.0, "offset": 0.0},
                ...
            }
            digital_channels (dict): {
                "0": {"enabled": False, "input": True, "role": -1},
                ...
            }
            trigger (dict): {
                "channel": 0,       -- trigger channel index
                "slope": 0,         -- 0=RisingEdge, 1=FallingEdge
                "delay_us": 0.0,    -- trigger delay in microseconds
                "level": 0.0        -- trigger level in volts
            }
            sample_rate (float):    -- samples per second (e.g. 10e9 = 10 GSa/s)
            record_length (int):    -- number of samples per record
            bytes_per_point (int):  -- 1 or 2
            byte_order (int):       -- 0=LittleEndian, 1=BigEndian
            block_average (bool):   -- enable hardware block averaging
            num_averages (int):     -- number of averages (if block_average=True)
            multi_record (bool):    -- enable multi-record acquisition
            num_records (int):      -- records per acquisition (if multi_record=True)
            fid_channel (int):      -- analog channel index carrying the FID

        Args:
            analog_channels (dict):  Channel configuration (see above)
            digital_channels (dict): Channel configuration (see above)
            trigger (dict):          Trigger configuration (see above)
            sample_rate (float):     Desired sample rate in Sa/s
            record_length (int):     Desired record length in samples
            bytes_per_point (int):   1 or 2 bytes per sample
            byte_order (int):        0=LittleEndian, 1=BigEndian
            block_average (bool):    Hardware block averaging enable
            num_averages (int):      Number of averages
            multi_record (bool):     Multi-record acquisition enable
            num_records (int):       Number of records per acquisition
            fid_channel (int):       Channel carrying the FID signal
            **kwargs:                Reserved for future use

        Returns:
            dict: Must contain:
                - 'success' (bool): True if configuration applied OK
                - 'config' (dict): The validated configuration. Should have
                  the same structure as the input arguments. Omitted keys
                  are left unchanged on the C++ side.
        """
        # Store config for use in _acquisition_loop()
        self._record_length   = record_length
        self._bytes_per_point = bytes_per_point
        self._byte_order      = byte_order
        self._multi_record    = multi_record
        self._num_records     = num_records
        self._fid_channel     = fid_channel
        self._block_average   = block_average
        self._num_averages    = num_averages
        self._configured      = True

        enabled_analog  = sum(1 for ch in (analog_channels  or {}).values()
                              if ch.get("enabled"))
        enabled_digital = sum(1 for ch in (digital_channels or {}).values()
                              if ch.get("enabled"))
        self.log.debug(
            f"Configured: rate={sample_rate/1e9:.1f} GSa/s, "
            f"reclen={record_length}, bpp={bytes_per_point}, "
            f"{enabled_analog} analog / {enabled_digital} digital channels enabled, "
            f"FID ch={fid_channel}"
        )

        # Build validated config dict (virtual: return input unchanged)
        validated = {
            "analog_channels":  analog_channels  or {},
            "digital_channels": digital_channels or {},
            "trigger":          trigger          or {},
            "sample_rate":      sample_rate,
            "record_length":    record_length,
            "bytes_per_point":  bytes_per_point,
            "byte_order":       byte_order,
            "block_average":    block_average,
            "num_averages":     num_averages,
            "multi_record":     multi_record,
            "num_records":      num_records,
            "fid_channel":      fid_channel,
        }

        return {"success": True, "config": validated}

    def begin_acquisition(self):
        """Start the acquisition loop.

        Called when a Blackchirp experiment starts. Launches a background
        thread that reads waveforms from hardware and pushes them to C++
        via self.digi.emit_shot(). The main thread remains free to handle
        IPC messages (e.g., end_acquisition).

        Examples:
            # Arm the digitizer before starting the loop:
            # self.comm.write("ARM\\n")
        """
        self.log.debug("FTMW Digitizer beginning acquisition")
        self._acquiring = True
        self._acq_thread = threading.Thread(target=self._acquisition_loop,
                                            daemon=True)
        self._acq_thread.start()

    def end_acquisition(self):
        """Stop the acquisition loop.

        Called when a Blackchirp experiment stops or is aborted. Signals
        the acquisition thread to stop and waits for it to finish.

        Examples:
            # Disarm the digitizer:
            # self.comm.write("STOP\\n")
        """
        self.log.debug("FTMW Digitizer ending acquisition")
        self._acquiring = False
        if self._acq_thread is not None:
            self._acq_thread.join(timeout=5.0)
            self._acq_thread = None

    def _acquisition_loop(self):
        """Background thread: read waveforms and push them to C++.

        Runs until self._acquiring is False. Replace the body with real
        hardware reads for your digitizer.
        """
        while self._acquiring:
            raw = self._generate_virtual_waveform()
            # For real hardware: raw = self.comm.read_bytes(total_bytes)
            self.digi.emit_shot(raw)
            time.sleep(0.2)  # virtual mode: ~5 Hz

    def _generate_virtual_waveform(self):
        """Generate random waveform bytes matching the configured format."""
        if not self._configured:
            return b""

        num_records = self._num_records if self._multi_record else 1
        total_samples = self._record_length * num_records

        if self._bytes_per_point == 1:
            fmt = f"{total_samples}b"
            samples = [random.randint(-128, 127) for _ in range(total_samples)]
            return struct.pack(fmt, *samples)
        else:
            endian = "<" if self._byte_order == 0 else ">"
            fmt = f"{endian}{total_samples}h"
            samples = [random.randint(-32768, 32767) for _ in range(total_samples)]
            return struct.pack(fmt, *samples)

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("FTMW Digitizer entering sleep mode")
        else:
            self.log.debug("FTMW Digitizer waking from sleep mode")

    def read_settings(self):
        """Reload settings from Blackchirp without restarting the process.

        Called by PythonFtmwDigitizer::ftmwReadSettings() when the user changes
        hardware settings in the GUI. Use self.settings.get() to re-read
        any configuration values that affect operation.

        Examples:
            # self._timeout_ms = self.settings.get("timeoutMs", 5000)
        """
        self.log.debug("FTMW Digitizer reloading settings")
