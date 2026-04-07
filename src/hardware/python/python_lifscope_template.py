"""
Blackchirp Python LIF Digitizer Driver Template

This script is loaded by the PythonLifScope C++ trampoline class. It provides
a complete virtual LIF digitizer implementation that you can customize for
your hardware.

The LIF digitizer in Blackchirp captures laser-induced fluorescence waveforms.
The C++ LifScope base class handles:
  - Acquisition gating and discard logic (setAcquisitionGated / discardCount)
  - Waveform dispatching via the waveformRead signal to the LIF processing chain
  - Experiment configuration (calls configure() via prepareForExperiment())

Your Python script implements configure() to apply/validate settings and
begin_acquisition() / end_acquisition() to control a background acquisition
thread that pushes waveforms via self.scope.emit_shot().

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "LifScopeDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel
    self.scope    -- push waveform data to C++ (call self.scope.emit_shot())

Configuration lifecycle:
    configure() is called after a successful connection and before each
    experiment (via LifScope::prepareForExperiment()). Your script should
    apply settings to the hardware and return a dict with 'success' (bool)
    and 'config' (the validated configuration). The config dict is
    deserialized back into C++ LifDigitizerConfig, so any adjustments
    (e.g., clamped record length) are reflected in the application.

Waveform data format:
    self.scope.emit_shot(raw_bytes) must be called with raw bytes where each
    element is a signed integer (qint8 / int8). The byte layout depends on
    the channel_order setting:

    Sequential layout (channel_order=0):
        [lif_channel data] [ref_channel data]  (if ref_enabled=True)
        or just [lif_channel data]             (if ref_enabled=False)
        Each channel block: record_length × bytes_per_point bytes

    Interleaved layout (channel_order=1):
        Pairs of (lif sample, ref sample) for each point across record_length
        Each sample: bytes_per_point bytes

    Sample encoding:
        bytes_per_point=1: int8  (range -128 to 127)
        bytes_per_point=2: int16 (range -32768 to 32767), byte_order respected
"""

import random
import struct
import threading
import time


class LifScopeDriver:
    """Python LIF Digitizer hardware driver.

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
        """Called once when the hardware object is first created."""
        self.log.log("LIF Scope driver initialized")

        # Internal config state (populated by configure())
        self._record_length   = 1000
        self._bytes_per_point = 1
        self._byte_order      = 0      # 0 = LittleEndian, 1 = BigEndian
        self._lif_channel     = 1
        self._ref_channel     = 2
        self._ref_enabled     = False
        self._channel_order   = 0      # 0 = Sequential, 1 = Interleaved
        self._configured      = False

        # Acquisition thread state
        self._acquiring  = False
        self._acq_thread = None

    def test_connection(self):
        """Verify communication with the digitizer.

        Returns:
            bool: True if communication is working, False otherwise.
        """
        self.log.log("Testing LIF Scope connection")
        return True

    def configure(self, analog_channels=None, digital_channels=None,
                  trigger=None, sample_rate=0.0, record_length=1000,
                  bytes_per_point=1, byte_order=0, block_average=False,
                  num_averages=1, multi_record=False, num_records=1,
                  lif_channel=1, ref_channel=2, ref_enabled=False,
                  channel_order=0, **kwargs):
        """Apply and validate the digitizer configuration.

        Called after a successful connection and before each experiment.
        The C++ side sends the desired configuration; this method should:

        1. Apply settings to the hardware
        2. Read back actual values if they may differ
        3. Return a dict with 'success' (bool) and 'config' (the validated
           configuration dict). Omitted keys are left unchanged on the C++ side.

        Config dict structure:
            analog_channels (dict): {
                "1": {"enabled": True, "full_scale": 1.0, "offset": 0.0},
                ...
            }
            digital_channels (dict): {
                "0": {"enabled": False, "input": True, "role": -1},
                ...
            }
            trigger (dict): {
                "channel": 0,
                "slope": 0,         -- 0=RisingEdge, 1=FallingEdge
                "delay_us": 0.0,
                "level": 0.0
            }
            sample_rate (float):    -- samples per second
            record_length (int):    -- samples per record
            bytes_per_point (int):  -- 1 or 2
            byte_order (int):       -- 0=LittleEndian, 1=BigEndian
            block_average (bool):   -- hardware block averaging
            num_averages (int):     -- averages (if block_average=True)
            multi_record (bool):    -- multi-record acquisition
            num_records (int):      -- records per acquisition
            lif_channel (int):      -- analog channel carrying the LIF signal
            ref_channel (int):      -- analog channel carrying the reference
            ref_enabled (bool):     -- whether reference channel is active
            channel_order (int):    -- 0=Sequential, 1=Interleaved

        Returns:
            dict: Must contain 'success' (bool) and 'config' (dict).
        """
        self._record_length   = record_length
        self._bytes_per_point = bytes_per_point
        self._byte_order      = byte_order
        self._lif_channel     = lif_channel
        self._ref_channel     = ref_channel
        self._ref_enabled     = ref_enabled
        self._channel_order   = channel_order
        self._configured      = True

        self.log.debug(
            f"Configured: rate={sample_rate/1e6:.1f} MSa/s, "
            f"reclen={record_length}, bpp={bytes_per_point}, "
            f"lif_ch={lif_channel}, ref_enabled={ref_enabled}, "
            f"channel_order={'Sequential' if channel_order == 0 else 'Interleaved'}"
        )

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
            "lif_channel":      lif_channel,
            "ref_channel":      ref_channel,
            "ref_enabled":      ref_enabled,
            "channel_order":    0,  # virtual always uses Sequential
        }

        return {"success": True, "config": validated}

    def begin_acquisition(self):
        """Start the acquisition loop.

        Launches a background thread that reads waveforms and pushes them
        to C++ via self.scope.emit_shot(). The main thread remains free
        to handle IPC messages (e.g., end_acquisition).
        """
        self.log.debug("LIF Scope beginning acquisition")
        self._acquiring  = True
        self._acq_thread = threading.Thread(target=self._acquisition_loop,
                                            daemon=True)
        self._acq_thread.start()

    def end_acquisition(self):
        """Stop the acquisition loop."""
        self.log.debug("LIF Scope ending acquisition")
        self._acquiring = False
        if self._acq_thread is not None:
            self._acq_thread.join(timeout=5.0)
            self._acq_thread = None

    def _acquisition_loop(self):
        """Background thread: read waveforms and push them to C++."""
        while self._acquiring:
            raw = self._generate_virtual_waveform()
            self.scope.emit_shot(raw)
            time.sleep(0.2)  # virtual mode: ~5 Hz

    def _generate_virtual_waveform(self):
        """Generate random waveform bytes in Sequential layout."""
        if not self._configured:
            return b""

        num_channels = 2 if self._ref_enabled else 1
        total_samples = self._record_length * num_channels

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
        """Called when hardware enters or exits standby mode."""
        if sleeping:
            self.log.debug("LIF Scope entering sleep mode")
        else:
            self.log.debug("LIF Scope waking from sleep mode")

    def read_settings(self):
        """Reload settings from Blackchirp without restarting the process."""
        self.log.debug("LIF Scope reloading settings")
