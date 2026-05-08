"""
Blackchirp Python Pulse Generator Driver Template

This script is loaded by the PythonPulseGenerator C++ trampoline class. It
provides a complete virtual pulse generator implementation that you can
customize for your hardware.

The pulse generator in Blackchirp controls the timing of pulsed experiment
components (gas valve, microwave protection switch, AWG gate, amplifier, etc.).
The C++ PulseGenerator base class handles:
  - Experiment configuration (applies the full channel config at start via setAll())
  - GUI updates (channel table, rep rate, enabled state)
  - Validation (range checking on width, delay, rep rate)
  - Role-based channel lookup (e.g., LIF delay channel)

Your Python script implements the low-level hw_* methods that communicate
with the actual hardware. The base class calls these one at a time, so each
method performs a single hardware operation.

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "PulseGeneratorDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel

Enum values passed as integers:
    ActiveLevel:   ActiveLow=0, ActiveHigh=1
    ChannelMode:   Normal=0, DutyCycle=1
    PGenMode:      Continuous=0, Triggered_Rising=1, Triggered_Falling=2

Sleep behavior:
    PulseGenerator.sleep(True) is declared final in C++ and calls
    set_hw_pulse_enabled(False) internally. Your set_hw_pulse_enabled()
    method is therefore called on sleep -- you do not need a separate
    sleep() method unless you want additional logic on wake.
"""


class PulseGeneratorDriver:
    """Python Pulse Generator hardware driver.

    The PulseGenerator base class calls these methods through its public
    slot interface (setPGenSetting, setRepRate, setPulseEnabled, readAll).

    Channel set methods (8 per channel):
        set_ch_width(channel, width)          -> bool
        set_ch_delay(channel, delay)          -> bool
        set_ch_active_level(channel, level)   -> bool   (level: 0=ActiveLow, 1=ActiveHigh)
        set_ch_enabled(channel, enabled)      -> bool
        set_ch_sync_ch(channel, sync_ch)      -> bool
        set_ch_mode(channel, mode)            -> bool   (mode: 0=Normal, 1=DutyCycle)
        set_ch_duty_on(channel, pulses)       -> bool
        set_ch_duty_off(channel, pulses)      -> bool

    Channel read methods (8 per channel):
        read_ch_width(channel)                -> float
        read_ch_delay(channel)                -> float
        read_ch_active_level(channel)         -> int    (0=ActiveLow, 1=ActiveHigh)
        read_ch_enabled(channel)              -> bool
        read_ch_sync_ch(channel)              -> int    (0 = no sync)
        read_ch_mode(channel)                 -> int    (0=Normal, 1=DutyCycle)
        read_ch_duty_on(channel)              -> int
        read_ch_duty_off(channel)             -> int

    Global set methods:
        set_hw_rep_rate(rep_rate)             -> bool
        set_hw_pulse_mode(mode)               -> bool   (mode: 0=Continuous, 1=Rising, 2=Falling)
        set_hw_pulse_enabled(enabled)         -> bool

    Global read methods:
        read_hw_rep_rate()                    -> float
        read_hw_pulse_mode()                  -> int    (0=Continuous, 1=Rising, 2=Falling)
        read_hw_pulse_enabled()               -> bool

    Lifecycle methods:
        initialize()      -- called once on startup
        test_connection() -- called to verify hardware
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        This is called from PulseGenerator::initialize() (via initializePGen).
        Use this to set up internal state such as channel dictionaries and
        defaults. The comm proxy is available but the connection has not been
        tested yet.
        """
        self.log.log("Pulse Generator driver initialized")

        num_channels = self.settings.get("numChannels", 8)

        # Internal state for virtual mode
        self._rep_rate = 10.0       # Hz
        self._pulse_enabled = True
        self._pulse_mode = 0        # Continuous
        self._channels = [
            {
                "width":   1e-6,    # seconds
                "delay":   0.0,     # seconds
                "level":   1,       # ActiveHigh
                "enabled": False,
                "sync_ch": 0,
                "mode":    0,       # Normal
                "duty_on": 1,
                "duty_off": 1,
            }
            for _ in range(num_channels)
        ]

    def test_connection(self):
        """Verify communication with the pulse generator.

        Called from PulseGenerator::testConnection(). If this returns True,
        the base class proceeds to readAll() (which reads back the full
        channel state from hardware).

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity:
            # response = self.comm.query("*IDN?\\n")
            # return len(response.strip()) > 0
        """
        self.log.log("Testing Pulse Generator connection")
        return True

    # =========================================================================
    # Per-channel set methods
    # =========================================================================

    def set_ch_width(self, channel, width):
        """Set the pulse width for a channel.

        Called by PulseGenerator::setPGenSetting when the user changes the
        width of a channel in the GUI or when setAll() applies an experiment
        config. The base class validates that width is within [minWidth, maxWidth]
        before calling this method.

        Args:
            channel (int): Zero-based channel index.
            width (float): Pulse width in seconds.

        Returns:
            bool: True if the hardware accepted the value, False on error.

        Examples:
            # self.comm.write(f"WIDTH {channel+1} {width:.9f}\\n")
            # return True
        """
        self._channels[channel]["width"] = width
        return True

    def set_ch_delay(self, channel, delay):
        """Set the pulse delay for a channel.

        The delay is the time from the trigger (or rep-rate clock) to the
        rising edge of the pulse. The base class validates that delay is within
        [minDelay, maxDelay] before calling this method.

        Args:
            channel (int): Zero-based channel index.
            delay (float): Pulse delay in seconds.

        Returns:
            bool: True if the hardware accepted the value, False on error.
        """
        self._channels[channel]["delay"] = delay
        return True

    def set_ch_active_level(self, channel, level):
        """Set the active (output) level for a channel.

        Controls whether the channel output is normally low (active high pulse)
        or normally high (active low pulse).

        Args:
            channel (int): Zero-based channel index.
            level (int): 0 = ActiveLow, 1 = ActiveHigh.

        Returns:
            bool: True if the hardware accepted the value, False on error.
        """
        self._channels[channel]["level"] = level
        return True

    def set_ch_enabled(self, channel, enabled):
        """Enable or disable a channel output.

        Called when the user toggles a channel in the GUI or when setAll()
        applies an experiment config. Only called if the hardware capability
        setting canDisableChannels is True (default True).

        Args:
            channel (int): Zero-based channel index.
            enabled (bool): True to enable the channel, False to disable.

        Returns:
            bool: True if the hardware accepted the value, False on error.
        """
        self._channels[channel]["enabled"] = enabled
        return True

    def set_ch_sync_ch(self, channel, sync_ch):
        """Set the sync (trigger) source channel for a channel.

        When sync_ch is non-zero, this channel's delay is measured from the
        output of sync_ch instead of from the global trigger. A value of 0
        means the channel triggers from the global source. Only called if the
        hardware capability setting canSyncToChannel is True.

        Args:
            channel (int): Zero-based channel index.
            sync_ch (int): Source channel index (1-based), or 0 for global.

        Returns:
            bool: True if the hardware accepted the value, False on error.
        """
        self._channels[channel]["sync_ch"] = sync_ch
        return True

    def set_ch_mode(self, channel, mode):
        """Set the output mode for a channel.

        In DutyCycle mode (1) the channel fires for duty_on pulses then
        inhibits for duty_off pulses in a repeating pattern. In Normal mode
        (0) the channel fires every trigger. Only called if the hardware
        capability setting canDutyCycle is True.

        Args:
            channel (int): Zero-based channel index.
            mode (int): 0 = Normal, 1 = DutyCycle.

        Returns:
            bool: True if the hardware accepted the value, False on error.
        """
        self._channels[channel]["mode"] = mode
        return True

    def set_ch_duty_on(self, channel, pulses):
        """Set the number of active pulses in the duty cycle pattern.

        Only meaningful when the channel is in DutyCycle mode. Only called if
        canDutyCycle is True.

        Args:
            channel (int): Zero-based channel index.
            pulses (int): Number of pulses to fire before inhibiting (>= 1).

        Returns:
            bool: True if the hardware accepted the value, False on error.
        """
        self._channels[channel]["duty_on"] = pulses
        return True

    def set_ch_duty_off(self, channel, pulses):
        """Set the number of inhibited pulses in the duty cycle pattern.

        Only meaningful when the channel is in DutyCycle mode. Only called if
        canDutyCycle is True.

        Args:
            channel (int): Zero-based channel index.
            pulses (int): Number of pulses to inhibit before firing again (>= 1).

        Returns:
            bool: True if the hardware accepted the value, False on error.
        """
        self._channels[channel]["duty_off"] = pulses
        return True

    # =========================================================================
    # Per-channel read methods
    # =========================================================================

    def read_ch_width(self, channel):
        """Read the current pulse width for a channel from hardware.

        Called by PulseGenerator::readChannel() which is in turn called by
        readAll() and setPGenSetting() (to verify a write succeeded).

        Args:
            channel (int): Zero-based channel index.

        Returns:
            float: Pulse width in seconds.

        Examples:
            # response = self.comm.query(f"WIDTH? {channel+1}\\n")
            # return float(response.strip())
        """
        return self._channels[channel]["width"]

    def read_ch_delay(self, channel):
        """Read the current pulse delay for a channel from hardware.

        Args:
            channel (int): Zero-based channel index.

        Returns:
            float: Pulse delay in seconds.
        """
        return self._channels[channel]["delay"]

    def read_ch_active_level(self, channel):
        """Read the active level setting for a channel from hardware.

        Args:
            channel (int): Zero-based channel index.

        Returns:
            int: 0 = ActiveLow, 1 = ActiveHigh.
        """
        return self._channels[channel]["level"]

    def read_ch_enabled(self, channel):
        """Read whether a channel output is currently enabled.

        Args:
            channel (int): Zero-based channel index.

        Returns:
            bool: True if the channel is enabled, False if disabled.
        """
        return self._channels[channel]["enabled"]

    def read_ch_sync_ch(self, channel):
        """Read the sync source channel for a channel.

        Args:
            channel (int): Zero-based channel index.

        Returns:
            int: Source channel index (1-based), or 0 for global trigger.
        """
        return self._channels[channel]["sync_ch"]

    def read_ch_mode(self, channel):
        """Read the output mode for a channel.

        Args:
            channel (int): Zero-based channel index.

        Returns:
            int: 0 = Normal, 1 = DutyCycle.
        """
        return self._channels[channel]["mode"]

    def read_ch_duty_on(self, channel):
        """Read the number of active duty cycle pulses for a channel.

        Args:
            channel (int): Zero-based channel index.

        Returns:
            int: Number of pulses to fire before inhibiting.
        """
        return self._channels[channel]["duty_on"]

    def read_ch_duty_off(self, channel):
        """Read the number of inhibited duty cycle pulses for a channel.

        Args:
            channel (int): Zero-based channel index.

        Returns:
            int: Number of pulses to inhibit before firing again.
        """
        return self._channels[channel]["duty_off"]

    # =========================================================================
    # Global set methods
    # =========================================================================

    def set_hw_rep_rate(self, rep_rate):
        """Set the global repetition rate.

        The base class validates that rep_rate is within [minRepRate, maxRepRate]
        before calling this method. After this call, readHwRepRate() is called
        to confirm the actual programmed rate.

        Args:
            rep_rate (float): Repetition rate in Hz.

        Returns:
            bool: True if the hardware accepted the value, False on error.

        Examples:
            # self.comm.write(f"RATE {rep_rate:.4f}\\n")
            # return True
        """
        self._rep_rate = rep_rate
        return True

    def set_hw_pulse_mode(self, mode):
        """Set the global trigger/repetition mode.

        Controls whether the pulse generator fires continuously at the
        programmed rep rate or waits for an external trigger. Only called
        if canTrigger is True for non-Continuous modes.

        Args:
            mode (int): 0 = Continuous, 1 = Triggered_Rising, 2 = Triggered_Falling.

        Returns:
            bool: True if the hardware accepted the value, False on error.
        """
        self._pulse_mode = mode
        return True

    def set_hw_pulse_enabled(self, enabled):
        """Enable or disable all pulse outputs globally.

        Called by the base class when the user toggles the global enable in
        the GUI, during setAll() (disables before reconfiguring, then
        re-enables), and during sleep (sleep=True disables the outputs).

        Args:
            enabled (bool): True to enable all outputs, False to disable.

        Returns:
            bool: True if the hardware accepted the value, False on error.

        Examples:
            # self.comm.write(f"OUTPUT {'ON' if enabled else 'OFF'}\\n")
            # return True
        """
        self._pulse_enabled = enabled
        return True

    # =========================================================================
    # Global read methods
    # =========================================================================

    def read_hw_rep_rate(self):
        """Read the current repetition rate from hardware.

        Called after set_hw_rep_rate() and by readAll() to confirm the
        programmed value. The base class range-checks this value against
        [minRepRate, maxRepRate] and logs an error if out of range.

        Returns:
            float: Repetition rate in Hz.

        Examples:
            # response = self.comm.query("RATE?\\n")
            # return float(response.strip())
        """
        return self._rep_rate

    def read_hw_pulse_mode(self):
        """Read the current trigger/repetition mode from hardware.

        Returns:
            int: 0 = Continuous, 1 = Triggered_Rising, 2 = Triggered_Falling.
        """
        return self._pulse_mode

    def read_hw_pulse_enabled(self):
        """Read whether pulse outputs are currently globally enabled.

        Returns:
            bool: True if outputs are enabled, False if disabled.
        """
        return self._pulse_enabled

    # =========================================================================
    # Lifecycle methods
    # =========================================================================

    def read_settings(self):
        """Reload settings from Blackchirp without restarting the process.

        Called by PythonPulseGenerator::pgReadSettings() when the user changes
        hardware settings in the GUI. Use self.settings.get() to re-read any
        configuration values that affect operation.

        Examples:
            # self._my_param = self.settings.get("myParam", default_value)
        """
        self.log.debug("Pulse Generator reloading settings")
