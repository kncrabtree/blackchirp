"""
Blackchirp Python IO Board Driver Template

This script is loaded by the PythonIOBoard C++ trampoline class. It provides
a complete virtual IO Board implementation that you can customize.

The IO Board in Blackchirp reads analog voltages and digital input states.
Analog channels appear in rolling data plots during experiments; digital
channels can be used as validation conditions (e.g., abort if interlock open).

The C++ IOBoard base class handles aux data formatting, validation, and
experiment integration. Your script only needs to return raw channel readings
for the channels requested by the C++ side.

Class name must match the pythonClass setting (default: "IOBoardDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel

Configuration lifecycle:
    After a successful connection and before each experiment, the C++ side
    calls ``configure()`` with the full IOBoard configuration. Your script
    should apply settings to the hardware, validate them, and return a dict
    with ``success`` (bool) and ``config`` (the validated configuration).
    The ``config`` dict in the return value will be deserialized back into
    the C++ IOBoardConfig, so any changes (e.g., clamped full_scale values)
    are reflected in the application.
"""

import random


class IOBoardDriver:
    """Python IO Board hardware driver.

    You must implement:
        configure(...)         -- apply/validate config, return success + config
        read_analog_channels(channels)  -- returns dict[int, float]
        read_digital_channels(channels) -- returns dict[int, bool]

    The ``channels`` argument is a list of enabled channel indices that the
    C++ side expects you to read. Only return data for these channels.

    All other methods are optional.
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        Use this to set up internal state. The comm proxy is available
        but the connection has not been tested yet.
        """
        self.config = {}
        self.log.log("IO Board driver initialized")

    def test_connection(self):
        """Verify communication with the IO board hardware.

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # For a LabJack or DAQ device:
            # response = self.comm.query("*IDN?\\n")
            # return len(response.strip()) > 0

            # For virtual/testing:
            return True
        """
        self.log.log("Testing IO Board connection")
        return True

    def configure(self, analog_channels=None, digital_channels=None,
                  trigger=None, **kwargs):
        """Apply and validate the IOBoard configuration on the hardware.

        Called after a successful connection and before each experiment.
        The C++ side sends the desired configuration; this method should:

        1. Apply settings to the hardware (e.g., set channel voltage ranges)
        2. Read back actual values from hardware if they may differ
        3. Update the config dict with validated/actual values
        4. Return a dict with 'success' (bool) and 'config' (the validated
           configuration dict)

        If the hardware cannot satisfy a requested setting, the script
        decides whether to adjust silently, log a warning, or fail.

        Args:
            analog_channels: dict of {index_str: {enabled, full_scale, offset, name?}}
            digital_channels: dict of {index_str: {enabled, input, role, name?}}
            trigger: dict of {channel, slope, delay_us, level}
            **kwargs: sample_rate, record_length, bytes_per_point, byte_order,
                      block_average, num_averages, multi_record, num_records

        Returns:
            dict: Must contain:
                - 'success' (bool): True if configuration applied OK
                - 'config' (dict): The validated configuration. Should have
                  the same structure as the input (analog_channels,
                  digital_channels, trigger, plus scalar fields in kwargs).
                  Omitted keys are left unchanged on the C++ side.
        """
        self.config = {
            "analog_channels": analog_channels or {},
            "digital_channels": digital_channels or {},
            "trigger": trigger or {},
            **kwargs,
        }

        n_an = sum(1 for ch in (analog_channels or {}).values()
                   if ch.get("enabled"))
        n_dig = sum(1 for ch in (digital_channels or {}).values()
                    if ch.get("enabled"))
        self.log.debug(
            f"Configured: {n_an} analog, {n_dig} digital channels enabled"
        )

        # Return success and the (possibly modified) config
        return {"success": True, "config": self.config}

    def read_analog_channels(self, channels=None):
        """Read the requested analog input channels.

        Called periodically by the IOBoard base class (via readAuxData).
        Analog readings appear in rolling data plots and are recorded
        during experiments.

        Args:
            channels (list[int]): Enabled channel indices to read. Only
                return data for these channels. May be empty if no analog
                channels are enabled.

        Returns:
            dict[int, float]: Channel index -> voltage reading.
                Keys must be from the ``channels`` list.
                Return an empty dict if channels is empty.

        Examples:
            # Read from a real device:
            # voltages = {}
            # for ch in (channels or []):
            #     resp = self.comm.query(f"AIN{ch}?\\n")
            #     voltages[ch] = float(resp.strip())
            # return voltages

            # Virtual: return random voltages for requested channels
        """
        if not channels:
            return {}
        return {ch: random.uniform(0.0, 2.44) for ch in channels}

    def read_digital_channels(self, channels=None):
        """Read the requested digital input channels.

        Called periodically by the IOBoard base class (via readValidationData).
        Digital channel states are checked against validation conditions.
        For example, a safety interlock channel can be configured to abort
        an experiment if it goes False.

        Args:
            channels (list[int]): Enabled channel indices to read. Only
                return data for these channels. May be empty if no digital
                channels are enabled.

        Returns:
            dict[int, bool]: Channel index -> state (True/False).
                Keys must be from the ``channels`` list.
                Return an empty dict if channels is empty.

        Examples:
            # Read from a real device:
            # states = {}
            # for ch in (channels or []):
            #     resp = self.comm.query(f"DIN{ch}?\\n")
            #     states[ch] = resp.strip() == "1"
            # return states

            # Virtual: all requested channels read True (safe)
        """
        if not channels:
            return {}
        return {ch: True for ch in channels}

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("IO Board entering sleep mode")
        else:
            self.log.debug("IO Board waking from sleep mode")
