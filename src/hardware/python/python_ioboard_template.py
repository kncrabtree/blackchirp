"""
Blackchirp Python IO Board Driver Template

This script is loaded by the PythonIOBoard C++ trampoline class. It provides
a complete virtual IO Board implementation that you can customize.

The IO Board in Blackchirp reads analog voltages and digital input states.
Analog channels appear in rolling data plots during experiments; digital
channels can be used as validation conditions (e.g., abort if interlock open).

The C++ IOBoard base class handles aux data formatting, validation, and
experiment integration. Your script only needs to return raw channel readings.

Class name must match the pythonClass setting (default: "IOBoardDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel
"""

import random


class IOBoardDriver:
    """Python IO Board hardware driver.

    You must implement two methods:
        read_analog_channels()  -- returns dict[int, float]
        read_digital_channels() -- returns dict[int, bool]

    Channel indices should match the numAnalogChannels / numDigitalChannels
    configured in Blackchirp's hardware settings. Only return data for
    channels that are enabled.

    All other methods are optional.
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        Use this to set up internal state. The comm proxy is available
        but the connection has not been tested yet.
        """
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

    def read_analog_channels(self):
        """Read all enabled analog input channels.

        Called periodically by the IOBoard base class (via readAuxData).
        Analog readings appear in rolling data plots and are recorded
        during experiments.

        Returns:
            dict[int, float]: Channel index -> voltage reading.
                Keys are 1-based channel indices matching the hardware
                configuration in Blackchirp. Only include enabled channels.
                Return an empty dict if no channels are configured.

        Examples:
            # Read from a real device:
            # voltages = {}
            # for ch in range(1, 9):
            #     resp = self.comm.query(f"AIN{ch}?\\n")
            #     voltages[ch] = float(resp.strip())
            # return voltages

            # Virtual: return random voltages for channels 1-4
            return {i: random.uniform(0.0, 2.44) for i in range(1, 5)}
        """
        return {i: random.uniform(0.0, 2.44) for i in range(1, 5)}

    def read_digital_channels(self):
        """Read all enabled digital input channels.

        Called periodically by the IOBoard base class (via readValidationData).
        Digital channel states are checked against validation conditions.
        For example, a safety interlock channel can be configured to abort
        an experiment if it goes False.

        Returns:
            dict[int, bool]: Channel index -> state (True/False).
                Keys are 1-based channel indices matching the hardware
                configuration. Only include enabled channels.
                Return an empty dict if no channels are configured.

        Examples:
            # Read from a real device:
            # states = {}
            # for ch in range(1, 9):
            #     resp = self.comm.query(f"DIN{ch}?\\n")
            #     states[ch] = resp.strip() == "1"
            # return states

            # Virtual: all channels read True (safe)
            return {i: True for i in range(1, 5)}
        """
        return {i: True for i in range(1, 5)}

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("IO Board entering sleep mode")
        else:
            self.log.debug("IO Board waking from sleep mode")
