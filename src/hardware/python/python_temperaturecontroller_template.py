"""
Blackchirp Python Temperature Controller Driver Template

This script is loaded by the PythonTemperatureController C++ trampoline class. It
provides a complete virtual temperature controller implementation that you can
customize for your hardware.

The temperature controller in Blackchirp reads temperatures from one or more
channels and records them as auxiliary data during experiments. The C++
TemperatureController base class handles:
  - Periodic polling (calls hw_read_temperature() on a timer for each enabled channel)
  - GUI updates (temperature displays per channel)
  - Experiment configuration (channel enable/disable, channel names)
  - Aux data recording (temperatures in rolling data plots and experiment headers)

Your Python script implements the low-level hw_* method that communicates with
the actual hardware.

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "TemperatureControllerDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel
"""

import math
import random


class TemperatureControllerDriver:
    """Python Temperature Controller hardware driver.

    The TemperatureController base class calls these methods through its polling
    timer. The number of temperature channels is configured in Blackchirp's
    hardware settings (default: 4).

    Required methods:
        hw_read_temperature(channel) -> float  (temperature, or NaN on error)

    Lifecycle methods (called by base class):
        initialize()      -- called once on startup (via tcInitialize)
        test_connection() -- called to verify hardware (via tcTestConnection)
        sleep(sleeping)   -- called on hardware standby transitions
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        This is called from TemperatureController::initialize() (via tcInitialize).
        The base class creates the poll timer before calling this. Use this
        to set up internal state.

        The comm proxy is available but the connection has not been tested yet.
        """
        self.log.log("Temperature Controller driver initialized")

        # Internal state for virtual mode
        self._temperatures = {}

    def test_connection(self):
        """Verify communication with the temperature controller.

        Called from TemperatureController::testConnection() (via tcTestConnection).
        If this returns True, the base class starts the poll timer and schedules
        an initial readAll().

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity:
            # response = self.comm.query("*IDN?\\n")
            # return len(response.strip()) > 0
        """
        self.log.log("Testing Temperature Controller connection")

        # Read number of channels from settings (matches C++ numChannels default)
        num_channels = int(self.settings.get("numChannels", 4))
        self.log.log(f"Configured for {num_channels} temperature channels")

        # Initialize virtual channel state (4-6 K range, like VirtualTemperatureController)
        for ch in range(num_channels):
            self._temperatures[ch] = random.uniform(4.0, 6.0)

        return True

    # =========================================================================
    # Temperature Channel Methods
    # =========================================================================

    def hw_read_temperature(self, channel):
        """Read the current temperature for a channel.

        Called periodically by the TemperatureController poll timer for each
        enabled channel. The base class uses the return value to update the
        GUI and aux data. If NaN is returned, the base class treats it as an
        error and stops polling.

        Args:
            channel (int): 0-based channel index.

        Returns:
            float: Temperature in the channel's configured units (default: K).
                   Return float('nan') to indicate an error.

        Examples:
            # Query a Lakeshore-style instrument:
            # response = self.comm.query(f"KRDG? {channel + 1}\\n")
            # try:
            #     return float(response.strip())
            # except ValueError:
            #     return float('nan')
        """
        # Virtual: simulate temperature near 4-6 K with small noise
        base = self._temperatures.get(channel, 5.0)
        temperature = base + random.gauss(0, 0.01)
        self._temperatures[channel] = temperature
        return temperature

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("Temperature Controller entering sleep mode")
        else:
            self.log.debug("Temperature Controller waking from sleep mode")
