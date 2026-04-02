"""
Blackchirp Python Flow Controller Driver Template

This script is loaded by the PythonFlowController C++ trampoline class. It
provides a complete virtual flow controller implementation that you can
customize for your hardware.

The flow controller in Blackchirp manages mass flow controllers (MFCs) and
a pressure gauge/controller. The C++ FlowController base class handles:
  - Periodic polling (calls hw_read_* methods on a timer)
  - GUI updates (flow rates, pressure, setpoints)
  - Experiment configuration (setpoints, pressure control mode)
  - Aux data recording (flows and pressure in rolling data plots)

Your Python script implements the low-level hw_* methods that communicate
with the actual hardware.

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "FlowControllerDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel
"""

import random


class FlowControllerDriver:
    """Python Flow Controller hardware driver.

    The FlowController base class calls these methods through its polling
    timer and public slot interface. The number of flow channels is
    configured in Blackchirp's hardware settings (default: 4).

    Required methods (10 total):
        hw_read_flow(channel)            -> float  (flow rate, or -1.0 on error)
        hw_read_flow_setpoint(channel)   -> float  (setpoint, or -1.0 on error)
        hw_read_pressure()               -> float  (pressure, or -1.0 on error)
        hw_read_pressure_setpoint()      -> float  (setpoint, or -1.0 on error)
        hw_read_pressure_control_mode()  -> int    (1=on, 0=off, -1=error)
        hw_set_flow_setpoint(channel, value) -> None
        hw_set_pressure_setpoint(value)      -> None
        hw_set_pressure_control_mode(enabled) -> None

    Lifecycle methods (called by base class):
        initialize()      -- called once on startup (via fcInitialize)
        test_connection() -- called to verify hardware (via fcTestConnection)
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        This is called from FlowController::initialize() (via fcInitialize).
        The base class creates the poll timer before calling this. Use this
        to set up internal state.

        The comm proxy is available but the connection has not been tested yet.
        """
        self.log.log("Flow Controller driver initialized")

        # Internal state for virtual mode
        self._flows = {}
        self._flow_setpoints = {}
        self._pressure = 0.0
        self._pressure_setpoint = 0.0
        self._pressure_control = False

    def test_connection(self):
        """Verify communication with the flow controller.

        Called from FlowController::testConnection() (via fcTestConnection).
        If this returns True, the base class starts the poll timer and
        schedules an initial readAll() after 1 second.

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity:
            # response = self.comm.query("*IDN?\\n")
            # return len(response.strip()) > 0
        """
        self.log.log("Testing Flow Controller connection")

        # Read number of channels from settings (matches C++ flowChannels default)
        num_channels = int(self.settings.get("flowChannels", 4))
        self.log.log(f"Configured for {num_channels} flow channels")

        # Initialize virtual channel state
        for ch in range(num_channels):
            self._flows[ch] = 0.0
            self._flow_setpoints[ch] = 0.0

        return True

    # =========================================================================
    # Flow Channel Methods
    # =========================================================================

    def hw_read_flow(self, channel):
        """Read the current flow rate for a channel.

        Called periodically by the FlowController poll timer (via readAll).
        The base class uses the return value to update the GUI and aux data.

        Args:
            channel (int): 0-based channel index.

        Returns:
            float: Flow rate in the channel's configured units (e.g., sccm).
                   Return -1.0 to indicate an error (value will be ignored).

        Examples:
            # Read from hardware:
            # response = self.comm.query(f"FLOW{channel+1}?\\n")
            # return float(response.strip())
        """
        # Virtual: simulate flow near setpoint with noise
        sp = self._flow_setpoints.get(channel, 0.0)
        if sp > 0:
            flow = sp + random.gauss(0, sp * 0.01)
        else:
            flow = abs(random.gauss(0, 0.1))
        self._flows[channel] = flow
        return flow

    def hw_read_flow_setpoint(self, channel):
        """Read the flow setpoint for a channel.

        Called by the base class after setting a setpoint or during readAll.

        Args:
            channel (int): 0-based channel index.

        Returns:
            float: Flow setpoint in the channel's units.
                   Return -1.0 on error.
        """
        return self._flow_setpoints.get(channel, 0.0)

    def hw_set_flow_setpoint(self, channel, value):
        """Set the flow setpoint for a channel.

        Called when the user changes a flow setpoint in the GUI or when
        an experiment configures flows.

        Args:
            channel (int): 0-based channel index.
            value (float): Desired flow setpoint in the channel's units.

        Examples:
            # Send to hardware:
            # self.comm.write(f"FLOW{channel+1}:SP {value}\\n")
        """
        self.log.debug(f"Setting flow ch{channel} setpoint to {value}")
        self._flow_setpoints[channel] = value

    # =========================================================================
    # Pressure Methods
    # =========================================================================

    def hw_read_pressure(self):
        """Read the current chamber pressure.

        Called periodically by the poll timer. The return value is displayed
        in the GUI and recorded in aux data during experiments.

        Returns:
            float: Pressure in the configured units (e.g., Torr).
                   Return -1.0 on error.

        Examples:
            # Read from hardware:
            # response = self.comm.query("PRESSURE?\\n")
            # return float(response.strip())
        """
        # Virtual: simulate pressure with noise
        self._pressure = max(0.0, self._pressure_setpoint + random.gauss(0, 0.1))
        return self._pressure

    def hw_read_pressure_setpoint(self):
        """Read the pressure setpoint from the controller.

        Returns:
            float: Pressure setpoint in configured units. Return -1.0 on error.
        """
        return self._pressure_setpoint

    def hw_set_pressure_setpoint(self, value):
        """Set the target pressure for the pressure controller.

        Args:
            value (float): Desired pressure in configured units.

        Examples:
            # self.comm.write(f"PRESSURE:SP {value}\\n")
        """
        self.log.debug(f"Setting pressure setpoint to {value}")
        self._pressure_setpoint = value

    def hw_read_pressure_control_mode(self):
        """Read whether automatic pressure control is enabled.

        Returns:
            int: 1 if pressure control is active, 0 if not, -1 on error.
        """
        return 1 if self._pressure_control else 0

    def hw_set_pressure_control_mode(self, enabled):
        """Enable or disable automatic pressure control.

        When enabled, the controller adjusts flow rates to maintain the
        pressure setpoint. When disabled, flow setpoints are manual.

        Args:
            enabled (bool): True to enable pressure control.

        Examples:
            # self.comm.write(f"PRESSURE:CTRL {'ON' if enabled else 'OFF'}\\n")
        """
        self.log.debug(f"Pressure control mode: {'ON' if enabled else 'OFF'}")
        self._pressure_control = enabled

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("Flow Controller entering sleep mode")
        else:
            self.log.debug("Flow Controller waking from sleep mode")
