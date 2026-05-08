"""
Blackchirp Python Pressure Controller Driver Template

This script is loaded by the PythonPressureController C++ trampoline class. It
provides a complete virtual pressure controller implementation that you can
customize for your hardware.

The pressure controller in Blackchirp monitors and controls chamber pressure.
The C++ PressureController base class handles:
  - Periodic polling (calls hw_read_pressure() on a timer)
  - GUI updates (current pressure, setpoint, control mode)
  - Experiment configuration (applies setpoint and control mode at start)
  - Aux data recording (pressure in rolling data plots)

Your Python script implements the low-level hw_* methods that communicate
with the actual hardware.

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "PressureControllerDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel

Read-only mode:
    If the controller is configured as read-only (readOnly = True in hardware
    settings), the base class will not call hw_set_pressure_setpoint(),
    hw_set_pressure_control_mode(), hw_open_gate_valve(), or
    hw_close_gate_valve(). Those methods are still present below for
    completeness; they simply do nothing in virtual/read-only mode.
"""

import math


class PressureControllerDriver:
    """Python Pressure Controller hardware driver.

    The PressureController base class calls these methods through its polling
    timer and public slot interface.

    Required methods (9 total):
        hw_read_pressure()                    -> float  (pressure value, or NaN on error)
        hw_read_pressure_setpoint()           -> float  (setpoint, or NaN on error)
        hw_set_pressure_setpoint(value)       -> float  (actual setpoint applied, or NaN on error)
        hw_read_pressure_control_mode()       -> int    (1=on, 0=off, -1=error)
        hw_set_pressure_control_mode(enabled) -> None
        hw_open_gate_valve()                  -> None
        hw_close_gate_valve()                 -> None

    Lifecycle methods (called by base class):
        initialize()      -- called once on startup (via pcInitialize)
        test_connection() -- called to verify hardware (via pcTestConnection)
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        This is called from PressureController::initialize() (via pcInitialize).
        The base class creates the poll timer before calling this. Use this
        to set up internal state.

        The comm proxy is available but the connection has not been tested yet.
        """
        self.log.log("Pressure Controller driver initialized")

        # Internal state for virtual mode
        self._pressure = 0.0
        self._setpoint = 0.0
        self._control_mode = False

    def test_connection(self):
        """Verify communication with the pressure controller.

        Called from PressureController::testConnection() (via pcTestConnection).
        If this returns True, the base class starts the poll timer.

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity:
            # response = self.comm.query("*IDN?\\n")
            # return len(response.strip()) > 0
        """
        self.log.log("Testing Pressure Controller connection")
        return True

    # =========================================================================
    # Pressure Methods
    # =========================================================================

    def hw_read_pressure(self):
        """Read the current chamber pressure.

        Called periodically by the PressureController poll timer. The return
        value is used to update the GUI and recorded in aux data during
        experiments. The base class checks isnan() on the return value; return
        math.nan to signal an error (the reading will be discarded).

        Returns:
            float: Current pressure in the configured units (e.g., Torr).
                   Return math.nan to indicate an error.

        Examples:
            # Read from hardware:
            # response = self.comm.query("PRESSURE?\\n")
            # try:
            #     return float(response.strip())
            # except ValueError:
            #     return math.nan
        """
        return self._pressure

    def hw_set_pressure_setpoint(self, value):
        """Set the target pressure setpoint on the controller.

        Called when the user changes the setpoint in the GUI or when an
        experiment applies its configured pressure. The return value is the
        actual setpoint as read back from the hardware (or the requested value
        if the hardware does not report one). The base class checks isnan() on
        the return value; return math.nan to signal failure.

        Not called if the controller is configured as read-only.

        Args:
            value (float): Desired pressure setpoint in configured units.

        Returns:
            float: Actual setpoint applied by the hardware, or math.nan on error.

        Examples:
            # self.comm.write(f"SETPOINT {value}\\n")
            # readback = self.comm.query("SETPOINT?\\n")
            # return float(readback.strip())
        """
        self.log.debug(f"Setting pressure setpoint to {value}")
        self._setpoint = value
        return self._setpoint

    def hw_read_pressure_setpoint(self):
        """Read the current pressure setpoint from the controller.

        Called after setting a new setpoint to confirm the value. The base
        class checks isnan() on the return value; return math.nan on error.

        Returns:
            float: Current pressure setpoint in configured units, or math.nan on error.

        Examples:
            # response = self.comm.query("SETPOINT?\\n")
            # return float(response.strip())
        """
        return self._setpoint

    def hw_read_pressure_control_mode(self):
        """Read whether automatic pressure control is currently enabled.

        Called after toggling control mode to confirm the state. Return -1 on
        any communication error; the base class will ignore the result if -1
        is returned.

        Returns:
            int: 1 if pressure control feedback is active,
                 0 if disabled (manual mode),
                -1 on error.

        Examples:
            # response = self.comm.query("CTRL?\\n").strip()
            # if response == "ON":
            #     return 1
            # elif response == "OFF":
            #     return 0
            # return -1
        """
        return 1 if self._control_mode else 0

    def hw_set_pressure_control_mode(self, enabled):
        """Enable or disable automatic pressure feedback control.

        When enabled, the controller adjusts the valve position to maintain
        the pressure setpoint. When disabled, the valve is in manual mode.

        Not called if the controller is configured as read-only.

        Args:
            enabled (bool): True to enable feedback control, False to disable.

        Examples:
            # self.comm.write(f"CTRL {'ON' if enabled else 'OFF'}\\n")
        """
        self.log.debug(f"Pressure control mode: {'ON' if enabled else 'OFF'}")
        self._control_mode = enabled

    def hw_open_gate_valve(self):
        """Fully open the gate valve.

        Called from PressureController::openGateValve(), which first disables
        pressure control mode before opening the valve. Use this to vent the
        chamber or expose the vacuum system.

        Not called if the controller is configured as read-only.

        Examples:
            # self.comm.write("VALVE OPEN\\n")
        """
        self.log.debug("Opening gate valve")

    def hw_close_gate_valve(self):
        """Fully close the gate valve.

        Called from PressureController::closeGateValve(), which first disables
        pressure control mode before closing the valve. Use this to isolate
        the vacuum system.

        Not called if the controller is configured as read-only.

        Examples:
            # self.comm.write("VALVE CLOSE\\n")
        """
        self.log.debug("Closing gate valve")

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Use this to park the hardware in a safe state (e.g., close the valve,
        disable feedback) when Blackchirp goes to sleep, and restore operation
        on wake.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("Pressure Controller entering sleep mode")
        else:
            self.log.debug("Pressure Controller waking from sleep mode")

    def read_settings(self):
        """Reload settings from Blackchirp without restarting the process.

        Called by PythonPressureController::pcReadSettings() when the user
        changes hardware settings in the GUI. Use self.settings.get() to
        re-read any configuration values that affect operation.

        Examples:
            # units = self.settings.get("units", "Torr")
            # self._units = units
        """
        self.log.debug("Pressure Controller reloading settings")
