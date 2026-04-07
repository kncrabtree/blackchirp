"""
Blackchirp Python LIF Laser Driver Template

This script is loaded by the PythonLifLaser C++ trampoline class. It provides
a complete virtual LIF laser implementation that you can customize for your
hardware.

The LIF (Laser-Induced Fluorescence) laser in Blackchirp tunes to a target
wavelength/frequency and controls a flashlamp for pulsed operation. The C++
LifLaser base class handles:
  - Position validation (clamps set_pos() requests to the [minPos, maxPos] range)
  - Signal emission to the GUI (laserPosUpdate, laserFlashlampUpdate)
  - beginAcquisition() — automatically fires the flashlamp at experiment start
  - endAcquisition()   — automatically disables the flashlamp when autoDisable is set
  - hwPrepareForExperiment() — reads the autoDisable flag from experiment config

Your Python script implements the low-level hardware methods that communicate
with the actual laser controller.

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "LifLaserDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel

Position units and range are configurable in Blackchirp's hardware settings
(defaults: nm, 250–2000). Return -1.0 from read_pos() to signal an error.
"""


class LifLaserDriver:
    """Python LIF Laser hardware driver.

    The LifLaser base class calls these methods in response to GUI commands and
    experiment lifecycle events. Position validation (min/max clamping) is
    performed by the base class before set_pos() is called.

    Required methods:
        read_pos()          -> float  (current position, -1.0 on error)
        set_pos(pos)        -> None   (set target position)
        read_fl()           -> bool   (flashlamp state)
        set_fl(enabled)     -> bool   (True if setting was successful)

    Lifecycle methods (called by base class):
        initialize()        -- called once on startup
        test_connection()   -- called to verify hardware (via testConnection)
        sleep(sleeping)     -- called on hardware standby transitions
        read_settings()     -- called to reload settings without restarting
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        The comm proxy is available but the connection has not been tested yet.
        Use this to set up internal state.
        """
        self.log.log("LIF Laser driver initialized")

        # Internal state for virtual mode
        self._position  = 500.0   # current laser position (default nm)
        self._flashlamp = False   # flashlamp state

    def test_connection(self):
        """Verify communication with the laser controller.

        Called from LifLaser::testConnection(). If this returns True, the base
        class calls readPosition() and readFlashLamp() to populate the GUI.

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity:
            # response = self.comm.query("*IDN?\\n")
            # return len(response.strip()) > 0
        """
        self.log.log("Testing LIF Laser connection")

        # Read configured range from settings for informational logging
        min_pos  = float(self.settings.get("minPos",  250.0))
        max_pos  = float(self.settings.get("maxPos", 2000.0))
        units    = str(self.settings.get("units", "nm"))
        self.log.log(
            f"Laser range: {min_pos} – {max_pos} {units}"
        )

        # Virtual mode: initialize to a known position
        self._position  = 500.0
        self._flashlamp = False

        return True

    # =========================================================================
    # Position Methods
    # =========================================================================

    def read_pos(self):
        """Read the current laser position.

        Called by LifLaser::readPos() whenever the position is queried. The
        base class emits laserPosUpdate() with the returned value and updates
        the GUI.

        Returns:
            float: Current laser position in configured units (default: nm).
                   Return -1.0 to signal an error.

        Examples:
            # Query a GPIB-controlled OPO:
            # response = self.comm.query("POS?\\n")
            # try:
            #     return float(response.strip())
            # except ValueError:
            #     return -1.0
        """
        return self._position

    def set_pos(self, pos):
        """Set the laser to a target position.

        Called by LifLaser::setPos() after the base class has validated that
        pos is within [minPos, maxPos]. After this returns, the base class
        calls readPosition() to confirm the new position.

        Args:
            pos (float): Target position in configured units (default: nm).

        Examples:
            # Send position command to hardware:
            # self.comm.write(f"GOTO {pos:.4f}\\n")
        """
        self._position = pos

    # =========================================================================
    # Flashlamp Methods
    # =========================================================================

    def read_fl(self):
        """Read the current flashlamp state.

        Called by LifLaser::readFl(). The base class emits
        laserFlashlampUpdate() with the returned value.

        Returns:
            bool: True if the flashlamp is enabled, False if disabled.

        Examples:
            # Query flashlamp state:
            # response = self.comm.query("FL?\\n")
            # return response.strip() == "1"
        """
        return self._flashlamp

    def set_fl(self, enabled):
        """Set the flashlamp state.

        Called by LifLaser::setFl(). This should return True if the command
        was accepted by the hardware, False otherwise (NOT the new state).
        After this returns, the base class calls readFlashLamp() to confirm.

        The base class calls setFlashLamp(True) from beginAcquisition() and
        setFlashLamp(False) from endAcquisition() (when autoDisable is set).

        Args:
            enabled (bool): True to enable the flashlamp, False to disable it.

        Returns:
            bool: True if the setting was successfully applied, False on error.

        Examples:
            # Send flashlamp enable command:
            # self.comm.write(f"FL {'ON' if enabled else 'OFF'}\\n")
            # return True
        """
        self._flashlamp = enabled
        return True

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("LIF Laser entering sleep mode")
        else:
            self.log.debug("LIF Laser waking from sleep mode")

    def read_settings(self):
        """Reload settings from Blackchirp without restarting the process.

        Called when hardware settings are changed at runtime. Use this to
        re-read any configurable parameters from self.settings.
        """
        self.log.debug("LIF Laser reloading settings")
