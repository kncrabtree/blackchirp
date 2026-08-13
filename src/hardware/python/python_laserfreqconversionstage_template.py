"""
Blackchirp Python Laser Frequency-Conversion Stage Driver Template

This script is loaded by the PythonLaserFreqConversionStage C++ trampoline
class. It provides a complete virtual conversion-stage implementation that
you can customize for your hardware.

A LaserFreqConversionStage is one optical conversion node — a nonlinear
crystal or compensator — in the conversion topology that carries a tunable
fundamental laser to the final output beam. The caller assembles the
topology and computes, for each node, the LOCAL INPUT-BEAM WAVENUMBER
(cm^-1) that node must be phase-matched for; the driver has no knowledge of
the topology itself. The C++ LaserFreqConversionStage base class handles:
  - Move dispatch: setPosition(localCm1) calls this driver's set_pos(), then
    calls readPos() to verify the move.
  - Move verification: the readback is compared against the requested
    wavenumber within a tolerance (the "verifyToleranceCm1" setting). If
    "verifyMove" is on, a mismatch fails the move; if off, it only warns.
  - The hard error sentinel: a NEGATIVE readPos() value is always treated
    as a communication error and fails the move, regardless of the verify
    flag.
  - The registered conversionOp()/harmonicOrder() settings, used by callers
    to assemble the topology. The driver does not implement conversion math.

Your Python script implements only the low-level device methods that move
and query the phase-match actuator (typically a motor or rotation stage).

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "LaserFreqConversionStageDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel

Return a negative value from read_pos() to signal a communication error.
"""


class LaserFreqConversionStageDriver:
    """Python Laser Frequency-Conversion Stage hardware driver.

    The LaserFreqConversionStage base class calls these methods in response
    to setPosition()/readPosition() dispatches and lifecycle events.

    Required methods:
        read_pos()          -> float  (achieved local input-beam wavenumber
                                        in cm^-1; negative on error)
        set_pos(local_cm1)  -> None   (move the phase-match actuator)

    Lifecycle methods (called by base class):
        initialize()        -- called once on startup
        test_connection()   -- called to verify hardware (via testConnection)
        sleep(sleeping)     -- called on hardware standby transitions
        read_settings()     -- called to reload settings without restarting
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        The comm proxy is available but the connection has not been tested
        yet. Use this to set up internal state.
        """
        self.log.log("Laser Frequency-Conversion Stage driver initialized")

        # Internal state for virtual mode: the achieved local input-beam
        # wavenumber (cm^-1). 10000.0 matches the default seeded by
        # VirtualLaserFreqConversionStage.
        self._position = 10000.0

    def test_connection(self):
        """Verify communication with the phase-match actuator.

        Called from LaserFreqConversionStage::testConnection(). If this
        returns True, the base class calls readPosition() to prime state.

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity:
            # response = self.comm.query("*IDN?\\n")
            # return len(response.strip()) > 0
        """
        self.log.log("Testing Laser Frequency-Conversion Stage connection")

        # These settings are primarily consumed by the C++ base class (for
        # move verification and topology assembly); read them here only if
        # your actuation math needs them. Logged for informational purposes,
        # mirroring how other templates log their configured ranges.
        op        = str(self.settings.get("conversionOp", "NHG"))
        harmonic  = int(self.settings.get("harmonicOrder", 2))
        verify    = bool(self.settings.get("verifyMove", True))
        tolerance = float(self.settings.get("verifyToleranceCm1", 1.0))
        self.log.log(
            f"Stage operation: {op} (harmonic order {harmonic}), "
            f"verifyMove={verify}, tolerance={tolerance} cm^-1"
        )

        # Virtual mode: initialize to a known position
        self._position = 10000.0

        return True

    # =========================================================================
    # Position Methods
    # =========================================================================

    def read_pos(self):
        """Read the achieved local input-beam wavenumber for this node.

        Called by LaserFreqConversionStage::readPosition() after a move (and
        whenever the position is otherwise queried). The base class compares
        this value against the requested wavenumber to verify the move.

        A NEGATIVE return value is a hard communication-error sentinel: it is
        never treated as an actual reading, and it always fails the move
        regardless of the "verifyMove" setting. Only return a negative value
        when communication with the actuator has genuinely failed — not to
        indicate an out-of-tolerance-but-otherwise-valid reading.

        Returns:
            float: Achieved local input-beam wavenumber in cm^-1.
                   Return a negative value (e.g. -1.0) to signal a
                   communication error.

        Examples:
            # Query a motorized rotation stage and convert to wavenumber:
            # response = self.comm.query("POS?\\n")
            # try:
            #     angle = float(response.strip())
            #     return self._angle_to_cm1(angle)
            # except ValueError:
            #     return -1.0
        """
        return self._position

    def set_pos(self, local_cm1):
        """Move the phase-match actuator for a requested local wavenumber.

        Called by LaserFreqConversionStage::setPosition() with the local
        input-beam wavenumber (cm^-1) this node must be phase-matched for.
        The caller has already computed local_cm1 from the assembled
        conversion topology; this driver needs no knowledge of the topology,
        only how to translate a wavenumber into actuator motion (e.g. a
        crystal or compensator angle from a calibration curve). After this
        returns, the base class calls readPos() and validates/handles move
        verification.

        Args:
            local_cm1 (float): Requested local input-beam wavenumber, cm^-1.

        Examples:
            # Convert wavenumber to actuator angle and move:
            # angle = self._cm1_to_angle(local_cm1)
            # self.comm.write(f"GOTO {angle:.4f}\\n")
        """
        self._position = local_cm1

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("Laser Frequency-Conversion Stage entering sleep mode")
        else:
            self.log.debug("Laser Frequency-Conversion Stage waking from sleep mode")

    def read_settings(self):
        """Reload settings from Blackchirp without restarting the process.

        Called when hardware settings are changed at runtime. The
        conversionOp, harmonicOrder, verifyMove, and verifyToleranceCm1
        settings are read directly by the C++ base class as needed; use this
        hook to re-read any additional configuration values that affect this
        driver's own actuation math (e.g. a calibration file path).
        """
        self.log.debug("Laser Frequency-Conversion Stage reloading settings")
