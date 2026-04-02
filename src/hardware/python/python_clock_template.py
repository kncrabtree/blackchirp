"""
Blackchirp Python Clock Driver Template

This script is loaded by the PythonClock C++ trampoline class. It provides a
complete virtual clock implementation that you can customize for your hardware.

A Clock in Blackchirp is an oscillator that can be assigned one or more roles
at runtime. Common roles include:

    UpLO      -- local oscillator for the up-conversion mixer
    DownLO    -- local oscillator for the down-conversion mixer
    AwgRef    -- reference clock for the AWG
    Common    -- a single source serving multiple roles simultaneously

The C++ Clock base class handles:
  - Role assignment and tracking (which output serves which role)
  - Multiplication factors (e.g., if hardware output is divided externally)
  - prepareForExperiment() -- iterates clock assignments, calls setFrequency()
    for each assigned role, which in turn calls hw_set_frequency() after
    dividing by the multiplier and checking the min/max range
  - readAll() -- reads all outputs via hw_read_frequency() and emits
    frequencyUpdate signals; stops on the first output that returns < 0

Frequency values passed to hw_set_frequency() and returned by
hw_read_frequency() are the raw hardware frequencies in MHz, i.e., the
requested frequency already divided by the multiplier factor.

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "ClockDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel
"""


class ClockDriver:
    """Python Clock hardware driver.

    Required methods (4 pure virtuals from Clock base):
        hw_set_frequency(freq_mhz, output) -> bool  (True on success)
        hw_read_frequency(output)          -> float (MHz, or -1.0 on error)

    Lifecycle methods (called by base class):
        initialize()      -- called once on startup (via initializeClock)
        test_connection() -- verify hardware (via testClockConnection)

    Optional methods:
        sleep(sleeping)   -- enter/exit standby mode
        read_settings()   -- reload settings without restarting

    Note: readAll() in the base class stops iterating outputs on the first
    output that returns a frequency < 0.0, so return -1.0 consistently for
    outputs that are not available or on error.
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        This is called from Clock::initialize() (via initializeClock).
        Use this to set up internal state. The comm proxy is available
        but the connection has not been tested yet.
        """
        self.log.log("Clock driver initialized")

        # Internal state for virtual mode: store per-output frequencies
        self._frequencies = {}
        self._num_outputs = 1

    def test_connection(self):
        """Verify communication with the clock hardware.

        Called from Clock::testConnection() (via testClockConnection).
        If this returns True, the base class calls readAll() to populate
        the initial frequency values and emits frequencyUpdate signals.

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity:
            # response = self.comm.query("*IDN?\\n")
            # return len(response.strip()) > 0
        """
        self.log.log("Testing Clock connection")

        # Read number of outputs from settings (matches C++ numOutputs default)
        self._num_outputs = int(self.settings.get("numOutputs", 1))
        self.log.log(f"Configured for {self._num_outputs} output(s)")

        # Initialize virtual frequency state (default 0.0 MHz per output)
        for i in range(self._num_outputs):
            self._frequencies.setdefault(i, 0.0)

        return True

    # =========================================================================
    # Hardware Interface Methods (pure virtuals in C++)
    # =========================================================================

    def hw_set_frequency(self, freq_mhz, output=0):
        """Set the hardware output frequency.

        Called by Clock::setFrequency() after dividing the requested
        frequency by the output's multiplication factor and verifying it
        is within the configured min/max range.

        Args:
            freq_mhz (float): Hardware frequency to set, in MHz. This is
                the raw hardware frequency (after dividing by mult factor).
            output (int): 0-based output index.

        Returns:
            bool: True if the frequency was set successfully, False on error.

        Examples:
            # Send command to hardware:
            # self.comm.write(f":FREQ:CW {freq_mhz}MHZ\\n")
            # response = self.comm.query(":FREQ:CW?\\n")
            # return abs(float(response.strip()) / 1e6 - freq_mhz) < 0.001
        """
        self.log.debug(f"Setting output {output} frequency to {freq_mhz} MHz")
        self._frequencies[output] = freq_mhz
        return True

    def hw_read_frequency(self, output=0):
        """Read the current hardware output frequency.

        Called by Clock::readAll() for each output index in sequence.
        readAll() stops iterating if this method returns a value < 0.0,
        so return -1.0 consistently for outputs that are unavailable or
        on any communication error.

        Also called by Clock::setFrequency() after setting the frequency
        to read back the actual frequency achieved.

        Args:
            output (int): 0-based output index.

        Returns:
            float: Current hardware frequency in MHz, or -1.0 on error.
                   Do NOT apply the multiplier factor here; the base class
                   multiplies this return value by the mult factor itself.

        Examples:
            # Query from hardware:
            # response = self.comm.query(":FREQ:CW?\\n")
            # return float(response.strip()) / 1e6   # convert Hz to MHz
        """
        return self._frequencies.get(output, -1.0)

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        The Clock base class calls this via PythonClock::sleep(), which is
        triggered by the HardwareObject sleep mechanism.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.

        Examples:
            # Disable RF output when sleeping:
            # self.comm.write(f":OUTP {'OFF' if sleeping else 'ON'}\\n")
        """
        if sleeping:
            self.log.debug("Clock entering sleep mode")
        else:
            self.log.debug("Clock waking from sleep mode")

    def read_settings(self):
        """Reload hardware settings without restarting the driver.

        Called by PythonClock::readSettings(), which is invoked when
        the user modifies hardware settings in the Blackchirp GUI.
        Use this to apply updated configuration to the hardware without
        requiring a full reconnect.

        Examples:
            # Re-read and apply a configurable parameter:
            # new_level = float(self.settings.get("outputPowerDbm", 0.0))
            # self.comm.write(f":POW {new_level}DBM\\n")
        """
        pass
