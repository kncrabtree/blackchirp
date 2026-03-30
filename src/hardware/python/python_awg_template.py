"""
Blackchirp Python AWG Driver Template

This script is loaded by the PythonAwg C++ trampoline class. It provides a
complete virtual AWG implementation that you can customize for your hardware.

The AWG (Arbitrary Waveform Generator) in Blackchirp generates chirped-pulse
waveforms for CP-FTMW spectroscopy. The C++ base class handles waveform
configuration and experiment integration; your Python script handles
communication with the actual hardware.

Class name must match the pythonClass setting (default: "AwgDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel
"""


class AwgDriver:
    """Python AWG hardware driver.

    The AWG class in Blackchirp is a minimal interface: the base class has no
    type-specific polling methods. The primary interaction is through
    prepare_for_experiment(), where the C++ side sends waveform configuration
    and the driver programs the hardware.

    All methods are optional. Unimplemented methods return safe defaults.
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        Use this to set up any internal state. The comm proxy is available
        but the connection has not been tested yet.
        """
        self.log.log("AWG driver initialized")

    def test_connection(self):
        """Verify communication with the AWG hardware.

        Called when Blackchirp tests the hardware connection (e.g., on
        startup or when the user clicks "Test Connection").

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity
            response = self.comm.query("*IDN?\\n")
            return len(response.strip()) > 0

            # For virtual/testing, just return True
            return True
        """
        self.log.log("Testing AWG connection")

        # Example: query device identity via comm proxy
        # response = self.comm.query("*IDN?\\n")
        # if not response.strip():
        #     self.log.error("No response from AWG")
        #     return False
        # self.log.log(f"AWG identified: {response.strip()}")

        return True

    def read_aux_data(self):
        """Return auxiliary data for rolling data plots.

        Called periodically by Blackchirp's polling timer. Return a dict
        mapping string keys to float values. These appear in the rolling
        data plots and are saved to disk during experiments.

        Returns:
            dict[str, float]: Key-value pairs of auxiliary data.
                Keys should be descriptive (e.g., "Temperature", "Power").
                Return an empty dict if no auxiliary data is available.
        """
        # Example: read AWG temperature or output power
        # temp = float(self.comm.query("TEMP?\\n").strip())
        # return {"AWG.Temperature": temp}
        return {}

    def prepare_for_experiment(self, config):
        """Configure the AWG for an upcoming experiment.

        Called before each experiment starts. The config dict contains
        experiment parameters from the C++ side. Use this to program
        waveforms, set sample rates, configure triggers, etc.

        Args:
            config (dict): Experiment configuration. Currently contains:
                - "number" (int): Experiment number (0 if not yet assigned)

        Returns:
            bool: True if preparation succeeded, False to abort experiment.
        """
        self.log.log(f"Preparing AWG for experiment {config.get('number', '?')}")

        # Example: program a waveform
        # self.comm.write("WAVEFORM:LOAD DEFAULT\\n")
        # self.comm.write("OUTPUT ON\\n")

        return True

    def begin_acquisition(self):
        """Called when experiment data acquisition starts.

        The experiment has been fully configured and is now running.
        Use this to enable outputs, start triggering, etc.
        """
        self.log.debug("AWG acquisition started")

    def end_acquisition(self):
        """Called when experiment data acquisition ends.

        Use this to disable outputs, stop triggering, return to idle, etc.
        """
        self.log.debug("AWG acquisition ended")

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Blackchirp may put hardware to sleep between experiments to reduce
        wear or power consumption.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("AWG entering sleep mode")
            # Example: self.comm.write("OUTPUT OFF\\n")
        else:
            self.log.debug("AWG waking from sleep mode")
            # Example: self.comm.write("OUTPUT ON\\n")
