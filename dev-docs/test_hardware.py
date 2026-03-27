"""
Test hardware script for PythonTestHardware proof-of-concept.

This script is loaded by python_hw_host.py when launched by the C++
PythonTestHardware class. It exercises all HardwareObject virtual methods
via the injected proxies:
  - self.comm     (CommProxy)     -- query/write to the echo server
  - self.settings (SettingsProxy) -- read/write persistent settings
  - self.log      (LogProxy)     -- emit log messages to Blackchirp UI

Point PythonTestHardware's pythonScript setting to this file, and run
echo_server.py on localhost:12345.
"""

import json
import time


class TestHardware:
    """Proof-of-concept Python hardware implementation."""

    def initialize(self):
        """Called once after script is loaded and wrappers are injected."""
        self.log.highlight("=== initialize() called ===")
        self.log.log(f"  Hardware key: {self.settings.key}")
        self.log.log(f"  Hardware model: {self.settings.model}")
        self.log.log("  (no data passed from C++ for this method)")

        # Store an initial setting to verify SettingsProxy write
        self.settings.set("pyInitTime", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.log.debug("  Wrote pyInitTime to settings")

    def test_connection(self):
        """Verify communication with the echo server."""
        self.log.highlight("=== test_connection() called ===")
        self.log.log("  (no data passed from C++ for this method)")

        response = self.comm.query("*IDN?\n")
        self.log.log(f"  Comm query '*IDN?' -> {response!r}")

        if response.strip():
            self.log.highlight(f"  Connection OK -- echo server responded: {response.strip()}")
            return True
        else:
            self.log.error("  Connection FAILED -- empty response from echo server")
            return False

    def read_aux_data(self):
        """Return auxiliary data for aux/rolling data plots."""
        self.log.debug("=== read_aux_data() called ===")
        self.log.debug("  (no data passed from C++ for this method)")

        response = self.comm.query("AUX?\n")
        self.log.debug(f"  Comm query 'AUX?' -> {response!r}")

        data = {
            "PythonTest.EchoLen": float(len(response.strip())),
            "PythonTest.Timestamp": float(int(time.time()) % 1000),
        }
        self.log.debug(f"  Returning aux data: {data}")
        return data

    def read_validation_data(self):
        """Return validation data for experiment validation."""
        self.log.debug("=== read_validation_data() called ===")
        self.log.debug("  (no data passed from C++ for this method)")

        data = {
            "PythonTest.Valid": 1.0,
            "PythonTest.Check": 42.0,
        }
        self.log.debug(f"  Returning validation data: {data}")
        return data

    def prepare_for_experiment(self, config):
        """Configure hardware for an upcoming experiment."""
        self.log.highlight("=== prepare_for_experiment() called ===")
        self.log.log(f"  Config received from C++ ({len(config)} keys):")
        for key, value in sorted(config.items()):
            val_repr = json.dumps(value) if isinstance(value, (dict, list)) else repr(value)
            self.log.log(f"    {key} = {val_repr}")

        if len(config) <= 1:
            self.log.warning("  NOTE: Only experiment number received. "
                             "No other experiment data is serialized from C++ yet.")

        # Write something to settings to test persistence during experiment
        self.settings.set("lastExptNumber", config.get("number", -1))
        self.log.debug("  Wrote lastExptNumber to settings")

        response = self.comm.query("PREP?\n")
        self.log.debug(f"  Comm query 'PREP?' -> {response!r}")

        return True

    def begin_acquisition(self):
        """Called when experiment acquisition starts."""
        self.log.highlight("=== begin_acquisition() called ===")
        self.log.log("  (no data passed from C++ for this method)")

        response = self.comm.query("ACQ_START\n")
        self.log.debug(f"  Comm query 'ACQ_START' -> {response!r}")

    def end_acquisition(self):
        """Called when experiment acquisition ends."""
        self.log.highlight("=== end_acquisition() called ===")
        self.log.log("  (no data passed from C++ for this method)")

        response = self.comm.query("ACQ_END\n")
        self.log.debug(f"  Comm query 'ACQ_END' -> {response!r}")

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode."""
        self.log.highlight("=== sleep() called ===")
        self.log.log(f"  Data from C++: sleeping = {sleeping!r} (type: {type(sleeping).__name__})")

        state = "SLEEPING" if sleeping else "AWAKE"
        response = self.comm.query(f"SLEEP={state}\n")
        self.log.debug(f"  Comm query 'SLEEP={state}' -> {response!r}")

    def read_settings(self):
        """Called when settings are reloaded (subprocess was restarted)."""
        self.log.highlight("=== read_settings() called ===")
        self.log.log("  (C++ kills and restarts the Python process for this method)")

        # Read back the settings we previously wrote
        init_time = self.settings.get("pyInitTime", "not set")
        last_expt = self.settings.get("lastExptNumber", -1)
        self.log.log(f"  Settings read-back: pyInitTime = {init_time}")
        self.log.log(f"  Settings read-back: lastExptNumber = {last_expt}")
