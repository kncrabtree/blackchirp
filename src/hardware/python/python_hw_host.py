#!/usr/bin/env python3
"""
Blackchirp Python Hardware IPC Host

This script is launched by the C++ PythonTestHardware class as a subprocess.
It communicates with the C++ side via JSON-lines over stdin/stdout pipes.

Usage:
    python3 python_hw_host.py /path/to/user_script.py ClassName

The host loads the user's Python script, instantiates the specified class,
injects self.comm, self.settings, and self.log proxy objects, and dispatches
method calls received from the C++ side.

Protocol:
    - Each message is a single JSON object on one line (compact, no embedded newlines)
    - C++ sends requests with "id" and "method" fields
    - Python sends responses with matching "id" and "result" or "error" fields
    - Python sends unsolicited log messages with "log" and "level" fields
    - Relay requests (comm/settings) use "relay" field and block for "relay_result"
"""

import base64
import importlib.util
import json
import os
import sys
import threading
import traceback


# =============================================================================
# Thread-safe stdout writing
# =============================================================================

_stdout_lock = threading.Lock()


def _send_json(obj):
    """Write a JSON object as a single line to stdout (thread-safe)."""
    line = json.dumps(obj, separators=(',', ':')) + '\n'
    with _stdout_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _read_line():
    """Read a single line from stdin. Returns None on EOF."""
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip()


# =============================================================================
# Relay helpers (Python -> C++ -> Python round-trip)
# =============================================================================

# When Python needs to call back to C++ (e.g., self.comm.query()), it sends
# a relay request and then reads stdin for the relay_result. Since the main
# dispatch loop is the one reading stdin, relay calls can only happen from
# within a method dispatch (which is always on the main thread in our design).
# This means we can safely read stdin here without races.

def _relay_request(request_obj):
    """Send a relay request to C++ and block until the relay response arrives.

    The C++ side reads this relay request while waiting for our method response,
    handles it, and writes back a relay result line.
    """
    _send_json(request_obj)
    # Block until C++ sends back the relay result
    while True:
        line = _read_line()
        if line is None:
            raise ConnectionError("C++ process closed stdin during relay")
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip malformed lines
        if "relay_result" in resp or "relay_error" in resp:
            return resp
        # If it's something else (shouldn't happen during relay), skip it


# =============================================================================
# Proxy classes injected into user's hardware object
# =============================================================================

class CommProxy:
    """Proxy for hardware communication. Relays calls to C++ p_comm."""

    def query(self, cmd):
        """Send command, read response string."""
        resp = _relay_request({"relay": "comm_query", "cmd": str(cmd)})
        if "relay_error" in resp:
            raise ConnectionError(resp["relay_error"])
        return str(resp.get("relay_result", ""))

    def write(self, cmd):
        """Send command, return success bool."""
        resp = _relay_request({"relay": "comm_write", "cmd": str(cmd)})
        if "relay_error" in resp:
            raise ConnectionError(resp["relay_error"])
        return bool(resp.get("relay_result", False))

    def read_bytes(self, n):
        """Read n bytes from device, returned as bytes."""
        resp = _relay_request({"relay": "comm_read_bytes", "n": int(n)})
        if "relay_error" in resp:
            raise ConnectionError(resp["relay_error"])
        b64 = resp.get("relay_result", "")
        return base64.b64decode(b64) if b64 else b""

    def write_binary(self, data):
        """Write binary data to device, return success bool."""
        b64 = base64.b64encode(bytes(data)).decode("ascii")
        resp = _relay_request({"relay": "comm_write_binary", "data": b64})
        if "relay_error" in resp:
            raise ConnectionError(resp["relay_error"])
        return bool(resp.get("relay_result", False))


class SettingsProxy:
    """Proxy for persistent settings. Relays calls to C++ SettingsStorage."""

    def __init__(self):
        self._key = ""
        self._model = ""

    @property
    def key(self):
        return self._key

    @property
    def model(self):
        return self._model

    def get(self, key, default=None):
        """Read a persistent setting."""
        resp = _relay_request({
            "relay": "get_setting",
            "key": str(key),
            "default": default
        })
        if "relay_error" in resp:
            return default
        return resp.get("relay_result", default)

    def set(self, key, value):
        """Write a persistent setting."""
        resp = _relay_request({
            "relay": "set_setting",
            "key": str(key),
            "value": value
        })
        if "relay_error" in resp:
            raise RuntimeError(resp["relay_error"])


class LogProxy:
    """Proxy for log messages. Sends unsolicited JSON to C++."""

    def log(self, msg):
        _send_json({"log": str(msg), "level": "Normal"})

    def debug(self, msg):
        _send_json({"log": str(msg), "level": "Debug"})

    def warning(self, msg):
        _send_json({"log": str(msg), "level": "Warning"})

    def error(self, msg):
        _send_json({"log": str(msg), "level": "Error"})

    def highlight(self, msg):
        _send_json({"log": str(msg), "level": "Highlight"})


# =============================================================================
# Script loading
# =============================================================================

def load_user_class(script_path, class_name):
    """Load a Python module from a file path and return the specified class."""
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("user_hardware", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cls = getattr(module, class_name, None)
    if cls is None:
        available = [n for n in dir(module) if not n.startswith('_')]
        raise AttributeError(
            f"Class '{class_name}' not found in {script_path}. "
            f"Available names: {available}"
        )

    return cls


# =============================================================================
# Method dispatch
# =============================================================================

# Map of method names to how they should be called
_SIMPLE_METHODS = {
    "initialize",
    "test_connection",
    "read_aux_data",
    "read_validation_data",
    "begin_acquisition",
    "end_acquisition",
    "read_settings",
}


def dispatch(user_obj, request):
    """Dispatch a method call to the user object and return the result."""
    method = request.get("method", "")

    if method == "_init":
        # Initialization: inject proxies
        user_obj.comm = CommProxy()
        user_obj.settings = SettingsProxy()
        user_obj.settings._key = request.get("key", "")
        user_obj.settings._model = request.get("model", "")
        user_obj.log = LogProxy()
        return True

    if method in _SIMPLE_METHODS:
        fn = getattr(user_obj, method, None)
        if fn is None:
            # Method not implemented -- return safe default
            if method == "test_connection":
                return True
            if method in ("read_aux_data", "read_validation_data"):
                return {}
            return None
        return fn()

    if method == "prepare_for_experiment":
        fn = getattr(user_obj, "prepare_for_experiment", None)
        if fn is None:
            return True
        config = request.get("config", {})
        return fn(config)

    if method == "sleep":
        fn = getattr(user_obj, "sleep", None)
        if fn is None:
            return None
        sleeping = request.get("sleeping", False)
        return fn(sleeping)

    # Generic dispatch for type-specific methods (e.g., hw_read_flow,
    # read_analog_channels). All request keys other than "id" and "method"
    # are passed as keyword arguments.
    fn = getattr(user_obj, method, None)
    if fn is None:
        return None  # Method not implemented -- return safe default
    kwargs = {k: v for k, v in request.items() if k not in ("id", "method")}
    return fn(**kwargs)


# =============================================================================
# Main loop
# =============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python_hw_host.py <script_path> <class_name>",
              file=sys.stderr)
        sys.exit(1)

    script_path = sys.argv[1]
    class_name = sys.argv[2]

    # Load the user's hardware class
    try:
        cls = load_user_class(script_path, class_name)
        user_obj = cls()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Main IPC loop
    while True:
        line = _read_line()
        if line is None:
            break  # EOF -- C++ closed the pipe

        if not line:
            continue  # blank line

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}: {line!r}", file=sys.stderr)
            continue

        req_id = request.get("id")

        try:
            result = dispatch(user_obj, request)
            response = {"id": req_id, "result": result}
        except Exception as e:
            tb = traceback.format_exc()
            response = {
                "id": req_id,
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
            }
            print(tb, file=sys.stderr)

        _send_json(response)


if __name__ == "__main__":
    main()
