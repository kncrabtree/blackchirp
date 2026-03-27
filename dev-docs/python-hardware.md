# Python Hardware Implementations

## Motivation

Users should be able to write hardware drivers in Python without modifying or
recompiling Blackchirp. This enables:

- Rapid prototyping of new hardware support
- Use of vendor-provided Python libraries (pyvisa, manufacturer SDKs)
- User-customizable drivers that can be tuned without C++ knowledge
- Community-contributed implementations shared as simple `.py` files

## Architecture

### QProcess + JSON IPC

Python hardware runs in a **separate subprocess**, communicating with the C++
application via JSON-lines over stdin/stdout pipes. This provides complete
memory isolation -- crashes in Python cannot corrupt the Qt application.

```
C++ PythonTestHardware          IPC (JSON over pipes)         Python subprocess
---                             ---                           ---
initialize()              ->    {"method": "initialize"}  ->  initialize()
testConnection()          ->    {"method": "test_connection"} -> test_connection()
  (p_comm available)            {"relay": "comm_query",       (self.comm.query()
                                 "cmd": "*IDN?\n"}        ->    via IPC relay)
readAuxData()             ->    {"method": "read_aux_data"} -> read_aux_data()
prepareForExperiment()    ->    {"method": "prepare_for_experiment", ...} -> prepare_for_experiment(cfg)
beginAcquisition()        ->    {"method": "begin_acquisition"} -> begin_acquisition()
endAcquisition()          ->    {"method": "end_acquisition"} -> end_acquisition()
sleep(b)                  ->    {"method": "sleep", ...}  ->  sleep(sleeping)
readSettings()            ->    (kill and restart process) -> fresh load
```

### Why QProcess (not pybind11)

An earlier attempt used pybind11 to embed the Python interpreter directly.
This was abandoned due to persistent heap corruption from pybind11 3.0.2 +
Python 3.13 mimalloc incompatibility in multi-threaded Qt applications.
The pybind11 attempt is preserved on the `python-hw-pybind11` branch.

QProcess advantages:
- **Complete memory isolation** -- Python's heap is in a separate process
- **No GIL concerns** -- the GIL is entirely within the subprocess
- **Python version independence** -- no pybind11 version coupling
- **No build dependency on Python** -- C++ side only uses QProcess (Qt6::Core)
- **Crash resilience** -- Python crashes produce a clean error on the C++ side

### Communication Design

**Transport**: stdin/stdout pipes via `QProcess`. JSON-lines protocol (one
JSON object per line, compact format).

**Request (C++ -> Python)**:
```json
{"id": 1, "method": "test_connection"}
{"id": 2, "method": "prepare_for_experiment", "config": {"number": 42}}
```

**Response (Python -> C++)**:
```json
{"id": 1, "result": true}
{"id": 2, "result": true}
```

**Error response**:
```json
{"id": 1, "error": "ConnectionError: no response", "traceback": "..."}
```

**Log messages (unsolicited)**:
```json
{"log": "Connected to device", "level": "Normal"}
```

### Communication Relay

Python can't directly access C++ `p_comm`. When Python calls
`self.comm.query("*IDN?\n")`, the comm proxy sends a relay request back
to C++ over the pipe:

1. Python sends: `{"relay": "comm_query", "cmd": "*IDN?\n"}`
2. C++ reads this, calls `p_comm->queryCmd("*IDN?\n")`
3. C++ sends back: `{"relay_result": "response text"}`
4. Python's `comm.query()` returns `"response text"`

The C++ `sendRequest()` read loop handles these interleaved relay requests
while waiting for the final method response.

### Settings Relay

Similarly, `self.settings.get(key, default)` and `self.settings.set(key, value)`
are relayed through IPC to the C++ side, which performs the actual
`SettingsStorage` operations.

### File Layout

```
src/hardware/python/
    pythonprocess.h/.cpp          # QProcess wrapper: launch, IPC protocol, lifecycle
    pythontesthardware.h/.cpp     # HardwareObject subclass, dispatches via IPC
dev-docs/
    echo_server.py                # TCP echo server for testing
    test_hardware.py              # Example Python hardware script
    python_hw_host.py             # Python-side IPC host: reads stdin, dispatches to user script
```

## Python API Contract

Each hardware type defines a Python class with methods matching the C++
virtual interface. Methods use snake_case. `self.comm`, `self.settings`,
and `self.log` are injected by the IPC host script.

### Common Methods (all types)

```python
def initialize(self):
    """Called once after script is loaded."""

def test_connection(self) -> bool:
    """Verify communication with hardware. Return True on success."""

def read_aux_data(self) -> dict[str, float]:          # optional
def read_validation_data(self) -> dict[str, float]:    # optional
def prepare_for_experiment(self, config: dict) -> bool: # optional
def begin_acquisition(self):                            # optional
def end_acquisition(self):                              # optional
def sleep(self, sleeping: bool):                        # optional
def read_settings(self):                                # optional
```

### Injected Objects

**self.comm** (communication proxy):
- `query(cmd: str) -> str` -- send command, read response
- `write(cmd: str) -> bool` -- send command, no response
- `read_bytes(n: int) -> bytes` -- read n bytes
- `write_binary(data: bytes) -> bool` -- send binary data

**self.settings** (settings proxy):
- `get(key: str, default=None)` -- read persistent setting
- `set(key: str, value)` -- write persistent setting
- `key` (read-only) -- hardware key (e.g., "PythonTestHardware.Default")
- `model` (read-only) -- hardware model name

**self.log** (logging proxy):
- `log(msg)`, `debug(msg)`, `warning(msg)`, `error(msg)`, `highlight(msg)`

## Build System

The QProcess approach requires no external dependencies:

- `BC_ENABLE_PYTHON_HARDWARE` option in `BuildConfig.cmake` (default OFF)
- When enabled: adds `src/hardware/python/` sources to `blackchirp-hardware`
  library and defines `BC_PYTHON_HARDWARE` compile definition
- No `find_package(Python3)` or `find_package(pybind11)` needed
- Python is a runtime dependency only (system `python3` on PATH)

## Current Status

### What's Done

The core QProcess infrastructure is complete and compiles cleanly:

- **`PythonProcess`** (`src/hardware/python/pythonprocess.h/.cpp`):
  QProcess wrapper with JSON-lines IPC, interleaved relay handling,
  log forwarding, and timeout management.
- **`PythonTestHardware`** (`src/hardware/python/pythontesthardware.h/.cpp`):
  HardwareObject subclass dispatching all virtual methods via IPC.
- **`python_hw_host.py`** (`dev-docs/python_hw_host.py`):
  Python-side IPC host with CommProxy, SettingsProxy, LogProxy, and
  method dispatch.
- **`test_hardware.py`** and **`echo_server.py`** (`dev-docs/`):
  Test script exercising all methods, and TCP echo server for testing.
- **Build system**: `BC_ENABLE_PYTHON_HARDWARE` option, conditional
  compilation, `#ifdef` guards. Builds with ON and OFF.
- **HardwareProfileManager**: `pythonScriptPath` field with getter/setter
  and load/save persistence.
- **HardwareManager**: PythonTestHardware conditionally registered in
  `d_optHwTypes` when `BC_PYTHON_HARDWARE` is defined.
- **pybind11 reference**: The abandoned pybind11 attempt is preserved on
  the `python-hw-pybind11` branch for reference.

### Proof-of-Concept Validated (2026-03-26)

All manual testing passed. The full IPC pipeline works end-to-end:

- **Script path UI**: `RuntimeHardwareConfigDialog` Advanced section has
  a QLineEdit + Browse button for Python script path, with per-profile
  persistence via `HardwareProfileManager::setPythonScriptPath()`.
- **Connection**: `test_connection` succeeds via comm relay to echo server.
  Log messages from Python appear in the hardware log panel.
- **Experiment lifecycle**: `prepareForExperiment`, `beginAcquisition`,
  `endAcquisition` all dispatch correctly. `sleep(bool)` works.
- **Rolling data**: `readAuxData` returns synthetic data at the configured
  interval. Rolling data files are written correctly.
- **Settings relay**: `self.settings.get/set` round-trips through IPC to
  C++ SettingsStorage. Values persist across restarts.
- **Hot-reload**: Accepting the Hardware Settings dialog triggers
  `readSettings`, which kills and restarts the Python subprocess.
- **Multi-profile**: Multiple PythonTestHardware profiles work correctly
  with independent script paths and settings.

### Known Issues / Gaps

1. **`exp.d_number` is 0 at `prepareForExperiment` time**: The experiment
   number is assigned in `Experiment::initialize()`, which runs after
   hardware prep. This is by design in the current architecture. Need to
   decide what experiment data to serialize for Python hardware.
2. **Experiment aux data**: `readAuxData` data appears in rolling data
   plots but not in `auxdata.csv` for experiments. The experiment aux data
   path (`bcReadAuxData` via `getAuxData`) is separate from the rolling
   data timer path.
3. **Host script deployment**: `findHostScript()` uses relative paths from
   the build directory. Need a CMake install rule for deployment.
4. **Python hardware type detection**: The script path UI is conditional
   on `hardwareType == "PythonTestHardware"`. This will need to become
   a registry-based check when Phase 2 trampolines are added.

### Remaining Work (after proof-of-concept validated)

#### Phase 2: Trampolines for Other Hardware Types

For each hardware interface class (FlowController, Clock, etc.), create a
`PythonXxx` trampoline class that inherits the interface and dispatches
type-specific virtual methods via IPC. The Python host script already supports
arbitrary method dispatch; only the C++ trampoline needs to be created.

Priority order:
1. `PythonFlowController` (simple, 3 required methods)
2. `PythonTemperatureController` (3 required methods)
3. `PythonPressureController` (9 required methods)
4. `PythonIOBoard` (2 required methods)
5. `PythonClock` (4 required methods)
6. `PythonPulseGenerator` (21 methods -- most complex)
7. `PythonAwg` (relies on prepareForExperiment only)
8. `PythonFtmwScope` / `PythonLifScope` (performance-sensitive polling)

#### Phase 3: Polish

- Template/skeleton scripts for each hardware type
- Script hot-reload improvements
- Python environment support (venv/conda path per-profile)
- Inline script editor in Hardware Settings dialog
- User documentation

## Open Questions

1. **Security**: Python scripts have full system access. Is a warning on
   first use sufficient, or do we need sandboxing?

2. **Performance**: The IPC round-trip adds ~1ms per operation, negligible
   for instrument I/O (10-100ms). But digitizer polling (FtmwScope/LifScope
   `readWaveform`) may need optimization -- possibly batching data or
   using shared memory for large transfers.

3. **Experiment data exposure**: How much of the Experiment object should
   `prepare_for_experiment` receive? Currently just `{"number": N}`.
