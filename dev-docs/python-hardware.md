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

### Generic Method Dispatch

The Python host script (`python_hw_host.py`) uses generic keyword-argument
dispatch for type-specific methods. When the C++ trampoline sends:
```json
{"id": 5, "method": "hw_read_flow", "channel": 2}
```

The host calls `user_obj.hw_read_flow(channel=2)`. Any keys in the request
other than `"id"` and `"method"` are passed as keyword arguments. This allows
all hardware-type-specific methods to work without modifying the host script.

Methods not implemented by the user's class return a safe default (`None`,
`True`, or `{}` depending on the method category).

### File Layout

```
src/hardware/python/
    pythonprocess.h/.cpp              # QProcess wrapper: launch, IPC protocol, lifecycle
    pythontesthardware.h/.cpp         # HardwareObject subclass (proof-of-concept)
    pythonawg.h/.cpp                  # AWG trampoline
    pythonioboard.h/.cpp              # IOBoard trampoline
    pythonflowcontroller.h/.cpp       # FlowController trampoline
    python_hw_host.py                 # Python-side IPC host script
    python_awg_template.py            # Template AWG driver script
    python_ioboard_template.py        # Template IOBoard driver script
    python_flowcontroller_template.py # Template FlowController driver script
dev-docs/
    echo_server.py                    # TCP echo server for testing
    test_hardware.py                  # Example Python hardware script
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

### Deployment

The following Python files are deployed alongside the application:
- `python_hw_host.py` -- IPC host script (required at runtime)
- Template scripts (`python_*_template.py`) -- copied to build dir and
  installed to `share/blackchirp/`

Template scripts are offered to the user when creating a new Python hardware
profile (see Template Script Workflow below).

## HwConfigParam Registry

Some hardware base classes require constructor parameters that must be known
before the C++ object is constructed (e.g., `numChannels` for
TemperatureController, `tunable` for Clock). Python trampolines cannot
hard-code these since the values are user-defined.

### Design

A `HwConfigParam` struct in `hardwareregistry.h` declares a parameter:

```cpp
struct HwConfigParam {
    QString key;           // SettingsStorage key
    QString label;         // Display label for UI
    QVariant defaultValue; // Type-aware default (determines widget type)
    QVariant minimum;      // Optional min for numeric types
    QVariant maximum;      // Optional max for numeric types
};
```

Each trampoline that needs constructor params defines a static function:

```cpp
static QVector<HwConfigParam> configParams();
```

This is registered via the `REGISTER_HARDWARE_PARAMS(CLASS)` macro, which
stores the params in `HardwareRegistry` alongside the factory and protocol
registrations.

### UI Integration

When a user adds a hardware profile in `RuntimeHardwareConfigDialog`, the
dialog queries `HardwareRegistry::getConfigParams()` for the selected
implementation. If non-empty, a "Configuration Parameters" group box is
shown with auto-generated widgets:

| QVariant Type | Widget |
|---|---|
| `int` / `uint` | QSpinBox (with min/max) |
| `double` | QDoubleSpinBox (with min/max) |
| `bool` | QCheckBox |
| `QString` | QLineEdit |

Values are written to QSettings under the hardware's SettingsStorage key
*before* the hardware object is constructed, so base class constructors
(e.g., `FlowController`'s `getOrSetDefault(flowChannels, 4)`) find them.

### Which Trampolines Need Config Params

| Trampoline | Params | Status |
|---|---|---|
| PythonAwg | none | Done (Wave 1) |
| PythonIOBoard | none | Done (Wave 1) |
| PythonFlowController | none (reads from settings) | Done (Wave 1) |
| PythonTemperatureController | `numChannels` (uint) | Wave 2 |
| PythonPressureController | `readOnly` (bool) | Wave 2 |
| PythonClock | `numOutputs` (int), `tunable` (bool) | Wave 2 |
| PythonPulseGenerator | `numChannels` (int) | Wave 2 |

## Template Script Workflow

### Requirements

Each Python hardware trampoline class must have a corresponding template
script that:

1. **Works out of the box** with the Virtual communication protocol as a
   functional "virtual" hardware implementation
2. **Documents every method** with docstrings explaining:
   - What the method does in the context of Blackchirp
   - When it is called during the hardware/experiment lifecycle
   - What it is expected to return (type and semantics)
   - Error return values (e.g., `-1.0` for failed reads)
3. **Demonstrates** usage of `self.comm`, `self.settings`, and `self.log`
4. **Uses the correct class name** matching the trampoline's default
   `pythonClass` setting (e.g., `AwgDriver`, `IOBoardDriver`,
   `FlowControllerDriver`)

### Script Naming Convention

Template scripts follow the pattern `python_<type>_template.py` and live in
`src/hardware/python/` alongside the C++ sources:

| Trampoline | Template File | Default Class Name |
|---|---|---|
| PythonTestHardware | `python_test_template.py` | `TestHardware` |
| PythonAwg | `python_awg_template.py` | `AwgDriver` |
| PythonIOBoard | `python_ioboard_template.py` | `IOBoardDriver` |
| PythonFlowController | `python_flowcontroller_template.py` | `FlowControllerDriver` |
| PythonTemperatureController | `python_temperaturecontroller_template.py` | `TemperatureControllerDriver` |
| PythonPressureController | `python_pressurecontroller_template.py` | `PressureControllerDriver` |
| PythonClock | `python_clock_template.py` | `ClockDriver` |
| PythonPulseGenerator | `python_pulsegenerator_template.py` | `PulseGeneratorDriver` |

### User Workflow

When a user creates a new Python hardware profile in
`RuntimeHardwareConfigDialog`:

1. The standard "Add Profile" dialog collects implementation, protocol,
   label, and any config params
2. After the profile is created, if the implementation is a Python hardware
   type, the dialog asks: *"Would you like to create a copy of the template
   script to customize?"*
3. If the user clicks **Yes**, a file-save dialog opens with a suggested
   filename (e.g., `my_flow_controller.py`)
4. The template script is copied to the user's chosen location
5. The saved path is automatically set as the profile's Python script path

This ensures users always start from a working, well-documented script
rather than an empty file.

### Template-to-Trampoline Mapping

Each trampoline class provides a static method to identify its template:

```cpp
// In the trampoline .cpp file, the template filename is used by the dialog
// to locate the template in the application's resource directory.
```

The dialog locates the template using the same search paths as
`findHostScript()` (application dir, `../share/blackchirp/`).

## Current Status

### What's Done

#### Phase 1: Core Infrastructure (complete)

- **`PythonProcess`** (`src/hardware/python/pythonprocess.h/.cpp`):
  QProcess wrapper with JSON-lines IPC, interleaved relay handling,
  log forwarding, and timeout management.
- **`PythonTestHardware`** (`src/hardware/python/pythontesthardware.h/.cpp`):
  HardwareObject subclass dispatching all virtual methods via IPC.
- **`python_hw_host.py`** (`src/hardware/python/python_hw_host.py`):
  Python-side IPC host with CommProxy, SettingsProxy, LogProxy, and
  method dispatch. Supports generic keyword-argument dispatch for
  type-specific methods.
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

#### Phase 1 Proof-of-Concept Validated (2026-03-26)

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

#### Phase 2: Wave 1 Trampolines (complete)

Three trampoline classes created for hardware types with no constructor
parameter issues:

- **`PythonAwg`** (`pythonawg.h/.cpp`): Inherits AWG. Dispatches all
  HardwareObject virtuals via IPC. No AWG-specific pure virtuals.
- **`PythonIOBoard`** (`pythonioboard.h/.cpp`): Inherits IOBoard.
  Dispatches `readAnalogChannels()` and `readDigitalChannels()` via IPC.
  Base class handles `readAuxData`/`readValidationData` by calling these.
- **`PythonFlowController`** (`pythonflowcontroller.h/.cpp`): Inherits
  FlowController. Dispatches `fcInitialize()`, `fcTestConnection()`, and
  8 `hw*` pure virtuals via IPC. Base class handles polling, `readAll()`,
  and `prepareForExperiment()`.

All three compile cleanly, tests pass.

#### Phase 2: HwConfigParam Registry (complete)

Infrastructure for declaring constructor parameters that need UI input:

- **`HwConfigParam` struct** in `hardwareregistry.h`
- **`REGISTER_HARDWARE_PARAMS` macro** in `hardwareregistration.h`
- **`addConfigParams()`/`getConfigParams()`** in `HardwareRegistry`
- **Dynamic UI** in `RuntimeHardwareConfigDialog::onAddProfile()` that
  auto-generates widgets from config params and writes values to QSettings
  before hardware construction

#### Phase 2: Template Scripts & Host Update (in progress)

Completed:
- **`python_hw_host.py`** updated with generic keyword-argument dispatch
  for type-specific methods (replaces `ValueError` for unknown methods)
- **Template scripts created** for Wave 1 trampolines:
  - `python_awg_template.py` (class `AwgDriver`)
  - `python_ioboard_template.py` (class `IOBoardDriver`)
  - `python_flowcontroller_template.py` (class `FlowControllerDriver`)
- **CMake deployment** updated in `BlackchirpHardware.cmake`: uses
  `file(GLOB ... python_*_template.py)` to copy templates to build dir
  and install them to `share/blackchirp/`

Remaining:
- Add template-copy dialog to `RuntimeHardwareConfigDialog::onAddProfile()`
  (ask user if they want a copy, show save dialog, set as script path).
  This is the **next immediate task**. The modification point is in
  `runtimehardwareconfigdialog.cpp` in the `onAddProfile()` method, after
  the profile is successfully created (~line 1110). The dialog needs to:
  1. Detect if the implementation is a Python hardware type (check if class
     name starts with "Python" or check registry for a template filename)
  2. Ask the user with QMessageBox::question
  3. If yes, locate the template file using the same search paths as
     `findHostScript()` (app dir, `../share/blackchirp/`)
  4. Show QFileDialog::getSaveFileName
  5. Copy the template to the chosen path
  6. Set `d_previewPythonScriptConfig[profileKey] = savedPath`
- Create template scripts for Wave 2 trampolines (after those classes exist)

### Remaining Work

#### Phase 2: Wave 2 Trampolines

For each remaining hardware type, create a trampoline class using the
HwConfigParam registry for constructor parameters:

1. `PythonTemperatureController` (3 pure virtuals + `configParams`)
2. `PythonPressureController` (9 pure virtuals + `configParams`)
3. `PythonClock` (4 pure virtuals + `configParams`)
4. `PythonPulseGenerator` (24 pure virtuals + `configParams`)

Each trampoline needs:
- A `static QVector<HwConfigParam> configParams()` method
- `REGISTER_HARDWARE_PARAMS(ClassName)` in the .cpp file
- Constructor reads params from SettingsStorage (written by dialog before
  construction) and passes to base class constructor
- For TemperatureController/PulseGenerator/Clock: the param value is read
  from `HardwareProfileManager::instance().getConstructorParam()` or
  directly from QSettings in the member initializer list since
  SettingsStorage hasn't been constructed yet at that point. **Alternative
  approach**: read from QSettings directly using a static helper, since the
  dialog writes values under `hwKey/implementation/key` before construction.

#### Phase 2: FtmwScope / LifScope (deferred)

Performance-sensitive polling types. May need batched data transfer or
shared memory for large waveforms. Deferred until after all other types.

#### Phase 3: Polish

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
