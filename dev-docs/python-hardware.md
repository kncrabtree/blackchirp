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
C++ PythonXxx class             IPC (JSON over pipes)         Python subprocess
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

### PythonHardwareBase Mixin

All Python trampoline classes inherit from `PythonHardwareBase` (via
multiple inheritance alongside their hardware base class) to eliminate
boilerplate. The mixin provides:

- **`initPythonProcess(comm, getter, setter)`** — creates the
  `PythonProcess`, sets hardware info, wires up settings callbacks
- **`testPythonConnection(comm)`** — starts the subprocess if needed,
  sends `test_connection`, returns success/failure
- **`startPythonProcess()`** — looks up `pythonScriptPath` and
  `pythonClassName` from `HardwareProfileManager`; returns false with
  an error if either is empty (no silent fallbacks)
- **`findHostScript()`** — static method to locate `python_hw_host.py`
- **`pythonSleep(b)`** / **`pythonReadSettings()`** — common IPC
  dispatches
- **`pythonForbiddenKeys()`** — static, returns `{commType, model}`
- **Destructor** — stops `pu_process` if running

The mixin's constructor takes `(key, model)` strings — no back-pointer
to `HardwareObject` is needed. Each concrete class passes `d_key` and
`d_model` from its `HardwareObject` base in the initializer list.

The Python script path and class name are managed entirely by
`HardwareProfileManager` (per-profile). There are no per-class settings
keys for these values.

### File Layout

```
src/hardware/python/
    pythonhardwarebase.h/.cpp         # Mixin base class: common Python subprocess management
    pythonprocess.h/.cpp              # QProcess wrapper: launch, IPC protocol, lifecycle
    pythonawg.h/.cpp                  # AWG trampoline
    pythonclock.h/.cpp                # Clock trampoline
    pythonflowcontroller.h/.cpp       # FlowController trampoline
    pythonioboard.h/.cpp              # IOBoard trampoline
    pythonpressurecontroller.h/.cpp   # PressureController trampoline
    pythontemperaturecontroller.h/.cpp # TemperatureController trampoline
    python_hw_host.py                 # Python-side IPC host script
    python_awg_template.py            # Template AWG driver script
    python_clock_template.py          # Template Clock driver script
    python_flowcontroller_template.py # Template FlowController driver script
    python_ioboard_template.py        # Template IOBoard driver script
    python_pressurecontroller_template.py # Template PressureController driver script
    python_temperaturecontroller_template.py # Template TemperatureController driver script
```

## Python API Contract

Each hardware type defines a Python class with methods matching the C++
virtual interface. Methods use snake_case. `self.comm`, `self.settings`,
and `self.log` are injected by the IPC host script.

### Subprocess Lifecycle

`PythonProcess::start()` establishes the subprocess and runs two
handshake steps before returning:

1. **`_init`** -- sets up `self.comm`, `self.settings`, and `self.log`
   proxies on the user object.
2. **`initialize`** -- calls the user script's `initialize()` method so
   it can set up internal state (e.g., channel dicts, calibration data).

After `start()` returns, the C++ trampoline sends `test_connection`.
This mirrors the C++ `HardwareObject` lifecycle:
**constructor → initialize() → testConnection()**.

**Key constraint**: `initialize()` is called exactly once per subprocess
start. If the subprocess is killed and restarted (e.g., due to script
path change), `initialize()` runs again on the new process. The
`readSettings()` path does **not** restart the subprocess; it sends a
`read_settings` IPC message to the running process instead.

### Common Methods (all types)

```python
def initialize(self):
    """Called once when the subprocess starts, after proxies are injected.

    Use this to set up internal state (dicts, calibration, etc.).
    self.comm is available but the connection has not been tested yet.
    Called automatically by PythonProcess::start() before testConnection.
    """

def test_connection(self) -> bool:
    """Verify communication with hardware. Return True on success.

    May be called multiple times (e.g., on reconnect). initialize()
    is guaranteed to have run before the first call.
    """

def read_aux_data(self) -> dict[str, float]:          # optional
def read_validation_data(self) -> dict[str, float]:    # optional
def prepare_for_experiment(self, config: dict) -> bool: # optional
def begin_acquisition(self):                            # optional
def end_acquisition(self):                              # optional
def sleep(self, sleeping: bool):                        # optional
def read_settings(self):                                # optional -- reload settings without restart
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
- `key` (read-only) -- hardware key (e.g., "PythonAwg.Default")
- `model` (read-only) -- hardware model name

**self.log** (logging proxy):
- `log(msg)`, `debug(msg)`, `warning(msg)`, `error(msg)`, `highlight(msg)`

## Trampoline Implementation Contract

These rules apply to all C++ Python trampoline classes (PythonAwg,
PythonFlowController, etc.) and must be followed when creating new ones.

### Creating a New Trampoline

1. Inherit from both the hardware base class and `PythonHardwareBase`
2. Initialize `PythonHardwareBase(d_key, d_model)` in the constructor
3. In the initialize override, call `initPythonProcess(p_comm, getter, setter)`
   and connect `pu_process->logMessage` to `this->logMessage`
4. In the test connection override, call `testPythonConnection(p_comm)`
5. Delegate `sleep()` to `pythonSleep()`, `readSettings()` to
   `pythonReadSettings()`, and build `forbiddenKeys()` from
   `pythonForbiddenKeys()` plus any class-specific keys
6. Implement hardware-specific virtual methods as JSON IPC dispatches
   using `pu_process->sendRequest()`

### Process Lifecycle

1. **`initialize()` / `fcInitialize()`**: Call `initPythonProcess()` to
   create the `PythonProcess` and wire up callbacks. Connect the log
   signal. Do **not** start the subprocess here — it is started lazily
   in `testConnection`.

2. **`testConnection()` / `fcTestConnection()`**: Call
   `testPythonConnection()`, which starts the subprocess if needed
   (via `startPythonProcess()` → `PythonProcess::start()`, which sends
   `_init` and `initialize` IPC) and then sends `test_connection`.

3. **`readSettings()`**: Call `pythonReadSettings()` to send
   `{"method": "read_settings"}` to the running process. Do **not** kill
   and restart the subprocess — that would trigger a spurious
   `initialize()` call and disrupt connected state.

### QSettings Key Paths

When writing config params and commType to QSettings before hardware
construction (in `onAddProfile()`), values must be written directly
under the `hwKey` group:

```
QSettings path: <hwType:label>/commType     ← CORRECT
QSettings path: <hwType:label>/flowChannels ← CORRECT
QSettings path: <hwType:label>/<impl>/commType ← WRONG (extra subgroup)
```

`HardwareObject`'s `SettingsStorage` root is `hwKey` (e.g.,
`FlowController:Default`). It reads `commType` and config params from
that level, not from an implementation subgroup.

### Python Script Path UI

The Python script path widget (QLineEdit + Browse button) in the
Advanced section of `RuntimeHardwareConfigDialog` is created for all
hardware types (under `#ifdef BC_PYTHON_HARDWARE`) but shown/hidden
dynamically based on whether the selected profile's implementation
contains "Python". This is checked via
`HardwareProfileManager::getImplementation()`.

### Base Class State Management Patterns

Hardware base classes fall into three categories based on how they manage
internal state. The category determines how the Python trampoline must
interact with config and how `prepareForExperiment` works.

#### Pattern A: Bulk Configure (complex inherited config state)

These classes **inherit** from a config class (e.g., `DigitizerConfig`)
with many coupled fields (channel maps, trigger settings, sample rates,
etc.). State is not modified by individual setter calls during normal
operation. Instead, the experiment provides a desired config object, the
subclass applies it to hardware in a single `configure()` call, reads
back actual values, and the base class updates its internal state from
the validated result.

**C++ pattern** (established by `LifScope`, now also `IOBoard`):
```
hwPrepareForExperiment:
  1. Get desired config from Experiment (or use current state)
  2. Call virtual configure(config&)
     → subclass applies settings to hardware
     → subclass reads back actual values, updates config reference
     → returns true/false
  3. If success: copy validated config to *this, store in Experiment
```

**Python trampoline pattern**: The trampoline serializes the config to
JSON and sends it as a `configure` IPC call. The Python script receives
the full config, applies settings, and returns
`{"success": True, "config": {...validated...}}`. The trampoline
deserializes the response back into the config reference.

Additionally, per-call channel selection (e.g., enabled channel indices
sent with each `readAnalogChannels()` call) ensures the Python side
always has current information even if config changes between calls.

| Base Class | Config Class | configure() | Status |
|---|---|---|---|
| **IOBoard** | IOBoardConfig (→ DigitizerConfig) | `bool configure(IOBoardConfig&)` pure virtual | Done |
| **LifScope** | LifDigitizerConfig (→ DigitizerConfig) | `bool configure(const LifDigitizerConfig&)` pure virtual | Existing (deferred) |
| **FtmwScope** | FtmwDigitizerConfig (→ DigitizerConfig) | No base virtual; subclasses override `prepareForExperiment` directly | Needs refactor (deferred) |

**Note on FtmwScope**: Unlike IOBoard and LifScope, FtmwScope does not
currently have a `configure()` virtual. Each subclass (e.g., DSA71604C)
overrides `prepareForExperiment()` and applies config directly. A future
PythonFtmwScope would likely need a `configure()` virtual added to
`FtmwScope`, following the same pattern as IOBoard and LifScope.

#### Pattern B: Granular Methods (base class manages state)

These classes **contain** a config object as a member (not inherited).
The base class owns all state and updates it through individual
getter/setter methods, each of which delegates to a `hw*` pure virtual
for the actual hardware I/O. The base class decides *when* and *which*
calls to make (e.g., polling order).

**Python trampoline pattern**: Each `hw*` virtual is dispatched as a
simple IPC call with the relevant arguments. The Python script only
sees one value at a time. No bulk config serialization is needed — the
base class handles all state management.

| Base Class | Config Class | Virtual Style | Status |
|---|---|---|---|
| **FlowController** | FlowConfig (contained) | 8 `hw*` pure virtuals (per-channel reads/writes) | Done |
| **PulseGenerator** | PulseGenConfig (contained) | ~24 `hw*` pure virtuals + `setAll()` bulk | Pending |
| **TemperatureController** | TemperatureControllerConfig (contained) | 3 `hw*` pure virtuals (per-channel) | Done |
| **PressureController** | PressureControllerConfig (contained) | 9 `hw*` pure virtuals (scalar reads/writes) | Done |

**PulseGenerator note**: Although PulseGenerator has a `setAll()` method,
it delegates to individual `hwSetChannel()` calls internally. The
trampoline only needs to implement the granular `hw*` methods.

#### Pattern C: Stateless / Experiment-Data Pass-Through

These classes have no complex internal config state. Instead, they
receive data from the Experiment at `prepareForExperiment` time and
program the hardware with it.

| Base Class | What it receives | Status |
|---|---|---|
| **AWG** | ChirpConfig waveform data + markers | Done |
| **Clock** | Frequency assignments per role | Done |

### Base Class Integration Details

Trampolines must respect the base class's ownership of state and
signals. For example:

- **FlowController**: `d_config`, `d_numChannels`, and
  `QPrivateSignal`-guarded signals are private. Trampolines update state
  only through the base class public slots (`readFlow()`,
  `readPressure()`, etc.), not by directly modifying `d_config`.
- **FlowController::poll()**: Non-virtual. The base class implements
  sequential channel cycling (one `readFlow(ch)` per tick, then
  `readPressure()`). Trampolines do not override this; the IPC round-trip
  per `hwRead*` call is the correct granularity since Python scripts may
  be communicating over slow serial links.
- **IOBoard**: `d_analogChannels` and `d_digitalChannels` (from
  `DigitizerConfig`) are updated only through the `configure()` virtual.
  The trampoline sends enabled channel indices with each read call so
  the Python script knows which channels to poll.

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
| PythonAwg | none | Done |
| PythonClock | `numOutputs` (int), `tunable` (bool) | Done |
| PythonFlowController | none (reads from settings) | Done |
| PythonIOBoard | `numAnalogChannels` (int), `numDigitalChannels` (int) | Done |
| PythonPressureController | `readOnly` (bool) | Done |
| PythonTemperatureController | `numChannels` (uint) | Done |
| PythonPulseGenerator | `numChannels` (int) | Done |

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
4. **Uses the correct class name** matching the `pythonClassName`
   configured in `HardwareProfileManager` (e.g., `AwgDriver`,
   `IOBoardDriver`, `FlowControllerDriver`)

### Script Naming Convention

Template scripts follow the pattern `python_<type>_template.py` and live in
`src/hardware/python/` alongside the C++ sources:

| Trampoline | Template File | Default Class Name |
|---|---|---|
| PythonAwg | `python_awg_template.py` | `AwgDriver` |
| PythonIOBoard | `python_ioboard_template.py` | `IOBoardDriver` |
| PythonFlowController | `python_flowcontroller_template.py` | `FlowControllerDriver` |
| PythonTemperatureController | `python_temperaturecontroller_template.py` | `TemperatureControllerDriver` |
| PythonPressureController | `python_pressurecontroller_template.py` | `PressureControllerDriver` |
| PythonClock | `python_clock_template.py` | `ClockDriver` |
| PythonPulseGenerator | `python_pulsegenerator_template.py` | `PulseGeneratorDriver` | Done |

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
6. The script is scanned for class definitions, which are then placed into
   a QComboBox for the user to select the correct class name (e.g.,
   `FlowControllerDriver`)

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

- **`PythonHardwareBase`** (`src/hardware/python/pythonhardwarebase.h/.cpp`):
  Mixin base class providing common Python subprocess management
  (process init, test connection, sleep, readSettings, findHostScript).
- **`PythonProcess`** (`src/hardware/python/pythonprocess.h/.cpp`):
  QProcess wrapper with JSON-lines IPC, interleaved relay handling,
  log forwarding, and timeout management.
- **`python_hw_host.py`** (`src/hardware/python/python_hw_host.py`):
  Python-side IPC host with CommProxy, SettingsProxy, LogProxy, and
  method dispatch. Supports generic keyword-argument dispatch for
  type-specific methods.
- **Build system**: `BC_ENABLE_PYTHON_HARDWARE` option, conditional
  compilation, `#ifdef` guards. Builds with ON and OFF.
- **HardwareProfileManager**: `pythonScriptPath` and `pythonClassName`
  fields with getter/setter and load/save persistence.
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
- **Settings reload**: Accepting the Hardware Settings dialog triggers
  `readSettings`, which sends `read_settings` IPC to the running process.
- **Multi-profile**: Multiple Python hardware profiles work correctly
  with independent script paths and settings.

#### Phase 2: Trampoline Classes (complete)

All non-scope hardware types have Python trampoline classes. Each
inherits from both its hardware base class and `PythonHardwareBase`,
eliminating boilerplate for process lifecycle, sleep, readSettings,
findHostScript, and forbiddenKeys.

- **`PythonAwg`** (`pythonawg.h/.cpp`): Inherits AWG. Dispatches
  `prepareForExperiment` with full chirp/RF config serialization,
  `beginAcquisition`, `endAcquisition`, `readAuxData`,
  `readValidationData` via IPC.
- **`PythonClock`** (`pythonclock.h/.cpp`): Inherits Clock. Dispatches
  `setHwFrequency` and `readHwFrequency` via IPC. Constructor reads
  `numOutputs` and `tunable` from QSettings before Clock construction.
- **`PythonFlowController`** (`pythonflowcontroller.h/.cpp`): Inherits
  FlowController. Dispatches 8 `hw*` pure virtuals via IPC. Base class
  handles polling, `readAll()`, and `prepareForExperiment()`.
- **`PythonIOBoard`** (`pythonioboard.h/.cpp`): Inherits IOBoard.
  Implements `configure(IOBoardConfig&)` to serialize/deserialize the
  full digitizer config via IPC. Sends enabled channel indices with
  each read call.
- **`PythonPressureController`** (`pythonpressurecontroller.h/.cpp`):
  Inherits PressureController. Dispatches 7 `hw*` pure virtuals via
  IPC. Constructor reads `readOnly` from QSettings before construction.
- **`PythonTemperatureController`**
  (`pythontemperaturecontroller.h/.cpp`): Inherits TemperatureController.
  Dispatches `readHwTemperature` via IPC. Constructor reads `numChannels`
  from QSettings before construction.

#### Phase 2: HwConfigParam Registry (complete)

Infrastructure for declaring constructor parameters that need UI input:

- **`HwConfigParam` struct** in `hardwareregistry.h`
- **`REGISTER_HARDWARE_PARAMS` macro** in `hardwareregistration.h`
- **`addConfigParams()`/`getConfigParams()`** in `HardwareRegistry`
- **Dynamic UI** in `RuntimeHardwareConfigDialog::onAddProfile()` that
  auto-generates widgets from config params and writes values to QSettings
  before hardware construction

#### Phase 2: Template Scripts & Host Update (complete)

- **`python_hw_host.py`** updated with generic keyword-argument dispatch
  for type-specific methods (replaces `ValueError` for unknown methods)
- **Template scripts created** for all trampolines:
  - `python_awg_template.py` (class `AwgDriver`)
  - `python_clock_template.py` (class `ClockDriver`)
  - `python_flowcontroller_template.py` (class `FlowControllerDriver`)
  - `python_ioboard_template.py` (class `IOBoardDriver`)
  - `python_pressurecontroller_template.py` (class `PressureControllerDriver`)
  - `python_temperaturecontroller_template.py` (class `TemperatureControllerDriver`)
- **CMake deployment** updated in `BlackchirpHardware.cmake`: uses
  `file(GLOB ... python_*_template.py)` to copy templates to build dir
  and install them to `share/blackchirp/`
- **Template-copy dialog** in `onAddProfile()`: when a Python hardware
  profile is created, offers to copy the template to a user-chosen
  location (file dialog initializes to app save path), then sets the
  script path in `d_previewPythonScriptConfig`. Template filename is
  derived from the implementation class name (e.g., `PythonAwg` →
  `python_awg_template.py`).

### Next Steps

#### Template Validation

Exercise each existing template script with Virtual hardware to verify
consistent behavior across all trampoline classes. Each template should
successfully:
- Connect via Virtual protocol
- Return valid data from all `hw*` methods
- Handle `sleep`/`readSettings` round-trips
- Pass through `prepareForExperiment` (where applicable)

#### PythonPulseGenerator (complete)

`PythonPulseGenerator` (`pythonpulsegenerator.h/.cpp`) dispatches all 22
`hw*` pure virtuals via IPC. Constructor reads `numChannels` from QSettings.
`initializePGen()` initializes the process; `testConnection()` is overridden
directly (not via a helper virtual, since PulseGenerator doesn't define one).
`sleep()` is final in PulseGenerator and calls `setHwPulseEnabled(false)`
internally, so Python sleep is handled automatically through IPC.
Template: `python_pulsegenerator_template.py` (class `PulseGeneratorDriver`).

#### FtmwScope / LifScope (deferred)

Performance-sensitive Pattern A types. Both inherit from DigitizerConfig
and need bulk configure. May also need batched data transfer or shared
memory for large waveforms (readWaveform).

- **LifScope**: Already has `virtual bool configure(const LifDigitizerConfig&)`
  — the same pattern now used by IOBoard. PythonLifScope would follow
  the PythonIOBoard pattern directly.
- **FtmwScope**: Does **not** have a `configure()` virtual. Subclasses
  override `prepareForExperiment()` directly. A PythonFtmwScope would
  require adding a `configure()` virtual to FtmwScope first.

#### Polish

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
   using shared memory for large transfers. See `digitizer-data-flow.md`.
