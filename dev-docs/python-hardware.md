# Python Hardware Implementations

## Motivation

Users should be able to write hardware drivers in Python without modifying or
recompiling Blackchirp. This enables:

- Rapid prototyping of new hardware support
- Use of vendor-provided Python libraries (pyvisa, manufacturer SDKs)
- User-customizable drivers that can be tuned without C++ knowledge
- Community-contributed implementations shared as simple `.py` files

## Architecture

### Trampoline Pattern

For each hardware interface class (FlowController, Clock, etc.), a C++
**trampoline** class inherits the interface and dispatches every virtual method
to a user-supplied Python object via pybind11's embedded interpreter:

```
Blackchirp C++ lifecycle        C++ Trampoline              User's Python class
────────────────────────        ──────────────              ───────────────────
bcInitInstrument()          →   PythonFlowController
  initialize()              →     loads .py, creates obj  → __init__()
  testConnection()          →     calls py test_connection  → test_connection()
hwReadFlow(ch)              →     calls py read_flow(ch)    → read_flow(ch)
hwSetFlowSetpoint(ch,val)   →     calls py set_setpoint     → set_flow_setpoint(ch,val)
prepareForExperiment(exp)   →     packs config dict         → prepare_for_experiment(config)
beginAcquisition()          →     calls py begin            → begin_acquisition()
endAcquisition()            →     calls py end              → end_acquisition()
```

The trampoline handles:
- Loading the Python script from a user-configured path
- Injecting a `comm` wrapper so Python can use Blackchirp's CommunicationProtocol
- Translating C++ types (Experiment, QByteArray, enums) to Python-friendly types
  (dicts, bytes, strings)
- Catching Python exceptions and mapping them to `d_errorString` + failure return
- Providing no-op defaults for optional methods the user didn't implement

### Communication

Python implementations can communicate with hardware in two ways:

1. **Via Blackchirp's CommunicationProtocol** (`self.comm`): The trampoline
   injects a wrapper around `p_comm` supporting `query()`, `write()`, and
   `read_bytes()`. The user selects TCP/RS232/GPIB/Custom in the Hardware
   Configuration dialog as usual.

2. **Via external Python libraries**: The user sets CommunicationProtocol to
   Virtual (or a new "Python-managed" type), imports their own library
   (pyvisa, pyserial, a vendor SDK), and handles all communication themselves.
   Blackchirp's `p_comm` goes unused.

Both approaches are valid and can coexist across different hardware objects.

### Settings Access

Python implementations need access to persistent settings for device-specific
configuration (number of channels, calibration values, custom parameters). The
trampoline injects a `self.settings` wrapper scoped to the hardware object's
own SettingsStorage group:

```python
def initialize(self):
    self.num_channels = self.settings.get("numChannels", 4)

def test_connection(self):
    resp = self.comm.query("CHANNELS?\n")
    n = int(resp.strip())
    self.settings.set("numChannels", n)  # persists across restarts
```

The wrapper exposes:
- `get(key, default)` / `set(key, value)` for scalar values
- `get_array(key)` / `set_array(key, list_of_dicts)` for array settings
- Read-only access to hardware keys: `self.settings.key` (the `d_key`),
  `self.settings.model` (the `d_model`)

Settings written by Python are visible in the Hardware Settings dialog
tree view and persist in the same QSettings file as C++ implementations.
This lets users configure Python hardware through the normal Blackchirp UI.

### Threading & GIL

Many hardware types run in their own QThread (`d_threaded = true`). On the
`devel` branch, the threading defaults are:

| Threaded (own QThread) | Non-threaded (main thread) |
|---|---|
| FtmwScope, LifScope, LifLaser | PulseGenerator, AWG |
| Clock, IOBoard, GpibController | FlowController, PressureController |
| (base class default: true) | TemperatureController |

Python hardware must support both modes from the start. The key mechanism
is pybind11's GIL management:

- **Threaded hardware**: Each QThread that calls into Python must acquire
  the GIL via `py::gil_scoped_acquire` before dispatching to the Python
  object, and release it during C++ I/O waits. pybind11 manages thread
  state (`PyThreadState`) automatically when using these scoped guards.
- **Non-threaded hardware**: Runs on the main thread where the interpreter
  was initialized. GIL acquisition is simpler but still required.

Performance characteristics:

- **I/O-bound calls** (query/response): The GIL is released during C++ I/O
  waits (via `py::gil_scoped_release` in the comm wrapper). Multiple Python
  hardware objects in different threads interleave fine — while one waits
  for a device response, others can execute Python code.
- **CPU-bound calls** (waveform computation, data parsing): Hold the GIL.
  For most hardware this is negligible (microseconds). AWG waveform
  generation could be slower but runs only once during `prepareForExperiment`.
- **Polling loops** (FtmwScope/LifScope `readWaveform`): Called repeatedly
  during acquisition. The poll frequency should be user-adjustable so users
  can tune the tradeoff between acquisition speed and system responsiveness.
  Users can also mitigate by configuring hardware-side block averaging.

### Registration & Discovery

Python hardware registers through the existing HardwareRegistry system. Each
trampoline class (e.g., `PythonFlowController`) registers as a normal
implementation. At runtime, the trampoline reads a `pythonScript` setting
from SettingsStorage to locate the user's `.py` file.

Script management options:
- **Per-profile setting**: The script path is stored in the hardware profile's
  settings, configurable in the Hardware Settings dialog
- **Standard location**: `~/.config/blackchirp/python/` as a default search path
- **Validation**: The trampoline checks that the script defines the required
  class and methods at load time, reporting clear errors for missing methods

### Python Environment

Users often need third-party packages (pyvisa, vendor SDKs) that require
`pip install` into an isolated environment. The trampoline must support:

- **System Python**: Default — uses whatever `python3` is on PATH
- **Virtual environment**: User specifies a venv path in settings. The
  trampoline activates it before importing the script by prepending the
  venv's `site-packages` to `sys.path`
- **Conda environment**: User specifies a conda env name or prefix. The
  trampoline resolves its `site-packages` path similarly

The environment setting is per-profile (different hardware can use different
environments) and configurable in the Hardware Settings dialog. The embedded
interpreter is initialized once, but `sys.path` is adjusted per-script to
pick up environment-specific packages.

### Script Editing

The Hardware Settings dialog for Python implementations includes:

- **Script path selector** with Browse button
- **Inline editor** (QPlainTextEdit with basic syntax highlighting) for
  quick edits and hot testing — changes can be saved and the module
  reloaded via `testConnection()` without restarting Blackchirp
- **Skeleton generator**: A button that creates a new `.py` file
  pre-populated with all required and optional methods for the hardware
  type, with docstrings explaining each method's contract
- **Log/traceback viewer**: Python exceptions during method dispatch are
  shown with full tracebacks in the hardware log panel

## Implementation Plan

### Phase 1: Infrastructure

**Goal**: Embedded Python interpreter + comm/settings wrappers + thread-safe
dispatch + one working trampoline.

1. Add pybind11 as a build dependency (CMake `find_package(pybind11)` or
   bundled headers)
2. Initialize the Python interpreter once at application startup
   (`py::scoped_interpreter` in main.cpp or a lazy singleton); call
   `PyEval_SaveThread()` to release the GIL so worker threads can acquire it
3. Create `CommWrapper` class exposing `p_comm` methods to Python:
   - `query(cmd: str) -> str`
   - `write(cmd: str) -> bool`
   - `read_bytes(n: int) -> bytes`
   - `write_binary(data: bytes) -> bool`
   - All methods release the GIL during the underlying C++ I/O call
4. Create `SettingsWrapper` class exposing the hardware object's
   SettingsStorage to Python:
   - `get(key, default)` / `set(key, value)` for scalar settings
   - `get_array(key)` / `set_array(key, list_of_dicts)` for arrays
   - Read-only `key` and `model` properties
5. Create `PythonHardwareBase` utility class/namespace with:
   - Script loading with venv/conda `sys.path` injection
   - Class instantiation and `self.comm` / `self.settings` injection
   - Thread-safe method dispatch: `py::gil_scoped_acquire` before every
     Python call, with proper `PyThreadState` handling for worker threads
   - Exception handling: catch Python exceptions, extract traceback,
     store in `d_errorString`, emit to log with full traceback
   - Optional-method detection (hasattr checks, no-op defaults)
   - Module reload support for hot testing
6. Implement `PythonFlowController` as the first trampoline
7. Register it in HardwareRegistry alongside existing implementations
8. Add Python-specific settings UI in Hardware Settings dialog:
   script path, environment path, inline editor, skeleton generator

**Deliverable**: A user can write a `.py` file implementing a flow controller,
select "PythonFlowController" in the Hardware Configuration dialog, point it
at their script, and use it in an experiment.

### Phase 2: Simple Hardware Types

**Goal**: Trampolines for all polling-based, non-streaming hardware.

Create trampoline classes for:
- `PythonTemperatureController` (3 required methods)
- `PythonPressureController` (9 required methods)
- `PythonIOBoard` (2 required methods)
- `PythonClock` (4 required methods)
- `PythonLifLaser` (4 required methods)
- `PythonGpibController` (2 required methods)

Each trampoline follows the same pattern established in Phase 1. The Python
API for each type is documented with a template `.py` file showing all
required and optional methods with docstrings.

Also in this phase:
- Template/example scripts installed to a standard location
- Error reporting improvements (line numbers, tracebacks in log)
- Script hot-reload: re-import the module on `bcTestConnection()` so users
  can edit scripts without restarting Blackchirp

### Phase 3: PulseGenerator

**Goal**: Trampoline for the most method-heavy interface.

PulseGenerator has 21 pure virtual methods covering channel configuration
(width, delay, active level, enabled, sync, mode, duty cycle) plus global
settings (rep rate, pulse mode, pulse enabled). The Python API should flatten
these into a clean interface:

```python
def set_channel_width(self, channel: int, width: float) -> bool
def read_channel_width(self, channel: int) -> float
def set_rep_rate(self, rate: float) -> bool
# ... etc
```

This phase also addresses enum translation: `PulseGenConfig::ActiveLevel`,
`PulseGenConfig::ChannelMode`, and `PulseGenConfig::PGenMode` must be exposed
as Python string constants or IntEnums.

### Phase 4: AWG

**Goal**: Trampoline for chirp source / arbitrary waveform generators.

AWG has no pure virtual methods of its own — it relies on
`prepareForExperiment()` from `HardwareObject`. The trampoline must:

1. Extract chirp waveform data, sample rate, marker data, and relevant
   RF config from the Experiment object
2. Pack it into a Python dict with bytes/float/int values
3. Call `prepare_for_experiment(config)` on the Python object

The Python implementation handles all waveform upload logic using either
`self.comm` or its own vendor library.

```python
def prepare_for_experiment(self, config: dict) -> bool:
    waveform = config["chirp_data"]      # bytes
    sample_rate = config["sample_rate"]   # float (Hz)
    markers = config["marker_data"]       # bytes
    # ... upload to hardware
    return True
```

### Phase 5: Digitizers (FtmwScope, LifScope)

**Goal**: Trampolines for data-streaming hardware with performance tuning.

These are the most performance-sensitive types. The trampoline must:

1. Call `readWaveform()` on the Python object, which returns raw bytes
2. Feed those bytes back into the C++ signal emission path
   (`emitShot()` / `emit waveformRead()`)

```python
def read_waveform(self) -> bytes:
    # Read from scope via vendor library or self.comm
    data = self.scope.read_binary_values(...)
    return bytes(data)
```

Performance considerations:
- **Poll frequency**: Expose as a user-adjustable setting. Default to a
  conservative rate; let users increase if their Python implementation
  is fast enough.
- **Block averaging**: Users can configure the scope to average N shots
  in hardware before transferring, reducing the poll rate needed.
- **Zero-copy**: For `self.comm`-based implementations, investigate whether
  the bytes can be passed without copying between Python and C++.
- **Timeout handling**: The trampoline must handle the case where Python
  raises a timeout or returns empty data gracefully.

The `configure()` method on LifScope needs the LifDigitizerConfig exposed
as a dict (sample rate, record length, trigger settings, etc.).

### Phase 6: Documentation & Templates

- User guide: how to write a Python hardware implementation
- API reference for each hardware type's Python interface
- Template scripts for every hardware type (installable examples)
- Troubleshooting guide (common errors, GIL gotchas, performance tips)

## Python API Contract

Each hardware type defines a Python class with methods matching the C++
virtual interface. Methods use snake_case. The `self.comm` object is
injected by the trampoline if a Blackchirp CommunicationProtocol is selected.

### Common Methods (all types)

```python
def initialize(self):
    """Called once at startup. Set up any resources."""

def test_connection(self) -> bool:
    """Verify communication with hardware. Return True on success,
    raise an exception with a descriptive message on failure."""

def read_aux_data(self) -> dict[str, float]:          # optional
def read_validation_data(self) -> dict[str, float]:    # optional
def prepare_for_experiment(self, config: dict) -> bool: # optional
def begin_acquisition(self):                            # optional
def end_acquisition(self):                              # optional
def sleep(self, sleeping: bool):                        # optional
```

### Type-Specific Methods

See individual trampoline phases above for the method signatures each
type requires. The general principle: every C++ `hw*` pure virtual becomes
a Python method with the `hw` prefix stripped and the name converted to
snake_case.

## Build System

- pybind11 added as an optional dependency (`BC_ENABLE_PYTHON_HARDWARE` in
  BuildConfig.cmake, default OFF)
- When enabled, the Python trampoline classes are compiled into a separate
  static library (`blackchirp-python-hw`) linked into the main application
- Python 3.8+ required (for stable embedding API)
- The embedded interpreter is initialized lazily on first use of a Python
  hardware object

## Resolved Decisions

- **Script editing**: Inline editor in Hardware Settings dialog for hot
  testing, plus skeleton generator for new scripts
- **Python environments**: Per-profile venv/conda path setting; trampoline
  adjusts `sys.path` before importing
- **Module reloading**: Reload on every `testConnection()` call for fast
  iteration during development
- **Multiple scripts**: Each hardware profile has its own script path;
  all share the embedded interpreter and GIL
- **Settings access**: Python gets scoped read/write to its own
  SettingsStorage group via `self.settings`
- **Threading**: Supported from the start via `py::gil_scoped_acquire` /
  `py::gil_scoped_release`; no restriction to non-threaded types

## Open Questions

1. **Security**: Python scripts have full system access. Is a warning on
   first use sufficient, or do we need sandboxing?

2. **Interpreter lifecycle**: Should the embedded interpreter persist for
   the entire application lifetime, or be finalized/restarted when all
   Python hardware is removed? (pybind11's `scoped_interpreter` does not
   support re-initialization — this may force a persistent interpreter.)

3. **Experiment data exposure**: How much of the Experiment object should
   `prepare_for_experiment` receive? A flat dict of relevant settings, or
   a richer nested structure? The answer may vary by hardware type.

4. **FtmwScope poll tuning**: Should the poll interval be a simple
   SettingsStorage value (adjustable in Hardware Settings), or does it
   need a real-time slider in the acquisition UI?
