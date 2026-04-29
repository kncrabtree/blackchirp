# Bundle 06 — Python Hardware (User Guide)

A new chapter introducing user-written hardware drivers in Python.

## Scope

Add a new chapter at `doc/source/user_guide/python_hardware.rst`
with the following sub-pages under
`doc/source/user_guide/python_hardware/`:

- `overview.rst` — what Python hardware is, why it exists, the
  architectural one-paragraph summary (subprocess + JSON IPC, no
  pybind11 dependency on the user's machine), the security warning
  (Python scripts run with the same permissions as Blackchirp),
  the supported hardware types table (which `Python*` trampolines
  exist).
- `selecting.rst` — how to choose a Python implementation in
  `RuntimeHardwareConfigDialog`, the script-path field, the class-
  name dropdown, the per-profile Python environment field
  (`pythonEnvPath`) for venv/conda setups, and the offered template
  copy on profile creation.
- `writing_a_driver.rst` — the Python API contract: subclass the
  driver class from a template, the lifecycle methods
  (`initialize`, `test_connection`, `prepare_for_experiment`,
  `begin_acquisition`, `end_acquisition`, `sleep`, `read_settings`,
  `read_aux_data`, `read_validation_data`), the injected proxies
  (`self.comm`, `self.settings`, `self.log`), the optional
  `self.scope` proxy for digitizer push, return-type expectations,
  default values for unimplemented methods.
- `hot_reload.rst` — the Reload Script and Open in Editor buttons in
  HwDialog for Python hardware, what state survives a reload, what
  errors look like (script-path errors, syntax errors, missing
  class).
- `per_type_capabilities.rst` — table summarising what each
  trampoline type expects: AWG (chirp/RF data passed in
  prepare_for_experiment), Clock (frequency assignments), Flow /
  Temperature / Pressure / IOBoard (granular `hw_*` methods or bulk
  `configure`), PulseGenerator (`hw_*` methods), FtmwScope (push via
  `self.scope.emit_shot`), LifScope (push), GpibController, LifLaser.
  This is the table users consult when deciding what to implement.

## Out of scope

- Internal IPC protocol details, JSON message formats, the
  trampoline contract for adding a new `Python*` C++ class — that
  is developer-guide content (bundle 12).
- The `WaveformBuffer` and pre-accumulation behaviour — bundle 12.

## Sources

- `dev-docs/python-hardware.md` — primary; extract user-facing
  sections (Architecture overview, API Contract, Template Script
  Workflow, environment support).
- `dev-docs/python-script-reload.md` — for the hot-reload and
  open-in-editor pages.
- `dev-docs/python-env-support.md` — for the env-path field
  documentation.
- `dev-docs/python-process-push-refactor.md` — for the
  `self.scope` push model used by `PythonFtmwScope` and
  `PythonLifScope`.
- Source: each `python_*_template.py` — read to confirm method
  signatures and defaults.
- Source: `src/hardware/python/python_hw_host.py` — confirm the
  proxy API surface.

## Sphinx file deltas

**Created:**
- `doc/source/user_guide/python_hardware.rst`
- `doc/source/user_guide/python_hardware/overview.rst`
- `doc/source/user_guide/python_hardware/selecting.rst`
- `doc/source/user_guide/python_hardware/writing_a_driver.rst`
- `doc/source/user_guide/python_hardware/hot_reload.rst`
- `doc/source/user_guide/python_hardware/per_type_capabilities.rst`

## Toctree delta

In `user_guide.rst` (after `hardware_config`):

```
   user_guide/python_hardware
```

In `python_hardware.rst` (new):

```rst
.. toctree::
   :hidden:

   python_hardware/overview
   python_hardware/selecting
   python_hardware/writing_a_driver
   python_hardware/hot_reload
   python_hardware/per_type_capabilities
```

## Screenshots

- `_static/user_guide/python_hardware/profile_creation.png` —
  RuntimeHardwareConfigDialog Add Profile flow with Python script
  path and class-name dropdown visible.
- `_static/user_guide/python_hardware/template_copy_prompt.png` —
  the prompt that offers to copy the template script.
- `_static/user_guide/python_hardware/hwdialog_python.png` — HwDialog
  for a Python device showing the Open in Editor / Reload Script
  controls and status label.
- `_static/user_guide/python_hardware/error_state.png` — example of
  the error state shown in the Python control widget when the
  script fails to load.

## Acceptance criteria

- A user with no C++ knowledge can read this chapter and write a
  working virtual Python driver for any of the supported types
  using the template script as a starting point.
- The per-type capabilities table includes every trampoline type
  that ships in the current build with the correct default class
  name (`AwgDriver`, `IOBoardDriver`, etc.).
- The security warning quoted in `dev-docs/python-hardware.md`
  appears verbatim or nearly so on the overview page.
- The page set never includes JSON message-format details or
  references to `PythonProcess::sendRequest()`.
