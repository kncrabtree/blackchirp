# HwStatusBox Updates

This project has 2 main purposes:

1. Make HwStatusBox display more compact.
2. Improve user experience by showing more data in tooltips and clickable targets to
   quickly open associated configuration dialogs.

## Goal 1: QSpinBox/QDoubleSpinBox -> QLabel

Numerical data is currently displayed in read-only QSpinBox and QDoubleSpinBox, which
takes up considerable space. Replace these with QLabels. Use std::to_chars with general
format to configure display, replacing the "eXX" suffix on scientific notation with
" x 10^XX" (see ScientificSpinBox::applySuperscript). Consider extracting some utility
functions from ScientificSpinBox into bcglobals.cpp or another appropriate file for
reuse. Preserve suffixes that are currently in QSpinBox/QDoubleSpinBox and append to
QLabel instead.

## Goal 2: UX Improvements

HwStatusBox (and other sections shown on the left panel of the MainWindow) are intended
to provide at-a-glance information about hardware status, program status, experiment
progress, etc. It is important to surface key information and hide inactive/redundant
information in order to keep the display compact, but it is also desirable for the user
to be able to access more detailed status data via well-considered tooltips. Then,
the user should be able to easily open the associated HwDialog (or other associated
dialog) directly from the status panel without having to navigate menus. Consider using
a "configure" icon from src/resources/icons as clickable target.

What is **not** in scope: direct control of hardware from a status box. Control is
consolidated in the Control widgets embedded in the HwDialog. 

## Notes on Implementation

Apply principles above if the class is not mentioned here. Most other classes should
be straightforward to implement.

### Experiment information display

- Experiment number QSpinBox -> QLabel
- Add (possibly truncated) display of save path; tooltip shows full path
- Add clickable target that opens ApplicationConfigManager
- Consider moving to place next to progress bars?

### ClockDisplayBox

- QDoubleSpinBox -> QLabel for each clock
- Hide rows for clocks that are not configured
- Tooltip shows physical clock and output number
- Clickable target for opening RfConfigDialog
- Clickable target per clock for opening HwDialog for that clock

### PulseStatusBox

- Do not hide inactive channels; Led shows enabled/disabled status
- Rep rate/triggered display already correct.
- Mouseover on Led or name label shows popup with syncCh, delay, width, active level. If
  channel is Duty Cycle mode (yellow Led), show on/off values. Format should be
  tabular.
- Clickable target for opening HwDialog

### FlowStatusBox

- QDoubleSpinBox -> QLabel.
- Hide inactive channels (Led off). Consider moving to 2 channels per column for flow
  channels, with pressure always on its own line. Pressure never hidden.
- Mouseover on channel name label, numeric label, or Led shows setpoint (including pressure)
- Clickable target for opening HwDialog

---

## Implementation Plan

Branch: `cmakemigration`. Build dir: `build/Desktop-Debug/`. Test dir: `build/tests/`.

### Architectural decisions (agreed)

- **Base widget:** `HardwareStatusBox` switches from `QGroupBox` to `QFrame` to gain
  corner widgets (configure cog) and a collapsible body. Title row layout:
  `[collapse-arrow] [Title label] [stretch] [configure-cog]`. Body widgets are added
  via a protected `body()` accessor that returns a content `QWidget` whose layout
  subclasses populate (instead of calling `setLayout` directly on the frame).
- **Configure-button signal:** `HardwareStatusBox` emits parameterless
  `configureRequested()`. `MainWindow` already has `key` in scope where it wires
  status boxes, so no need to plumb the key through the signal.
- **Numeric formatting:** extract `applySuperscript` and a general "format double for
  display" helper from `ScientificSpinBox` into `bcglobals.{h,cpp}` (or a new
  `gui/util/numericformat.{h,cpp}` if `bcglobals` is the wrong layer — implementer
  decides). `ScientificSpinBox::formatForDisplay` then delegates to the helper.
  Keep `removeSuperscript` private to `ScientificSpinBox`.
- **GasFlowDisplayBox:** "inactive" trigger remains `setpoint == 0`, matching current
  Led behavior. Tooltip captures the `val` already passed to `updateFlowSetpoint`.
- **ClockDisplayBox:** refactor to inherit `HardwareStatusBox` (with synthetic/empty
  key — base must tolerate this). Adds an extra `clockHardwareRequested(QString hwKey)`
  signal alongside the inherited `configureRequested` (which opens `RfConfigDialog`).
  `HardwareManager` gains a new `clockHardwareUpdate(ClockType, QString hwKey, int output)`
  signal feeding the box's hwKey/output map.
- **Menus stay intact:** `menuHardware` actions remain the entry path for hardware
  without status boxes; cog buttons supplement, do not replace.

### Step 1 — Foundation utilities (do first; blocking)

Single agent, then build + run `tst_scientificspinboxtest` via ctest.

- Add free functions to `bcglobals.{h,cpp}` (or new `gui/util/numericformat.{h,cpp}`):
  - `QString formatScientificSuperscript(const QString &)` — pulled from
    `ScientificSpinBox::applySuperscript`.
  - `QString formatNumberForDisplay(double value, int precision = -1, ScientificSpinBox::DisplayMode mode = Auto)`
    — wraps `std::to_chars` + superscript.
- Refactor `ScientificSpinBox::formatForDisplay` and `applySuperscript` to delegate to
  the new free functions. `removeSuperscript` stays inside `ScientificSpinBox`.
- Update CMakeLists if a new file is added.
- **Verification:** build target `blackchirp` succeeds; `ctest --test-dir build/tests
  -R scientificspinbox` passes (the test sets a required env var via ctest fixtures —
  do not run the binary directly).

### Step 2 — Base class refactor (blocking)

Single agent.

- Convert `HardwareStatusBox` from `QGroupBox` to `QFrame`.
- Title row: `QLabel` (title), optional collapse `QToolButton` (chevron icon),
  configure `QToolButton` (cog-6-tooth.svg) on the right.
- Add protected `QWidget *body()` accessor; subclasses populate `body()->setLayout(...)`.
- Tolerate empty key (for `ClockDisplayBox` reuse).
- Signal `void configureRequested()`.

### Step 3 — Per-status-box refactor (parallelizable; six agents)

Each subclass: spinbox→QLabel using Step-1 formatter, preserve suffix, switch from
`setLayout(this)` to populating `body()`, expose configure cog via base class.

- **Agent A — `PressureStatusBox`:** spinbox→QLabel only.
- **Agent B — `TemperatureStatusBox`:** spinbox→QLabel; preserve hide/show in
  `setChannelEnabled`.
- **Agent C — `GasFlowDisplayBox`:** spinbox→QLabel for channels and pressure; hide
  inactive channel rows (`setpoint == 0`); pressure row never hidden; re-layout to
  2 channels per column; tooltip with setpoint on name/value/Led (capture `val` in
  `updateFlowSetpoint`).
- **Agent D — `PulseStatusBox`:** no spinbox swap; add tabular tooltip on
  (label, Led) showing syncCh, delay, width, active level (and on/off for Duty Cycle
  mode); refresh tooltip in `updateAll` and `updatePulseSetting`.
- **Agent E — `LifLaserStatusBox`:** spinbox→QLabel.
- **Agent F — `ClockDisplayBox`:** inherit `HardwareStatusBox`; spinbox→QLabel; hide
  unconfigured rows; tooltip = `"{hwKey} output {n}"`; new slot
  `setClockHardware(ClockType, QString hwKey, int output)`; new signal
  `clockHardwareRequested(QString hwKey)`; per-row cog button.

### Step 4 — Experiment information panel

Single agent.

- Replace `ui->exptSpinBox` (mainwindow_ui.h:251) with `QLabel`. Update setters at
  mainwindow.cpp:100, 226.
- Add elided save-path `QLabel` with full-path tooltip; refresh on
  `appConfigAction` accept (mainwindow.cpp:97).
- Add small cog `QToolButton` opening `ApplicationConfigDialog` (alongside save path).
- Evaluate moving block next to progress bars; report findings if relocation chosen.

### Step 5 — MainWindow signal wiring

Single agent (after Steps 2 + 3).

- Connect `HardwareStatusBox::configureRequested → act->trigger()` for each status box
  in `buildHardwareUI` (key in lambda scope).
- Add `HardwareManager::clockHardwareUpdate(ClockType, QString, int)` signal; emit
  during clock configuration; connect to `ClockDisplayBox::setClockHardware`.
- Connect `ClockDisplayBox::configureRequested → MainWindow::launchRfConfigDialog`.
- Connect `ClockDisplayBox::clockHardwareRequested(QString) → trigger matching menu
  action lookup` (or directly call `createHWDialog`).

### Step 6 — Manual UI verification

Build + launch the app; confirm tooltips, cog buttons, hide/show, layout.
Run full test suite (`ctest --test-dir build/tests`).

### Delegation order

1. Step 1 (single agent, blocking) → build + ctest scientificspinbox.
2. Step 2 (single agent, blocking) → build.
3. Step 3 Agents A–F (six agents in parallel) → build + ctest.
4. Steps 4 + 5 (two agents in parallel) → build + ctest.
5. Step 6 (manual).

### Progress tracking

- [x] Step 1 — formatting utility extracted (`src/gui/util/numericformat.{h,cpp}`, namespace `BC::Gui`), scientificspinbox test passes
- [ ] Step 2 — `HardwareStatusBox` converted to `QFrame` with title row
- [ ] Step 3A — PressureStatusBox
- [ ] Step 3B — TemperatureStatusBox
- [ ] Step 3C — GasFlowDisplayBox
- [ ] Step 3D — PulseStatusBox
- [ ] Step 3E — LifLaserStatusBox
- [ ] Step 3F — ClockDisplayBox
- [ ] Step 4 — Experiment information panel
- [ ] Step 5 — MainWindow wiring + HardwareManager clockHardwareUpdate signal
- [ ] Step 6 — Manual UI verification + full ctest
