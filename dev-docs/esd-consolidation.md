# ESD Scan Page Consolidation

Merge `ExperimentLOScanConfigPage` and `ExperimentDRScanConfigPage` into
`ExperimentTypePage` (the first ESD page), removing two navigation nodes.

## Phase 1 — Extract scan widgets ✓

- Create `LOScanConfigWidget` (`src/gui/expsetup/loscanconfigwidget.{h,cpp}`):
  move all controls and logic from `ExperimentLOScanConfigPage`, inheriting
  `QWidget` + `SettingsStorage`. Retain `initialize()`, `validate()`, `apply()`
  as public methods (not overrides).
- Create `DRScanConfigWidget` (`src/gui/expsetup/drscanconfigwidget.{h,cpp}`)
  the same way from `ExperimentDRScanConfigPage`.
- Keep the old page classes alive and delegating to the new widgets so nothing
  breaks yet.

## Phase 2 — Add `FtmwViewWidget::clockHwChanged` signal

- Add `clockHwChanged()` signal to `FtmwViewWidget`
  (`src/gui/widget/ftmwviewwidget.{h,cpp}`).
- Emit it whenever the clock hardware selection changes inside that widget.
- Wire the signal into `ExperimentSetupDialog` (connect to a slot that will
  later trigger constraint reapplication and validation).

## Phase 3 — Integrate into `ExperimentTypePage`

- Add a `QStackedWidget` to `ExperimentTypePage` between the type-selector
  combo box and the phase-correction checkbox.
- Pages of the stacked widget:
  - Simple FTMW modes (Target Shots, Target Time, Forever, Peak Up): inline
    controls managed directly in `ExperimentTypePage` (shots spinbox, duration
    spinbox + themed Est. completion label, or empty widget).
  - LO Scan: `LOScanConfigWidget`.
  - DR Scan: `DRScanConfigWidget`.
- Switch the active stack page when `p_ftmwTypeBox` selection changes.
- Move scan `initialize()`, `validate()`, `apply()` calls into
  `ExperimentTypePage`'s corresponding overrides.
- Apply clock-range constraints on construction and on `clockHwChanged`; remove
  any `pageChanged`-based constraint logic for scan pages.
- Revise layouts as needed so the page is taller than it is wide.
- Update Est. completion label to use theme colors.

## Phase 4 — Remove old pages

- Delete `ExperimentLOScanConfigPage` and `ExperimentDRScanConfigPage` source
  files and remove their entries from `CMakeLists.txt`.
- Remove their tree nodes from `ExperimentSetupDialog` constructor; remove any
  `pageChanged` handling that routed to those nodes.
- Verify no remaining references in other source files.

## Phase 5 — Cleanup and testing

- Remove the temporary delegation wrappers from Phase 1 if any remain.
- Rename keys in `BC::Key::WizLoScan` / `BC::Key::WizDR` namespaces if the new
  widget uses a different storage key (or keep them for settings compatibility).
- Build and run; exercise all FTMW modes through the ESD and confirm
  validation catches out-of-range scan parameters correctly.
