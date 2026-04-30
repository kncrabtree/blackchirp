# Bundle 04 — Hardware Menu, Communication, Status Panel

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-04-29: drafted → complete. Landed as commit dc921003 ("Refresh
  Hardware menu and document Hardware Dialog and status panel"). All
  four screenshots captured by the user prior to commit; HWDialog
  setWindowIcon fix included in the same commit.
- 2026-04-29: drafted (screenshot validation pass). User captured the
  remaining four screenshots: hardware_menu/menu.png,
  hardware_menu/communication.png, hwdialog/settings_tab.png,
  hwdialog/control_tab.png. Validating images against drafted prose
  surfaced four corrections, all applied: (a) hardware_menu.rst index
  block carried a stale "Application Configuration; Hardware menu"
  entry — removed (the action is in the Settings menu and is no longer
  referenced from this page); (b) the Communication and Test All
  Connections menu entries display Ctrl+H and Ctrl+T shortcuts, which
  the prose now surfaces; (c) the Settings-tab section in hwdialog.rst
  described Required/Important/Advanced as inline groups with Advanced
  being a collapsible section. The actual UI (hwsettingswidget.cpp:52,
  56, 80, 229, 235) is a nested QTabWidget with a "Settings" sub-tab
  containing Required + Important QGroupBoxes and a separate
  "Advanced" sub-tab that is added only when advanced settings exist.
  Rewrote the Settings-tab section to match, and added a sentence
  noting which sub-tab the screenshot shows. (d) Pulse Generator
  Control-tab entry under-described the actual UI; expanded to cover
  the System Settings group, per-channel configuration table (Sync,
  Mode, Cfg cog), and the timing-diagram preview pane. Replaced the
  TODO :alt:/caption strings on all four .. figure:: directives with
  descriptive alt text and informative captions. "Ok" → "OK" in two
  places to match the actual button label. Side fix outside the
  doc tree: hwdialog.cpp did not call setWindowIcon, so the dialog
  rendered without the Blackchirp logo unlike its siblings — added
  ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ...)
  alongside the existing setAttribute calls. Verified by rebuilding
  the blackchirp target (clean compile).
- 2026-04-29: drafted (screenshot pass — partial). User refreshed
  ui.png and decided it subsumes the status_panel screenshot
  requirement; help_menu and about_dialog screenshots deemed
  unnecessary. Removed three .. figure:: blocks from
  ui_overview.rst: status_panel.png (Instrument Status section now
  refers the reader to the left panel of the page-top ui.png),
  help_menu.png (Help Menu section), and about_dialog.png (About
  Blackchirp section). User also opportunistically softened the
  status-box variant prose to drop class-name parentheticals; the
  page-top index entries were trimmed to match (Clock Display Box,
  Gas Flow Display Box, etc., as user-facing terms rather than C++
  class names). Remaining screenshots awaiting capture:
  hardware_menu/menu.png, hardware_menu/communication.png,
  hwdialog/settings_tab.png, hwdialog/control_tab.png.
- 2026-04-29: in progress → drafted. Drafter delivered hardware_menu.rst
  rewrite, ui_overview.rst Status/Help refresh, and new hwdialog.rst;
  toctree updated. Verifier flagged one load-bearing factual error: the
  ClockDisplayBox title-bar configure button opens the RF Configuration
  dialog (clockdisplaybox.cpp:58 sets tooltip "Open Rf Configuration
  Dialog"), not a per-device HwDialog; the per-row cog icons emit
  clockHardwareRequested and open the individual clock hardware dialog
  (clockdisplaybox.cpp:91). Fixed via Edit: distinguished the two
  configure controls in the ClockDisplayBox entry and softened the
  general status-box description in ui_overview.rst to note variant
  overrides. Two scope drift items confirmed against source and noted
  for the master plan: (a) appConfigAction lives in settingsMenu, not
  menuHardware (mainwindow_ui.h:462) — the bundle's Scope listing of
  "Application Configuration" as a Hardware-menu entry was wrong; the
  drafter correctly omitted it. (b) the Hardware menu entry that opens
  RuntimeHardwareConfigDialog is labelled "Hardware Selection" in the
  current UI; bundle text said "Hardware Configuration". Drafted prose
  uses the actual UI label. All seven screenshot TODO markers in place
  with the bundle's specified filenames. Awaiting user review.
- 2026-04-29: not started → in progress. Drafter dispatched in worktree
  (Sonnet, isolation: "worktree"). Scope verified against current source:
  rebuildLoadoutMenu, rebuildFtmwPresetMenu, CommunicationDialog, HWDialog,
  HwSettingsWidget, HardwareStatusBox, and AboutDialog all present in the
  indexed graph; cited dev-docs (settings-registry.md, loadout-system.md)
  exist.
-->

Rewrites the Hardware menu page and the day-to-day instrument-control
UI surfaces (Communication dialog, HwDialog Settings/Control tabs,
status-box family).

## Scope

- Rewrite `doc/source/user_guide/hardware_menu.rst` to reflect the
  current Hardware menu structure: Communication, Hardware
  Configuration (opens `RuntimeHardwareConfigDialog`), Application
  Configuration, Loadouts submenu, FTMW Configuration entry, FTMW
  Preset submenu, per-device entries (each opening `HwDialog`).
- Rewrite the Communication subsection to describe runtime protocol
  selection: each device exposes a per-implementation set of
  available protocols (RS232, TCP, GPIB, Custom, Virtual); the
  Communication dialog now uses protocol-specific widgets backed by
  group-based settings storage. Keep the existing FTDI/udev tips
  but mark them as Linux-specific in a clearly delimited subsection.
- Add `doc/source/user_guide/hwdialog.rst` describing what a user
  sees when they open Hardware → \[Device\]: a tabbed dialog with a
  Settings tab (the `HwSettingsWidget` in Edit mode) and, when
  applicable, a Control tab (e.g., the `PulseGenerator` channel
  table, `FlowController` gas controls). Document the Required-vs-
  Important-vs-Advanced sections, that Required settings are
  read-only after profile creation, and the Apply/Cancel semantics.
- Rewrite the Instrument Status section in
  `doc/source/user_guide/ui_overview.rst` to cover the new
  collapsible `HwStatusBox` family (now a `QFrame` with collapsible
  body), the per-status-box configure action that opens the
  associated `HwDialog`, and the experiment info panel.
- Add a Help menu subsection to `ui_overview.rst` covering the new
  Help menu entries (About dialog, online resources, library info).

## Out of scope

- Per-device settings reference (bundle 05).
- RF Configuration dialog (moved to bundle 07 because it is now
  reached via Hardware → FTMW Configuration, not Hardware → Rf
  Configuration directly, and ties into the preset bar).

## Sources

- `dev-docs/settings-registry.md` — Required/Important/Optional UI
  semantics.
- `dev-docs/loadout-system.md` — Hardware menu structure.
- Source: `src/gui/mainwindow.cpp` — `rebuildLoadoutMenu`,
  `rebuildFtmwPresetMenu`, Help menu wiring.
- Source: `src/gui/dialog/communicationdialog.{h,cpp}` — runtime
  protocol selection, protocol-specific widget wiring.
- Source: `src/gui/dialog/hwdialog.{h,cpp}`,
  `src/gui/widget/hwsettingswidget.{h,cpp}`,
  `src/gui/dialog/hwarrayeditdialog.{h,cpp}` — Settings/Control tab
  composition.
- Source: `src/gui/widget/hardwarestatusbox.{h,cpp}` and the
  HwStatusBox subclasses (`gasflowdisplaybox`,
  `temperaturestatusbox`, `pulsestatusbox`, etc.).
- Source: `src/gui/dialog/aboutdialog.{h,cpp}` and the Help menu
  wiring (commit `dbac45bd`).

## Sphinx file deltas

**Modified:**
- `doc/source/user_guide/hardware_menu.rst` — full rewrite.
- `doc/source/user_guide/ui_overview.rst` — Instrument Status
  section + new Help menu section.

**Created:**
- `doc/source/user_guide/hwdialog.rst`

## Toctree delta in `user_guide.rst`

Insert `user_guide/hwdialog` after `user_guide/hardware_menu`.

## Screenshots

- `_static/user_guide/hardware_menu/menu.png` — refresh; new menu
  contents include Loadouts, FTMW Configuration, FTMW Preset,
  Application Configuration, Help.
- `_static/user_guide/hardware_menu/communication.png` — refresh
  with the protocol-widget UI.
- `_static/user_guide/hwdialog/settings_tab.png` — HwDialog
  Settings tab showing Required (read-only) / Important / Advanced
  sections.
- `_static/user_guide/hwdialog/control_tab.png` — HwDialog Control
  tab using `PulseGenerator` as the example.
- `_static/user_guide/ui_overview/status_panel.png` — refresh of
  the status panel with collapsible status boxes.
- `_static/user_guide/ui_overview/help_menu.png` — Help menu open.
- `_static/user_guide/ui_overview/about_dialog.png` — About dialog.

## Acceptance criteria

- The Hardware menu page accurately lists every menu entry that
  appears in the current build, in the current order.
- The Communication subsection no longer implies a fixed
  per-device protocol; runtime protocol selection is the documented
  default.
- `hwdialog.rst` clearly distinguishes Settings (registry-backed,
  persistent) from Control (live device commands), and explains why
  closing the dialog with the X button discards Settings changes
  but Control changes have already been sent.
- The Instrument Status section enumerates every status-box
  variant that ships in the current build, with one sentence per
  variant on what readings it surfaces.
- The Help menu section covers About, Library Info, and the online
  resource links from commit `dbac45bd`.
