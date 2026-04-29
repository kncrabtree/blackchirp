# Bundle 04 — Hardware Menu, Communication, Status Panel

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
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
