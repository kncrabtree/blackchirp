# SettingsTable backport — evaluation handoff

Ephemeral scratchpad for a fresh-context session. Not load-bearing
documentation; purged before release. Branch: `settingstable-backport`
(off `master` after the `ftmw-view-dock-refactor` merge).

## Goal

`SettingsTable` (`gui/widget/settingstable.{h,cpp}`) is now the
canonical content-sized, scrollbarless, section-aware **two-column
label/value** settings grid. The overlay creation dialog and the three
FTMW side panels were ported to it and the `tablefit.h` stopgap was
deleted.

This task is to **evaluate, per widget, whether the remaining
hand-rolled settings tables should also move onto `SettingsTable`**,
then port the ones that fit. It is an assessment first, not a blanket
port — several candidates are true matrices or data grids that the
2-column model does not serve.

## Read these first (the exemplars)

The just-completed ports are the reference implementations. Skim them
before touching anything:

- `gui/widget/ftmwacquisitionpanel.cpp` — minimal port (2 rows, a
  viewer-hidden row via the index `addSettingRow` returns).
- `gui/widget/ftmwprocessingpanel.cpp` — straight static port, label
  tooltips on both the value widget and the label cell.
- `gui/widget/ftmwplotpanel.{h,cpp}` — fixed + dynamically built
  rows; `static constexpr` row indices replaced by stored indices
  from `addSettingRow`.
- `gui/overlay/catalogoverlaywidget.cpp`,
  `gui/overlay/genericxyoverlaywidget.cpp`,
  `gui/overlay/overlaytypespecificwidget.{h,cpp}` — section tiers,
  the checkable-tier pattern, and the "no consecutive section bands"
  rule.

Git: the work is the run of commits ending at `435bf3ae` on `master`
(`Port Ftmw*Panel onto SettingsTable`, `Rebuild the overlay type
panel on a single SettingsTable`, `Generalize SettingsTable section
rows`, etc.).

## SettingsTable API cheat-sheet

From `gui/widget/settingstable.h`:

- `int addSettingRow(label, value, tooltip={})` — label in col 0,
  one widget in col 1 (stretched). Returns the row index.
- `int addSettingRow(label, first, second, tooltip={})` — two widgets
  side by side in the value cell (e.g. box + button, min + max).
- `int addSectionRow(title)` — bold, centered, theme-shaded spanning
  heading row.
- `int addCheckableSectionRow(title, checked, QCheckBox** =nullptr)`
  — spanning heading with a leading checkbox; bound rows collapse
  when unchecked.
- `bindSectionRows(sectionRow, QList<int>)`,
  `setSectionTitle`, `setSectionCheckable`, `setBoundRowsEnabled`,
  `setSectionVisible`, `applySectionVisibility`,
  `sectionCheckBox`.
- `minimumSizeHint()` floors the vertical minimum at the content
  height (the table never scrolls, so it must never be squeezed
  shorter than its rows). Harmless for dialog-embedded tables;
  required for ones in a dock.

Construction is borderless / headerless / non-selectable / read-only
/ `AdjustToContents` / no vertical scrollbar — the widget owns all of
that, callers just add rows.

## Precedents settled during the FTMW/overlay work

Follow these so the ported widgets stay consistent with what shipped:

- Replace `setVerticalHeaderLabels({...})` + `setCellWidget(r,0,w)`
  with one `addSettingRow("Label", w)` per row. Drop the
  `horizontalHeader/verticalHeader/ScrollBar/SelectionMode`
  boilerplate — `SettingsTable` sets it.
- Checkboxes: **do not** use `centerCellWidget`; a plain
  `addSettingRow("Label", checkBox)` (checkbox sits left in the value
  column) is the established look across the app now.
- Never stack a generic section heading directly on top of a more
  specific one ("consecutive section bands"). Either retitle the
  enclosing tier (`setSectionTitle`) or, for a single gated group,
  make the tier itself the checkable section
  (`setSectionCheckable` + the
  `typeSpecificSection{Checkable,Title,InitiallyChecked}` virtual-hook
  pattern in `overlaytypespecificwidget`).
- Row heights are computed once when the row is added (not live
  font-tracked). Accepted tradeoff; do not add machinery for it.
- The two-widget value-cell variant covers "label + spinbox +
  button" and "min … max" pairs without a real second column.
- A new or moved widget file must be registered in **both**
  `cmake/BlackchirpGui.cmake` and `cmake/BlackchirpViewerGui.cmake`.
- `tablefit.h` is gone — do not reintroduce a manual height cap.

## Candidate survey (current state, triaged)

Grep used: `new QTableWidget` / `setVerticalHeaderLabels` /
`setCellWidget` under `gui/`. Triage is a starting hypothesis —
confirm by reading each before porting.

**Strong fit — single value column, label/value rows (port):**

- `gui/expsetup/drscanconfigwidget.cpp` — `QTableWidget(5,1)`,
  `setVerticalHeaderLabels` + `setCellWidget(r,0,...)`. Exactly the
  FTMW-panel pattern. Best first port / proof.
- `gui/expsetup/experimenttypepage.cpp` — several small
  `QTableWidget(rows,1)` sub-tables (`forever`, `shots`, `duration`,
  …), same pattern. Port each sub-table.

**Borderline — needs a judgment call:**

- `gui/expsetup/loscanconfigwidget.cpp` — `QTableWidget(6,2)`, two
  value columns "Up LO" / "Down LO". Could map onto the
  `addSettingRow(label, upBox, downBox)` two-widget variant, but the
  column headers are lost; assess whether that reads acceptably or
  whether it should stay a real matrix.
- `gui/widget/temperaturecontrolwidget.cpp` (`n×2`, Name/Enabled),
  `gui/widget/gascontrolwidget.cpp` (`n×3`, Name/Setpoint/Enabled)
  — per-channel matrices. They were *deliberately* moved to
  `QTableWidget` in the recent UI pass (see
  `dev-docs/user-guide-cleanup.md` screenshot notes). Likely **leave
  as-is**; only revisit if a row genuinely reduces to label/value.

**Poor fit — true matrices or data grids, leave alone:**

- `gui/widget/digitizerconfigwidget.cpp` — multi-column channel
  tables (analog/digital × params). Matrix; not label/value.
- `gui/widget/pulseconfigwidget.cpp` — standard/advanced per-channel
  multi-column tables. Matrix.
- `gui/widget/peakfindwidget.cpp` — `p_filterGrid` is a 2×2
  Freq/Intensity × Min/Max matrix (not label/value); `p_peakListView`
  is a data list. Neither is a SettingsTable candidate.
- `gui/widget/librarystatuswidget.cpp`, `gui/dialog/aboutdialog.cpp`
  — read-only tabular data (exactly what plain tables are *for* per
  `doc/AGENTS.md`). Not candidates.
- `gui/dialog/hwarrayeditdialog.cpp` — dynamic N-column array editor.
  Not label/value.
- `gui/overlay/overlaymanagerwidget.cpp` — the overlay list table is
  a data grid, not a settings form. Not a candidate.

**Reclassified from the original memory note:**

- "ExperimentSetupPage" → the concrete targets are the
  `gui/expsetup/*.cpp` tables above (there is no single
  `ExperimentSetupPage` file).
- `gui/lif/gui/lifcontrolwidget.cpp` — uses `QFormLayout` +
  `QGroupBox`, **not** a raw table. Converting it would be a layout
  *restyle* (form → SettingsTable), a different and larger decision
  than a like-for-like table swap. Treat separately / probably out
  of scope for this pass.

## Suggested order

1. `drscanconfigwidget` — smallest, cleanest proof; validates the
   pattern end to end (build + runtime).
2. `experimenttypepage` sub-tables — same pattern, slightly more rows.
3. Decide `loscanconfigwidget` (two-widget variant vs. keep matrix);
   implement only if the two-widget rendering reads well.
4. Write up the verdict for the "poor fit / leave" set so the
   evaluation is closed, not silently dropped.

One widget per commit; user runtime-verifies each before the next
(the discipline used throughout the FTMW/overlay work — live-UI
regression stakes).

## Build / verify

```bash
cmake --build /home/kncrabtree/github/blackchirp/src/build/Desktop-Debug/ \
  --target blackchirp blackchirp-viewer -j$(nproc)
```

~2:30–3:00; allow 300000 ms. After adding/moving a file in a
`*.cmake` list, run `cmake . -B build/Desktop-Debug/` once first so
the source list regenerates.

## Branch state at handoff

`settingstable-backport` branched from `master` @ `435bf3ae`
(post-merge, clean). This doc is the only change. The
`dev-docs/user-guide-cleanup.md` remaining-pages checklist is the
*other* independent post-merge workstream — keep the two separate.
