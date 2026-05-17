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

## Evaluation outcome (closed)

Per-widget verdicts after reading each candidate. A standing user
decision applies throughout: **any `QTableWidget` whose row names live
in the vertical header must move them into a regular column-0 cell**
(built like `SettingsTable`'s label cells: `QTableWidgetItem`,
`Qt::ItemIsEnabled`, `NoEditTriggers`, vertical header hidden), for
visual consistency with the ported settings tables — *whether or not*
the widget itself becomes a `SettingsTable`. This makes the old
"poor fit / leave entirely" bucket mostly a **restyle** bucket.

### A. Ported to `SettingsTable` — done

- `gui/expsetup/drscanconfigwidget.cpp` — `1554d518`.
- `gui/expsetup/experimenttypepage.cpp` Forever / Shot Settings /
  Duration Settings stack sub-tables — `ff3a1aa1`.

### B. Kept as `QTableWidget`, vertical-header → column-0 restyle — done

Two real value columns with meaningful headers ("Up LO/Down LO",
"Delay/Laser") — not a 2-column `SettingsTable` fit, restyled instead:

- `gui/expsetup/loscanconfigwidget.cpp` `p_loTable` — `d819cdd4`.
- `gui/expsetup/experimenttypepage.cpp` LIF `scanTable` — `dd944422`.

### C. Genuine `SettingsTable` candidates — NOT yet done

The original triage missed these; they are real label/value forms:

- `gui/widget/digitizerconfigwidget.cpp` is **mixed**: `dtTable`
  (`QTableWidget(4,1)`: Record Length / Sample Rate / …) and
  `trigTable` (`QTableWidget(4,1)`: Source / Slope / Delay / Level)
  are single-column label/value — exactly the `drscanconfigwidget`
  pattern, strong `SettingsTable` fits. Its analog/digital channel
  tables and `aTable` (2×2) are matrices → bucket D.
- `gui/lif/gui/lifprocessingwidget.cpp` — a `QGroupBox`-based
  processing form structurally analogous to the already-ported
  `FtmwProcessingPanel`. Real candidate, but a **substantial** port,
  not a quick swap: the "Gates" block is a 2×2 LIF/Reference ×
  Start/End matrix (bucket-D restyle, not label/value), the
  Savitzky-Golay checkable `QGroupBox` maps to a checkable section,
  and α / window / order map to setting rows. Effort comparable to
  the overlay type-panel port.

### D. Restyle-only — true matrices/data grids, NOT yet done

Stay `QTableWidget`; apply the vertical-header → column-0 restyle
only (the standing decision). Not `SettingsTable` ports:

- `gui/widget/peakfindwidget.cpp` `p_filterGrid` (2×2 Freq/Intensity
  × Min/Max). `p_peakListView` is a data list — leave entirely.
- `gui/widget/digitizerconfigwidget.cpp` analog/digital channel
  tables + `aTable` (the non-C parts above).
- `gui/widget/pulseconfigwidget.cpp` standard/advanced per-channel
  multi-column tables.
- `gui/widget/temperaturecontrolwidget.cpp` (`n×2`),
  `gui/widget/gascontrolwidget.cpp` (`n×3`) — deliberate UI-pass
  per-channel matrices; kept as matrices, vertical header restyled.

### E. Leave entirely — no change

- `gui/lif/gui/liflaserwidget.cpp` — small `QGridLayout` of live
  hardware-command buttons (set position, toggle flashlamp), not a
  persisted-settings form. Weak fit; no table.
- `gui/lif/gui/lifcontrolwidget.cpp` — a `QVBox/QHBox/QGroupBox`
  shell embedding child widgets; no table. (Its `<QFormLayout>`
  include is stale/unused — the doc's "QFormLayout restyle" premise
  was wrong; the real LIF candidates are its children, see C/D/E.)
- `gui/widget/librarystatuswidget.cpp`, `gui/dialog/aboutdialog.cpp`
  — read-only tabular data (what plain tables are *for*).
- `gui/dialog/hwarrayeditdialog.cpp` — dynamic N-column array editor.
- `gui/overlay/overlaymanagerwidget.cpp` — overlay-list data grid.

### Remaining work (decision point)

Buckets C and D are not yet executed and materially exceed the
original "assess + port the easy ones" scope. Sequence when resumed,
one widget per commit, user runtime-verifies each:

1. `digitizerconfigwidget` `dtTable` + `trigTable` → `SettingsTable`
   (drscan pattern); its other tables → D restyle.
2. D restyles: `peakfindwidget`, `pulseconfigwidget`,
   `temperaturecontrolwidget`, `gascontrolwidget`.
3. `lifprocessingwidget` → `SettingsTable` (largest; do last).

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
