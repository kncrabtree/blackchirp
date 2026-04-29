# Bundle 08 — Experiment Workflow Refresh

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Refreshes the experiment-wizard pages to match the consolidated
LO/DR scan widgets, updated optional-hardware initialization,
and the current quick-experiment / batch-sequence flows.

## Scope

- Refresh `doc/source/user_guide/experiment_setup.rst` (parent page).
  The intro stays substantially the same; update the "key points"
  list to reference the FTMW Configuration page (bundle 07) and
  the quick-experiment hardware-compatibility check.
- Refresh `doc/source/user_guide/experiment/acquisition_types.rst`:
  - LO Scan and DR Scan setup were extracted into
    `LOScanConfigWidget` / `DRScanConfigWidget` and folded into the
    Type page. Update prose to reflect that the LO/DR configuration
    UI now appears on the same wizard page as the type selection.
  - Verify the screenshots and text still match the current widget
    layout.
- Refresh `doc/source/user_guide/experiment/quick_experiment.rst`:
  - Hardware-configuration compatibility check uses the new
    runtime profile system; quick experiment requires the same
    hardware map (loadout) as the original experiment, not just
    the same compile-time hardware list.
  - Mention that quick experiment can copy overlays from the
    source experiment (commit `b682122d`).
- Refresh `doc/source/user_guide/experiment/sequence_mode.rst`:
  - Mention the abort-during-sequence fix in passing (no longer
    a known bug; just confirm wording is accurate).
- Refresh `doc/source/user_guide/experiment/validation.rst`:
  - Light refresh; confirm the object-key/value-key list reflects
    the label-based hardware identification.
- Add a new page at
  `doc/source/user_guide/experiment/optional_hardware.rst`
  describing the experiment-wizard pages for optional hardware
  (FlowController, PressureController, TemperatureController, IO
  Board, PulseGenerator), how each is initialized to live settings
  by default, and how to override them.

## Out of scope

- The chirp setup page (bundle 07).
- The digitizer setup page (bundle 07).

## Sources

- Source: `src/gui/expsetup/experimenttypepage.{h,cpp}`,
  `src/gui/expsetup/loscanconfigwidget.{h,cpp}`,
  `src/gui/expsetup/drscanconfigwidget.{h,cpp}` — for the
  consolidated Type page layout.
- Source: `src/gui/expsetup/experiment*configpage.{h,cpp}` (one
  per optional hardware type) — for the optional-hardware page.
- Source: `src/gui/dialog/quickexptdialog.{h,cpp}` — for the
  quick-experiment compatibility check.
- Source: `src/gui/dialog/batchsequencedialog.{h,cpp}` — for
  sequence mode behaviour.
- Source: `src/gui/expsetup/experimentvalidatorconfigpage.{h,cpp}`
  — for the validation page.

## Sphinx file deltas

**Modified:**
- `doc/source/user_guide/experiment_setup.rst`
- `doc/source/user_guide/experiment/acquisition_types.rst`
- `doc/source/user_guide/experiment/quick_experiment.rst`
- `doc/source/user_guide/experiment/sequence_mode.rst`
- `doc/source/user_guide/experiment/validation.rst`

**Created:**
- `doc/source/user_guide/experiment/optional_hardware.rst`

## Toctree delta

The `experiment/` toctree in `experiment_setup.rst` uses a
`:glob:` directive, so the new `optional_hardware.rst` is picked
up automatically.

## Screenshots

- `_static/user_guide/experiment/startpage.png` — refresh; the
  Type page now includes LO/DR scan settings inline.
- `_static/user_guide/experiment/loscan.png` — refresh.
- `_static/user_guide/experiment/drscan.png` — refresh.
- `_static/user_guide/experiment/optional_hardware_pgen.png` —
  example optional-hardware page.
- `_static/user_guide/experiment/quickexpt_1.png` — refresh if
  the dialog has changed.

## Acceptance criteria

- `acquisition_types.rst` accurately describes the consolidated
  Type page (no implication that LO/DR scan settings are on a
  separate later page).
- `quick_experiment.rst` describes the loadout-based hardware
  compatibility check.
- `optional_hardware.rst` exists and walks through every
  optional-hardware experiment-wizard page that ships in the
  current build.
- The validation page accurately reflects the label-based
  object-key dropdown contents.
