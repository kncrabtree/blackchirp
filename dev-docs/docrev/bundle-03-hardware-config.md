# Bundle 03 — Hardware Configuration: Profiles, Loadouts, FTMW Presets

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-04-29: drafted → complete. Landed as commit c67f84f9
  ("Add hardware configuration chapter: profiles, loadouts, FTMW
  presets"). All six screenshots captured by the user prior to
  commit. Bundle 02's first_run.rst leftmost-panel label updated to
  "Loadout" (singular) for consistency with bundle 03 prose and the
  actual UI.
- 2026-04-29: drafted (screenshot + label pass). User captured all six
  screenshots into doc/source/_static/user_guide/hardware_config/.
  Inspecting the screenshots surfaced label discrepancies between the
  drafted prose and the actual UI. Verified labels in source: the
  Hardware menu item that opens the runtime dialog is **Hardware
  Selection** (mainwindow_ui.h:181, not "Hardware Configuration"); the
  Hardware submenu is **Loadout** singular (mainwindow_ui.h:448, not
  "Loadouts"); the dialog's accept button is **Apply Configuration**
  (runtimehardwareconfigdialog_ui.h:248, not "Save"). Reading
  onDialogAccepted (runtimehardwareconfigdialog.cpp:1102-1120) showed
  a four-button unsaved-changes prompt that was not previously
  documented. Edits: replaced all six "Screenshot:" TODO markers with
  .. figure:: directives + captions; corrected the three menu/panel
  labels in hardware_config.rst, profiles.rst, and loadouts.rst;
  rewrote loadouts.rst Preview-State section to describe Apply
  Configuration / Cancel as the dialog accept and to document the
  four-button unsaved-changes prompt (Save and apply / Apply without
  saving / Save to new loadout... / Cancel). Bundle 02's first_run.rst
  uses "Loadouts" plural for the panel; left as-is since it was
  committed under that wording — flag for a follow-up consistency pass
  if desired.
- 2026-04-29: in progress → drafted. Verifier graded A/D PASS, B
  PARTIAL (AWG sample-rate wording slightly imprecise but acceptable),
  C FAIL (lead-in said "three choices" but listed four; corrected).
  Drafter verified the four source-of-truth claims (active preset
  cannot be deleted; __LastUsed__ hidden from dropdowns; AWG sample
  rate not stored in presets; preset cannot exist outside a loadout)
  against codebase-memory and noted the loadout/preset menu gating
  is Idle-only (not Idle/Disconnected as one dev-doc suggested) —
  prose matches the source. Orchestrator made four inline fixes:
  (1) ftmw_presets.rst lead-in "three choices" → "the following
  options"; (2) removed duplicate preset_bar.png screenshot TODO at
  the top of ftmw_presets.rst, retaining the one inside the Preset
  Bar section; (3) loadouts.rst preview-state button label
  "Apply / Save" → "Save" (verified actual QPushButton label in
  runtimehardwareconfigdialog_ui.h:199); (4) loadouts.rst Save As
  description now mentions the user-prompted copy of FTMW presets
  when the FTMW-relevant hardware is unchanged (verified in
  onLoadoutSaveAs at runtimehardwareconfigdialog.cpp:1454-1507).
  Bundle 02's first_run.rst forward-reference TODO was resolved.
  Six screenshot TODOs (one per bundle-spec filename) remain for a
  later screenshot pass. Awaiting user review and commit.
- 2026-04-29: not started → in progress. Drafter dispatched (Sonnet,
  isolated worktree). Bundle file scope verified against current
  source: all six cited headers and both dev-docs exist at the paths
  the bundle gives. Bundle 02 (commit 3d5da2cb) corrected the runtime
  config dialog to a four-panel layout; bundle 03 prose should match
  that vocabulary.
-->

Introduces the runtime hardware configuration system: profiles
(label-based hardware identification), loadouts (named hardware maps),
and FTMW presets (named operating points within a loadout). This is
the largest conceptual shift from v1.x and is referenced by nearly
every later chapter.

## Scope

- Add a new chapter at `doc/source/user_guide/hardware_config.rst`
  that owns the toctree for this subsystem.
- Sub-pages under `doc/source/user_guide/hardware_config/`:
  - `profiles.rst` — what a hardware profile is, the
    Type/Label/Implementation triple, system vs. user profiles,
    profile creation via `RuntimeHardwareConfigDialog`'s "Add
    Profile" flow, the `HwSettingsWidget` priority sections
    (Required / Important / Optional / Advanced) the user sees at
    profile creation time, enable/disable, deletion.
  - `loadouts.rst` — what a loadout is, the Default loadout, how to
    create/rename/duplicate/delete loadouts, switching loadouts via
    the Hardware → Loadouts submenu, the drift-detection prompt that
    appears when switching loadouts invalidates an FTMW preset.
  - `ftmw_presets.rst` — what an FTMW preset captures (Rf chain,
    clock frequencies, chirp waveform, marker definitions, digitizer
    configuration), the per-loadout `__LastUsed__` sentinel, the
    `currentFtmwPreset` notion, why the active preset cannot be
    deleted, switching presets via Hardware → FTMW Preset submenu
    and via the preset bar in `FtmwConfigDialog`.
- Cross-link from each sub-page back to the chapter overview and to
  the Hardware menu page (bundle 04).
- Preview-state semantics (Apply, Cancel, drift-detection prompt
  outcomes Discard / Save As / Cancel) documented prominently — these
  are the parts users get wrong without explanation.

## Out of scope

- Communication settings (bundle 04).
- The HwDialog Settings/Control tabs that appear *after* a profile is
  created (bundle 04).
- The chirp-relative marker table within an FTMW preset (bundle 07).

## Sources

- `dev-docs/loadout-system.md` — primary source; extract user-facing
  content.
- `dev-docs/settings-registry.md` — for the priority levels (Required,
  Important, Optional) the user encounters in `HwSettingsWidget`.
- Source: `src/data/loadout/hardwareloadout.{h,cpp}`,
  `src/data/loadout/loadoutmanager.{h,cpp}` — confirm the user-visible
  vocabulary.
- Source: `src/gui/dialog/runtimehardwareconfigdialog.{h,cpp}` and
  `src/gui/dialog/addprofiledialog.{h,cpp}` — UI layout the prose
  describes.
- Source: `src/gui/widget/hwsettingswidget.{h,cpp}` — for the
  Required/Important/Advanced layout.
- Source: `src/gui/dialog/ftmwconfigdialog.{h,cpp}` and
  `src/gui/widget/ftmwconfigwidget.{h,cpp}` — preset-bar behaviour.

## Sphinx file deltas

**Created:**
- `doc/source/user_guide/hardware_config.rst`
- `doc/source/user_guide/hardware_config/profiles.rst`
- `doc/source/user_guide/hardware_config/loadouts.rst`
- `doc/source/user_guide/hardware_config/ftmw_presets.rst`

## Toctree delta

In `user_guide.rst` (after `library_status`):

```
   user_guide/hardware_config
```

In `hardware_config.rst` (new):

```rst
.. toctree::
   :hidden:

   hardware_config/profiles
   hardware_config/loadouts
   hardware_config/ftmw_presets
```

## Screenshots

- `_static/user_guide/hardware_config/runtimedialog.png` — full
  Runtime Hardware Configuration dialog with left/middle/right panels
  visible.
- `_static/user_guide/hardware_config/addprofile.png` — Add Profile
  flow showing the Required/Important/Advanced layout.
- `_static/user_guide/hardware_config/loadouts_menu.png` — Hardware →
  Loadouts submenu.
- `_static/user_guide/hardware_config/ftmw_presets_menu.png` —
  Hardware → FTMW Preset submenu.
- `_static/user_guide/hardware_config/preset_bar.png` — preset bar
  inside `FtmwConfigDialog`.
- `_static/user_guide/hardware_config/drift_prompt.png` — drift-
  detection prompt with Discard / Save As / Cancel options.

## Acceptance criteria

- A user who has only ever used v1.x's `config.pri` flow can read
  this chapter and understand why hardware is now selected at
  runtime, what a loadout is, and what an FTMW preset is.
- The chapter explicitly states the constraints: an FTMW preset
  cannot exist outside a loadout, the active preset cannot be
  deleted, AWG sample rate is not stored in presets,
  `__LastUsed__` is hidden from dropdowns and never user-deletable.
- The drift-detection user flow (Discard / Save As / Cancel) is
  documented with one paragraph per outcome.
- Cross-references to bundle 04 (Hardware menu) and bundle 07
  (FTMW configuration / preset bar) are in place.
