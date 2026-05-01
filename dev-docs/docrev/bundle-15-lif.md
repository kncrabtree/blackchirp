# Bundle 15 — LIF Module User Guide

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-05-01: in progress → complete. Content commit 6f470752. After
  initial verifier acceptance, user manual-verification pass corrected
  factual errors on every sub-page (wizard framing, end-box read-only,
  flashlamp purpose, dialog layout, slice/2D linkage via crosshairs and
  context menu, Save semantics, accumulated-sums data convention,
  header.csv LifConfig/LifScope sections). Three screenshots already on
  disk; no additional screenshots required.
- 2026-05-01: not started → in progress. Drafter dispatched (Sonnet,
  general-purpose). Three screenshots already provided in
  doc/source/_static/user_guide/lif/ (lif_exp_setup.png, lif_config.png,
  lif_tab.png); drafter instructed to inspect and embed those rather than
  leaving TODOs.
-->

Replaces the one-paragraph LIF stub with a full chapter covering
the laser-induced fluorescence (and REMPI-style laser-scan)
acquisition module: how it appears in the experiment wizard, how
it is configured, how its runtime tab is used, and what it writes
to disk.

## Scope

The chapter is organised into four sub-pages plus an intro.
Filenames and ordering below.

### Intro page

- `doc/source/user_guide/lif.rst` — converted from stub to a brief
  intro plus a `:hidden:` toctree pointing at the four sub-pages.
  One paragraph on what the LIF module does (time-gated integration
  of an analog signal as a function of laser frequency and/or
  delay), how it is enabled (per-profile runtime toggle in
  Application Configuration; LIF is built into every binary),
  and that it can run alongside CP-FTMW or standalone.

### Experiment Setup sub-page

- `doc/source/user_guide/lif/experiment_setup.rst` — the LIF page
  in the experiment wizard.
  - When the LIF page appears in the wizard (only when LIF is
    enabled in the active profile and an LIF acquisition is
    requested).
  - Selecting laser-frequency vs. delay axes (one or both).
  - Setting axis ranges, step sizes, and shots per point.
  - Randomised delay-point ordering (commit `09b05738`) — what it
    does and when to use it.
  - Initialising the wizard page from a previous LIF acquisition.
  - Cross-link to :doc:`configuration` for the channel/gate setup
    that lives in the LIF tab rather than the wizard.

### LIF Configuration sub-page

- `doc/source/user_guide/lif/configuration.rst` — channel and
  gate setup.
  - LIF channel vs. Reference channel concepts; what the reference
    channel is for (laser-power normalisation).
  - Integration gates: how they are placed, units, and the
    relationship between the gate and the digitizer record window.
  - Laser-power normalisation behaviour and when it kicks in.
  - The runtime LIF toggle in Application Configuration (and the
    fact that profile-level enablement is required before an LIF
    page can be added to a wizard).
  - Cross-link to :doc:`/user_guide/hw/lifdigitizer` and
    :doc:`/user_guide/hw/liflaser` for hardware-side wiring.

### LIF Tab sub-page

- `doc/source/user_guide/lif/lif_tab.rst` — the LIF Display tab
  used during and after acquisition.
  - Tab tour: time trace (`LifTracePlot`), delay slice plot,
    laser slice plot, 2D plot (`LifSpectrogramPlot`), processing
    panel (`LifProcessingWidget`), laser control (`LifLaserWidget`).
  - Refresh interval / live-update behaviour.
  - Processing options exposed in the panel (smoothing,
    integration window override, normalisation toggle).
  - Slice plot interaction (click to set slice index in the 2D
    plot).
  - Manual laser tuning from the laser control box outside of an
    acquisition.

### LIF Data Storage sub-page

- `doc/source/user_guide/lif/data_storage.rst` — what the LIF
  module writes to disk.
  - The LIF data files inside an experiment folder, alongside the
    files documented in :doc:`/user_guide/data_storage`.
  - The trace-array layout (per delay × per laser-frequency point,
    with shot count metadata) and how to load it externally.
  - The configuration record (`LifConfig` serialised header
    fields) and what each field means for re-analysis.
  - Cross-link to the FTMW :doc:`/user_guide/data_storage` page so
    that a reader can see the full per-experiment file inventory
    in one view.

## Out of scope

- LIF source-code internals — covered in the developer guide
  (bundle 12) and API reference (bundle 13e/13f).
- LIF hardware driver pages — those are bundle 05.
- The Application Configuration profile editor itself (already
  documented in :doc:`/user_guide/application_config`); only the
  LIF-specific toggle is mentioned here.
- The "experimental" caveat that the legacy stub carried — LIF is
  no longer flagged experimental in this revision.

## Sources

- `src/data/lif/lifconfig.{h,cpp}` — user-relevant configuration
  fields and serialisation surface.
- `src/data/lif/lifdigitizerconfig.{h,cpp}` — gate/channel
  semantics.
- `src/data/lif/lifstorage.{h,cpp}` — on-disk layout for the data
  storage page.
- `src/data/lif/liftrace.{h,cpp}` — single-trace structure.
- `src/gui/lif/gui/experimentlifconfigpage.{h,cpp}` — wizard page
  fields for the experiment-setup sub-page.
- `src/gui/lif/gui/lifdisplaywidget.{h,cpp}` plus the supporting
  widgets in the same directory (`lifcontrolwidget`,
  `lifprocessingwidget`, `lifsliceplot`, `lifspectrogramplot`,
  `liftraceplot`, `liflaserwidget`, `liflaserstatusbox`,
  `liflasercontroldoublespinbox`) — LIF tab tour.
- `src/data/storage/applicationconfigmanager.{h,cpp}` — runtime
  LIF-enable toggle key.
- `cmake/BlackchirpApplication.cmake` — confirms LIF is built
  into every binary (no compile-time flag).
- `dev-docs/devel-roadmap.md` — historical "experimental" caveat;
  do not propagate it into the new chapter.

## Sphinx file deltas

**Replaced/modified:**

- `doc/source/user_guide/lif.rst` — converted from stub to intro
  plus toctree.

**Created:**

- `doc/source/user_guide/lif/experiment_setup.rst`
- `doc/source/user_guide/lif/configuration.rst`
- `doc/source/user_guide/lif/lif_tab.rst`
- `doc/source/user_guide/lif/data_storage.rst`

## Toctree delta

`user_guide.rst` already lists `user_guide/lif` from the previous
stub; no top-level toctree change is required.

In the new `lif.rst`:

```rst
.. toctree::
   :hidden:

   lif/experiment_setup
   lif/configuration
   lif/lif_tab
   lif/data_storage
```

## Screenshots

Drafter leaves TODO markers with these filenames (capture is a
later human pass):

- `_static/user_guide/lif/experiment_setup.png` — LIF page in the
  experiment wizard with both axes enabled.
- `_static/user_guide/lif/configuration.png` — LIF channel and
  gate configuration UI in the LIF tab.
- `_static/user_guide/lif/lif_tab.png` — full LIF Display tab
  with time trace, delay slice, laser slice, and 2D plot
  populated.
- `_static/user_guide/lif/processing.png` — close-up of the
  processing panel.

## Acceptance criteria

- The LIF chapter is no longer a single paragraph; it covers
  experiment setup, channel/gate configuration, the LIF tab, and
  on-disk data storage as four discrete sub-pages.
- The chapter does not call LIF "experimental" or refer the
  reader to Discord for usage information.
- The runtime LIF-enable toggle in Application Configuration is
  documented in the configuration sub-page and is correctly
  described as a per-profile setting.
- The data-storage sub-page enumerates the LIF-specific files
  written into an experiment folder and cross-links to the main
  FTMW data-storage page rather than duplicating its content.
- All cited source files and class names exist in the current
  tree.
- All `:doc:` cross-references resolve.
