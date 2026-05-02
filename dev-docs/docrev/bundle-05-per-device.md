# Bundle 05 — Per-Device Hardware Pages

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-05-01: not started → complete. Content commit 2dfda1eb. Drafter
  fan-out (six parallel Sonnet drafters owning ≤2 files each) plus
  per-page and cross-page Sonnet verifiers; orchestrator applied the
  load-bearing punch-list fixes directly. Subsequent user review
  surfaced several factual corrections (virtual FTMW scope synthesizes
  an FID rather than shipping a resource; multiple GPIB controllers
  supported; LabJack U3 on macOS too; pulse generators reorganized by
  series rather than model) which were applied before staging. Final
  pass dropped the parenthetical short-key suffixes that were artifacts
  of compile-time hardware selection.
-->

Light refresh of the eleven `doc/source/user_guide/hw/*.rst` pages.

## Scope

The settings registry now provides labels and tooltips inline in the
UI, so the per-device pages no longer need to enumerate every
setting. Each page should retain its existing structure (Overview,
Settings, Implementations) but trim the Settings section to just the
non-obvious behaviour, defaults that matter for users to be aware of,
and per-implementation caveats.

Pages to refresh:

- `hw/awg.rst` — replace the `prot`/`amp` settings discussion with a
  short note pointing to the new `markerCount` setting and the
  Markers tab in the chirp configuration widget (forward-link to
  bundle 07). Refresh the implementations table to mention the
  trigger-as-marker behaviour on AWG5204.
- `hw/clock.rst` — minor refresh; mention the Apply Clock Settings
  button and that clock pushes happen on loadout accept.
- `hw/flowcontroller.rst` — runtime channel-count support, base-class
  default for pUnits, GasFlowDisplayBox layout updates.
- `hw/ftmwdigitizer.rst` — confirm transfer-rate tips are still
  accurate; add a brief mention of the WaveformBuffer
  pre-accumulation as a user-visible behaviour ("Blackchirp
  automatically accumulates shots locally if processing falls
  behind") with no internals.
- `hw/gpibcontroller.rst` — Prologix is still the only
  implementation; verify wording.
- `hw/ioboard.rst` — mention LabJack U3 cross-platform availability;
  point to the Library Status page (bundle 02) for the Windows UD
  install hint.
- `hw/lifdigitizer.rst` — minor refresh.
- `hw/liflaser.rst` — minor refresh.
- `hw/pressurecontroller.rst` — minor refresh.
- `hw/pulsegenerator.rst` — refresh screenshot; note the registry-
  driven settings UI.
- `hw/temperaturecontroller.rst` — minor refresh.

For every page:

- Remove `<../page.html>` style cross-references and replace with
  `:doc:` references. Replace `<../hardware_menu.html#anchor>` with
  `:ref:` to a target defined in the new `hardware_menu.rst`.
- Drop or rewrite any sentence that begins "As of Blackchirp v1.0…"
  or that references `config.pri`.

Also refresh `doc/source/user_guide/hardware_details.rst` (the
parent page): drop the `config.pri` mention; the only common
settings discussed here should be `critical` and
`rollingDataIntervalSec`, and the page should briefly note that
most other settings are surfaced with labels and tooltips by the
settings registry (forward to bundle 04 `hwdialog.rst`).

## Out of scope

- Adding new device pages.
- The hardware catalog table from the original revision plan (folded
  into these per-device pages — no standalone catalog page).
- Python hardware (bundle 06).

## Sources

- `dev-docs/settings-registry.md` — confirm priority semantics for
  any setting still mentioned.
- `dev-docs/awg-marker-system.md` — for the AWG implementations
  table.
- Source files for each hardware base class — read for any
  implementation-specific quirks worth surfacing.

## Sphinx file deltas

**Modified:**
- `doc/source/user_guide/hardware_details.rst`
- `doc/source/user_guide/hw/awg.rst`
- `doc/source/user_guide/hw/clock.rst`
- `doc/source/user_guide/hw/flowcontroller.rst`
- `doc/source/user_guide/hw/ftmwdigitizer.rst`
- `doc/source/user_guide/hw/gpibcontroller.rst`
- `doc/source/user_guide/hw/ioboard.rst`
- `doc/source/user_guide/hw/lifdigitizer.rst`
- `doc/source/user_guide/hw/liflaser.rst`
- `doc/source/user_guide/hw/pressurecontroller.rst`
- `doc/source/user_guide/hw/pulsegenerator.rst`
- `doc/source/user_guide/hw/temperaturecontroller.rst`

## Toctree delta

None.

## Screenshots

- `_static/hardware/pulsegenerator_menu.png` — refresh to show
  current HwDialog layout.
- AWG marker tab is owned by bundle 07; this bundle does not need
  a marker screenshot.

## Acceptance criteria

- No per-device page contains an exhaustive setting-by-setting
  table; each page documents only behaviour the settings-registry
  tooltips do not already convey.
- All `<page.html>`-style anchors have been converted to `:doc:`
  / `:ref:` directives.
- The AWG page's protection/gate spinbox content has been removed
  and replaced with a one-paragraph forward-link to the chirp
  setup page (bundle 07).
- The IOBoard page mentions LabJack cross-platform availability
  and links to the Library Status page.
