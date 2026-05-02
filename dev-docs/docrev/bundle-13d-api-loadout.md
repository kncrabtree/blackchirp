# Bundle 13d — API Reference: Loadout / Preset Classes

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-05-02: drafted → complete. Stage 1 content commit: 5925ba9f. User reviewed the rendered docs and approved without revisions.
- 2026-05-02: in progress → drafted. Verifier passed cleanly (no blockers). Three new RST pages landed under doc/source/classes/ (hardwareloadout, loadoutmanager, rfconfigsnapshot); three loadout headers refreshed with \brief on every public/protected member. Optional loadout_helpers.rst page omitted — chirpconfigloadout/ftmwdigitizerloadout are pure free-function persistence plumbing with no class state, already cross-referenced from the hardwareloadout page. Two should-fix items applied directly by orchestrator: single-backtick → :cpp:class: for two RfConfig references in rfconfigsnapshot.rst, and added :doc: cross-link to the user-guide ftmw_presets chapter from hardwareloadout.rst. Awaiting user review of rendered docs.
- 2026-05-02: not started → in progress. Drafter dispatched (Sonnet, delegated mode). Scope sanity-checked: HardwareLoadout and FtmwPreset are structs in hardwareloadout.h (→ doxygenstruct on a single page); LoadoutManager has the __LastUsed__ sentinel at loadoutmanager.h:35 and a BC::Store::LM key namespace; RfConfigSnapshot has its own header. Auxiliary loadout helpers (ChirpConfigLoadout, FtmwDigitizerLoadout) deferred to drafter judgment per bundle spec.
-->

Adds API reference pages for the loadout and FTMW preset data
model.

## Scope

New pages under `doc/source/classes/`:

- `hardwareloadout.rst` ← `src/data/loadout/hardwareloadout.h`
  (`HardwareLoadout`, `FtmwPreset` if it lives in this header).
- `loadoutmanager.rst` ← `src/data/loadout/loadoutmanager.h`
  (`LoadoutManager` singleton).
- `rfconfigsnapshot.rst` ←
  `src/data/loadout/rfconfigsnapshot.h` (`RfConfigSnapshot`).

If `FtmwPreset` lives in its own header
(`src/data/loadout/ftmwpreset.h` or similar), give it a separate
page. Otherwise document it within `hardwareloadout.rst`.

Auxiliary loadout classes (`ChirpConfigLoadout`,
`FtmwDigitizerLoadout`) can be combined into a single
`loadout_helpers.rst` page if their surface area is small;
otherwise omit them from API reference and let the developer guide
describe the relationship.

## Out of scope

- Loadout UI surfaces (`runtimehardwareconfigdialog`,
  `ftmwconfigwidget` preset bar) — those are GUI helpers and
  belong to bundle 13g if at all.

## Sources

- `dev-docs/loadout-system.md` — primary.
- The header files.

## Sphinx file deltas

**Created:**
- `doc/source/classes/hardwareloadout.rst`
- `doc/source/classes/loadoutmanager.rst`
- `doc/source/classes/rfconfigsnapshot.rst`
- (`doc/source/classes/loadout_helpers.rst` — optional)

**Possibly modified:**
- `src/data/loadout/hardwareloadout.h`
- `src/data/loadout/loadoutmanager.h`
- `src/data/loadout/rfconfigsnapshot.h`
- `src/data/loadout/chirpconfigloadout.h` (optional)
- `src/data/loadout/ftmwdigitizerloadout.h` (optional)

## Acceptance criteria

- `LoadoutManager` page documents the `__LastUsed__` sentinel
  semantics and the QSettings storage layout (the storage layout
  diagram from `dev-docs/loadout-system.md` may be quoted
  verbatim).
- `HardwareLoadout` page enumerates every field with a `\brief`.
- `RfConfigSnapshot` page explains its serialization role
  (helper class for `FtmwPreset`).
- All pages cross-link to the user-guide Hardware Configuration
  chapter (bundle 03).
