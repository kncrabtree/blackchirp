# Documentation Revision — Master Plan

The Sphinx/ReadTheDocs documentation under `doc/source/` was last refreshed at
commit `8bc115aeba017986786a1d70e1346be9cd08aaf9` on the `master` branch.
Since then, 420+ commits on `devel` and `cmakemigration` have introduced
substantial new functionality, replaced the build system (qmake → CMake),
added binary distribution, restructured the hardware subsystem, and
introduced Python-based hardware drivers. The existing user guide does not
reflect any of this.

This plan is broken into self-contained **bundles**. Each bundle is a single
file in `dev-docs/docrev/` describing a unit of work small enough to be
implemented in one focused session by a smaller model. A bundle file states
its scope, its inputs (which dev-docs and source files to mine), the
Sphinx files it creates or touches (with toctree deltas), screenshot
requirements, and acceptance criteria.

## Project layout

- `dev-docs/documentation-revision.md` — this plan (master roadmap)
- `dev-docs/docrev/bundle-NN-name.md` — one file per work bundle
- `doc/source/` — Sphinx source (target of the work)
- `doc/source/_static/user_guide/` — screenshots referenced in user-guide pages

## Bundles at a glance

| ID | Title | Estimated Effort | Depends on |
|----|-------|------------------|------------|
| 00 | Doc infrastructure, landing page, README | S | — |
| 01 | Installation: binary packages and CMake source build | M | 00 |
| 02 | First Run, Application Configuration, Hardware Onboarding | M | 00 |
| 03 | Hardware Configuration: profiles, loadouts, FTMW presets | L | 02 |
| 04 | Hardware Menu, Communication, Status Panel | M | 03 |
| 05 | Per-device hardware page refresh | M | 04 |
| 06 | Python hardware (user guide) | L | 03 |
| 07 | RF configuration, chirp setup, FTMW digitizer | M | 03, 04 |
| 08 | Experiment workflow refresh | M | 07 |
| 09 | FTMW data viewing, overlays, data storage refresh | M | 08 |
| 10 | LIF, Rolling/Aux, Log tab, Blackchirp-viewer | M | 08 |
| 11 | Migration guide v1.x → 2.0.0 + Changelog | S | most user-guide bundles |
| 12 | Developer Guide | L | — (independent) |
| 13a | API ref: refresh existing 5 (HardwareObject etc.) | S | — |
| 13b | API ref: hardware-management classes | S | 13a |
| 13c | API ref: Python hardware classes | S | 13a |
| 13d | API ref: loadout/preset classes | S | 13a |
| 13e | API ref: data/experiment classes | M | 13a |
| 13f | API ref: storage classes | M | 13a |
| 13g | API ref: GUI helper classes | M | 13a |
| 13h | API ref: file parsers | S | 13a |

Effort key: S ≈ 1 session, M ≈ 2 sessions, L ≈ 3+ sessions.

## Recommended order

The user-guide track (00 → 11) is mostly linear, with the API-reference
track (13a → 13h) and the developer guide (12) running independently in
parallel.

### Sequential critical path (user guide)

1. **00 — Doc infrastructure & landing.** Establishes the toctree
   skeleton and Sphinx changelog scaffold that every other bundle plugs
   into.
2. **01 — Installation.** Replaces qmake-era content; binary downloads
   are referenced from later bundles.
3. **02 — First Run & Application Configuration.** Introduces concepts
   (data path, profiles, library status) that bundles 03 and onwards
   reference.
4. **03 — Hardware Configuration: profiles, loadouts, FTMW presets.**
   This is the largest conceptual shift from v1.x and is referenced by
   nearly every later chapter.
5. **04 — Hardware Menu, Communication, Status Panel.** Updates the
   day-to-day UI navigation page.
6. **07 — RF, chirp, FTMW configuration.** Depends on 03 (`FtmwConfigDialog`
   and preset bar) and 04 (Hardware menu entry points).
7. **08 — Experiment workflow refresh.** Depends on 07 (chirp/RF setup
   pages are linked from the wizard walkthrough).
8. **09 — FTMW data viewing, overlays, data storage refresh.** Depends
   on 08 (data storage describes what the experiment writes).
9. **10 — LIF, Rolling/Aux, Log tab, Blackchirp-viewer.** Depends on 08
   (LIF acquisition setup is part of the experiment wizard).
10. **11 — Migration guide and changelog.** Best done last so it can
    cross-reference the new pages.

### Independent / parallelizable bundles

These can be tackled at any time, in any order, without blocking the
critical path:

- **05 — Per-device hardware pages.** Light refresh; can run in parallel
  with 03–10. Only depends on 04 for terminology.
- **06 — Python hardware user guide.** Depends conceptually on 03
  (profiles) but can be drafted in parallel; cross-link at the end.
- **12 — Developer Guide.** Sources are dev-docs and source code; no
  dependency on the user-guide bundles.
- **13a–13h — API reference bundles.** Each is independent of the others
  except that 13a establishes the Doxygen-comment style guide that
  13b–13h follow. After 13a, the remaining seven are fully parallel.

### Suggested parallel teams (if multiple bundles run concurrently)

- **Team A (user guide critical path):** 00 → 01 → 02 → 03 → 04 → 07 → 08 → 09 → 10 → 11
- **Team B (per-device + python):** 05, 06
- **Team C (developer + API):** 12, 13a, then 13b–13h fanned out

## Common conventions for bundle authors

These rules apply across every bundle and are intentionally not repeated
in each bundle file.

- **Voice and tense.** User-facing content is in the present tense.
  Avoid temporal markers ("now", "currently", "recently") and version
  labels in prose ("Phase 2", "v1.1.0 introduced"). Permanent
  version-keyed information lives in the changelog or migration guide.
- **Cross-references.** Use Sphinx `:doc:` and `:ref:` directives, not
  raw HTML links. Replace any existing `<page.html>`-style anchors when
  editing a page.
- **Screenshots.** All new or changed UI screenshots go in
  `doc/source/_static/user_guide/<page-name>/`. Each bundle's
  "Screenshots" section enumerates which ones it needs; the bundle is
  not "complete" until those exist (the author can leave a TODO and
  pre-record the filenames so the prose is correct).
- **Index entries.** Every new page begins with a `.. index::` block
  listing the key user-facing terms it introduces.
- **Settings-registry assumption.** Per-device settings are
  self-documenting in the UI via the registry's labels and tooltips.
  Documentation does **not** enumerate every setting; it documents the
  non-obvious ones, defaults that matter, and behavioural caveats.
- **dev-doc reuse.** Where a `dev-docs/*.md` already explains a
  subsystem (loadouts, settings registry, python hardware, etc.), the
  user-guide page is built by extracting the *user-relevant* portions
  and dropping internals. Cite the dev-doc path in the bundle's "Sources"
  section so the implementer knows where to look.
- **API reference style.** Prefer `.. doxygenclass::` over
  `.. doxygenfile::` so each class gets a focused page and member
  documentation is grouped by member. Bundle 13a establishes the
  template the rest follow.
- **No backwards-compatibility prose.** Documentation describes the
  current state of the program. Migration to 2.0.0 is concentrated in
  bundle 11; everywhere else, write as if 2.0.0 has always been the
  state of the world.

## Open coordination

- The hardware catalog (originally a separate goal) is intentionally
  folded into the per-device pages (bundle 05). No standalone catalog
  table is planned.
- The Python module (`python/blackchirp/`) and the example notebook
  documentation under `doc/source/python/` are out of scope for this
  revision unless a bundle explicitly touches them.
