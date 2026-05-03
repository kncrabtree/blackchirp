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

| ID | Title | Effort | Depends on | Status |
|----|-------|--------|------------|--------|
| 00 | Doc infrastructure, landing page, README | S | — | complete |
| 01 | Installation: binary packages and CMake source build | M | 00 | complete |
| 02 | First Run, Application Configuration, Hardware Onboarding | M | 00 | complete |
| 03 | Hardware Configuration: profiles, loadouts, FTMW presets | L | 02 | complete |
| 04 | Hardware Menu, Communication, Status Panel | M | 03 | complete |
| 05 | Per-device hardware page refresh | M | 04 | complete |
| 06 | Python hardware (user guide) | L | 03 | complete |
| 07 | RF configuration, chirp setup, FTMW digitizer | M | 03, 04 | complete |
| 08 | Experiment workflow refresh | M | 07 | complete |
| 09 | FTMW data viewing, overlays, data storage refresh | M | 08 | complete |
| 10 | Rolling/Aux, Log tab, Blackchirp-viewer | M | 08 | complete |
| 11 | Changelog + Migration Guide chapter umbrella | S | most user-guide bundles | complete |
| 11a | Changelog: 2.0.0 release notes (user-perspective) | M | 11 | not started |
| 11b | Migration Guide: v1.x → 2.0.0 | M | 11a | not started |
| 12 | Developer Guide chapter scaffold + intro | S | — (independent) | complete |
| 12a | Developer Guide: Build System & Project Layout | M | 12 | complete |
| 12b | Developer Guide: Coding Conventions | S | 12 | complete |
| 12c | Developer Guide: Architecture | M | 12 | complete |
| 12d | Developer Guide: Hardware Configuration | M | 12c | complete |
| 12e | Developer Guide: Hardware Runtime | M | 12d | complete |
| 12f | Developer Guide: Experiment Lifecycle | M | 12e | complete |
| 12g | Developer Guide: FTMW Acquisition & Visualization | M | 12f | complete |
| 12h | Developer Guide: LIF Acquisition & Visualization | M | 12f | complete |
| 12i | Developer Guide: Persistence | M | 12f | complete |
| 12j | Developer Guide: Python Hardware | L | 12e | complete |
| 12k | Developer Guide: Vendor Libraries | M | 12e | complete |
| 12l | Developer Guide: Adding a Driver | M | 12e | complete |
| 12m | Developer Guide: Adding a Hardware Type | M | 12e, 12j | complete |
| 12n | Developer Guide: Adding an Experiment Mode | M | 12f | complete |
| 13a | API ref: refresh existing 5 (HardwareObject etc.) | S | — | complete |
| 13b | API ref: hardware-management classes | S | 13a | complete |
| 13c | API ref: Python hardware classes | S | 13a | complete |
| 13d | API ref: loadout/preset classes | S | 13a | complete |
| 13e | API ref: data/experiment classes | M | 13a | complete |
| 13f | API ref: storage classes | M | 13a | complete |
| 13g | API ref: GUI helper classes | M | 13a | complete |
| 13h | API ref: file parsers | S | 13a | complete |
| 13i | API ref: orchestration managers (HardwareManager, AcquisitionManager, BatchManager, ClockManager) | M | 13a | complete |
| 14 | Final consistency pass — chapter umbrella | S | most user-guide bundles | not started |
| 14a | API page intro / header-comment harmonization | M | 14 | not started |
| 14b | Screenshot sizing pass | S | 14 | not started |
| 14c | File organization & menu layout audit | S | 14 | not started |
| 14d | American English sweep | S | 14 | not started |
| 14e | Implementation → driver terminology sweep | S | 14 | not started |
| 15 | LIF module (experiment setup, configuration, tab, storage) | M | 08 | complete |

Effort key: S ≈ 1 session, M ≈ 2 sessions, L ≈ 3+ sessions.

Status values:

- **not started** — no work has been done on this bundle.
- **in progress** — the bundle has been opened in an orchestrator
  session but is not yet accepted. The bundle's own header (see
  *Bundle status header* below) carries the detail (handoff notes,
  open questions, partial-landing summary).
- **complete** — the bundle's content commit (stage 1 of the
  two-stage commit pattern) has landed. The commit hash is recorded
  in the bundle's own status header for traceability.
- **blocked** — work has been attempted but cannot proceed without a
  human decision (scope drift, dependency on un-merged work, source
  ambiguity). The blocker is described in the bundle's header.

This table is the **single source of truth for progress.** The
orchestrator updates it (via `Edit`) when a bundle's status
transitions. The user is encouraged to confirm the table is correct
before each new orchestrator session.

The user-guide track (00–10, 15) and the API-reference track
(13a–13i) used a delegated drafter/verifier workflow that is no
longer in service — those tracks are complete. Every remaining
track (developer guide 12 + 12a–12n, changelog/migration 11 +
11a/11b, final consistency pass 14 + 14a–14e) uses the
**orchestrator-direct** workflow described in *Workflow* below.
Each chapter umbrella (`bundle-12-developer-guide.md`,
`bundle-11-changelog-migration.md`, `bundle-14-final-pass.md`)
restates the workflow as it applies to that track; this file
remains the canonical source of truth for **status** (the bundles
table above) for every track.

## Recommended order

The user-guide (00–10, 15), API-reference (13a–13i), and
developer-guide (12 + 12a–12n) tracks are complete. The remaining
work is:

1. **11 + 11a + 11b — Changelog and migration guide.** 11 is a
   chapter-umbrella commit. 11a populates the 2.0.0 changelog page
   (user perspective: what is new from a v1.0 user's point of view)
   and notes any items the migration guide will need that the 11b
   spec does not already enumerate. 11b writes the migration guide
   itself, consuming 11a's carry-forward list. 11a depends on 11;
   11b depends on 11a.
2. **14 + 14a + 14b–14e — Final consistency pass.** 14 is a
   chapter-umbrella commit. **14a runs first** because it may
   involve new API-page writing (re-balancing class-level prose
   between Doxygen headers and the `.rst` page) and writes the
   `api_style.rst` rule that future API edits should follow.
   14b–14e are mechanical sweeps (screenshot sizing, file
   organization audit, American-English sweep, implementation →
   driver terminology sweep) and can be tackled in any order
   after 14a.

## Common conventions for bundle authors

These rules apply across every bundle and are intentionally not repeated
in each bundle file.

- **Voice and tense.** User-facing content is in the present tense.
  Avoid temporal markers ("now", "currently", "recently") and version
  labels in prose ("Phase 2", "v1.1.0 introduced"). Permanent
  version-keyed information lives in the changelog or migration guide.
- **American English spelling.** All documentation prose uses American
  English: `normalize`/`normalization`, `behavior`, `color`,
  `visualization`, `randomize`, `initialize`, `analyze`, `co-averaging`,
  etc. Match UI labels exactly when quoting them; if a UI label uses an
  American spelling (e.g. "Randomize Delay Order"), do not "correct" it
  to British English in prose.
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

## Out-of-band page refreshes

Pages outside any bundle's declared scope may be refreshed directly
when a small gap is discovered. These edits do not require an
orchestrator/drafter/verifier cycle.

- `doc/source/user_guide/plot_controls.rst` — refreshed context-menu
  description, added the per-curve `Type` and `Autoscale` controls,
  and added a new **Curve Presets** section covering the nine default
  presets, the **Save Curve Appearance Preset** dialog, and the
  apply/save/delete workflows.

## Workflow

Every remaining bundle uses the **orchestrator-direct** workflow:
one bundle per orchestrator session, one Opus context, no
drafter/verifier subagents writing prose. The orchestrator reads
the chapter umbrella plus the sub-bundle file plus the cited
sources and writes the page directly. The five steps below are the
canonical sequence; each chapter umbrella restates them as they
apply to that track.

1. **Read the sub-bundle file and verify its scope is current.**
   Confirm cited paths still exist, that any class names match the
   current code, and that the listed sources are still
   load-bearing. The codebase keeps moving; a sub-bundle authored
   weeks ago may need a touch-up before drafting begins. If scope
   has drifted enough that the sub-bundle file itself needs
   revising, do that first and flag it to the user.

2. **Draft the page directly.** Read the chapter umbrella, the
   sub-bundle file, and the cited sources. Use the
   `codebase-memory` MCP tools (project name
   `home-kncrabtree-github-blackchirp-src`) for code exploration
   per `CLAUDE.md`. Produce only the RST files the sub-bundle
   declares in scope. Do not edit `MEMORY.md`, the master plan,
   the umbrella, or any other sub-bundle file (with the narrow
   exception of the sub-bundle's own status header in stage 5).

3. **Sanity-check and hand off to the user for review.** Build the
   docs:

   ```bash
   touch doc/source/index.rst && conda run -n breathe cmake --build build --target docs
   ```

   Note any new warnings or unresolved cross-references. Report
   to the user with: a one-paragraph summary of what landed; any
   gaps the sub-bundle's sources could not support; any deviations
   from stated scope and why; and the path to the rendered page
   so the user can review locally. Do not stage anything yet —
   the working tree stays unstaged so the user can experiment
   freely. Loop on user-directed revisions until the user signals
   approval.

4. **Stage and run the content commit (stage 1 of two).** Stage
   only the files the sub-bundle declared in scope. Do not stage
   anything under `dev-docs/` for stage 1. Commit with a subject
   that names the page's deliverable (not "Bundle 11a"; the
   reader of `git log` does not care which bundle it was).

5. **Stage and run the tracking commit (stage 2 of two).** Update
   the sub-bundle's status header (status → `complete`, append a
   status-log entry timestamped with the stage-1 commit hash) and
   the master-plan table cell in this file. Stage those two files
   and commit with subject
   `Update documentation revision tracking status`. The
   orchestrator may run both commits itself, or wait for the user
   to commit, per the user's preference for the session.

### Research-agent dispatch

The orchestrator may dispatch agents (Sonnet or Haiku) for
**context-gathering tasks only** — surveying the commit log
between two refs, pulling specific snippets out of dev-docs or
source headers, summarizing a long file, listing every page that
references a given symbol, etc. Research agents return data that
the orchestrator synthesizes; they do **not** write the RST or
take design decisions. This protects the orchestrator's context
while keeping high-level reasoning at the Opus tier.

### Resuming a partial session

If a session ends mid-bundle, the orchestrator must:

- update the sub-bundle's status header to `in progress` and
  append a status-log entry summarizing what landed, what
  remains, and any open questions or design decisions the next
  session needs;
- stop at a clean checkpoint (do not leave the docs in a
  half-built state);
- not run either commit; hand off to the user with the same
  summary so they can decide whether to commit the partial work
  or roll it back.

The next orchestrator session reads the status header before
continuing.

### Bundle status header

Each bundle file (`dev-docs/docrev/bundle-NN-*.md`) carries a
status block at the top, immediately under the H1, in the form:

```markdown
**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->
```

The orchestrator updates this block in lockstep with the
master-plan table whenever a status transition occurs. Each
transition appends one entry to the status log with: timestamp,
transition (e.g. `not started → complete`), and a short note
(handoff summary, blocker description, content-commit hash). When
a bundle reaches `complete`, record the stage-1 (content) commit
hash; the stage-2 tracking commit is the one that physically
writes `complete` and is not separately recorded.

### Resuming work in a fresh orchestrator session

On startup the orchestrator's first action (after reading the
master plan and `CLAUDE.md`) is:

1. Read the status column in the bundle table above.
2. For any bundle marked `in progress` or `blocked`, read that
   bundle file's status header for context.
3. Surface the current state to the user and wait for direction.

Do not assume that work marked `not started` actually has not
been attempted: glance at the latest few git log entries on the
working branch as a sanity check. If the table and git history
disagree, the user resolves it.
