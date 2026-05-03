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
| 11 | Migration guide v1.x → 2.0.0 + Changelog | S | most user-guide bundles | not started |
| 12 | Developer Guide chapter scaffold + intro | S | — (independent) | complete |
| 12a | Developer Guide: Build System & Project Layout | M | 12 | not started |
| 12b | Developer Guide: Coding Conventions | S | 12 | not started |
| 12c | Developer Guide: Architecture | M | 12 | not started |
| 12d | Developer Guide: Hardware Configuration | M | 12c | not started |
| 12e | Developer Guide: Hardware Runtime | M | 12d | not started |
| 12f | Developer Guide: Experiment Lifecycle | M | 12e | not started |
| 12g | Developer Guide: FTMW Acquisition & Visualization | M | 12f | not started |
| 12h | Developer Guide: LIF Acquisition & Visualization | M | 12f | not started |
| 12i | Developer Guide: Persistence | M | 12f | not started |
| 12j | Developer Guide: Python Hardware | L | 12e | not started |
| 12k | Developer Guide: Vendor Libraries | M | 12e | not started |
| 12l | Developer Guide: Adding a Driver | M | 12e | not started |
| 12m | Developer Guide: Adding a Hardware Type | M | 12e, 12j | not started |
| 12n | Developer Guide: Adding an Experiment Mode | M | 12f | not started |
| 13a | API ref: refresh existing 5 (HardwareObject etc.) | S | — | complete |
| 13b | API ref: hardware-management classes | S | 13a | complete |
| 13c | API ref: Python hardware classes | S | 13a | complete |
| 13d | API ref: loadout/preset classes | S | 13a | complete |
| 13e | API ref: data/experiment classes | M | 13a | complete |
| 13f | API ref: storage classes | M | 13a | complete |
| 13g | API ref: GUI helper classes | M | 13a | complete |
| 13h | API ref: file parsers | S | 13a | complete |
| 13i | API ref: orchestration managers (HardwareManager, AcquisitionManager, BatchManager, ClockManager) | M | 13a | complete |
| 14 | Final pass: screenshot sizing & navigation review | S | most user-guide bundles | not started |
| 15 | LIF module (experiment setup, configuration, tab, storage) | M | 08 | complete |

Effort key: S ≈ 1 session, M ≈ 2 sessions, L ≈ 3+ sessions.

Status values:

- **not started** — no work has been done on this bundle.
- **in progress** — drafter has been dispatched but the bundle is not
  yet accepted. The bundle's own header (see "Bundle status header"
  below) carries the detail (revision pass, open punch list).
- **drafted** — drafter output has landed in the working tree, the
  verifier punch list is resolved, and the orchestrator has handed
  off to the user for review of the rendered docs. The bundle stays
  in `drafted` through any user-directed revision rounds.
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

**Developer-guide track (12, 12a–12n).** Direct orchestrator
drafting, one bundle per session, no drafter/verifier subagents.
The "Per-bundle workflow (delegated bundles)", "Orchestrator
hygiene", and "Dispatch checklist" sections below describe the
delegated workflow used by the user-guide and API-reference
tracks; they do not apply to the developer-guide track. The
developer-guide track substitutes the workflow in
`dev-docs/docrev/bundle-12-developer-guide.md`. This file remains
the canonical source of truth for **status** (the bundles table
above) for every track.

## Recommended order

Bundles are tackled one at a time. The user-guide track (00 → 11) is
mostly linear; the API-reference track (13a → 13h) and the developer
guide (12) are independent of it and can be slotted in between
user-guide bundles in whatever order suits.

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
9. **10 — Rolling/Aux, Log tab, Blackchirp-viewer.** Depends on 08
   (the data-storage page describes what each tab consumes).
10. **15 — LIF module.** Depends on 08 (LIF acquisition setup is
    part of the experiment wizard). Peer of bundle 10; either can
    land first.
11. **11 — Migration guide and changelog.** Best done last so it can
    cross-reference the new pages.

### Independent bundles

These can be tackled at any point in the schedule without blocking the
critical path:

- **05 — Per-device hardware pages.** Light refresh; depends on 04 only
  for terminology.
- **06 — Python hardware user guide.** Depends conceptually on 03
  (profiles); cross-link to the relevant user-guide pages once both
  have landed.
- **12 — Developer Guide.** Sources are dev-docs and source code; no
  dependency on the user-guide bundles.
- **13a–13i — API reference bundles.** Each is independent of the others
  except that 13a establishes the Doxygen-comment style guide that
  13b–13i follow.

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

## Orchestrator instructions

This plan is designed to be driven by an **orchestrator** (Opus,
fresh context) that dispatches **drafter** subagents (Sonnet) to
implement individual bundles, then dispatches separate **verifier**
subagents to grade the output against acceptance criteria. The
orchestrator's own context stays clean — it reads bundle files,
briefs subagents, judges results, and merges.

### When to delegate vs. do yourself

| Bundle | Mode | Reason |
|--------|------|--------|
| 00 (infrastructure) | Direct | Small, fiddly Sphinx wiring; delegation overhead exceeds drafting time |
| 01–10 (user guide) | Delegate | Bulk RST writing; well-scoped; Sonnet handles cleanly |
| 11 (changelog/migration) | Direct | Synthesis-heavy; cross-references every other bundle's output |
| 12 + 12a–12n (developer guide) | Direct, one bundle per session | Each sub-page involves higher-level reasoning across multiple systems; orchestrator drafts directly without delegation, fresh context per sub-bundle |
| 13a (existing API refresh) | Direct | Establishes the API style convention 13b–13h follow |
| 13b–13i (API reference) | Delegate | Independent of each other once 13a is locked |

### Per-bundle workflow (delegated bundles)

1. **Read the bundle file and verify scope is current.** The
   codebase keeps moving; a bundle authored weeks ago may reference
   files or behaviour that has shifted. Skim the cited sources
   (dev-docs and source headers). If scope has drifted, revise the
   bundle file *before* dispatching — drafters cannot be expected
   to detect plan staleness.
2. **Dispatch drafter (Sonnet).** Brief with: the bundle file path,
   the codebase-memory project name
   (`home-kncrabtree-github-blackchirp-src`), the explicit
   instruction to use codebase-memory tools first per `CLAUDE.md`,
   and a hard scope: "produce only the Sphinx files the bundle
   lists; leave screenshot TODO markers with the bundle's specified
   filenames; do not edit `MEMORY.md` or the bundle file itself."
   The drafter edits the working tree directly.
3. **Dispatch verifier (Sonnet, fresh context).** Brief with: the
   working-tree diff, the bundle file, and the instruction "grade
   against each acceptance criterion; check that any cited file
   paths and class names exist; report a punch list under 300
   words." Fresh context matters — the drafter has motivated
   reasoning the verifier lacks.
4. **Orchestrator judges the punch list.** Decide which items are
   load-bearing. For load-bearing issues, dispatch a revision pass
   to the drafter with a focused prompt: "address items N, M, P
   from this punch list". For minor prose issues, fix directly via
   Edit. Do not loop through more than two revision passes; if a
   third is needed, the bundle scope is wrong and needs human
   input.
5. **Sanity-check and hand off to the user for review.** Once the
   verifier's punch list is resolved, do a quick orchestrator-level
   sanity check on the working-tree diff (file count plausible,
   nothing landed outside scope, headers and RSTs touched the
   classes the bundle declares). Then report to the user with
   anything that explicitly needs their attention: screenshots
   that need to be captured, design decisions the drafter made
   that they should sign off on, unexpected output, scope
   adjustments made mid-flight, or any acceptance criterion the
   drafter could not satisfy. Keep the report short — the user
   builds the docs locally (`cmake --build build --target docs`),
   reads the rendered output, and replies with revisions or
   approval. Loop on user-directed revisions (drafter or direct
   Edit, whichever is cheaper) until the user signals they are
   satisfied. Do not stage anything during this phase — the
   working tree stays unstaged so the user can experiment freely.
6. **Stage and run the two-stage commit** once the user has
   approved the bundle. Stage 1 is the **content commit**: every
   file the bundle declared as in scope (typically the new/edited
   files under `doc/source/`, plus any header refreshes the bundle
   authorized in `src/`). Stage only those files; do not stage
   anything under `dev-docs/` for stage 1. Commit with a subject
   that names the bundle's deliverable (not "Bundle 13b"; the
   reader of `git log` does not care which bundle it was).
7. **Stage 2 is the tracking commit.** Record the stage-1 commit
   hash in the bundle's status header (status → `complete`,
   append a status-log entry with the hash) and update the
   master-plan table (status → `complete`). Stage those two
   `dev-docs/` files and commit with the subject
   `Update documentation revision tracking status`. The
   orchestrator may run both commits itself, or wait for the user
   to commit, per the user's preference for the session.

### Orchestrator hygiene

- **One bundle per orchestrator session.** Running the
  orchestrator across many bundles in a single context defeats the
  chunking — context grows linearly and prompt-cache benefit is
  lost by the third or fourth bundle. Start a fresh Opus session
  per bundle.
- **Stay out of file-by-file writing.** If the orchestrator finds
  itself using `Edit`/`Write` for routine prose, push the work
  back to a drafter. The orchestrator's job is briefing,
  judging, and merging.
- **Pin codebase-memory in every subagent prompt.** Sonnet
  without that hint falls back to grep and misses indexed
  structural relationships.
- **Subagents do not touch this plan or MEMORY.md.** Drafters
  edit `doc/source/`, plus the header refreshes that bundles
  13a–13h explicitly authorize. Plan revisions, memory updates,
  and commits are the orchestrator's (and ultimately the user's)
  responsibility.
- **Screenshots are deferred.** Drafters leave TODO markers with
  the filenames specified in the bundle's Screenshots section.
  Screenshot capture is a separate human pass after a bundle is
  otherwise accepted.

### Dispatch checklist

Before each drafter dispatch, the orchestrator confirms:

- [ ] Bundle file scope is still accurate (cited paths exist;
      cited behaviour matches code).
- [ ] All bundle dependencies (per the table at the top of this
      plan) have status `complete`.
- [ ] The drafter prompt includes: bundle file path,
      codebase-memory project name, explicit scope limits, and a
      reminder to follow `CLAUDE.md` conventions (string
      literals, timeless prose, American English spelling, no
      emojis unless requested).
- [ ] The verifier dispatch is queued to follow drafter
      completion, and the verifier prompt explicitly instructs it
      to flag any British English spellings as part of the punch
      list (per the "Common conventions for bundle authors"
      American-English rule).
- [ ] Stage 1 (content commit) and stage 2 (tracking commit) are
      treated as separate commits — never combined. Both are run
      only after the user has reviewed the rendered docs and
      signed off on the bundle.

### Bundle status header

Each bundle file (`dev-docs/docrev/bundle-NN-*.md`) carries a status
block at the top, immediately under the H1, in the form:

```markdown
**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->
```

The orchestrator updates this block in lockstep with the master-plan
table whenever a status transition occurs. Each transition appends
one entry to the status log with: timestamp, transition (e.g.
`not started → in progress`), and a one-line note (verifier
outcome, blocker, commit hash, etc.). When a bundle reaches
`complete`, record the **content commit** hash (stage 1 of the
two-stage commit pattern). The tracking commit (stage 2) is the one
that physically writes the `complete` status into this block and into
the master-plan table; its hash is not separately recorded.

The status log gives a fresh orchestrator session enough context to
resume mid-bundle: it tells the orchestrator whether a verifier
punch list is outstanding and whether the previous attempt was
abandoned for a known reason.

### Resuming work in a fresh orchestrator session

On startup the orchestrator's first action (after reading the master
plan and `CLAUDE.md`) is:

1. Read the status column in the bundle table above.
2. For any bundle marked `in progress` or `blocked`, read that
   bundle file's status header for context.
3. Surface the current state to the user in the form: "Bundle X is
   in progress, awaiting Y. Bundle W is blocked on V. Recommended
   next action: …" — and wait for user direction.

Do not assume that work marked `not started` actually has not been
attempted: glance at the latest few git log entries on the working
branch as a sanity check. If the table and git history disagree,
the user resolves it.

## Bundle 14 — Final pass: screenshot sizing & navigation review

A short cleanup bundle to be run after the bulk of the user-guide work
has landed. Five scopes:

1. **Screenshot sizing.** Walk every `.. image::` and `.. figure::`
   directive under `doc/source/user_guide/`. For each referenced PNG,
   inspect the native pixel width:
   - **Native width ≤ 800 px:** drop any explicit `:width:` option (or
     set it to the native width) so the image renders at 1:1.
   - **Native width > 800 px:** cap the rendered width at 800 px and
     wrap the directive so the figure links to the full-resolution
     image (e.g. via `:target:` pointing at the same `_static` path,
     or a `figure` with a click-through). Pick one approach and apply
     it consistently.

2. **File organization & menu layout.** Reassess whether the
   `doc/source/user_guide/` directory layout still matches the final
   shape of the documentation. Candidates to reconsider:
   - Whether the `experiment/` subdirectory should remain (or its
     contents promoted) given that several pages there are now
     subpages of `ftmw_configuration` rather than `experiment_setup`.
   - Whether the `hw/` subdirectory should sit under
     `hardware_details` as a logical sub-tree or stay flat.
   - Toctree ordering and `:caption:` choices in the top-level
     `index.rst` and any `:hidden:` toctrees.

   The goal is a consistent navigation tree that matches how a reader
   would actually browse the manual; do not rearrange files unless a
   concrete improvement is identified.

3. **American English sweep.** Run a final pass for any British
   English spellings that slipped past per-bundle verifiers. Walk
   every `.rst` file under `doc/source/user_guide/` (and
   `doc/source/api/` if API-reference bundles have landed) and
   convert: `-ise/-isation` → `-ize/-ization` (normalize, organize,
   visualize, optimize, randomize, initialize, summarize,
   categorize, customize, etc.); `-yse` → `-yze` (analyze,
   catalyze); `colour` → `color`; `behaviour` → `behavior`;
   `centre` → `center`; `fibre` → `fiber`; `metre` → `meter`;
   `dialogue` → `dialog` only when referring to a UI dialog box
   (leave the literary sense alone); `programme` → `program`;
   `coaveraging` → `co-averaging`; plus any others surfaced by a
   regex sweep. Skip code blocks and source-identifier names; only
   change prose. Match UI labels exactly when quoting them.

4. **API page intro / header-comment harmonization.** Earlier API
   bundles (notably 13a `SettingsStorage`) put the bulk of the
   class-level prose in the header's Doxygen block and left the
   `.rst` page short. Later bundles (13e onward, formalized in 13f)
   moved orientation prose to the `.rst` and trimmed the header's
   class-level block to a tight `\brief` plus internals-as-needed.
   Walk every page under `doc/source/classes/` and align them to
   the later convention: the `.rst` carries the plain-language
   orientation, the header `\brief` block stays tight, and
   per-method `///` blocks remain rich. Update `api_style.rst` to
   make this split explicit so future API bundles do not drift back.

5. **Implementation → driver terminology sweep.** Replace
   user-facing uses of "implementation" with "driver" where the term
   refers to a concrete hardware backend for a hardware type
   (e.g. `AWG70002a`, `VirtualAwg`, `PythonAwg`). The mental model
   the manual should reinforce is *Hardware Object → Hardware Type
   → one of several drivers*: "implementation" reads as a
   programming abstraction, "driver" matches how users think about
   choosing a backend for a device. Walk every `.rst` file under
   `doc/source/user_guide/` and the API pages under
   `doc/source/classes/` and `doc/source/developer_guide/`. Skip
   code blocks, identifier names, and documentation comments that
   quote a Doxygen tag or registry macro literally
   (`REGISTER_HARDWARE_*` etc.). Headers and source files are out
   of scope; the C++ comments use whichever term reads naturally for
   each call site. After the sweep, update any glossary, index, or
   cross-reference that uses the old term.

Drafter mode: direct (orchestrator-driven). All scopes are mechanical
enough that delegation overhead exceeds the work, and a single
reviewer pass keeps the cross-page consistency tight.
