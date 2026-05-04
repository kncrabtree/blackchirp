# Bundle 11 — Changelog and Migration Guide (Chapter Umbrella)

**Status:** complete

<!--
Status log:
- 2026-05-03 — scope revision during the 11a session. Dropped the
  user-vs-developer audience split and the 2.0.0-page user-only
  exception; bug fixes are now in scope on every release page.
  Per-release pages organize as Highlights → component-level
  sections grouped by subsystem → Bug fixes (also grouped by
  subsystem). The 1.0.0 → 1.1.0 change set was split out onto its
  own page (11a now creates two pages — changelog/1.1.0.rst and
  changelog/2.0.0.rst — and the umbrella's toctree lists both).
  The "Keeping the changelog updated" H2 in changelog.rst was
  folded into the chapter-intro paragraphs so the toctree sits at
  the bottom of the page. The umbrella's content commit hash
  (642f5d9f) still stands; the re-edit lands as part of 11a's
  content commit. This entry records the spec change so the
  policy in this file matches the rendered page.
- 2026-05-03 — not started → complete (content commit 642f5d9f). Replaced
  the changelog.rst and migration.rst placeholders with chapter intros
  and explicit toctrees referencing changelog/2.0.0 and
  migration/v1_to_v2. Four transient build warnings remain until 11a
  and 11b land: two toctree-points-to-missing-document warnings, one
  orphan warning for the bundle-00 changelog/v2_0_0_alpha.rst placeholder
  (11a should remove it when it creates 2.0.0.rst), and one cross-link
  warning for /migration/v1_to_v2 in the changelog intro.
-->

Builds the chapter scaffolds for the **changelog** and the
**migration guide** and lands the chapter-level intros. The
substantive content is split across two sub-bundles:

- **11a** — populate `changelog.rst` (already converted from
  placeholder by this umbrella; 11a edits the intro if the
  per-release-page convention shifts) plus two per-release pages,
  `changelog/1.1.0.rst` (1.0.0 → 1.1.0; the devel-only commits
  that will be tagged 1.1.0 before merging cmakemigration) and
  `changelog/2.0.0.rst` (1.1.0 → 2.0.0; the cmakemigration-only
  commits that will be tagged 2.0.0 after merge). As 11a writes
  those pages it also builds a **carry-forward list** of items the
  migration guide will need that the 11b spec does not already
  enumerate; the list is appended to the 11b sub-bundle file (or
  recorded inline in the 11a status-log entry the user can move
  into 11b before 11b's session begins).
- **11b** — populate `migration.rst` plus the
  `migration/v1_to_v2.rst` page, consuming 11a's carry-forward
  list and the 11b spec.

11a depends on 11; 11b depends on 11a. Each sub-bundle runs in a
single orchestrator session.

## Why the chapter exists

The user-guide chapters (00–10, 15) describe the program as it is
*today*. They do not carry version-keyed prose ("v1.x did X, 2.0
does Y"); per the *Common conventions for bundle authors* in the
master plan, that material is concentrated here.

- The **changelog** is a forward-looking record. The 2.0.0 page
  this chapter creates is the first entry; future releases each
  get their own page under `doc/source/changelog/`. The chapter
  intro names that convention so subsequent maintainers know
  where new entries land.
- The **migration guide** is a one-shot upgrade aid for users
  coming from v1.x. It assumes the reader already knows v1.x and
  walks them through the changes that affect their workflow.

The two are written as a pair because they cover overlapping
ground from different angles: a changelog bullet says "what
changed"; the matching migration paragraph says "what you need
to do about it." Each direction cross-links to the other.

## Reader profile

- The **changelog** reader is a Blackchirp user (any version)
  who wants a scan of what is new in a given release. They are
  not necessarily a contributor and they may not read the
  developer guide; frame entries in terms of UI, workflow, file
  formats, and observable behavior, and lean toward language a
  user can act on. Backend or refactoring changes that show up
  in a release belong on the page only when they affect
  reliability or performance the user notices — internal-only
  rearrangements that do not surface in the UI are out of
  scope.
- The **migration guide** reader is specifically a v1.x user
  bringing a working installation and possibly a body of
  acquired data to 2.0. Frame the page as a checklist of upgrade
  actions, each with the v1.x starting condition, the 2.0 end
  state, and the steps to get from one to the other.

Neither audience overlaps with the developer-guide reader. The
changelog and migration guide are user-facing prose; they cite
user-guide pages, not developer-guide pages or API references.

## Sub-bundle map

| Sub-bundle | Page (under `doc/source/`) | Depends on |
|---|---|---|
| 11a | `changelog.rst`, `changelog/1.1.0.rst`, `changelog/2.0.0.rst` | — |
| 11b | `migration.rst`, `migration/v1_to_v2.rst` | 11a |

## Workflow

The changelog/migration track is **orchestrator-direct, one
sub-bundle per session**. The five-step workflow in the *Workflow*
section of `dev-docs/documentation-revision.md` applies unchanged.

The orchestrator may dispatch research agents (Sonnet or Haiku)
for context-gathering tasks: surveying
`git log 8bc115ae..HEAD --oneline` for user-visible changes,
listing every commit that touched a particular subsystem,
extracting a snippet from a dev-doc, or summarizing a long
changelog draft. Research agents return data; the orchestrator
synthesizes and writes the prose.

### Stage 5 detail for 11a's carry-forward list

If 11a discovers user-visible changes that the 11b spec does not
already enumerate (and that the user should know about when they
upgrade), 11a's status-log entry must list those items so the
next orchestrator session can incorporate them. Two options for
where the list lives:

- **Preferred:** append a short `### Carry-forward to 11b`
  section to `bundle-11b-migration-guide.md` under the existing
  *Scope* section. Stage that change as part of 11a's content
  commit (since the list is content the reader of 11b's session
  needs to see). This keeps the carry-forward visible in the
  same file 11b's orchestrator will read.
- **Fallback:** if appending to 11b feels presumptuous before
  11b runs, list the items in 11a's status-log entry and ask the
  user to move them into 11b before 11b's session begins.

The orchestrator picks whichever is cleaner for the specific
items found, and notes the choice in the handoff to the user.

## Cross-cutting conventions for sub-bundle drafters

Sub-bundle drafters (11a, 11b) **must read this section before
drafting**. These rules are stated once here and not repeated per
sub-bundle.

### Voice, tense, audience

- The changelog uses **past tense** for entries describing what
  changed in a specific release ("Replaced compile-time hardware
  selection with runtime profiles"). The chapter intro is
  present tense.
- The migration guide uses **second person, imperative present**
  ("Open the Hardware menu, click *Configure Profiles*…"). It
  addresses the v1.x user directly.
- Neither page describes development history beyond what the
  reader needs. "We rewrote X" is a development-history marker;
  "X is now configured at runtime instead of at compile time" is
  the user-visible delta and is fine.

### Per-page structure

Each release page is organized the same way:

1. A short summary paragraph framing the release.
2. A **Highlights** section with 3–5 bullets surfacing the
   largest changes.
3. **Component-level sections grouped by subsystem** — only the
   subsystems the release actually touched. The standard buckets
   are: build & distribution, hardware configuration, hardware
   drivers (Python hardware, communication, AWG/chirp, GPIB,
   per-driver behavior), acquisition and data flow, user
   interface, overlays, LIF, logging and diagnostics, file
   formats and data storage, and tooling. Merge or split as the
   change set warrants; do not list a category that has no
   entries.
4. A **Bug fixes** section, also grouped by the same subsystem
   labels, listing user-noticeable fixes (the kind the user
   would have filed, would have hit during a run, or would
   notice as restored or improved behavior on upgrade).

### What is in scope

- UI changes (new menus, dialogs, tabs; renamed widgets;
  reordered workflows).
- File-format changes (new CSV files; new columns; new on-disk
  directories).
- Build/install changes (qmake → CMake; binary distribution).
- Configuration model changes (compile-time → runtime hardware
  selection; loadouts; FTMW presets; QSettings versioning).
- New top-level features (Python hardware; overlays; viewer
  app; new acquisition modes).
- Behavior changes the user observes at runtime (parallel
  waveform processing; collapsible status boxes; theme-aware
  colors; etc.) when the change is large enough to be worth
  noting.
- Bug fixes that change observable behavior, restore a
  previously-broken feature, or fix instability the user would
  notice (crashes, hangs, races, data corruption). Group these
  under the **Bug fixes** section.

Out of scope:

- Pure refactors and internal API rearrangements that do not
  surface in the UI, output, or runtime behavior (developer
  guide territory).
- Per-class API changes (API reference territory).
- Test-only or build-script-only commits with no user
  observable effect.

### Cross-references

- Use `:doc:`path`` for whole-page cross-references.
- The changelog is the citation hub: nearly every bullet links
  to the user-guide page that documents the feature. The
  migration guide cites the changelog page once at the top and
  then cross-links into the user-guide pages as it walks each
  upgrade action.
- Neither page should link into `dev-docs/`. Dev-docs are
  temporary scaffolding; the rendered RST must be self-
  sufficient.
- The changelog and migration guide cross-reference each other:
  the migration guide's intro points to the changelog as the
  authoritative summary; the changelog's release page points to
  the migration guide as the place to learn how to upgrade.

### Source treatment

- **Commit history.** `git log <prev-release>..<this-release> --oneline`
  is the primary source. The orchestrator (or a research agent)
  walks it and builds a categorized list before drafting begins.
  Feature commits are condensed into one bullet per logical
  feature, regardless of how many commits implemented it. Bug
  fixes are also pulled in (see *What is in scope* above);
  group them under the page's **Bug fixes** section. For 1.1.0,
  the range is `8bc115ae..eec074ae` (the devel-only commits).
  For 2.0.0, the range is `eec074ae..<cmakemigration tip>` (the
  cmakemigration-only commits).
- **dev-docs.** `dev-docs/awg-marker-system.md`,
  `dev-docs/loadout-system.md`,
  `dev-docs/python-hardware.md`, etc. are useful for naming and
  framing a feature consistently with how the user guide
  describes it. Read for context only; do not link into
  `dev-docs/` from the rendered pages.
- **User-guide pages.** Each changelog bullet should cite the
  user-guide page that documents the feature. The page-name
  inventory under `doc/source/user_guide/` is the master list.
- **Existing scaffold.** `doc/source/changelog.rst` and
  `doc/source/migration.rst` are the chapter-landing files
  created by bundle 00; both are currently thin placeholders.

### Length

- The 2.0.0 changelog page can run long because it is dense
  bullets with cross-links. Aim to keep each bullet to one or
  two lines.
- The migration page should aim for tight, scannable steps —
  long enough to be complete, short enough that a v1.x user can
  walk through it linearly in one sitting.

### Index entries

Each new page begins with a `.. index::` directive. The
changelog page should index the major feature names that
readers might search for; the migration page should index the
v1.x → 2.0 transitions so a search lands on the right
checklist item.

### Screenshots

None. Both pages are prose with cross-links into the user-guide
chapters that already carry the screenshots.

## Sources for this bundle (11 itself)

### Related source files

- `doc/source/changelog.rst` — current chapter landing
  (placeholder).
- `doc/source/migration.rst` — current chapter landing
  (placeholder).
- `doc/source/index.rst` — confirm both chapters are in the
  master toctree at the expected position.

### Related dev-docs

- `dev-docs/documentation-revision.md` — master plan; the bundle
  table cell update for 11 + 11a + 11b is the housekeeping
  change for this commit.

### Related user-guide pages

None directly; this bundle's deliverable is the chapter scaffold.
The sub-bundles cite the user-guide pages.

### Related API reference pages

None.

## Sphinx file deltas

**Modified:**

- `doc/source/changelog.rst` — replace placeholder body with
  chapter intro and explicit toctree.
- `doc/source/migration.rst` — same.

**Created:**

- None directly. Sub-bundles 11a and 11b create their own pages.

## Acceptance criteria

- `doc/source/changelog.rst` carries a chapter intro describing
  the changelog's purpose, the per-release-page convention, the
  per-page section structure (Highlights → component-level
  sections by subsystem → Bug fixes), and the editorial
  conventions for adding new release pages. The intro is plain
  paragraphs above the toctree — no H2 between the intro and the
  toctree, so the per-release pages render flush at the bottom
  of the chapter landing.
- `doc/source/migration.rst` carries a 2–3 paragraph chapter
  intro describing what the migration guide is, who it is for,
  and how to use it; cross-link to the changelog as the
  authoritative summary of changes.
- Each chapter's toctree is explicit and lists its sub-bundle's
  RST file.
- The master-plan table is updated to enumerate 11 + 11a + 11b.
- This bundle does not edit any other RST file under
  `doc/source/`.
