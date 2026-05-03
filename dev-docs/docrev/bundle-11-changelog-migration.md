# Bundle 11 — Changelog and Migration Guide (Chapter Umbrella)

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Builds the chapter scaffolds for the **changelog** and the
**migration guide** and lands the chapter-level intros. The
substantive content is split across two sub-bundles:

- **11a** — populate `changelog.rst` plus the per-release page
  `changelog/2.0.0.rst` that summarizes what is new in 2.0.0
  *from the user's perspective*. As 11a writes that page it also
  builds a **carry-forward list** of items the migration guide
  will need that the 11b spec does not already enumerate; the list
  is appended to the 11b sub-bundle file (or recorded inline in
  the 11a status-log entry the user can move into 11b before
  11b's session begins).
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

- The **changelog** reader, *for the 2.0.0 page specifically*,
  is a current Blackchirp user (any version) who wants a quick
  scan of what is new from their seat. They are not a
  contributor; they will not read the developer guide. Frame
  2.0.0 entries in terms of UI, workflow, file formats, and
  observable behavior. "From a v1.0 user's perspective, what is
  different?" is the test. The 2.0.0 release is the exception:
  the change set is large enough that a developer-oriented log
  on the same page would drown out the user-visible content.
  **For every release after 2.0.0, the changelog page addresses
  developers as a secondary audience** — backend / non-user-
  visible changes (refactors, internal API rearrangements,
  threading or storage rework that does not surface in the UI)
  get their own short section on each release page so that
  contributors and integrators can scan a release page for
  relevant context. The chapter intro in `changelog.rst` should
  state this convention explicitly so future maintainers know
  the 2.0.0 framing is the one-off and that subsequent pages
  carry both audiences.
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
| 11a | `changelog.rst`, `changelog/2.0.0.rst` | — |
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
  selection with runtime profiles"). The chapter intro and the
  "best practices" note are present tense.
- The migration guide uses **second person, imperative present**
  ("Open the Hardware menu, click *Configure Profiles*…"). It
  addresses the v1.x user directly.
- Neither page describes development history beyond what the
  reader needs. "We rewrote X" is a development-history marker;
  "X is now configured at runtime instead of at compile time" is
  the user-visible delta and is fine.

### What "user-visible" means

In scope for both pages:

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

Out of scope for both pages:

- Bug fixes that do not change documented behavior. A user-
  visible bug fix (a previously broken feature that now works)
  may merit a one-line changelog entry; routine fixes do not.
- Internal refactors that do not change observable behavior
  (developer guide territory).
- Per-class API changes (API reference territory).

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

- **Commit history.** `git log 8bc115ae..HEAD --oneline` is the
  primary source. The orchestrator (or a research agent) walks
  it and builds a categorized list before drafting begins. Pure
  bug-fix commits are dropped; feature commits are condensed
  into one bullet per logical feature, regardless of how many
  commits implemented it.
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

- `doc/source/changelog.rst` carries a 2–3 paragraph chapter
  intro describing the changelog's purpose, the per-release
  page convention, the dual user/developer audience rule for
  releases after 2.0.0 (and the 2.0.0-page user-only
  exception), and a one-paragraph "best practices for keeping
  the changelog updated" note.
- `doc/source/migration.rst` carries a 2–3 paragraph chapter
  intro describing what the migration guide is, who it is for,
  and how to use it; cross-link to the changelog as the
  authoritative summary of changes.
- Each chapter's toctree is explicit and lists its sub-bundle's
  RST file.
- The master-plan table is updated to enumerate 11 + 11a + 11b.
- This bundle does not edit any other RST file under
  `doc/source/`.
