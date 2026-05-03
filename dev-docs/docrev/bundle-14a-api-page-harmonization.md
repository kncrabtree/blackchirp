# Bundle 14a — API page intro / header-comment harmonization

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Final Consistency Pass. Re-balances class-level
orientation between Doxygen `\brief` blocks in C++ headers and
the `.rst` page under `doc/source/classes/`, then writes the
rule into `api_style.rst` so future API edits do not drift
back.

## Background

Earlier API bundles (notably 13a `SettingsStorage`) put the
bulk of the class-level prose in the header's Doxygen block and
left the `.rst` page short. Later bundles (13e onward,
formalized in 13f) moved orientation prose to the `.rst` and
trimmed the header's class-level block to a tight `\brief`
plus internals-as-needed. The split this sub-bundle locks in:

- The **header `\brief` block** stays tight: one or two
  sentences naming what the class is and its primary
  collaborator(s). Anything that would benefit a future
  contributor reading the header in isolation is fine
  (lifecycle invariants, ownership rules, threading caveats);
  anything that reads as Sphinx-page prose moves out.
- The **per-method `///` blocks** in the header remain rich:
  one-line `\brief`, parameter and return docs, sample-rate
  / unit / range notes where relevant. These are the per-
  method API contract and Breathe surfaces them on the
  `.rst` page.
- The **`.rst` page** carries the plain-language orientation:
  what the class does in the system, who its collaborators
  are, the lifecycle the reader needs in mind to read the
  member list usefully, and any cross-references to user-
  guide or developer-guide pages that contextualize the
  class.

This sub-bundle is the only one in bundle 14 that involves
prose writing.

## Scope

1. **Walk every page under `doc/source/classes/`.** For each
   page, decide whether the class-level orientation is on the
   `.rst` page (the convention going forward) or in the
   header. If on the header, move the substantive prose to
   the `.rst` page and trim the header `\brief` block.
2. **Update `doc/source/developer_guide/api_style.rst`** to
   make the split explicit: the `.rst` carries orientation;
   the header `\brief` stays tight; per-method `///` blocks
   remain the per-method contract.
3. **Authorized header edits.** When prose moves out of a
   header, edit the header in the same content commit. The
   header edits are paired with the corresponding `.rst`
   edit; do not edit a header without the matching page
   change.

The orchestrator may dispatch a research agent to enumerate
which pages currently lean on the header for class-level
prose. The agent returns a list (one row per page: page path,
header path, where the orientation lives today). The
orchestrator decides per-row whether a move is warranted.

## Out of scope

- Per-method Doxygen comment edits beyond what is necessary
  to keep cross-references working after the orientation
  move.
- Adding new content to `.rst` pages beyond what already
  exists in the header (the goal is a relocation, not a
  rewrite).
- Restructuring the `.rst` page beyond what the relocation
  requires.
- Touching pages that already follow the convention.
- Touching `doc/source/developer_guide/` pages (those are
  developer-guide chapter content, not API pages).

## Sources

### Related source files

- `doc/source/classes/*.rst` — every API page.
- `doc/source/developer_guide/api_style.rst` — the rule the
  sub-bundle hardens.
- `src/**/*.h` — every header for which the corresponding
  `.rst` page exists; only the class-level block (the
  `/*! ... */` immediately preceding the class declaration)
  is in scope.

### Related dev-docs

None directly.

### Related user-guide pages

None directly. API pages may cross-link into the user guide;
existing cross-references are preserved as-is.

### Related API reference pages

The page set under `doc/source/classes/` is itself the work
unit.

## Sphinx file deltas

**Modified:**

- Selected pages under `doc/source/classes/` (the ones whose
  orientation needs to move).
- `doc/source/developer_guide/api_style.rst`.

**Authorized header edits:**

- The class-level Doxygen block in each header whose `.rst`
  page receives relocated prose. Per-method `///` blocks are
  not edited.

**Created:**

- None.

## Acceptance criteria

- Every page under `doc/source/classes/` carries its class-
  level orientation on the `.rst` page; the corresponding
  header `\brief` block is tight (1–2 sentences plus any
  internals notes a header reader genuinely needs).
- `api_style.rst` documents the split as the canonical rule.
- No per-method Doxygen comment is changed beyond what the
  orientation move required.
- Build is clean: no broken cross-references.
