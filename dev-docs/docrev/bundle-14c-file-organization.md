# Bundle 14c — File organization & menu layout audit

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Final Consistency Pass. Audits whether the
`doc/source/user_guide/` directory layout and toctree structure
still match the final shape of the documentation, and applies
any concrete improvements that surface.

## Scope

Reassess the `doc/source/user_guide/` directory structure and
the top-level `index.rst` toctrees against the documentation
as it actually exists today. Candidates to reconsider (the
exact list of action items will depend on what the audit
surfaces):

- Whether the `experiment/` subdirectory should remain (or its
  contents promoted) given that several pages there are now
  subpages of `ftmw_configuration` rather than
  `experiment_setup`.
- Whether the `hw/` subdirectory should sit under
  `hardware_details` as a logical sub-tree or stay flat.
- Whether toctree ordering and `:caption:` choices in the
  top-level `index.rst` and any `:hidden:` toctrees match the
  reading order a new user would expect.
- Whether any pages are orphaned (not referenced from any
  toctree) or double-listed.

The goal is a navigation tree that matches how a reader would
actually browse the manual. **Do not rearrange files unless a
concrete improvement is identified** — the bar is "this is
better," not "this is different."

## Approach

1. **Dispatch a research agent** to enumerate the current
   structure: every directory under `doc/source/user_guide/`,
   every toctree directive in `doc/source/index.rst` and any
   chapter landings, and every page that is referenced from
   no toctree (orphans) or from more than one (double-listed).
2. **Orchestrator reads** the audit and decides which moves,
   if any, are warranted. Each proposed move is evaluated
   for:
   - reader benefit (does it match how the user would browse?)
   - cross-link cost (how many `:doc:` references break?)
   - reader confusion (do bookmarks to the old path break?)
3. **Apply the moves** atomically: rename or move files,
   update every cross-reference, re-run the build to confirm
   no link is broken.
4. **If the audit finds nothing worth changing, that is a
   valid outcome.** The sub-bundle still ends with a status-
   log entry summarizing the audit and noting that no moves
   were warranted.

## Out of scope

- Renaming pages purely for stylistic reasons (no concrete
  reader benefit).
- Adding new pages or expanding existing ones.
- Changing prose content of pages beyond what a rename
  requires (cross-link updates, anchor renames).
- Audit of `doc/source/classes/`, `doc/source/developer_guide/`,
  `doc/source/changelog/`, `doc/source/migration/` — those
  are out of scope; this sub-bundle is the user-guide audit.

## Sources

### Related source files

- `doc/source/index.rst` — top-level toctree.
- `doc/source/user_guide.rst` — chapter landing.
- `doc/source/user_guide/**/*.rst` — every user-guide page.
- Subdirectory layout under `doc/source/user_guide/`.

### Related dev-docs

None directly. The user-guide track's bundle files document
the original placement decisions; the audit may reference them
for context but does not need to.

### Related user-guide pages

The page set under `doc/source/user_guide/` is itself the
work unit.

### Related API reference pages

None.

## Sphinx file deltas

**Modified:**

- `doc/source/index.rst` and any chapter landing whose
  toctree changes.
- Any page whose `:doc:` cross-reference targets a moved
  file.

**Renamed / moved:**

- Whatever the audit determines is warranted.

**Created / deleted:**

- None unless the audit identifies an orphan to remove.

## Acceptance criteria

- The audit has been performed and its findings are recorded
  in the status-log entry.
- Any moves that landed are paired with cross-reference
  updates so no `:doc:` reference is broken.
- Build is clean.
- The status-log entry justifies each move with a one-line
  reader-benefit rationale, or, if no moves landed, states
  that the audit found nothing warranted.
