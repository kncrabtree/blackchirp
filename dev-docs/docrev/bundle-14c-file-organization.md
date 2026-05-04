# Bundle 14c — File organization & menu layout audit

**Status:** complete

<!--
Status log:
- 2026-05-03 — not started → complete (content commits 751bcbbb +
  6b6773af). Audit performed via research-agent enumeration of every
  user-guide page, every toctree directive, the cross-link map
  between pages, and orphan/double-listed checks. Three concrete
  improvements applied:
  (1) Moved experiment/chirp_setup.rst and experiment/digitizer_setup.rst
      to ftmw_configuration/. They were already toctree'd from
      ftmw_configuration.rst, not experiment_setup.rst — the directory
      path no longer matched the chapter assignment. 17 :doc: and
      :ref: cross-references updated across 9 files spanning user
      guide pages, the FtmwConfig and ChirpConfig API class pages,
      and the 2.0.0 changelog.
  (2) Reordered the top-level user_guide.rst toctree to match new-
      user reading flow: getting-started orientation up front,
      hardware setup as a contiguous block, FTMW configuration and
      experiment setup before the during/after-acquisition tabs.
  (3) Split the flat top-level toctree into five captioned
      toctrees (Getting Started, Hardware Setup, Running Experiments,
      Inspecting Data, Modules) so the chapter landing renders
      visible section headings around the same 20 pages in the same
      order. The sphinx_rtd_theme sidebar does not propagate
      captions from chapter landings, so the sidebar stays flat;
      the structure shows up on user_guide.html.
  Two pre-existing broken :doc: targets fixed opportunistically
  (hwdialog.rst → communicationdialog became a :ref: into
  hardware_menu.rst; installation.rst → hardware_configuration
  became hardware_config). Items considered and held for follow-up:
  the hw/ → hardware_details/ rename (modest reader benefit vs.
  external-bookmark cost was judged borderline). Items considered
  and rejected: stub consolidation in experiment/ (each "stub" is
  actually a complete self-contained page on its own topic).
  Build is clean — every remaining warning is pre-existing
  (autosectionlabel duplicates in classes/, ipython3 lexer hits,
  python-hardware-custom-protocol and rf-configuration ref-target
  gaps in pages outside 14c scope). The 6b6773af follow-up commit
  is layered on the same in-flight working tree as the 751bcbbb
  content commit and shares its scope; recording 751bcbbb as the
  primary stage-1 hash for the master-plan table.
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
