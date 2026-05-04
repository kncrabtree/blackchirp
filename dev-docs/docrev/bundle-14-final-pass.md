# Bundle 14 — Final Consistency Pass (Chapter Umbrella)

**Status:** complete

<!--
Status log:
- 2026-05-03 — not started → complete. The umbrella's three
  acceptance criteria were already satisfied at the time of
  this status flip: the master-plan table enumerates 14 and
  14a–14e (commit 5ec30d8c, "Restructure documentation revision
  plan: split bundles 11 and 14"); the five sub-bundle files
  exist under dev-docs/docrev/ with their standard status
  headers and scope detail (same commit); and the umbrella
  authorizes no RST edits, so doc/source/ is untouched. This
  bundle therefore has no separate stage-1 content commit; the
  status flip is recorded as a single tracking commit.
-->

The cleanup pass run after the bulk of the documentation has
landed. Five mechanical-or-coordinated sweeps that each tighten
a different cross-cutting consistency dimension: API page
intro/header harmonization, screenshot sizing, file
organization, American-English spelling, and
implementation→driver terminology.

This bundle itself is a small chapter-scaffold commit; it has
no rendered output beyond updating the master plan to enumerate
the 5 sub-bundles. The substantive work happens in 14a–14e.

## Sub-bundle map

| Sub-bundle | Scope | Depends on |
|---|---|---|
| 14a | API page intro / header-comment harmonization | — |
| 14b | Screenshot sizing pass | 14 |
| 14c | File organization & menu layout audit | 14 |
| 14d | American English sweep | 14 |
| 14e | Implementation → driver terminology sweep | 14 |

**14a runs first** because it is the only sub-bundle that
involves new prose writing — re-balancing class-level
orientation between Doxygen `\brief` blocks and the `.rst`
page, and updating `api_style.rst` with the rule it enforces
so future API edits do not drift back. 14b–14e are mechanical
sweeps and can be tackled in any order after 14a; they do not
depend on each other.

## Workflow

The final-pass track is **orchestrator-direct, one sub-bundle
per session**. The five-step workflow in the *Workflow* section
of `dev-docs/documentation-revision.md` applies unchanged.

The orchestrator is **encouraged to dispatch research agents**
for the mechanical sweeps. 14b–14e are well-suited to this
pattern:

- 14b can dispatch a research agent to walk every `.. image::`
  / `.. figure::` directive and report the native pixel width
  of each referenced PNG plus the current `:width:` setting.
  The orchestrator decides which directives need editing and
  applies the changes.
- 14c can dispatch a research agent to enumerate the current
  toctree structure, the cross-link directionality between
  pages, and any orphaned files. The orchestrator decides
  whether reorganization is warranted.
- 14d can dispatch a research agent to grep for British
  spellings in `.rst` files and return file/line/spelling
  triples. The orchestrator confirms each match is in prose
  (not a code block or identifier) and applies the replacement.
- 14e can dispatch a research agent to find every prose
  occurrence of "implementation" in user-guide and API pages.
  The orchestrator decides per-occurrence whether the
  replacement applies.

14a is more synthesis-heavy and stays primarily in the
orchestrator's hands, though it may dispatch a research agent
to enumerate which API pages currently lean on the header for
class-level prose vs. on the `.rst` page.

## Cross-cutting conventions for sub-bundle drafters

- **Scope discipline.** Each sub-bundle stays inside its
  declared dimension. 14d does not relocate files; 14c does
  not change spellings; etc. Cross-track issues surfaced by
  one sub-bundle are flagged to the user and held for a later
  sub-bundle's session, not opportunistically fixed.
- **No new prose unless the sub-bundle authorizes it.** 14a
  authorizes prose writing on API pages and `api_style.rst`;
  14b–14e are pure mechanical edits. If a mechanical-sweep
  sub-bundle finds a prose gap, flag it; do not write fresh
  prose to fill it.
- **Build after each sub-bundle.** Even mechanical sweeps can
  break Sphinx (a renamed file breaks every cross-reference;
  a regex-replaced spelling can land inside a code block by
  accident). The build step in stage 3 of the workflow is
  non-optional.
- **One sub-bundle per session, one stage-1 commit per
  sub-bundle.** Even though 14b–14e are mechanical, they
  produce a meaningful commit log entry each ("Standardize
  user-guide screenshot sizing", "Harmonize American English
  spelling in user-guide prose", etc.). Do not bundle multiple
  sub-bundles into one commit.

## Sources for this bundle (14 itself)

- `dev-docs/documentation-revision.md` — master plan; the
  bundle table cell update for 14 + 14a–14e is the
  housekeeping change for this commit.

## Sphinx file deltas

None. This bundle is the chapter-scaffold + master-plan
update; it does not produce any rendered RST.

## Acceptance criteria

- The master-plan table enumerates 14 + 14a–14e with effort
  and dependency columns.
- Each sub-bundle file (14a–14e) exists under
  `dev-docs/docrev/` with its standard status header and
  scope detail.
- This bundle does not edit any RST file under
  `doc/source/`.
