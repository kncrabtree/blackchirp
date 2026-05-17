# Developer Guide refresh — continuation notes

Ephemeral scratchpad for the pre-2.0.0 developer-guide refresh. The
committed style rules live in `doc/AGENTS.md` and
`doc/source/developer_guide/conventions.rst`; this file is the running
plan and checklist for the pass. It is the direct successor to the
finished user-guide cleanup pass and inherits its working method.

**Cleanup-time note:** when this pass finishes and the doc is removed
from `dev-docs/`, also delete `.claude/commands/developer-guide-cleanup.md`
— the slash command exists only to re-enter this workflow across
sessions and has no purpose once the pass is done.

Work happens on the existing `user-guide-cleanup` branch (no rename;
it is a short-lived branch slated to merge back to `master` and be
discarded). Hygiene is not critical; commit in logical batches as the
user directs and only on explicit confirmation.

## State of the developer guide (entry assessment)

Reviewed at the start of this pass:

- **Prose is already disciplined.** Essentially zero first-person
  voice (only `conventions.rst`, inside rule examples); of ~28
  temporal-marker hits the large majority are legitimate runtime/state
  language ("a previously-connected device emits…") or the rule text
  itself. Internal heading hierarchy is sound. Design-rationale
  sections already exist and read well ("Why a ring buffer"). **This
  is a structure-and-consistency pass, not a heavy prose rewrite — do
  not re-write prose that is already sound.**
- **Navigation is the real problem.** The whole guide is a single
  entry (`developer_guide`) buried under the **Project Reference**
  captioned toctree, whose landing page then carries a second nested
  toctree of 17 flat, undifferentiated pages.
- **Genuine source-evolution lines are few** — sweep, do not assume
  volume. Known suspects: `python_hardware.rst:563`
  ("parameters that used to flow through the older…"),
  `persistence.rst:511` ("older fixtures may carry the legacy
  ``subKey``" — verify: this is back-compat for *reading old data*,
  likely legitimate runtime/state language, not source evolution).

## Ground rules (carried from the user-guide pass)

Same per-page discipline as the user-guide pass:

- **Strip source-evolution / temporal markers** in the
  source-evolution sense ("Phase N", "previously did X but now",
  "recently refactored", "in v1.x"). Runtime/state language stays
  ("after `initialize()` completes", "a previously-connected device",
  "older on-disk fixtures"). The five-years-from-now test from
  `AGENTS.md` applies unchanged.
- **American English, present tense, impersonal voice.** Second
  person is acceptable in the recipe/instruction pages
  (`adding_a_*`) and other explicit "do this" contexts, per
  `doc/AGENTS.md`. Already largely satisfied — do not churn.
- **Do not duplicate API-reference per-member content.** The
  developer guide carries cross-system flow and orientation; the
  per-method contract lives on the `classes/` page. Where a page
  touches a class with its own API page, give a brief orientation
  and cross-link. This is the `:ref:`api-reference-style`` contract
  in `conventions.rst`.
- **Match current code/architecture.** These pages describe live
  data flow and threading. Verify claims against the source, not a
  prior version of it. Flag and fix counts, class names, and
  signatures that have drifted.
- **Condense wide tables to definition lists** when columns are
  mostly prose; reserve tables for short tabular data.
- **Preserve `.. _labels:` referenced elsewhere.** `grep -rn
  ':ref:`<name>`' doc/source/` before renaming or removing an
  anchor; retarget every caller in the same commit. Page *paths*
  under `developer_guide/` do not change in this pass (only their
  toctree placement), so `:doc:` churn is minimal — the work is in
  `index.rst` and the landing page, not file moves.
- **`.. index::` block** on every page (most already have one;
  verify).

### The one deliberate exception: design rationale

The developer guide is the right place to explain **why** the design
is the way it is. The user-guide rule "strip apologia/marketing"
becomes here: **keep design rationale that helps a contributor reason
about the code; cut self-congratulation and change-history
narrative.** The line is timelessness, not topic:

- **Keep** — timeless rationale about the *current* design: "the FT
  runs on a worker thread because it is CPU-bound and would block the
  UI"; "a ring buffer decouples the producer from the drain loop".
- **Rewrite or cut** — rationale framed as history: "we switched from
  a queue to a ring buffer because the queue was slow"; "this was
  refactored to…". Make it timeless or remove it.

### conventions.rst is a spec, not narrative

`conventions.rst` is simultaneously a developer-guide page and the
normative contract referenced by `AGENTS.md` and other pages via
`:ref:`. Treat it as a specification: tighten prose only, preserve
every `.. _label:` (notably `api-reference-style`), and do not loosen
or restructure the rules. It is effectively out of scope for the
structural restructure; in scope only for light prose tightening and
the timeless sweep.

## Concrete defects to fix in this pass

- **Stale `api_style.rst` pointers.** `doc/AGENTS.md` (≈ lines 93,
  116) and `doc/source/AGENTS.md` reference
  `doc/source/developer_guide/api_style.rst`, which does not exist.
  The API contract lives in the `:ref:`api-reference-style`` section
  of `conventions.rst`. Retarget both pointers.
- The source-evolution lines noted in the entry assessment.

## Target navigation structure (decided)

Decisions taken with the user this session:

1. **Developer Guide becomes its own top-level band** of captioned
   sidebar sections (not nested under Project Reference), mirroring
   how the user guide reads in the sidebar.
2. **API Reference (`classes`) moves adjacent to it** — it is
   developer-facing and pairs with the guide. **Project Reference**
   is reduced to `migration` + `changelog`.
3. **Captioned subgroups**, not a flat list — multiple `:hidden:`
   captioned toctrees in `doc/source/index.rst`, exactly like the
   user-guide chapters. The `developer_guide.rst` landing page is
   **kept** (its three-audience routing prose is valuable and many
   inbound `:doc:`/developer_guide`` links resolve to it) but its
   nested toctree is dissolved into `index.rst`; it becomes the
   overview/first page of the first group (retitled as needed, no
   toctree of its own — same move as `user_guide.rst`'s removal,
   except the page itself stays as orientation).

Proposed `index.rst` toctree band (replacing the single
`developer_guide` line in Project Reference), reading order =
onboarding → core → subsystems → pipelines → recipes → python, then
the API reference:

```
:caption: Contributing
  developer_guide (overview; no toctree)
  developer_guide/conventions
  developer_guide/build_system
  developer_guide/packaging

:caption: Architecture
  developer_guide/architecture
  developer_guide/experiment_lifecycle
  developer_guide/persistence
  developer_guide/crash_handling

:caption: Hardware Subsystem
  developer_guide/hardware_configuration
  developer_guide/hardware_runtime
  developer_guide/python_hardware
  developer_guide/vendor_libraries

:caption: Acquisition Pipelines
  developer_guide/ftmw_acquisition
  developer_guide/lif_acquisition

:caption: Extending Blackchirp
  developer_guide/adding_a_driver
  developer_guide/adding_a_hardware_type
  developer_guide/adding_an_experiment_mode

:caption: Python Module
  developer_guide/python_module

:caption: API Reference
  classes

:caption: Project Reference
  migration
  changelog
```

Open sub-decisions for the fresh session to settle with the user
before editing `index.rst`:

- The first group's caption: "Contributing" with `developer_guide`
  (overview) as its lead page, vs. a one-item "Developer Guide"
  intro caption ahead of "Contributing". Recommendation:
  "Contributing" with the overview as lead page (fewer captions).
- Whether `python_module` (sole page, distinct audience) deserves
  its own caption or folds under "Contributing". Recommendation:
  keep its own caption — the landing page already frames it as a
  separate audience.
- Update `index.rst`'s "Where to start" body bullets
  (`:doc:`developer_guide``, `:doc:`classes``) to match the new
  band; keep the `developer_guide` page so those links stay valid.

## Per-page checklist

Grouped by the target cluster. Each page: timeless sweep, heading
hierarchy + `.. index::` check, API-duplication check, code-accuracy
spot-check, fix any stale cross-refs. Do not rewrite sound prose.

Working method addition (user instruction, mid-pass): for every
page touched, find the last commit that modified it before the
current pass, scan commits since then that touch the code/config the
page documents, and verify factual claims have not drifted. Surface
gaps for user review rather than silently rewriting.

Contributing: **done.**

- [x] `developer_guide.rst` (toctree dissolved into `index.rst` in
  the nav commit; three-audience prose kept; added missing
  `.. index::` block and the sidebar-grouping paragraph, reflowed)
- [x] `developer_guide/conventions.rst` (SPEC — reviewed; already
  clean, no prose change. `api_style.rst` pointers fixed in
  `doc/AGENTS.md`; `doc/source/AGENTS.md` does not exist)
- [x] `developer_guide/build_system.rst` (Packaging + CI sections
  condensed to a pointer at `:doc:\`packaging\`` per user decision;
  drift check clean — licenses/ split was repo-root not
  `python/blackchirp/`, labjack commit didn't touch the globs)
- [x] `developer_guide/packaging.rst` (drift fix: macOS bundle is
  now ad-hoc codesigned — corrected the false "not configured"
  claim and the signing table, added the symlink-collapse/codesign
  detail to the `QtDeployment.cmake` entry and the per-job
  skeleton, added the `MACOSX_DEPLOYMENT_TARGET=13.3` rationale to
  *Non-intuitive constructions*. Source: commits `cb91d54c`,
  `76639949`, `cb2d452d` postdated the page's last update)
- [x] `developer_guide/python_module.rst` (folded into Contributing;
  reviewed — clean, the 05-12 LIF-notebook `imshow` addition is
  consistent with existing prose)

Architecture:

- [ ] `developer_guide/architecture.rst` (has a Diagram section —
  verify it matches the current orchestration/threading layout)
- [ ] `developer_guide/experiment_lifecycle.rst`
- [ ] `developer_guide/persistence.rst` (check the `subKey`
  legacy-fixture line)
- [ ] `developer_guide/crash_handling.rst`

Hardware Subsystem:

- [ ] `developer_guide/hardware_configuration.rst`
- [ ] `developer_guide/hardware_runtime.rst`
- [ ] `developer_guide/python_hardware.rst` (fix the "parameters
  that used to flow through the older…" source-evolution line)
- [ ] `developer_guide/vendor_libraries.rst`

Acquisition Pipelines:

- [ ] `developer_guide/ftmw_acquisition.rst`
- [ ] `developer_guide/lif_acquisition.rst`

Extending Blackchirp:

- [ ] `developer_guide/adding_a_driver.rst`
- [ ] `developer_guide/adding_a_hardware_type.rst`
- [ ] `developer_guide/adding_an_experiment_mode.rst`

Python Module:

- [ ] `developer_guide/python_module.rst`

Navigation / cross-cutting:

- [x] `doc/source/index.rst` — replaced the Project-Reference
  `developer_guide` line with the captioned band (Contributing /
  Architecture / Hardware Subsystem / Acquisition Pipelines /
  Extending Blackchirp / API Reference / Project Reference);
  Project Reference reduced to `migration` + `changelog`; `classes`
  in its own `API Reference` caption. "Where to start" bullets left
  intact — they still resolve and read accurately (sound prose,
  structure-only discipline).
- [x] `developer_guide.rst` toctree dissolved; landing prose kept
  and given a short sidebar-grouping paragraph in place of the old
  toctree. Inbound `:doc:`/developer_guide`` links re-skimmed — all
  use absolute paths to unchanged page locations; none broken.

Sub-decisions settled with the user this session: first caption is
**"Contributing"** with the `developer_guide` overview as its lead
page; `python_module` **folds under Contributing** (no separate
caption). The `doc/source/AGENTS.md` referenced in "Concrete
defects" does not exist — the only stale `api_style.rst` pointers
were in `doc/AGENTS.md` (lines 93, 116), both retargeted to the
`api-reference-style` section of `conventions.rst`.

Single-page-section refinement (review feedback, same session;
sphinx_rtd_theme cannot collapse caption groups, so single-page
captions were eliminated instead):

- The user-guide `python` page moved out of **Inspecting Data**
  (which now holds only live in-app UI pages) and paired with
  `user_guide/viewer` under a new **Offline Analysis** caption,
  placed directly after Inspecting Data and before Data Format and
  Diagnostics. The old single-page **Blackchirp Viewer** caption is
  gone.
- `classes` folded into the **Architecture** caption (appended after
  `crash_handling`); the standalone **API Reference** caption is
  gone. `developer_guide.rst`'s sidebar-grouping paragraph updated
  to note the API reference now sits under Architecture.
- **Project Reference** retitled **Version History** (still
  `migration` + `changelog`).

Resulting band order: Getting Started / Hardware Setup / Running
Experiments / Inspecting Data / Offline Analysis / Data Format and
Diagnostics / Contributing / Architecture / Hardware Subsystem /
Acquisition Pipelines / Extending Blackchirp / Version History. No
single-page captions remain. Clean build, known nbsphinx warning
only.

## Diagrams / architecture accuracy

The developer guide has effectively no screenshots; the user-guide
"Screenshots to refresh" tracking is replaced by diagram/architecture
accuracy:

- [ ] `architecture.rst` "Diagram" section — confirm the rendered
  diagram (and the Orchestration-singletons / Threading-model prose)
  reflects the current `MainWindow`/manager wiring.
- [ ] Any ASCII / mermaid / data-flow figures in
  `ftmw_acquisition.rst`, `lif_acquisition.rst`,
  `experiment_lifecycle.rst`, `persistence.rst` — spot-check against
  current class names and signatures (recent renames, e.g.
  `FtmwScope`→`FtmwDigitizer`, must be reflected).

## Build / verify

```
touch doc/source/index.rst && conda run -n breathe cmake --build build --target docs
```

`touch index.rst` forces a toctree re-evaluation (mandatory after the
`index.rst` restructure). Check `doxygen.log` only if a page's
API-reference directive set changes. Expect the single known
`nbsphinx` warning; anything else is new and must be resolved.

## Reference

- `doc/AGENTS.md` — style rules, cross-reference conventions,
  index entries. (Fix its `api_style.rst` pointer in this pass.)
- `doc/source/developer_guide/conventions.rst` — the prose +
  API-reference-style contract; `:ref:`api-reference-style``.
- Method reference: the finished user-guide pass on the same branch
  (commits from `021bdc83` through the `Clean up the Blackchirp
  Viewer guide` commit) — same working rhythm: assess, surface
  decisions, edit, clean build, update this checklist, commit in
  batches on explicit confirmation.
