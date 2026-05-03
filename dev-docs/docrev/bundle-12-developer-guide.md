# Bundle 12 — Developer Guide (Chapter Umbrella)

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Builds the Developer Guide chapter scaffold and lands the chapter-level
intro. The substantive content is split across sub-bundles 12a–12n,
each of which produces one Sphinx page under
`doc/source/developer_guide/`.

This bundle is independent of the user-guide track (00–11) and the API
reference track (13a–13i). It can be tackled in parallel with either.

## Why the chapter exists

The Developer Guide chapter exists to cover what the **API reference**
and the **user guide** cannot:

- The **API reference** documents individual classes — what each method
  does, what its parameters mean, what invariants it preserves. It is
  generated from Doxygen comments in headers and surfaced through
  Sphinx via Breathe. A reader looking for "what does class X do" goes
  there.
- The **user guide** documents how to operate the program — how to
  configure hardware, set up experiments, view data. A reader who is
  *using* Blackchirp goes there.
- The **developer guide** documents the things that span both: the
  build system that produces the binaries, the conventions that hold
  the code together, the architecture and threading layout, the
  cross-manager experiment lifecycle, the data-flow pipelines for
  FTMW and LIF, the persistence model, and the "how to add" recipes
  that walk a contributor through a multi-file change.

A useful litmus test for any candidate developer-guide topic: does it
require coordination across multiple files or subsystems to explain?
If the answer is no, it probably belongs in the API reference (one
class) or the user guide (one user task). If the answer is yes, it
belongs here.

## Reader profile

The target audience is a contributor with strong C++/Qt skills who is
new to Blackchirp's source tree. Topics assume Qt6 fluency
(`QObject`, signal/slot, `QThread`, `QtConcurrent`, `QSettings`,
metaobject system) but do not assume any prior knowledge of
Blackchirp's architecture.

## Sub-bundle map

Each sub-bundle below produces one Sphinx page. Dispatch is direct
(no drafter/verifier subagents) — the orchestrator reads this umbrella
plus the sub-bundle file plus the listed sources, then writes the page.

| Sub-bundle | Page (under `doc/source/developer_guide/`) | Depends on |
|---|---|---|
| 12a | `build_system.rst` | — |
| 12b | `conventions.rst` | — |
| 12c | `architecture.rst` | — |
| 12d | `hardware_configuration.rst` | 12c |
| 12e | `hardware_runtime.rst` | 12d |
| 12f | `experiment_lifecycle.rst` | 12e |
| 12g | `ftmw_acquisition.rst` | 12f |
| 12h | `lif_acquisition.rst` | 12f |
| 12i | `persistence.rst` | 12f |
| 12j | `python_hardware.rst` | 12e |
| 12k | `vendor_libraries.rst` | 12e |
| 12l | `adding_a_driver.rst` | 12e |
| 12m | `adding_a_hardware_type.rst` | 12e, 12j |
| 12n | `adding_an_experiment_mode.rst` | 12f |

Dependencies are read-order dependencies. A sub-bundle's page should
be drafted *after* its dependencies have landed (or at minimum after
the dependent sub-bundle's RST has been written), because the later
page expects to cross-reference the earlier one with `:doc:` and
relies on its terminology.

## What this bundle (12) produces

This bundle is the chapter-level scaffold. Its deliverables are
small:

1. **Re-shape `doc/source/developer_guide.rst`** from the current
   thin landing page into the chapter's intro. The page should:

   - Open with a 2–3 paragraph welcome that mirrors *Why the chapter
     exists* and *Reader profile* from this bundle file (rephrased
     for end-user prose, no bundle/dev-doc references).
   - Replace the current `:glob:` toctree with an explicit one that
     lists every sub-bundle's RST file in the order shown in the
     *Sub-bundle map* above. Use `:maxdepth: 2` and a meaningful
     `:caption:` (e.g., `Developer Guide`).
   - Cross-link forward to `:doc:`/classes`` (API reference) and
     `:doc:`/user_guide`` (user guide) so readers landing here from
     a search know where the adjacent chapters live.

2. **Delete `doc/source/developer_guide/overview.rst`.** It is a
   placeholder that will be superseded by the chapter intro on
   `developer_guide.rst`. Remove it from any toctree references at
   the same time. (`api_style.rst` stays — it is the API-reference
   style contract that bundle 14 hardens.)

3. **Confirm the `:glob:` toctree is gone.** Once explicit entries
   are in place, glob expansion would silently double-list pages or
   pick up the unwanted `overview.rst` placeholder. Replace, do not
   coexist.

The 14 sub-bundle pages themselves are *not* produced here — each
sub-bundle owns its own page.

## Cross-cutting conventions for sub-bundle drafters

Sub-bundle drafters (12a–12n) **must read this section before
drafting** their page. These rules are stated once here and not
repeated per sub-bundle.

### Voice, tense, audience

- Present tense, timeless prose. Describe how the system *works
  today*. No development-history markers ("now", "currently",
  "recently", "Phase X", "we recently changed", "added in v1.2").
  Markers describing *runtime* program state are fine
  ("after `initialize()` completes", "while the experiment is
  acquiring") — see the *Critical Rules* section of `CLAUDE.md`.
- Address the contributor directly when giving recipes ("Subclass
  `HardwareObject`. Call `REGISTER_HARDWARE_META` …"). Avoid
  passive-voice circumlocutions when imperative instructions are
  clearer.
- Assume Qt6 fluency. Do not explain `QObject`/signal-slot/`QThread`
  basics; a one-line refresher is fine where a Blackchirp-specific
  twist applies.

### American English

`normalize`, `behavior`, `color`, `visualize`, `randomize`,
`initialize`, `analyze`, `co-averaging`. Match UI labels exactly
when quoting them. Match identifier names exactly when quoting them
(do not "correct" a British spelling that exists in the code; flag
it for a follow-up rename if it bothers you).

### Cross-references

- Use `:doc:`path`` for whole-page cross-references.
- Use `:cpp:class:`Foo``, `:cpp:func:`Foo::bar()``, `:cpp:struct:`,
  `:cpp:enum:` when referring to C++ symbols mid-prose. Breathe
  resolves these against the Doxygen XML.
- Use `:ref:`anchor`` for in-page or sibling-page anchors when they
  exist (e.g., `:ref:`acquisitionmanager-state-machine``).
- **Never** embed raw HTML links (`<page.html>`-style) or Markdown
  link syntax.
- The chapter assumes the reader has access to the API reference; do
  not duplicate class-level prose. Where a topic touches a class
  with an existing API page, write a one-paragraph orientation that
  names the class in `:cpp:class:` form and cross-link the API page
  with `:doc:`. The API page carries the per-method detail; the
  developer-guide page carries the cross-system flow.

### Source treatment

Each sub-bundle file lists four buckets of sources:

- **Related source files** — actual `.h`/`.cpp`/`.cmake` files. The
  drafter is expected to read these and run additional
  `grep`/`find`/`codebase-memory` queries until every claim in the
  draft can be supported from the code.
- **Related dev-docs** — files under `dev-docs/`. **Read for context
  only.** Dev-docs are temporary scaffolding that will be removed
  when the documentation project completes. The rendered RST page
  must be self-sufficient and **must not contain any link, `:doc:`
  reference, or prose mention pointing into `dev-docs/`**. Any
  sentence shaped "see `dev-docs/foo.md` for details" is a bug.
  Extract the necessary detail into the page or omit the claim.
- **Related user-guide pages** — files under
  `doc/source/user_guide/`. Cross-link with `:doc:` when forwarding
  the reader to user-facing operational detail (e.g., from a
  developer-guide hardware page to the user-guide hardware-config
  walkthrough). Do not duplicate user-task instructions.
- **Related API reference pages** — files under
  `doc/source/classes/`. Cross-link with `:doc:` when forwarding to
  class-level detail. The API ref carries the per-method contract;
  the developer guide carries the cross-system flow.

If the listed sources are **insufficient** to explain a topic the
sub-bundle includes in scope — i.e., the drafter encounters a gap
between what the page should say and what the sources support —
**flag it in the orchestrator's hand-off report**. Do not fabricate.
Possible gaps include: a system whose only documentation was a
since-removed dev-doc; a user-guide page whose terminology has
shifted; an API page whose orientation prose contradicts the code.

### Doxygen comments are not in scope

Sub-bundles in this chapter **do not edit C++ headers**. The API
reference style (Doxygen `\brief` blocks, `///` triple-slash
comments) is the contract of bundles 13a–13i and is enforced by
`doc/source/developer_guide/api_style.rst`. Developer-guide pages
may *cite* Doxygen tags or `\brief` lines as illustration, but do
not modify them.

### Exception: a single source-tree change is permitted per sub-bundle

When a sub-bundle's research surfaces a clearly-dead piece of code
or configuration (the `BC_ALLHARDWARE` cmake option in 12a is the
canonical case — a leftover from the qmake-era compile-time
hardware-selection model that no longer affects the build), **the
sub-bundle may delete it as part of the same commit** that lands
the RST page. The criterion is: removal must be uncontroversial
and confined to a single conceptual change. If the contemplated
removal touches more than ~50 lines or affects multiple
subsystems, scope it out as a separate task and flag it for user
consideration instead.

### File budget

Each sub-bundle produces **one** RST file under
`doc/source/developer_guide/` (plus the toctree update on
`developer_guide.rst`, already established by this umbrella).
Sub-bundles do not create new top-level chapter pages.

### Length

Aim for ~1500–3500 words of RST per sub-page. Pages can run shorter
when the topic is genuinely narrow, longer when it is genuinely
load-bearing. Do not pad. Diagrams (Mermaid blocks, ASCII art
inside `.. code-block:: text`) are encouraged where they replace
prose.

### Index entries

Each new page begins with a `.. index::` directive listing the key
developer-facing terms it introduces. Match the conventions
already in use on existing API pages (see e.g. the index blocks on
`doc/source/classes/hardwaremanager.rst`).

### Screenshots

None for the developer guide. ASCII or Mermaid diagrams are fine
inline; no `.. image::` or `.. figure::` directives.

## Sub-bundle file format

Each `dev-docs/docrev/bundle-12<X>-<topic>.md` carries the standard
status header followed by sections for *Scope*, *Out of scope*,
*Sources* (with the four buckets above), *Sphinx file deltas*,
*Page structure*, and *Acceptance criteria*. The sub-bundle file is
self-contained for orchestrator dispatch but **assumes the
orchestrator has read this umbrella first** for the cross-cutting
conventions.

## Sources for this bundle (12 itself)

### Related source files

- `doc/source/developer_guide.rst` — current chapter landing.
- `doc/source/developer_guide/overview.rst` — placeholder to delete.
- `doc/source/index.rst` — confirm `developer_guide` is in the
  master toctree at the expected position.
- `CLAUDE.md` (project root) — *Critical Rules* on timeless prose.

### Related dev-docs

- `dev-docs/documentation-revision.md` — master plan; cross-reference
  the bundle 12 → 12 + 12a–12n table update is a separate
  housekeeping task (see *Master plan update*, below).

### Related user-guide pages

- `doc/source/user_guide.rst` and its sub-pages — for terminology
  alignment when the developer-guide intro mentions user-facing
  concepts in passing.

### Related API reference pages

- `doc/source/classes.rst` — for the forward-link from the chapter
  intro.
- `doc/source/developer_guide/api_style.rst` — already exists and
  governs the API reference; the chapter intro forward-links to it
  so readers know how the API pages are built.

## Sphinx file deltas

**Modified:**

- `doc/source/developer_guide.rst` — replace the placeholder body
  with the chapter intro and explicit toctree.

**Deleted:**

- `doc/source/developer_guide/overview.rst`.

**Created:**

- None directly. Each sub-bundle creates its own RST file.

## Toctree delta

In `doc/source/developer_guide.rst`, replace the existing toctree
with the explicit list:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/build_system
   developer_guide/conventions
   developer_guide/architecture
   developer_guide/hardware_configuration
   developer_guide/hardware_runtime
   developer_guide/experiment_lifecycle
   developer_guide/ftmw_acquisition
   developer_guide/lif_acquisition
   developer_guide/persistence
   developer_guide/python_hardware
   developer_guide/vendor_libraries
   developer_guide/adding_a_driver
   developer_guide/adding_a_hardware_type
   developer_guide/adding_an_experiment_mode
   developer_guide/api_style
```

`api_style.rst` stays at the bottom of the toctree as the
API-reference style contract; it is not authored as part of this
chapter but is logically part of the developer guide.

The sub-bundle toctree entries reference files that do **not yet
exist** when this umbrella bundle lands. That is acceptable: Sphinx
will warn about missing files, the warnings will resolve as each
sub-bundle lands, and the explicit toctree avoids the silent
double-listing risk that `:glob:` expansion produces. If the warning
noise is undesirable, comment out the not-yet-written entries and
uncomment them as each sub-bundle lands.

## Master plan update

Update the bundle table at the top of
`dev-docs/documentation-revision.md`: the single `12 | Developer
Guide` row is replaced with rows for `12 | Developer Guide chapter
intro` plus rows for each `12a`–`12n` sub-bundle. Keep effort
estimates honest (12 itself is S; sub-bundles are mostly M, with
12d/12e/12f/12j leaning toward L). This is a small mechanical edit
and is in scope for the umbrella commit.

## Acceptance criteria

- `doc/source/developer_guide.rst` carries a 2–3 paragraph chapter
  intro that frames the developer guide's audience and its
  relationship to the API reference and the user guide.
- `doc/source/developer_guide/overview.rst` is deleted; no
  references to it remain in the docs tree.
- The chapter toctree is explicit (not `:glob:`) and lists every
  sub-bundle page in the order in the *Sub-bundle map*.
- `api_style.rst` continues to render in the developer-guide chapter.
- The master-plan table in `dev-docs/documentation-revision.md` is
  updated to enumerate 12 + 12a–12n with effort and dependency
  columns.
- This bundle does not edit any RST file under
  `doc/source/developer_guide/` other than the chapter landing,
  and does not edit any source file under `src/`.
