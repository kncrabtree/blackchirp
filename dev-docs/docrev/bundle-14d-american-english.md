# Bundle 14d — American English sweep

**Status:** complete

<!--
Status log:
- 2026-05-03 — not started → complete (content commit
  8a3e847e). Converted 44 prose occurrences across 24 RST files
  in the user guide, developer guide, and migration guide:
  initialisation/initialises/re-initialise → initialization/
  initializes/re-initialize; summarise(s) → summarize(s);
  organis(ed) → organiz(ed); optimised → optimized; digitised →
  digitized; specialised → specialized; maximising → maximizing;
  customised → customized; labelled → labeled (12 sites);
  cancelled → canceled (2 sites); modelled → modeled; greyed →
  grayed; behaviour → behavior (8 sites); colours → colors;
  coaverage* → co-average* (5 sites). Borderline call: a
  Mermaid diagram label in developer_guide/hardware_configuration
  (`Static initialisation` → `Static initialization`) was
  included because the text is rendered to the user even though
  the directive isn't `.. code-block::`. Authorized source-tree
  edits made in the same pass: brace-initialised → brace-
  initialized in src/hardware/core/hardwareregistration.h
  Doxygen comment; initialises → initializes in two Doxygen
  comments in src/data/experiment/rfconfig.h and one in
  src/gui/widget/customprotocolwidget.h; recognised →
  recognized in a user-visible HTML message string in
  src/hardware/library/labjacklibrary.cpp. The `Cancelled`
  enum/class identifiers (overlay processing) and other
  `cancelled`/`cancellation` source-code uses were left as-is per
  user direction (American English accepts both, and these often
  follow Qt API style). Build clean: 103 warnings, all
  pre-existing, none introduced by this sweep.
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Final Consistency Pass. Walks every prose
file under `doc/source/` for British English spellings and
converts them to American English per the project's
*American English spelling* convention.

## Scope

Walk every `.rst` file under:

- `doc/source/user_guide/` (and subdirectories)
- `doc/source/developer_guide/`
- `doc/source/classes/`
- `doc/source/changelog/`
- `doc/source/migration/`

Convert prose occurrences:

- `-ise` / `-isation` → `-ize` / `-ization`: normalize,
  organize, visualize, optimize, randomize, initialize,
  summarize, categorize, customize, recognize, etc.
- `-yse` → `-yze`: analyze, catalyze.
- `colour` → `color`.
- `behaviour` → `behavior`.
- `centre` → `center`.
- `fibre` → `fiber`.
- `metre` → `meter`.
- `dialogue` → `dialog` only when referring to a UI dialog
  box; leave the literary sense alone.
- `programme` → `program`.
- `coaveraging` → `co-averaging`.
- Plus any others surfaced by a regex sweep.

Do **not** change:

- Code blocks (anything inside `.. code-block::` or
  `.. literalinclude::` or backticks).
- Identifier names: class names, function names, file paths,
  enum values, registry keys, etc. (a British spelling
  inside an identifier is a code issue, not a doc issue;
  flag it for the user but do not "correct" it).
- UI labels quoted exactly. If the UI label uses a British
  spelling, the prose around it can use American spelling
  but the quoted label stays as the UI shows it.
- Doxygen-tag literals or `REGISTER_HARDWARE_*` macro
  names.

## Approach

1. **Dispatch a research agent** to grep for British
   spellings in `.rst` files under `doc/source/` and report
   a table: file, line, matched spelling, surrounding
   context (one line each side). Use the spelling list above
   as the regex set.
2. **Orchestrator confirms each match is in prose** — not
   inside a code block, identifier, or literal quote of a UI
   label. The research agent can pre-flag matches that
   appear inside backticks or directives so the orchestrator
   only judges the genuine prose hits.
3. **Apply replacements.** For each confirmed prose
   occurrence, replace the British spelling with the
   American spelling.
4. **Build** to confirm no inadvertent change broke a
   `:ref:` anchor.

## Out of scope

- Style edits beyond the spelling sweep.
- Source code spelling corrections (headers, `.cpp` files,
  identifier names). Flag these to the user; do not edit
  them in this sub-bundle.
- Rewording sentences to avoid a borderline word.

## Sources

### Related source files

- Every `.rst` file under `doc/source/`.

### Related dev-docs

None.

### Related user-guide pages

The pages themselves are the work unit.

### Related API reference pages

The pages themselves are the work unit.

## Sphinx file deltas

**Modified:**

- Every page that contained a prose British spelling.

**Created / deleted:**

- None.

## Acceptance criteria

- Every prose occurrence of the listed British spellings
  under `doc/source/` is converted to the American
  equivalent.
- Code blocks, identifier names, and quoted UI labels are
  preserved unchanged.
- Build is clean.
- The status-log entry lists any source-code British
  spellings flagged for the user.
