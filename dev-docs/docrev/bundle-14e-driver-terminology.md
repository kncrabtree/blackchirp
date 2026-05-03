# Bundle 14e — Implementation → driver terminology sweep

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Final Consistency Pass. Replaces user-facing
prose uses of "implementation" with "driver" where the term
refers to a concrete hardware backend for a hardware type.

## Background

The mental model the documentation should reinforce is
*Hardware Object → Hardware Type → one of several drivers*.
"Implementation" reads as a programming abstraction;
"driver" matches how users think about choosing a backend
for a device (`AWG70002a`, `VirtualAwg`, `PythonAwg`, etc.).

The user-guide track has already been mostly aligned, but
late-landing pages and developer-guide pages may still mix
the two. This sub-bundle is the final pass.

## Scope

Walk every prose file under:

- `doc/source/user_guide/`
- `doc/source/classes/`
- `doc/source/developer_guide/`

For each occurrence of "implementation" / "implementations" /
"implement" / "implemented" / "implementing", judge whether
the term refers to a concrete hardware backend:

- **Replace with "driver" / "drivers"** when the term is
  naming the concrete hardware-backend role
  (`AWG70002a`, `VirtualAwg`, `PythonAwg`, etc.).
- **Leave as-is** in every other sense: implementing an
  interface in the C++ sense, an implementation strategy,
  the implementation of a feature, etc.

## Approach

1. **Dispatch a research agent** to enumerate every prose
   occurrence of "implementation" / "implement" in the page
   set above. The agent returns: file, line, matched word,
   one-line context above and below.
2. **Orchestrator decides per-occurrence** whether the
   replacement applies. The decision is contextual; this is
   not a regex-replace pass.
3. **Apply replacements** for the confirmed cases.
4. **Update any glossary, index, or cross-reference** that
   used the old term. The bundle's commit should leave the
   documentation internally consistent.
5. **Build** to confirm no `:ref:` or `:doc:` references
   are broken.

Do **not** change:

- Code blocks.
- Identifier names: function or class names that contain
  "implement" or "implementation" stay as-is.
- Documentation comments that quote a Doxygen tag or
  registry macro literally (`REGISTER_HARDWARE_*` etc.).
- C++ source files (headers and `.cpp`). The C++ comments
  use whichever term reads naturally for each call site;
  this sub-bundle is documentation only.

## Out of scope

- Header / source-file edits.
- Renaming registry macros, classes, or files.
- Rewording sentences beyond what the term swap requires.
- Other terminology sweeps (e.g. "module" vs. "subsystem",
  "config" vs. "configuration"). Flag any candidate to the
  user; do not act.

## Sources

### Related source files

- Every `.rst` file under `doc/source/user_guide/`,
  `doc/source/classes/`, and `doc/source/developer_guide/`.

### Related dev-docs

None.

### Related user-guide pages

The pages themselves are the work unit.

### Related API reference pages

The pages themselves are the work unit.

## Sphinx file deltas

**Modified:**

- Every page that contained a prose use of "implementation"
  in the hardware-backend sense.

**Created / deleted:**

- None.

## Acceptance criteria

- Every prose occurrence of "implementation" in the
  hardware-backend sense, in the page set above, has been
  replaced with "driver".
- Other senses of "implementation" are preserved.
- Code blocks, identifier names, and macro literals are
  preserved unchanged.
- The glossary / index entries reflect the new term.
- Build is clean.
- The status-log entry lists any borderline cases the
  orchestrator decided one way or the other so the user can
  spot-check.
