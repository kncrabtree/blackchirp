# Bundle 14b — Screenshot sizing pass

**Status:** complete

<!--
Status log:
- 2026-05-03 — not started → complete (content commit ffca502a).
  All 50 image/figure directives under doc/source/user_guide/ now
  follow the rule: native ≤800 px renders 1:1 (no :width:); native
  >800 px caps at :width: 800 with :target: pointing at
  ../_images/<basename>.png (1-level pages) or
  ../../_images/<basename>.png (2-level pages under experiment/,
  lif/, hw/, python_hardware/, hardware_config/). Sphinx flattens
  every directive image to _images/<basename>, so the relative
  :target: paths resolve under ReadTheDocs as well as locally.
  Opportunistic edits, authorized by the user: 8 orphan PNGs under
  _static/user_guide/ were git-rm'd; one HTML-anchor cross-reference
  in data_storage.rst was rewritten as a :ref:; three
  "implementation" prose uses naming a hardware backend (in
  experiment/quick_experiment.rst and first_run.rst) were swapped to
  "driver". Build is clean — the 105 warnings are pre-existing
  (Pygments lexer + auto-generated C++ duplicate-declaration
  warnings in classes/) and not introduced by this work. Click-
  through is a static link; a JS lightbox upgrade was discussed and
  deferred (would need a Sphinx extension dependency, out of 14b
  scope).
-->

Sub-page of the Final Consistency Pass. Standardizes how
screenshots render on user-guide pages.

## Scope

Walk every `.. image::` and `.. figure::` directive under
`doc/source/user_guide/` (and any other `doc/source/`
chapter that references a screenshot). For each referenced
PNG, inspect the native pixel width and apply one of:

- **Native width ≤ 800 px:** drop any explicit `:width:`
  option (or set it to the native width) so the image renders
  at 1:1.
- **Native width > 800 px:** cap the rendered width at 800 px
  and wrap the directive so the figure links to the full-
  resolution image. Pick one of:
  - `:target:` on the `.. image::` directive pointing at the
    same `_static` path (click-through to the full-resolution
    PNG).
  - A `.. figure::` with the `:target:` option set similarly.

Pick one approach (probably `:target:`) and apply it
consistently across every directive that needs the cap.

## Approach

1. **Dispatch a research agent** to walk the directives and
   report a table: page path, directive line, image path,
   native pixel width, current `:width:` value (if any).
2. **Orchestrator decides** per row whether the directive
   needs editing (some are already correctly sized).
3. **Apply edits.** When the same `_static` image is
   referenced from multiple pages, apply the same treatment
   on every reference.
4. **Build the docs** and visually spot-check at least the
   most-changed pages in a browser to confirm rendering.

## Out of scope

- Capturing new screenshots or recapturing existing ones at a
  different resolution. This sub-bundle works with the PNGs
  on disk as they currently are.
- Adding `:alt:` text where it is missing (a separate pass if
  the user wants one).
- Editing screenshots themselves (cropping, annotating).
- Touching anything other than the `:width:`, `:target:`, and
  surrounding directive options.

## Sources

### Related source files

- `doc/source/user_guide/*.rst` and subdirectory pages.
- `doc/source/_static/user_guide/**/*.png` — the actual
  images whose native pixel widths drive the decisions.
- Any other `doc/source/` chapter that uses screenshots
  (currently the user guide carries them all, but the sweep
  should confirm).

### Related dev-docs

None.

### Related user-guide pages

The page set under `doc/source/user_guide/` is itself the
work unit.

### Related API reference pages

None.

## Sphinx file deltas

**Modified:**

- Every page that contains an `.. image::` or `.. figure::`
  directive whose width treatment needs adjusting.

**Created / deleted:**

- None.

## Acceptance criteria

- Every screenshot directive under `doc/source/user_guide/`
  follows the rule: ≤800 px renders 1:1, >800 px is capped
  with click-through to the full-resolution image.
- The click-through approach is consistent across all
  directives that need it.
- Build is clean.
- Spot-check in a browser confirms the rendered sizes are
  what the rule prescribes.
