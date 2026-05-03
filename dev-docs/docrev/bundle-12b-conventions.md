# Bundle 12b — Developer Guide: Coding Conventions

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. Authored
  doc/source/developer_guide/conventions.rst per spec: naming/indent,
  4-form string-literal table, function-signature policy, container
  rule, three key-declaration patterns with anti-example, Doxygen
  section forwarding to api_style, logging section naming all five
  free functions plus four hw* helpers and the five severity levels,
  persistent-settings section establishing the protected-set/public-get
  split and static-key rule. Forward-links land on /classes/loghandler,
  /classes/settingsstorage, /classes/hardwareregistry,
  /developer_guide/api_style, and /developer_guide/hardware_configuration
  (12d, not yet written — Sphinx warning expected per umbrella). No
  dev-docs/ links in rendered output. No source-tree change required.
  Content commit 836c73afe944f77c91dc28eb256973917909278f.
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Developer Guide chapter. Documents the coding
conventions a contributor must follow to fit a patch into the
existing source: naming, member prefixes, string-literal selection,
function-signature policy, container choices, key-declaration
patterns, Doxygen comments, and the canonical entry points for
logging and persistent settings.

This page replaces the original bundle 12's `code_style.rst`,
`logging.rst`, and `settings_storage.rst` outline entries: the API
reference now carries the per-class detail for `LogHandler` and
`SettingsStorage`, so this page documents only the conventions that
hold across the codebase and forwards class-level concerns to the
API ref.

## Scope

Single Sphinx file: `doc/source/developer_guide/conventions.rst`.

The page should answer the following for a contributor:

1. **Naming.** Mirror `CLAUDE.md`:

   - Classes / structs / enums: `UpperCamelCase`.
   - Functions and variables: `lowerCamelCase`.
   - Member-variable prefixes:
     - `d_` for value members,
     - `p_` for raw pointers,
     - `pu_` for `std::unique_ptr`,
     - `ps_` for `std::shared_ptr`.
   - 4-space indentation, spaces only.

2. **String literals.** Mirror `dev-docs/string-usage.md` (research
   only; do not link). One concise table:

   | Form | Type | When to use |
   |------|------|-------------|
   | `"..."_L1` | `QLatin1StringView` | ASCII content; **default**. |
   | `u"..."_s` | `QString` | Non-ASCII (e.g., `u"μs"_s`); also when the call site requires a `QString` and the literal contains non-ASCII. |
   | `"..."_s` | `QString` | Only when the call site genuinely requires a `QString` (`.arg()` receivers, widget constructors, `QStringList`, `QRegularExpression`). |
   | `QStringLiteral(...)` | `QString` | **Do not use in new code.** Replace existing occurrences opportunistically. |

   Note that `_s` and `_L1` are pulled in globally via
   `data/loghandler.h` (`using namespace
   Qt::Literals::StringLiterals`), so any TU that includes the log
   header transitively gets the suffixes for free.

3. **Function signatures.** Mirror `dev-docs/string-usage.md`:

   - Never pass `QString` by value unless the callee takes ownership
     and moves.
   - Prefer `QAnyStringView` for pure lookup/comparison/passthrough.
   - Use `const QString&` when the callee genuinely needs a `QString`
     (`.arg()` calls, storing as `QString`, calling a `const
     QString&` API).

4. **Containers.** `std::map<QString, T, std::less<>>` for new
   maps keyed on `QString`, so heterogeneous lookup with
   `QLatin1StringView` / `QStringView` / `const char*` does not
   construct a temporary `QString`. `QHash<QString, T>` does not
   need the special declaration; Qt6 supports heterogeneous hashing
   for it natively.

5. **Key declaration patterns.** Three patterns, each with a
   one-line example and a one-line "use when" guideline:

   - **Pattern A — `inline const QString`.** For keys consumed as
     `QString` (e.g., `.arg()` is called on them directly). One heap
     allocation per process, not per TU.
   - **Pattern B — `inline constexpr QLatin1StringView`.** For ASCII
     keys consumed by `QAnyStringView` parameters or stored in
     `std::map<QString, T, std::less<>>` with heterogeneous lookup.
     True `constexpr`, zero runtime cost. In headers, use the
     constructor form `{"..."}` (not `"..."_L1`) to avoid pulling in
     a `using namespace`.
   - **Pattern C — `inline constexpr QStringView`.** Same trade-offs
     as Pattern B but UTF-16, so non-ASCII keys are safe.

   Note the anti-example: `inline constexpr auto k = "key"_s;` does
   not compile because `QString` is not a literal type in the Qt
   versions Blackchirp targets.

   Cross-reference the canonical key namespaces:
   `BC::Key::` family in `src/data/settings/hardwarekeys.h` for
   hardware settings; `BC::Store::` family scattered across data
   classes for persistent storage keys; `BC::CSV::` for canonical
   experiment-directory filenames.

6. **Doxygen comments.** Two paragraphs, no more:

   - Triple-slash (`///`) on consecutive lines is preferred for new
     code; `///<` for trailing field/parameter comments. Multi-line
     `/*! ... */` blocks remain valid; match the surrounding file.
   - Standard tags: `\brief`, `\param`, `\return`, `\note`,
     `\warning`, `\sa`. Begin every documented entity with a
     single-sentence `\brief`. Document every public and protected
     member that a subclass author or external caller would reach
     for.
   - **The canonical reference is `:doc:`/developer_guide/api_style``.**
     That page (already shipped) describes where prose lives
     (header vs RST), the Sphinx directive choices
     (`.. doxygenclass::` over `.. doxygenfile::`), and the refresh
     checklist when editing a class. Do not duplicate it here;
     forward-link.

7. **Logging.** The contributor uses the free-function helpers
   declared in `data/loghandler.h`:
   `bcLog`, `bcDebug`, `bcWarn`, `bcError`, `bcHighlight`. Inside
   `HardwareObject` subclasses, the `hw*` member helpers
   (`hwLog`/`hwDebug`/`hwWarn`/`hwError`) prepend the device key
   automatically and are preferred. Do **not** use `qDebug()` or
   raw `emit logMessage(...)` in new code. Briefly summarize the
   five severity levels (Error / Warning / Normal / Highlight /
   Debug) and what they map to. Forward-link to
   `:doc:`/classes/loghandler`` for the full API surface, on-disk
   format, and per-experiment log lifecycle.

8. **Persistent settings.** The contributor uses `SettingsStorage`
   for any persistent state. Two key conventions:

   - The `set` family is **protected**: anywhere in the codebase
     may *read* settings (by constructing a transient
     `SettingsStorage` over the appropriate group), but only the
     owning class (or a declared `friend`) can *write*. This is
     the central guarantee: settings cannot drift behind the
     owner's back.
   - Setting keys are **statically declared** in the appropriate
     `BC::Key::` namespace (or `BC::Store::` for non-hardware
     persistent state). No string literals at call sites.

   For hardware settings specifically, defaults are supplied by
   the hardware-settings registry
   (`REGISTER_HARDWARE_SETTINGS` and friends), applied from
   `HardwareObject::applyRegisteredSettings()`, not by ad-hoc
   `setDefault` calls in subclass constructors. The full
   registration story is in `:doc:`/developer_guide/hardware_configuration``
   (12d). Forward-link to `:doc:`/classes/settingsstorage`` for the
   class-level API and to `:doc:`/classes/hardwareregistry`` for
   the registration macros.

## Out of scope

- The full API of `LogHandler` and `SettingsStorage` — covered in
  their respective API pages. Cross-link, do not duplicate.
- The Doxygen comment template, refresh checklist, and Sphinx
  directive choices — covered in `api_style.rst`. Cross-link.
- The hardware-settings registry macros (`REGISTER_HARDWARE_*`),
  which are conventions plus runtime mechanism — that mechanism is
  bundle 12d's job. Mention them in passing here only as the
  canonical source of hardware-settings defaults.
- CLAUDE.md-style critical rules (timeless prose, full paths in
  Bash commands) — those are contributor-instruction-level rules,
  not user-facing documentation; describe coding conventions, not
  contribution-process rules.

## Sources

### Related source files

- `data/loghandler.h` — free-function declarations and the
  `using namespace Qt::Literals::StringLiterals` global pull-in.
- `data/storage/settingsstorage.h` — confirm the protected `set`
  policy.
- `data/bcglobals.h` — for canonical `BC::Key`/`BC::Store`
  namespace examples.
- `data/settings/hardwarekeys.h` — for the hardware-key namespace
  layout to cite as the canonical example.
- A representative spread of source files demonstrating each
  string-literal form, each container choice, and each key-
  declaration pattern (use `grep` to confirm current usage). The
  drafter does not need to enumerate examples in the page; they
  serve as ground truth.

### Related dev-docs

- `dev-docs/string-usage.md` — research material for the
  string-literal table, function-signature policy, container
  policy, key-declaration patterns. Do not link from the rendered
  page.

### Related user-guide pages

None directly.

### Related API reference pages

- `doc/source/classes/loghandler.rst` — forward-link from the
  Logging section.
- `doc/source/classes/settingsstorage.rst` — forward-link from the
  Persistent settings section.
- `doc/source/classes/hardwareregistry.rst` — forward-link from
  the persistent-settings note about registry-supplied defaults.
- `doc/source/developer_guide/api_style.rst` — forward-link from
  the Doxygen comments section as the canonical style reference.

## Sphinx file deltas

**Modified:**

- `doc/source/developer_guide.rst` — only if the explicit toctree
  has not yet been written by bundle 12. If 12 has landed, no
  edit; the toctree already references this page.

**Created:**

- `doc/source/developer_guide/conventions.rst`.

## Page structure

H1 intro: 1–2 paragraphs framing the page as the contributor's
quick-reference for "how Blackchirp code looks". Cross-link to
`:doc:`/developer_guide/api_style`` for the documentation-comment
contract.

H2 sections (use `-` underlines):

- *Naming and indentation*
- *String literals*
- *Function signatures*
- *Containers*
- *Key declaration patterns*
- *Doxygen comments* (forward-link to `api_style`)
- *Logging* (forward-link to `:doc:`/classes/loghandler``)
- *Persistent settings* (forward-link to
  `:doc:`/classes/settingsstorage`` and
  `:doc:`/classes/hardwareregistry``)

## Acceptance criteria

- The string-literal table contains the four forms and is
  consistent with `dev-docs/string-usage.md` (research source).
- The four member-variable prefixes (`d_`/`p_`/`pu_`/`ps_`) are
  documented with one-line meanings.
- The three key-declaration patterns are each shown with a code
  example and a "use when" guideline; the failing
  `inline constexpr auto k = "key"_s;` anti-example is noted.
- The Doxygen-comments section is short (no more than a few
  paragraphs) and forwards to `api_style` rather than duplicating
  it.
- The logging section names the five free functions, the four
  `hw*` helpers, and the five severity levels; forwards to the
  `LogHandler` API page for the full surface.
- The persistent-settings section establishes the protected-`set`
  /public-`get` split and the static-key-declaration rule;
  forwards to `SettingsStorage` and `HardwareRegistry` API pages.
- No content duplicates the API reference at the per-method level.
- No rendered link points into `dev-docs/`.
