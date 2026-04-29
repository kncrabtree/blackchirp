# Bundle 13a — API Reference: Refresh Existing Five

Audits and refreshes the five existing API reference pages so they
match the current code, and establishes the conventions that
13b–13h follow.

## Scope

The five existing pages currently use `.. doxygenfile::`. This
bundle:

- Reviews each header's existing Doxygen comments. Where comments
  are missing, stale, or tied to v1.x behaviour, refresh them
  in-place in the header file.
- Switches each `.rst` page from `.. doxygenfile::` to
  `.. doxygenclass::` with `:members:` and `:undoc-members:` (or
  selectively-listed members) so each class gets a focused page
  rather than a file dump. Establishes this as the convention for
  13b–13h.
- Adds a one-paragraph prose intro at the top of each `.rst`
  describing the class's role in plain language and linking to
  the relevant developer-guide sub-page (bundle 12).

Pages and their headers:

- `classes/hardwareobject.rst` ← `src/hardware/core/hardwareobject.h`
- `classes/communicationprotocol.rst` ←
  `src/hardware/core/communication/communicationprotocol.h`
- `classes/custominstrument.rst` ←
  `src/hardware/core/communication/custominstrument.h`
- `classes/settingsstorage.rst` ←
  `src/data/storage/settingsstorage.h`
- `classes/headerstorage.rst` ← `src/data/storage/headerstorage.h`

This bundle also produces a short "API Reference Style" section in
the developer guide (or as a comment block in this bundle's
output) covering:

- Use `.. doxygenclass::` not `.. doxygenfile::`.
- Class `.rst` page begins with a 1–3 paragraph plain-language
  intro before any directive.
- Doxygen comment style: triple-slash `///` on consecutive lines
  for multi-line descriptions; `///<` for trailing field comments;
  use `\brief`, `\param`, `\return`, `\note`, `\warning`,
  `\sa` tags.
- Cross-reference between API and developer guide using
  `:doc:` / `:cpp:class:` / `:cpp:func:`.

## Out of scope

- Adding new API pages (bundles 13b–13h).
- Doxygen-config changes (`Doxyfile.in`) — the existing setup is
  fine.

## Sources

- The five header files listed above.
- `dev-docs/string-usage.md` — for `LogHandler` interactions
  documented on `HardwareObject`.
- `dev-docs/settings-registry.md` — for the registry-based
  defaults applied in `HardwareObject::applyRegisteredSettings`.

## Sphinx file deltas

**Modified:**
- `doc/source/classes/hardwareobject.rst`
- `doc/source/classes/communicationprotocol.rst`
- `doc/source/classes/custominstrument.rst`
- `doc/source/classes/settingsstorage.rst`
- `doc/source/classes/headerstorage.rst`

**Possibly modified (header comment refresh):**
- `src/hardware/core/hardwareobject.h`
- `src/hardware/core/communication/communicationprotocol.h`
- `src/hardware/core/communication/custominstrument.h`
- `src/data/storage/settingsstorage.h`
- `src/data/storage/headerstorage.h`

## Toctree delta

The `classes.rst` page uses a `:glob:` directive, so no toctree
edits are required.

## Acceptance criteria

- Each of the five RST pages uses `.. doxygenclass::` and begins
  with a plain-language intro.
- Headers no longer reference v1.x-specific behaviour
  (compile-time hardware selection, the old `prot`/`amp` AWG
  settings, etc.).
- Building the docs with `cmake --build build --target docs`
  produces no Doxygen warnings about undocumented public members
  for these classes (or, if some remain, they are intentional and
  noted in a comment block).
- The "API Reference Style" guide is captured somewhere durable
  (developer guide or top of `classes.rst`).
