# Bundle 13b — API Reference: Hardware-Management Classes

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-05-02: drafted → complete. Content commit 13635525. User-review
  pass added an "API Reference" wrapper section to all 9 pages under
  doc/source/classes/ (4 new + 5 existing) so Doxygen directives stop
  appearing nested under preceding prose subsections, restored the
  HardwareProfileManager prose subsections, refreshed the
  HardwareProfileManager class-level Doxygen and the
  generateDefaultLabel() candidate list to match the actual code, and
  documented the wrapper convention plus the directive-vs-entity-kind
  rule in doc/source/developer_guide/api_style.rst.
- 2026-05-02: in progress → drafted. Verifier punch list (3 load-bearing,
  2 minor) addressed in one revision pass: doxygenstruct fix, VendorLibrary
  ctor/dtor briefs, removal of private syncWithProfiles() reference,
  HardwareRegistration ctor briefs, "previously" temporal marker rewritten.
  Awaiting user content commit (stage 1).
- 2026-05-02: not started → in progress. Bundle scope refreshed
  to include REGISTER_CUSTOM_COMM / REGISTER_CUSTOM_COMM_BASE
  macros and CustomCommType enum (added in 67ee5dfa); BC::Key::Custom
  retirement and Python Custom-protocol semantics noted in Sources.
  Drafter dispatched.
-->

Adds API reference pages for the runtime hardware configuration and
vendor-library subsystems.

## Scope

New pages under `doc/source/classes/`:

- `hardwareregistry.rst` ← `src/hardware/core/hardwareregistry.h`
  (`HardwareRegistry`, `HardwareRegistration`, registration
  macros, `HwSettingDef`, `HwArraySettingDef`,
  `CustomCommDef`, `CustomCommType`).
- `hardwareprofilemanager.rst` ←
  `src/hardware/core/hardwareprofilemanager.h`
  (`HardwareProfileManager`, `ProfileInfo`).
- `runtimehardwareconfig.rst` ←
  `src/hardware/core/runtimehardwareconfig.h`
  (`RuntimeHardwareConfig`).
- `vendorlibrary.rst` ← `src/hardware/library/vendorlibrary.h`
  (`VendorLibrary` base class). Concrete subclasses
  (`LabjackLibrary`, `SpectrumLibrary`) get a single combined
  child page or are mentioned in the base class's prose; choose
  whichever yields the cleaner navigation given Doxygen output.

Each page follows the conventions established by 13a:

- Plain-language intro paragraph linking to the developer-guide
  page that explains usage in context.
- `.. doxygenclass::` directive (not `doxygenfile`).
- Header refresh where Doxygen comments are missing or stale.

## Out of scope

- The Python hardware classes (bundle 13c).
- The settings-registry macros' usage examples (developer guide).

## Sources

- `dev-docs/settings-registry.md` — for `HwSettingDef` and the
  registration macro semantics that the API page links into.
- `dev-docs/loadout-system.md` — for how `HardwareProfileManager`
  cooperates with `LoadoutManager`.
- `dev-docs/labjack-cross-platform-support.md` — for
  `VendorLibrary` semantics.
- The header files listed above.

The `REGISTER_CUSTOM_COMM` / `REGISTER_CUSTOM_COMM_BASE` macros
and the `CustomCommType` enum (`String`, `Int`, `FilePath`) were
added in commit `67ee5dfa` and are part of the registration-macro
surface this bundle documents. The legacy `BC::Key::Custom`
namespace was retired in the same commit; do not document or
cross-reference it. Connection-parameter declarations for Python
drivers remain script-side by design — `CommunicationProtocol::Custom`
on a Python implementation is the explicit "comm is handled by
the .py script" indicator (commit `7fdc8a4c`); the API page should
note this only briefly, since Python-specific docs belong to 13c.

## Sphinx file deltas

**Created:**
- `doc/source/classes/hardwareregistry.rst`
- `doc/source/classes/hardwareprofilemanager.rst`
- `doc/source/classes/runtimehardwareconfig.rst`
- `doc/source/classes/vendorlibrary.rst`

**Possibly modified (Doxygen comment refresh):**
- `src/hardware/core/hardwareregistry.h`
- `src/hardware/core/hardwareregistration.h`
- `src/hardware/core/hardwareprofilemanager.h`
- `src/hardware/core/runtimehardwareconfig.h`
- `src/hardware/library/vendorlibrary.h`

## Toctree delta

`classes.rst` uses a `:glob:` directive, so the new files are
picked up automatically. No edit needed.

## Acceptance criteria

- Every public method on each class has at least a `\brief`
  comment in the header.
- The `HardwareRegistry` page documents the registration-macro
  surface (`REGISTER_HARDWARE_META`, `REGISTER_HARDWARE_PROTOCOLS`,
  `REGISTER_HARDWARE_SETTINGS`, `REGISTER_CUSTOM_COMM`, base-class
  variants such as `REGISTER_CUSTOM_COMM_BASE`) and the
  `CustomCommType` enum values.
- The `HardwareProfileManager` page documents the per-profile
  fields (`pythonScriptPath`, `pythonClassName`, `pythonEnvPath`)
  including which are persisted to QSettings.
- The `VendorLibrary` page documents the dynamic-loading API
  shape and the relationship to `LibraryStatusWidget`.
