# Bundle 13b — API Reference: Hardware-Management Classes

Adds API reference pages for the runtime hardware configuration and
vendor-library subsystems.

## Scope

New pages under `doc/source/classes/`:

- `hardwareregistry.rst` ← `src/hardware/core/hardwareregistry.h`
  (`HardwareRegistry`, `HardwareRegistration`, registration
  macros, `HwSettingDef`, `HwArraySettingDef`).
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
  `REGISTER_HARDWARE_SETTINGS`, base-class variants).
- The `HardwareProfileManager` page documents the per-profile
  fields (`pythonScriptPath`, `pythonClassName`, `pythonEnvPath`)
  including which are persisted to QSettings.
- The `VendorLibrary` page documents the dynamic-loading API
  shape and the relationship to `LibraryStatusWidget`.
