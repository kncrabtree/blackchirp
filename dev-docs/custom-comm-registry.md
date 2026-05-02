# Custom Communication Parameter Registry

## Problem

`CommunicationProtocol::Custom` hardware (devices that do not use RS-232,
TCP, or GPIB) declares the connection parameters it needs from the user
— device path, serial number, file handle, etc. — by writing a
`BC::Key::Custom::comm` array into its own `SettingsStorage` from the
constructor. Each entry is a `SettingsMap` of hand-typed strings:

```cpp
if(!containsArray(BC::Key::Custom::comm))
    setArray(BC::Key::Custom::comm, {
        {{BC::Key::Custom::key,   "devPath"},
         {BC::Key::Custom::type,  BC::Key::Custom::stringKey},
         {BC::Key::Custom::label, "Device Path"}}
    });
```

`CustomProtocolWidget` reads the array out of `SettingsStorage` at
construction time and renders one `QLineEdit` or `QSpinBox` per entry.
Three things are wrong:

1. **Two divergent declaration mechanisms for what the user sees as the
   same kind of thing.** Hardware *settings* (sample rate, channel
   count, pressure units, etc.) are declared via
   `REGISTER_HARDWARE_SETTINGS` macros — typed, with labels,
   descriptions, priorities, and bounds, validated at registration
   time, available to the UI before any hardware object exists.
   Hardware *connection parameters* are declared via runtime
   `setArray` calls in the constructor — untyped maps of string
   values, available to the UI only after the object has been
   constructed at least once. The user's mental model is "things I
   have to fill in to talk to my hardware"; the code splits them on a
   distinction the user does not see.

2. **`AddProfileDialog` cannot show custom-comm fields at profile
   creation time.** The widget's `containsArray` check returns false
   for a brand-new profile because the constructor that would write
   the declaration has not run. The same problem the settings
   registry was built to solve.

3. **Boilerplate at the call site.** Every driver that uses the
   custom protocol writes the same `if(!containsArray(...))
   setArray(...)` block with literal-string `key`/`type`/`label`
   subkeys that no compiler checks.

## Proposed shape

Add a third registration macro family alongside
`REGISTER_HARDWARE_SETTINGS` / `REGISTER_HARDWARE_BASE`:

```cpp
REGISTER_CUSTOM_COMM(LabjackU3,
    {BC::Key::IOB::serialNo, "Serial Number", "USB serial number",
     CustomCommType::Int, 0, INT_MAX},
)

REGISTER_CUSTOM_COMM(M4i2220x8,
    {"devPath"_L1, "Device Path", "Spectrum card device node (/dev/spcm0)",
     CustomCommType::String, 64},
)

REGISTER_CUSTOM_COMM_BASE(PythonHardwareBase,
    {"scriptPath"_L1, "Driver Script", "Path to the .py driver file",
     CustomCommType::FilePath},
)
```

Stored on `HardwareRegistry` next to the existing
`HwSettingDef` / `HwArraySettingDef` tables. Three macros:
`REGISTER_CUSTOM_COMM` (concrete implementation),
`REGISTER_CUSTOM_COMM_BASE` (shared by all subclasses, merged by
inheritance chain just like base settings), and the typed-entry
struct `CustomCommDef`.

Consumed at two points:

- **Profile creation** (`AddProfileDialog`): when the user picks a
  hardware implementation whose `CommType` is `Custom`, the dialog
  asks `HardwareRegistry::getCustomCommDefs(hwType, hwImpl)` for the
  schema and renders the appropriate widgets *before* the object
  exists. Values are written to the per-profile
  `BC::Key::Comm::custom` group (the same on-disk location used today
  for the entered values), so `CustomInstrument` and the consuming
  driver's `readSettings` need no change.
- **Hardware settings dialog** (`HwDialog` →
  `CustomProtocolWidget`): same call, same on-disk location;
  the widget no longer reads from `SettingsStorage` for the schema
  (only for the values).

## Migration

- Add `CustomCommType` enum: `String`, `Int`, `FilePath` (new — gives
  the widget a file-picker affordance), and any others surfaced by
  existing drivers.
- Add `CustomCommDef` struct + `addCustomCommDefs` / `getCustomCommDefs`
  on `HardwareRegistry`.
- Add the three macros to `hardwareregistration.h`.
- Convert the three current direct uses
  (`labjacku3.cpp`, `m4i2220x8.cpp`, `m4i2211x8.cpp`) and the python
  hardware base to the new macros. Remove the
  `setArray(BC::Key::Custom::comm, ...)` blocks from each
  constructor.
- Rewrite `CustomProtocolWidget::generateDynamicUI` to pull the
  schema from the registry (passing the hardware type and
  implementation derived from the widget's `hardwareKey`) instead of
  the SettingsStorage array.
- Retire the `BC::Key::Custom::comm` array key and the
  `key`/`type`/`stringKey`/`intKey`/`maxLen`/`intMin`/`intMax`
  subkeys. Keep `BC::Key::Custom::label` only if it is still useful
  elsewhere.
- One-shot settings migration: on first launch after the change, walk
  every existing `Comm/custom` group and confirm the entered values
  are still readable; the *value* layout does not change, only the
  declaration mechanism, so no QSettings rewrite is required.

## Documentation follow-up (mandatory once the refactor lands)

Both touch points written in bundle 13a will be stale and must be
revised in the same PR as the refactor:

1. **`src/hardware/core/communication/custominstrument.h`** — the
   class-level Doxygen block currently walks the reader through the
   `setArray(BC::Key::Custom::comm, ...)` recipe with an inline code
   example. Replace with a one-paragraph note pointing to the
   `REGISTER_CUSTOM_COMM` macro family (analogous to how the current
   header points at `BC::Key::Custom::comm`).
2. **`doc/source/classes/custominstrument.rst`** — the orientation
   intro currently says custom-protocol drivers declare their fields
   "by writing a `BC::Key::Custom::comm` array into their
   `SettingsStorage` on first construction." Rewrite that paragraph
   to describe registration via `REGISTER_CUSTOM_COMM` at static
   initialization and reference the api_style page if appropriate.
3. **`doc/source/user_guide/python_hardware.rst`** (and any
   `python_hardware/` sub-page that mentions custom comm) —
   the user-facing description of "where do I tell Blackchirp the
   path to my .py file" is unchanged from the user's perspective,
   but the prose may reference the `BC::Key::Custom::comm` array
   indirectly; verify and update.
4. **`dev-docs/python-hardware.md`** and any other dev-doc that
   references the hand-written array pattern.

## Sequencing

- Bundle 13a documents *current* behavior. This dev-doc captures the
  refactor so it is not lost.
- Schedule the refactor as its own bundle after the user-guide /
  developer-guide work has settled (post-bundle-12 is a natural slot;
  it is not blocked by any other bundle).
- When the refactor lands, the documentation updates above are part
  of the same change.

## Out of scope

- The on-disk layout of the entered *values* (the `Comm/custom`
  group). This refactor only changes the *declaration* of which
  fields exist.
- Settings registry priority levels for custom-comm fields. All
  custom-comm fields are conceptually `Required` (the connection
  cannot be tested without them); keep the type simple unless a
  driver surfaces a reason to differentiate.
- Validation of entered values (e.g., "this file does not exist").
  That belongs in `testConnection()` per existing convention.
