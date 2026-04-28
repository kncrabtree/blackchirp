# Hardware Settings Registry

## Overview

The hardware settings registry is a unified system for declaring hardware settings
with metadata (labels, descriptions, priority levels) at compile time via static
registration macros. Settings are available before any hardware object is
constructed, so they can be presented to the user at profile creation time and
displayed with human-readable labels and tooltips in the hardware settings dialog.

Before this system, settings were invisible until a hardware object was constructed
and the user opened Hardware > [device] to inspect raw key/value pairs with no
context. Required parameters (e.g., digitizer channel count) used a separate
`HwConfigParam` system, while defaults used `setDefault` calls in constructors —
two divergent mechanisms serving the same purpose.

## How It Works

### Registration

Hardware classes register their settings in their `.cpp` file using static macros.
Registration happens at program startup before any objects are constructed.

```cpp
// Implementation scalar settings
REGISTER_HARDWARE_SETTINGS(MyClass,
    {BC::Key::MyHw::rate,    "Sample Rate (Hz)", "DAC output sample rate",
     16e9, 1e6, 100e9, HwSettingPriority::Important},
    {BC::Key::MyHw::enabled, "Enabled",          "Enable output",
     true, {}, {}, HwSettingPriority::Optional},
)

// Implementation array setting (metadata + one entry per REGISTER_HARDWARE_ARRAY_ENTRY call)
REGISTER_HARDWARE_ARRAY(MyClass, BC::Key::MyHw::sampleRates,
    "Sample Rates", "Available sample rates", HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(MyClass, BC::Key::MyHw::sampleRates,
    {{BC::Key::MyHw::srText, "2 GSa/s"}, {BC::Key::MyHw::srValue, 2e9}})

// Base-class scalar settings (inherited by all implementations)
REGISTER_HARDWARE_BASE(MyBaseClass,
    {BC::Key::MyHw::channels, "Channels", "Number of input channels",
     4, 1, 32, HwSettingPriority::Required},
)

// Base-class array placeholder (empty; implementations override with their entries)
REGISTER_HARDWARE_BASE_ARRAY(MyBaseClass, BC::Key::MyHw::sampleRates,
    "Sample Rates", "Available sample rates", HwSettingPriority::Important)
// Optional: supply default entries when a useful fallback exists
REGISTER_HARDWARE_BASE_ARRAY_ENTRY(MyBaseClass, BC::Key::MyHw::sampleRates,
    {{BC::Key::MyHw::srText, "1 GSa/s"}, {BC::Key::MyHw::srValue, 1e9}})
```

The `HwSettingDef` fields are:
- **key** — `SettingsStorage` key constant (never a string literal)
- **label** — user-facing display name
- **description** — tooltip text
- **defaultValue** — type determines the UI widget (`int` → `QSpinBox`,
  `double` → `ScientificSpinBox`, `bool` → `QCheckBox`, `QString` → `QLineEdit`)
- **minimum / maximum** — optional bounds for numeric types
- **priority** — controls visibility (see below)

### Priority Levels

| Priority | Meaning | UI placement |
|---|---|---|
| `Required` | Must be correct before construction (e.g., channel count) | Prominent form at top; read-only after profile creation |
| `Important` | Has a sensible default but the user should review it | Always-visible table below Required section |
| `Optional` | Rarely needs changing | Collapsible Advanced tab |

### Default Values

Registered defaults are applied in `HardwareObject::applyRegisteredSettings()`,
called from the base constructor. If a value already exists in `QSettings`
(from a previous session or from profile creation), the stored value takes
precedence. Subclass constructors no longer need `setDefault` calls for
registered settings.

### Base-Class Override Pattern

A setting declared with `REGISTER_HARDWARE_BASE` is shared by all
implementations. If an implementation needs a **different default** for the
same key, it can re-register that key with `REGISTER_HARDWARE_SETTINGS` —
`getSettingDefs` returns the implementation's entry first and skips the
base-class entry for that key, so no duplication occurs in the UI or in
`applyRegisteredSettings`.

```cpp
// Base class: sensible default for all implementations
REGISTER_HARDWARE_BASE(FlowController,
    {BC::Key::Flow::pUnits, "Pressure Units", "...", QString("kTorr"), ...},
)

// Derived class: override only pUnits; all other base-class settings inherited
REGISTER_HARDWARE_SETTINGS(PythonFlowController,
    {BC::Key::Flow::pUnits, "Pressure Units", "...", QString("Torr"), ...},
)
```

The same pattern applies to array settings: an implementation that registers
its own `REGISTER_HARDWARE_ARRAY` for a key defined by `REGISTER_HARDWARE_BASE_ARRAY`
will have its entries used in place of the base-class entries.

**Empty base-array declaration.** Calling `REGISTER_HARDWARE_BASE_ARRAY` with no
`REGISTER_HARDWARE_BASE_ARRAY_ENTRY` calls registers the array key with an empty
default. This guarantees the array always appears in the settings dialog (so the
user can add entries) even for implementations that do not supply their own
`REGISTER_HARDWARE_ARRAY`. Implementations that do supply entries override the
empty base completely. Use this for arrays like `sampleRates` where every
implementation defines its own entries but a user-facing Python or virtual
implementation might otherwise omit the key entirely.

### UI Integration

**Profile creation (`AddProfileDialog`):** When the user selects a hardware
implementation, an `HwSettingsWidget` (Create mode) is shown. Required settings
are editable form fields; Important settings are a table with live widgets;
Optional settings are under an Advanced tab. On accept, values are written to
`QSettings` before the object is constructed.

**Hardware settings dialog (`HWDialog`):** Opens from Hardware > [device]. The
Settings tab hosts an `HwSettingsWidget` (Edit mode), where Required settings are
shown read-only (they cannot be changed post-creation), and Important/Optional
settings are editable. The Control tab (when present) holds the hardware-specific
control widget (e.g., pulse generator channel table). Changes are written to
`QSettings` on Ok.

## Key Files

| File | Purpose |
|---|---|
| `src/hardware/core/hardwareregistration.h` | All registration macros: `REGISTER_HARDWARE_SETTINGS/ARRAY/ARRAY_ENTRY` (implementation), `REGISTER_HARDWARE_BASE/BASE_ARRAY/BASE_ARRAY_ENTRY` (base class); `HwSettingDef` and `HwArraySettingDef` structs |
| `src/hardware/core/hardwareregistry.h/.cpp` | `HardwareRegistry` singleton; stores and retrieves all registrations |
| `src/gui/widget/hwsettingswidget.h/.cpp` | `HwSettingsWidget` — shared embeddable widget used by both `AddProfileDialog` and `HWDialog` |
| `src/gui/dialog/hwarrayeditdialog.h/.cpp` | Sub-dialog for editing array setting entries |
| `src/gui/dialog/hwdialog.h/.cpp` | Hardware settings dialog (uses `HwSettingsWidget` in Edit mode) |
| `src/gui/dialog/addprofiledialog.h/.cpp` | Profile creation dialog (uses `HwSettingsWidget` in Create mode) |

## Adding Settings to a New Hardware Class

**For a concrete implementation:**

1. Declare key constants in the appropriate `BC::Key::` namespace in
   `src/data/settings/hardwarekeys.h`.
2. Add `REGISTER_HARDWARE_SETTINGS(...)` in the class `.cpp` file at file scope,
   after the existing `REGISTER_HARDWARE_META` and `REGISTER_HARDWARE_PROTOCOLS`
   macros. Omit any setting whose value matches the base-class default — it is
   inherited automatically.
3. For array settings, add `REGISTER_HARDWARE_ARRAY(...)` followed by one
   `REGISTER_HARDWARE_ARRAY_ENTRY(...)` per entry. Omit the whole block if the
   base-class array default is sufficient.
4. Remove any `setDefault` / `setArray` calls in the constructor for settings
   that are now registered — the base constructor handles them automatically.

**For an abstract base class:**

1. Declare key constants in `hardwarekeys.h`.
2. Add `REGISTER_HARDWARE_BASE(...)` in the base class `.cpp` file with the
   common settings and their sensible defaults. Include every setting shared by
   all implementations, even if the default value differs per implementation.
3. For array settings that every implementation defines (e.g. `sampleRates`),
   add `REGISTER_HARDWARE_BASE_ARRAY(...)` with no entries. This ensures the
   key always appears in the settings dialog even for implementations (such as
   Python-backed ones) that do not supply their own `REGISTER_HARDWARE_ARRAY`.
   If a useful default set of entries exists, add `REGISTER_HARDWARE_BASE_ARRAY_ENTRY`
   calls (see `FlowController` for an example).
4. Each implementation then registers only the settings that differ from the
   base defaults, or settings unique to that implementation.
