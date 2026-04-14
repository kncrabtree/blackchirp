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
// Scalar settings
REGISTER_HARDWARE_SETTINGS(MyClass,
    {BC::Key::MyHw::rate,    "Sample Rate (Hz)", "DAC output sample rate",
     16e9, 1e6, 100e9, HwSettingPriority::Important},
    {BC::Key::MyHw::enabled, "Enabled",          "Enable output",
     true, {}, {}, HwSettingPriority::Optional},
)

// Array setting (metadata + one entry per REGISTER_HARDWARE_ARRAY_ENTRY call)
REGISTER_HARDWARE_ARRAY(MyClass, BC::Key::MyHw::sampleRates,
    "Sample Rates", "Available sample rates", HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(MyClass, BC::Key::MyHw::sampleRates,
    {{BC::Key::MyHw::srText, "2 GSa/s"}, {BC::Key::MyHw::srValue, 2e9}})
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
| `src/hardware/core/hardwareregistration.h` | `REGISTER_HARDWARE_SETTINGS`, `REGISTER_HARDWARE_ARRAY`, `REGISTER_HARDWARE_ARRAY_ENTRY` macros; `HwSettingDef` and `HwArraySettingDef` structs |
| `src/hardware/core/hardwareregistry.h/.cpp` | `HardwareRegistry` singleton; stores and retrieves all registrations |
| `src/gui/widget/hwsettingswidget.h/.cpp` | `HwSettingsWidget` — shared embeddable widget used by both `AddProfileDialog` and `HWDialog` |
| `src/gui/dialog/hwarrayeditdialog.h/.cpp` | Sub-dialog for editing array setting entries |
| `src/gui/dialog/hwdialog.h/.cpp` | Hardware settings dialog (uses `HwSettingsWidget` in Edit mode) |
| `src/gui/dialog/addprofiledialog.h/.cpp` | Profile creation dialog (uses `HwSettingsWidget` in Create mode) |

## Adding Settings to a New Hardware Class

1. Declare key constants in the appropriate `BC::Key::` namespace in
   `src/data/settings/hardwarekeys.h`.
2. Add `REGISTER_HARDWARE_SETTINGS(...)` in the class `.cpp` file at file scope
   (outside any function), after the existing `REGISTER_HARDWARE_META` and
   `REGISTER_HARDWARE_PROTOCOLS` macros.
3. For array settings, add `REGISTER_HARDWARE_ARRAY(...)` followed by one
   `REGISTER_HARDWARE_ARRAY_ENTRY(...)` per default entry.
4. Remove any `setDefault` / `setArray` calls in the constructor for settings
   that are now registered — the base constructor handles them automatically.
