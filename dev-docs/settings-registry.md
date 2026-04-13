# Hardware Settings Registry

## Problem

Hardware settings (sample rates, voltage ranges, channel counts, etc.) are
defined inside constructors via `setDefault`/`setArray` calls. This means:

1. **Settings are invisible until construction.** A user creating a new profile
   cannot see or modify settings like AWG sample rate or digitizer sample rates
   until *after* the object is created. They must then navigate to
   Hardware > [device] to open the `HWDialog` and edit raw key/value pairs.
2. **No metadata.** Settings have no labels, descriptions, or priority
   information. The `HWDialog` tree view shows raw `SettingsStorage` keys
   (e.g., `sampleRateHz`) with no context about what they mean or which ones
   matter most.
3. **Two separate systems.** The `HwConfigParam`/`REGISTER_HARDWARE_PARAMS`
   system handles pre-construction "required" parameters, while `setDefault`
   handles everything else. These serve the same conceptual purpose but use
   entirely different code paths and data structures.

## Goal

A unified settings registry where every hardware implementation declares its
settings (with metadata) via static registration macros. Settings are available
before object construction, presented to the user at profile creation time, and
serve as the single source of truth for defaults.

## Design

### Data Structures

```cpp
/*!
 * \brief Priority level for hardware settings
 *
 * Controls visibility in the profile creation dialog and HWDialog:
 * - Required: Must be set before construction. Shown prominently.
 *   May not be editable after profile creation.
 * - Important: Has a sensible default but the user should review it.
 *   Shown in the main settings area.
 * - Optional: Rarely needs changing. Shown under a collapsible
 *   "Advanced Settings" section.
 */
enum class HwSettingPriority { Required, Important, Optional };

/*!
 * \brief Scalar setting definition with metadata
 *
 * Registered statically at program startup. The defaultValue's QVariant
 * type determines the UI widget (int -> QSpinBox, double -> QDoubleSpinBox,
 * bool -> QCheckBox, QString -> QLineEdit).
 */
struct HwSettingDef {
    QString key;              ///< SettingsStorage key
    QString label;            ///< User-facing display label
    QString description;      ///< Explanatory tooltip/help text
    QVariant defaultValue;    ///< Type-aware default value
    QVariant minimum;         ///< Optional min for numeric types (invalid = no limit)
    QVariant maximum;         ///< Optional max for numeric types (invalid = no limit)
    HwSettingPriority priority = HwSettingPriority::Optional;
};

/*!
 * \brief Array setting definition with metadata
 *
 * Describes an array-type setting (e.g., sampleRates). The entries vector
 * holds the default array contents; each entry is a SettingsMap.
 */
struct HwArraySettingDef {
    QString key;              ///< SettingsStorage array key
    QString label;            ///< User-facing display label
    QString description;      ///< Explanatory tooltip/help text
    std::vector<SettingsStorage::SettingsMap> entries;  ///< Default entries
    HwSettingPriority priority = HwSettingPriority::Optional;
};
```

### Registry API Additions

New fields in `HardwareRegistration`:

```cpp
struct HardwareRegistration {
    // ... existing fields ...
    QVector<HwSettingDef> settingDefs;
    QMap<QString, HwArraySettingDef> arraySettingDefs;
};
```

New methods in `HardwareRegistry`:

```cpp
// Scalar settings (bulk registration, one call per class)
bool addSettingDefs(const QString& key, const QString& subKey,
                    const QVector<HwSettingDef>& settings);
QVector<HwSettingDef> getSettingDefs(
    const QString& key, const QString& subKey) const;

// Array setting metadata (one call per array key per class)
bool addArraySettingDef(const QString& key, const QString& subKey,
                        const QString& arrayKey, const QString& label,
                        const QString& description, HwSettingPriority priority);

// Array setting entries (one call per entry)
bool addArraySettingEntry(const QString& key, const QString& subKey,
                          const QString& arrayKey,
                          const SettingsStorage::SettingsMap& entry);

// Retrieve array settings
QMap<QString, HwArraySettingDef> getArraySettingDefs(
    const QString& key, const QString& subKey) const;
```

### Registration Macros

In `hardwareregistration.h`:

```cpp
/// Register scalar settings for a hardware class.
/// Usage:
///   REGISTER_HARDWARE_SETTINGS(MyClass,
///       {key, "Label", "Description", defaultVal, min, max, priority},
///       ...
///   )
#define REGISTER_HARDWARE_SETTINGS(CLASS, ...) \
    static bool settings_registered_##CLASS = \
        HardwareRegistry::instance().addSettingDefs( \
            findHardwareBaseType(&CLASS::staticMetaObject), \
            QString(CLASS::staticMetaObject.className()), \
            {__VA_ARGS__} \
        );

/// Register array setting metadata (call once per array key).
/// Usage:
///   REGISTER_HARDWARE_ARRAY(MyClass, keyConstant,
///       "Display Label", "Description", HwSettingPriority::Important)
#define REGISTER_HARDWARE_ARRAY(CLASS, ARRAY_KEY, LABEL, DESC, PRIORITY) \
    static bool BC_ARRDEF_VAR(CLASS, ARRAY_KEY) = \
        HardwareRegistry::instance().addArraySettingDef( \
            findHardwareBaseType(&CLASS::staticMetaObject), \
            QString(CLASS::staticMetaObject.className()), \
            ARRAY_KEY, LABEL, DESC, PRIORITY);

/// Register one entry in an array setting (call once per entry).
/// Usage:
///   REGISTER_HARDWARE_ARRAY_ENTRY(MyClass, keyConstant,
///       {{subKey1, value1}, {subKey2, value2}})
#define REGISTER_HARDWARE_ARRAY_ENTRY(CLASS, ARRAY_KEY, ...) \
    static bool BC_ARRENTRY_VAR(CLASS, __COUNTER__) = \
        HardwareRegistry::instance().addArraySettingEntry( \
            findHardwareBaseType(&CLASS::staticMetaObject), \
            QString(CLASS::staticMetaObject.className()), \
            ARRAY_KEY, \
            SettingsStorage::SettingsMap{__VA_ARGS__} \
        );

// Helpers for unique static variable names
#define BC_ARRDEF_CONCAT(a, b) a##b
#define BC_ARRDEF_VAR(CLASS, KEY) BC_ARRDEF_CONCAT(arraydef_##CLASS##_, __LINE__)
#define BC_ARRENTRY_CONCAT(a, b) a##b
#define BC_ARRENTRY_VAR(CLASS, N) BC_ARRENTRY_CONCAT(arrayentry_##CLASS##_, N)
```

### Usage Examples

#### AWG (scalar settings only)

```cpp
// awg70002a.cpp
REGISTER_HARDWARE_META(AWG70002a, "Tektronix AWG70002A high-performance AWG")
REGISTER_HARDWARE_PROTOCOLS(AWG70002a, CommunicationProtocol::Tcp)
REGISTER_HARDWARE_SETTINGS(AWG70002a,
    {BC::Key::AWG::rate,      "Sample Rate (Hz)", "DAC output sample rate",
     16e9, 1e6, 100e9, HwSettingPriority::Important},
    {BC::Key::AWG::samples,   "Max Samples",      "Maximum waveform sample count",
     2e9, 0.0, {}, HwSettingPriority::Optional},
    {BC::Key::AWG::min,       "Min Freq (MHz)",    "Minimum chirp frequency",
     100.0, 0.0, {}, HwSettingPriority::Optional},
    {BC::Key::AWG::max,       "Max Freq (MHz)",    "Maximum chirp frequency",
     6250.0, 0.0, {}, HwSettingPriority::Optional},
    {BC::Key::AWG::prot,      "Protection Pulse",  "AWG outputs a protection pulse",
     true, {}, {}, HwSettingPriority::Optional},
    {BC::Key::AWG::amp,       "Amp Enable Pulse",  "AWG outputs an amplifier enable pulse",
     true, {}, {}, HwSettingPriority::Optional},
    {BC::Key::AWG::rampOnly,  "Ramp Only",         "Restrict to linear ramp chirps",
     false, {}, {}, HwSettingPriority::Optional},
    {BC::Key::AWG::triggered, "Triggered",         "AWG waits for external trigger",
     true, {}, {}, HwSettingPriority::Optional}
)
```

#### FtmwScope (scalar + array settings)

```cpp
// virtualftmwscope.cpp
REGISTER_HARDWARE_META(VirtualFtmwScope, "Virtual FTMW digitizer for testing")
REGISTER_HARDWARE_PROTOCOLS(VirtualFtmwScope, CommunicationProtocol::Virtual, ...)
REGISTER_HARDWARE_SETTINGS(VirtualFtmwScope,
    {numAnalogChannels,  "Analog Channels",  "Number of analog inputs",
     4, 1, 32, HwSettingPriority::Required},
    {numDigitalChannels, "Digital Channels",  "Number of digital inputs",
     0, 0, 32, HwSettingPriority::Required},
    {hasAuxTriggerChannel, "Aux Trigger",     "Has auxiliary trigger input",
     true, {}, {}, HwSettingPriority::Optional},
    {bandwidth,          "Bandwidth (MHz)",   "Analog bandwidth",
     16000.0, {}, {}, HwSettingPriority::Important},
    // ... remaining settings ...
)

using namespace BC::Key::Digi;
REGISTER_HARDWARE_ARRAY(VirtualFtmwScope, sampleRates,
    "Sample Rates", "Available digitizer sample rates",
    HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFtmwScope, sampleRates,
    {{srText, "2 GSa/s"}, {srValue, 2e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFtmwScope, sampleRates,
    {{srText, "5 GSa/s"}, {srValue, 5e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFtmwScope, sampleRates,
    {{srText, "10 GSa/s"}, {srValue, 10e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFtmwScope, sampleRates,
    {{srText, "20 GSa/s"}, {srValue, 20e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFtmwScope, sampleRates,
    {{srText, "50 GSa/s"}, {srValue, 50e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(VirtualFtmwScope, sampleRates,
    {{srText, "100 GSa/s"}, {srValue, 100e9}})
```

### Constructor Integration

The registry becomes the single source of truth for default values.
A new protected method in `HardwareObject` applies registered defaults:

```cpp
void HardwareObject::applyRegisteredSettings()
{
    auto& reg = HardwareRegistry::instance();

    // Scalar settings
    for (const auto& s : reg.getSettingDefs(d_hwType, d_model))
        setDefault(s.key, s.defaultValue);

    // Array settings
    auto arrays = reg.getArraySettingDefs(d_hwType, d_model);
    for (auto it = arrays.cbegin(); it != arrays.cend(); ++it) {
        if (!containsArray(it.key()) && !it.value().entries.empty())
            setArray(it.key(), it.value().entries);
    }
}
```

#### Where to call `applyRegisteredSettings()`

This should be called in the `HardwareObject` base constructor, after
the base-class setup (key, model, comm type) but before `save()`. Every
subclass gets it for free. Subclasses that have no registered settings
will simply iterate empty vectors (a no-op).

This means subclass constructors no longer need any `setDefault`/`setArray`
calls for settings that are in the registry. The only `setDefault` calls
remaining in subclass constructors would be for settings that are
intentionally *not* registered (if any exist).

### Profile Creation Flow

When a user adds a profile in `RuntimeHardwareConfigDialog`:

1. User selects hardware type and implementation
2. Dialog queries `HardwareRegistry::getSettingDefs()` and
   `getArraySettingDefs()` for the selected implementation
3. Dialog displays settings grouped by priority:
   - **Required**: Prominent form fields at the top. Must be filled.
   - **Important**: Visible form fields below Required. Pre-filled with
     defaults, user should review.
   - **Optional**: Hidden under a collapsible "Advanced Settings" group
     box. Tree view showing all remaining settings (HWSettingsModel-style).
4. User edits values and clicks OK
5. Dialog writes all values to QSettings under the hardware's
   `SettingsStorage` key (before the object is constructed)
6. On next startup, `HardwareObject` constructor calls
   `applyRegisteredSettings()` -> `setDefault()` finds existing values
   -> user edits are preserved

### HWDialog Enhancement

The existing `HWDialog` (Hardware > [device]) should also be updated to
use the registry metadata:

- Display `label` instead of raw key names
- Show `description` as tooltip text
- Group settings by priority (Required settings may be read-only after
  initial creation; Important/Optional are always editable)
- The `forbiddenKeys` mechanism is superseded by the registry: only
  registered settings are shown, and communication-related keys are
  excluded by nature (they are not registered)

This is a follow-up enhancement. The initial implementation can keep
the existing `HWDialog` unchanged and focus on the creation-time
experience.

### Deprecation of HwConfigParam

Once all hardware classes using `REGISTER_HARDWARE_PARAMS` have been
migrated to `REGISTER_HARDWARE_SETTINGS` with `HwSettingPriority::Required`,
the following are removed:

- `HwConfigParam` struct
- `REGISTER_HARDWARE_PARAMS` macro
- `HardwareRegistry::addConfigParams()` / `getConfigParams()`
- `HardwareRegistration::configParams` field
- `configParams()` static methods on trampoline classes

The RuntimeHardwareConfigDialog code that reads `configParams` is replaced
by the new priority-based rendering from `settingDefs`/`arraySettingDefs`.

### Communication Settings

Communication defaults (timeout, termination character) are **not**
registered in this system. They belong to the communication protocol
and should be configured in the CommunicationDialog alongside baud rate,
IP address, etc. A separate enhancement to the CommunicationDialog
should expose these settings there.

## Implementation Plan

### Phase 1: Registry Infrastructure **COMPLETE**

**Files modified:** `hardwareregistry.h`, `hardwareregistry.cpp`,
`hardwareregistration.h`, `hardwareobject.h`, `hardwareobject.cpp`

- Added `HwSettingPriority` enum, `HwSettingDef`, `HwArraySettingDef`
  structs to `hardwareregistry.h`
- Added `settingDefs` and `arraySettingDefs` fields to
  `HardwareRegistration`
- Implemented `addSettingDefs()`, `addArraySettingDef()`,
  `addArraySettingEntry()` and corresponding getters in
  `HardwareRegistry` (thread-safe, following existing patterns)
- Added `REGISTER_HARDWARE_SETTINGS`, `REGISTER_HARDWARE_ARRAY`,
  `REGISTER_HARDWARE_ARRAY_ENTRY` macros to `hardwareregistration.h`
- Added private `applyRegisteredSettings(const QString& hwType)` to
  `HardwareObject`, called from base constructor before `save()`

### Phase 2: Pilot Migration (4 classes) **COMPLETE**

**Files modified:** `virtualawg.cpp`, `pythonawg.cpp`,
`virtualftmwscope.cpp`, `pythonftmwscope.cpp`

- All four classes now use `REGISTER_HARDWARE_SETTINGS` macros with
  full metadata (labels, descriptions, priorities)
- VirtualFtmwScope and PythonFtmwScope also use `REGISTER_HARDWARE_ARRAY`
  and `REGISTER_HARDWARE_ARRAY_ENTRY` for sample rates
- Constructor `setDefault`/`setArray`/`save()` calls removed from all
  four classes (handled by `applyRegisteredSettings()` in base)
- PythonFtmwScope: `numAnalogChannels` and `numDigitalChannels` are
  `HwSettingPriority::Required`; `REGISTER_HARDWARE_PARAMS` and
  `configParams()` kept for backward compatibility until Phase 3
- All 26 tests pass, build clean

### Phase 3: Creation-Time UI **COMPLETE**

**Files modified:** `src/gui/dialog/addprofiledialog.h`,
`addprofiledialog.cpp`, `src/hardware/python/pythonftmwscope.h`,
`pythonftmwscope.cpp`

`AddProfileDialog::updateConfigParams()` was replaced with
`updateSettingsDefs()`, which builds three priority-grouped sections:

- **Required Settings** / **Important Settings**: `QFormLayout`-based,
  always visible when non-empty. Scalar widgets use `QSpinBox` (with
  `INT_MIN`/`INT_MAX` when no explicit range), `ScientificSpinBox` for
  doubles, `QCheckBox` for bools, and `QLineEdit` for strings. Tooltips
  show `HwSettingDef::description`. Array settings at these priorities
  are shown as read-only count labels ("N entries").

- **Advanced Settings**: checkable/collapsible `QGroupBox`, collapsed
  by default. Optional scalar settings are displayed in a scrollable
  two-column `QTableWidget` (Setting / Value), keeping the dialog
  compact even for implementations with many optional parameters.

A backward-compatibility fallback to `getConfigParams()` remains for
hardware classes not yet migrated to `REGISTER_HARDWARE_SETTINGS`
(these display under a "Configuration Parameters" group). The fallback
will be removed once Phase 4 is complete.

`REGISTER_HARDWARE_PARAMS` and `configParams()` were removed from
`PythonFtmwScope`, whose Required settings are now fully covered by its
`REGISTER_HARDWARE_SETTINGS` entries.

### Phase 4: Full Migration **COMPLETE**

All hardware implementations and base classes have been migrated to
`REGISTER_HARDWARE_SETTINGS`. The `dev-docs/phase4-settings/` working
directory has been removed.

#### Base Class Registrations (Complete)

| Class | Settings |
|---|---|
| HardwareObject | `critical`, `rInterval` |
| Clock | `manualTune` |
| FlowController | `interval` |
| IOBoard | `isTriggered` |
| TemperatureController | `interval` |

Classes with no base-class settings to register:
FtmwScope, AWG, PressureController, PulseGenerator, GpibController,
LifScope, LifLaser.

#### Implementation Migration (Complete)

| Hardware Type | Implementations |
|---|---|
| FtmwScope | ✓ 8 / 8 |
| ChirpSource | ✓ 8 / 8 |
| Clock | ✓ 5 / 5 |
| FlowController | ✓ 4 / 4 |
| PressureController | ✓ 3 / 3 |
| PulseGenerator | ✓ 7 / 7 |
| IOBoard | ✓ 3 / 3 |
| GpibController | ✓ 4 / 4 |
| LifScope | ✓ 4 / 4 |
| LifLaser | ✓ 4 / 4 |
| TemperatureController | ✓ 3 / 3 |

### Phase 5: Shared Settings Widget + HWDialog Replacement

**Status: Partially complete — `AddProfileDialog` wired; `HWDialog` deferred until Phase 4 is finished**

Rather than patching `HWSettingsModel` and keeping two divergent
code paths, Phase 5 introduces a shared `HwSettingsWidget` that both
`AddProfileDialog` and `HWDialog` embed. This gives a consistent
experience and a single maintenance point for the settings UI.

#### New classes

**`src/gui/widget/hwsettingswidget.h/.cpp`** — `HwSettingsWidget` ✓ Done

An embeddable `QWidget` that renders hardware settings from registry
metadata in two modes:

- `HwSettingsMode::Create` — Required settings shown as an editable
  `QFormLayout` (typed widgets). Used in `AddProfileDialog`.
- `HwSettingsMode::Edit` — Required settings shown as read-only text
  in a `QFormLayout`. Intended for `HWDialog` (not yet wired).

Settings are presented in a `QTabWidget` (fixed 350 px tall):
- **Settings tab**: Required group box (`QFormLayout`) + Important
  group box (two-column `QTableWidget`, Setting | Value). Either or
  both are hidden when empty.
- **Advanced tab**: second two-column `QTableWidget` in a `QScrollArea`;
  tab is only added when Optional/Advanced settings exist.
- If no settings exist at all, the tab widget is hidden and a
  "No settings available." `QLabel` is shown instead.

Row rendering within each table:
- Scalar rows: typed widget (`QSpinBox`, `ScientificSpinBox`,
  `QCheckBox`, or `QLineEdit`) in the value column.
- Array rows: inline `QLabel("N entries")` + `QPushButton("Edit...")`
  in the value column; button opens `HwArrayEditDialog`.

Each table uses `AdjustToContents` + `ScrollBarAlwaysOff`; scrolling
is handled by the `QScrollArea` wrapping it in each tab.

Public API:
```cpp
HwSettingsWidget(const QString& hwType, const QString& impl,
                 HwSettingsMode mode,
                 const QString& storageKey = {},  // non-empty → load current values from SettingsStorage
                 QWidget* parent = nullptr);

QHash<QString, QVariant> values() const;   // scalar values
QMap<QString, std::vector<SettingsStorage::SettingsMap>> arrayValues() const;
void saveToStorage(const QString& storageKey) const;
```

**`src/gui/dialog/hwarrayeditdialog.h/.cpp`** — `HwArrayEditDialog` ✓ Done

Sub-dialog opened from an array row's Edit button. Shows entries in a
`QTableWidget` (rows = entries, columns = sub-keys). Provides Add,
Remove, Move Up, Move Down buttons. Column names come from the
registered `HwArraySettingDef` entries, falling back to the keys of
the first stored entry.

```cpp
HwArrayEditDialog(const QString& label,
                  const QStringList& subKeys,
                  std::vector<SettingsStorage::SettingsMap> entries,
                  QWidget* parent = nullptr);

std::vector<SettingsStorage::SettingsMap> result() const;
```

#### Changes to existing classes

**`AddProfileDialog`** ✓ Done: Hosts an `HwSettingsWidget` (Create mode)
inside a plain container widget; dialog minimum width is 520 px.
`updateSettingsDefs()` deletes the old widget and constructs a new one
for the selected implementation. `accept()` calls
`widget->saveToStorage(key)` before creating the profile.

**`HWDialog`** — Pending (after Phase 4): Remove the `QTreeView`,
`HWSettingsModel` include, and `insertBefore`/`insertAfter`/`remove`
slots. Replace with an `HwSettingsWidget` instance (Edit mode,
initialized with the existing storage key). Because `HWDialog` hosts a
hardware-specific control widget (which can be complex, e.g. the pulse
generator channel table) and optionally a `PythonSettingsWidget`, the
`HwSettingsWidget` should be placed in its own tab within `HWDialog`'s
tab widget rather than stacked with the control widget.
`accept()` calls `widget->saveToStorage(key)`.

**`HWSettingsModel`** (`src/data/model/hwsettingsmodel.h/.cpp`):
Delete both files once HWDialog no longer depends on them.

## Open Questions

1. **Static initialization order:** The `REGISTER_HARDWARE_ARRAY_ENTRY`
   macros use `__COUNTER__` for unique variable names. All static
   registrations for a given translation unit execute in definition
   order, so entries within a single `.cpp` file are ordered correctly.
   Cross-TU ordering doesn't matter since each class's entries are
   independent. Verify this assumption holds on all target compilers
   (GCC, Clang, MSVC).

2. **Python hardware defaults:** PythonAwg and PythonFtmwScope register
   "generic" defaults (they don't know the actual hardware). These serve
   as sensible starting points that the user is expected to customize.
   The priority system (Important for key settings) draws attention to
   the ones that matter most.

3. **Settings that vary by instance:** Some multi-instance hardware
   types (e.g., multiple Clocks) might want different defaults per
   instance. The registry is per-implementation, not per-instance.
   Instance-specific customization happens at profile creation time
   or in the HWDialog. This is the correct behavior -- the registry
   defines the *defaults*, not the final values.
