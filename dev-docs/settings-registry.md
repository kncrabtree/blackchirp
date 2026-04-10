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

### Phase 3: Creation-Time UI

**Prerequisite:** `runtimehardwareconfigdialog.cpp` refactor **COMPLETE**.

**Goal:** Replace the `configParams`-based UI in the "Add Profile" dialog
with a priority-grouped UI driven by `getSettingDefs()` and
`getArraySettingDefs()`.

#### Current configParams UI (to be replaced)

The add-profile dialog is now a standalone class: `AddProfileDialog`
(`src/gui/dialog/addprofiledialog.h` / `addprofiledialog.cpp`).
It is instantiated by `RuntimeHardwareConfigDialog::onAddProfile()`.

The config params UI lives in `AddProfileDialog::updateConfigParams()`
(`addprofiledialog.cpp`), which:

1. Calls `HardwareRegistry::instance().getConfigParams(hardwareType, impl)`
2. For each `HwConfigParam`, creates a widget based on `defaultValue`
   type: `int` -> `QSpinBox`, `double` -> `QDoubleSpinBox`,
   `bool` -> `QCheckBox`, else -> `QLineEdit`. Min/max applied if valid.
3. Adds widgets to `p_configParamsLayout` (a `QFormLayout`) inside
   `p_configParamsGroup` (a `QGroupBox` labeled "Configuration Parameters",
   hidden when params list is empty)
4. Stores widgets in `d_paramWidgets` (`QHash<QString, QWidget*>`)
5. On dialog accept (`AddProfileDialog::accept()`), iterates
   `d_paramWidgets`, extracts values via `qobject_cast`, writes each to
   `QSettings` under the hardware's settings key before the hardware
   object is constructed

`updateConfigParams()` is connected to `p_implementationCombo`'s
`currentTextChanged` signal so it rebuilds when the user changes
implementation.

#### New UI structure

In `AddProfileDialog`, replace the single `p_configParamsGroup` with
three member group boxes:

```
p_requiredParamsGroup  (QGroupBox "Required Settings",  always visible when non-empty)
  - p_requiredParamsLayout (QFormLayout)
  - For HwSettingPriority::Required entries
  - These correspond to what configParams used to handle

p_importantParamsGroup (QGroupBox "Important Settings", always visible when non-empty)
  - p_importantParamsLayout (QFormLayout)
  - For HwSettingPriority::Important entries

p_advancedParamsGroup  (QGroupBox "Advanced Settings",  checkable/collapsible, collapsed by default)
  - p_advancedParamsLayout (QFormLayout)
  - For HwSettingPriority::Optional entries
  - Array settings shown as read-only summary (e.g., "6 sample rates")
```

`updateConfigParams()` is renamed `updateSettingsDefs()` and rebuilds
all three groups. The old `getConfigParams()` fallback path can live
alongside the new path during the transition (see Backward compatibility
below).

#### Widget creation

Reuse the same type-dispatch logic already in `updateConfigParams()`:
- `int` -> `QSpinBox` with min/max from `HwSettingDef`
- `double` -> `QDoubleSpinBox` with min/max
- `bool` -> `QCheckBox`
- `QString` -> `QLineEdit`
- Set `widget->setToolTip(setting.description)` for all widgets
- Use `setting.label` as the form row label

All widgets go into the same `d_paramWidgets` map (keyed by
`HwSettingDef::key`). The settings-writing logic in `accept()` is
identical to the current configParams flow — no changes needed there.

#### Backward compatibility

During the transition (before Phase 4 migrates all classes):
- Query `getSettingDefs()` first. If non-empty, use the new UI.
- If `getSettingDefs()` returns empty, fall back to `getConfigParams()`
  with the old UI behavior.
- Classes with BOTH (like PythonFtmwScope) should use the new UI;
  the old `configParams` entries are a subset of the `settingDefs`
  Required entries, so the new UI is strictly more complete.

#### After Phase 3 is complete

- The old `configParams`-based UI code path can be removed
- `REGISTER_HARDWARE_PARAMS` and `configParams()` on PythonFtmwScope
  can be removed (its Required settings are now in `settingDefs`)
- Other classes that still use only `configParams` will continue to
  work via the fallback until they are migrated in Phase 4

#### API methods to use

```cpp
// Get scalar settings (returns empty QVector if none registered)
QVector<HwSettingDef> settings =
    HardwareRegistry::instance().getSettingDefs(hardwareType, impl);

// Get array settings (returns empty QMap if none registered)
QMap<QString, HwArraySettingDef> arrays =
    HardwareRegistry::instance().getArraySettingDefs(hardwareType, impl);

// Existing fallback (returns empty QVector if none registered)
QVector<HwConfigParam> params =
    HardwareRegistry::instance().getConfigParams(hardwareType, impl);
```

#### Test cases

- Hardware type with no registered settings -> empty dialog (old behavior)
- VirtualAwg -> 1 Important (rate), 7 Optional, no arrays
- VirtualFtmwScope -> 2 Required (channels), 1 Important (bandwidth),
  18 Optional, 1 Important array (sample rates shown as summary)
- PythonFtmwScope -> same as VirtualFtmwScope minus `interval`, has
  both old configParams AND new settingDefs (new UI takes precedence)

### Phase 4: Full Migration

Migrate all remaining hardware implementations to
`REGISTER_HARDWARE_SETTINGS` (and array macros where applicable). This
includes all classes in:

- `src/hardware/core/ftmwdigitizer/` (Dpo71254b, M4i2220x8, etc.)
- `src/hardware/optional/chirpsource/` (AWG70002a, AD9914, etc.)
- `src/hardware/optional/` (flow controllers, pressure controllers, etc.)
- `src/hardware/core/clock/` (all clock implementations)
- All remaining Python trampolines

For each hardware type, scan though the implementations to collect
the current setDefault() calls. For the ones that are common between
implementations, suggest recommendations for which priority
each setting should be, and ask the user to confirm or change them
before finalizing the decision. Any `configParams()` must be Required.
For settings that show up only in a single implementation (or a subset),
ask the user if that setting should be:
- Added to the other implementations with an Optional or Important setting,
- Included as an Optional or Important priority setting for that
  implementation only,
- Left as a setDefault call in the constructor that is explicitly not 
  registered (i.e., a power user feature), or
- Removed entirely. In this case, scan the codebase to find any references
  to that setting that may be affected and present to the user for final 
  confirmation.
Once all parameters are confirmed, write the results to a file for use
in the implementation phase. **Note: This also applies to the classes that
were part of the pilot migration.** Some optional/important decisions may need
to be revisited; we did not walk through those prior to implementing 
the proof of concept.

To implement, for each class:
1. Add registration macros
2. Remove constructor `setDefault`/`setArray` calls
3. Migrate any `configParams()` to Required-priority settings
4. Remove `REGISTER_HARDWARE_PARAMS` and `configParams()` static method

After all classes are migrated, remove `HwConfigParam`,
`REGISTER_HARDWARE_PARAMS`, `addConfigParams()`, and `getConfigParams()`.

### Phase 5: HWDialog Enhancement (optional follow-up)

Update `HWSettingsModel` and `HWDialog` to use registry metadata:

- Replace raw key display with `label` from `HwSettingDef`
- Add `description` as tooltip
- Group by priority; mark Required settings read-only
- Replace `forbiddenKeys` filtering with registry-based visibility

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
