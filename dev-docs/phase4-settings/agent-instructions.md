# Phase 4 Step 3: Hardware Settings Migration — Agent Instructions

These instructions are given to each Haiku agent performing per-type migrations.
Agents work **directly in the source tree** (no worktrees). Do **not** build the
code; the user will build and fix any errors after the agent completes.
Use `codebase-memory-mcp` (project: `home-kncrabtree-github-blackchirp-src`)
if you need to explore the codebase beyond the files explicitly listed.

---

## What the migration does

Hardware classes currently call `setDefault(key, value)` in their constructors
to establish default values at runtime. We are replacing these with static
`REGISTER_HARDWARE_SETTINGS` macros that register settings before construction,
making them available in the profile-creation UI and the future HWDialog.

---

## Reference examples

Study these fully-migrated files before writing any code:

- `src/hardware/core/ftmwdigitizer/virtualftmwscope.cpp` — scalar settings,
  bool/numeric types, `QVariant{}` for absent min/max, array registration
- `src/hardware/optional/flowcontroller/flowcontroller.cpp` — base-class
  registration for a single Optional setting
- `src/hardware/optional/flowcontroller/virtualflowcontroller.cpp` — simple
  implementation with only common settings
- `src/hardware/optional/flowcontroller/mks946.cpp` — implementation with
  impl-specific settings alongside common settings

---

## Information sources

For each migration job you receive, you will be told:
- The **hardware type** (e.g., `PressureController`)
- The **TOML file** in `dev-docs/phase4-settings/` containing approved settings,
  labels, tooltips, priorities, defaults, min/max, and decisions
- The **base class section** in `dev-docs/phase4-settings/baseclass.toml` if
  the base class needs registration (not all types do)

Read these TOML files fully before touching any source file.

---

## Macro format

All macros go in the `.cpp` file at file scope, **after** existing
`REGISTER_HARDWARE_META` and `REGISTER_HARDWARE_PROTOCOLS` lines and
**before** the constructor.

```cpp
#include <hardware/core/hardwareregistration.h>   // must be present

// Already present — do NOT remove or duplicate:
REGISTER_HARDWARE_META(ClassName, "description")
REGISTER_HARDWARE_PROTOCOLS(ClassName, ...)

// Add after those:
REGISTER_HARDWARE_SETTINGS(ClassName,
    // Numeric with min and max:
    {key, "Label", "Tooltip text.", defaultValue, minValue, maxValue, HwSettingPriority::Important},
    // Numeric with min only (no upper bound):
    {key, "Label", "Tooltip text.", defaultValue, minValue, QVariant{}, HwSettingPriority::Optional},
    // Bool or string (no meaningful min/max):
    {key, "Label", "Tooltip text.", defaultValue, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    // QString default must be written as QString("..."), not bare "...":
    {key, "Label", "Tooltip text.", QString("kTorr"), QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

// Arrays (only when the TOML has an [arrays.KEY] entry with decision != "exclude"):
REGISTER_HARDWARE_ARRAY(ClassName, arrayKey,
    "Array Label", "Array tooltip.", HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(ClassName, arrayKey,
    {{subKey1, value1}, {subKey2, value2}})
// ...one REGISTER_HARDWARE_ARRAY_ENTRY per static entry
```

Priority values: `HwSettingPriority::Required`, `HwSettingPriority::Important`,
`HwSettingPriority::Optional`.

---

## Changes required per file

### Base class `.cpp` (if baseclass.toml lists settings for this type)

1. Add `#include <hardware/core/hardwareregistration.h>` if not present.
2. Add `REGISTER_HARDWARE_SETTINGS(BaseClassName, ...)` using the values from
   `baseclass.toml`.
3. Remove the corresponding `setDefault(...)` call(s) from the constructor.
4. Do **not** add a `readSettings()` override unless specifically instructed.

### Each implementation `.cpp`

1. Add `#include <hardware/core/hardwareregistration.h>` if not present.
2. Add `REGISTER_HARDWARE_SETTINGS(ImplClassName, ...)` containing:
   - All `[settings.KEY]` entries from the TOML (common to all impls), using
     this implementation's default value from the reference table.
   - Any `[partial.KEY]` entries where `decision` is `add_to_all` or
     `add_to_missing` and this impl is in the missing list.
   - Any `[impl_only.ImplName.KEY]` entries for this specific implementation.
   - Skip any entry whose `decision` is `setdefault_only`, `existing_only`,
     `remove`, or `exclude`.
3. Remove from the constructor every `setDefault(key, ...)` call for keys that
   are now registered. **Do not** remove:
   - `setDefault` calls for `BC::Key::Comm::*` keys (communication settings)
   - `setArray`, `setArrayValue`, `appendArrayMap`, `containsArray` calls
     (these manage runtime state, not registration)
   - Any non-`setDefault` constructor logic
4. If the constructor body becomes empty after removals, leave it intact.

### Migrating `configParams()` (Python hardware and some others)

Some classes use the old pre-construction parameter system. These classes have:
- `REGISTER_HARDWARE_PARAMS(ClassName)` in the `.cpp`
- A `static QVector<HwConfigParam> configParams()` method in `.cpp` and `.h`

**Migration steps:**

In the `.cpp`:
1. Remove the `REGISTER_HARDWARE_PARAMS(ClassName)` line.
2. Identify all entries in `configParams()`. Each one maps to a `Required`
   setting in the TOML (look for `[settings.KEY]` or `[impl_only.IMPL.KEY]`
   with `priority = "Required"`).
3. Include these keys as `HwSettingPriority::Required` entries in
   `REGISTER_HARDWARE_SETTINGS`.
4. Delete the entire `configParams()` function body.

In the `.h`:
5. Remove the `static QVector<HwConfigParam> configParams();` declaration.

**Example — before:**
```cpp
// In .cpp:
REGISTER_HARDWARE_PARAMS(PythonPressureController)

QVector<HwConfigParam> PythonPressureController::configParams()
{
    using namespace BC::Key::PController;
    return {
        { readOnly, QStringLiteral("Read Only"), QVariant(false), 0, 0 },
    };
}
```
```cpp
// In .h:
static QVector<HwConfigParam> configParams();
```

**Example — after:**
```cpp
// In .cpp: REGISTER_HARDWARE_PARAMS removed; readOnly is now Required in macro:
REGISTER_HARDWARE_SETTINGS(PythonPressureController,
    {BC::Key::PController::readOnly, "Read Only",
     "If enabled, this controller is monitored but not written to.",
     false, QVariant{}, QVariant{}, HwSettingPriority::Required},
    // ...other settings...
)
// configParams() function deleted entirely
```
```cpp
// In .h: configParams() declaration deleted
```

---

## Decision field reference

When reading TOML `[partial.KEY]` or `[impl_only.IMPL.KEY]` entries:

| `decision` value | Action |
|---|---|
| `add_to_all` | Register in all implementations |
| `add_to_missing` | Register only in implementations listed in `missing_from` |
| `existing_only` | Register only in implementations that already setDefault it |
| `setdefault_only` | Leave the existing setDefault in place; do not register |
| `remove` | Remove the setDefault; do not register |
| `exclude` | Leave completely as-is; skip |

---

## Constraints

- **Do not build.** The user builds after all agents finish.
- **Do not use worktrees.** Edit files in place.
- **Do not modify `.h` files** except to remove a `configParams()` declaration
  when migrating that pattern.
- **Do not remove** `REGISTER_HARDWARE_META` or `REGISTER_HARDWARE_PROTOCOLS`.
- **Do not remove** `setDefault` calls for `BC::Key::Comm::*` keys.
- **Do not remove** `setArray`, `setArrayValue`, `appendArrayMap`, or
  `containsArray` calls — these manage runtime state.
- **Do not remove** any non-`setDefault` constructor logic.
- **Do not add** error handling, comments, or refactors beyond what is listed.
- **Preserve** all existing `using namespace` statements and includes.
- After making changes, grep the modified `.cpp` files to confirm no
  Flow/Hw-settings-related `setDefault` calls remain (only `Comm::*` may stay).
