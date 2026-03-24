# Unified Application Configuration Dialog

## Motivation

Application settings are currently scattered across multiple access points:
- Font: Settings menu -> `QFontDialog`
- Save path: Settings menu -> `BCSavePathDialog`
- LIF/CUDA: no UI (compile-time flags, hardcoded in `ApplicationConfigManager`)
- Debug logging: no UI (signal plumbing exists but no toggle exposed)

There is no unified place to configure application-level options, and no first-run
onboarding experience -- the save path dialog appears on first launch but LIF and other
options are invisible to the user.

## Declarative Option Registry

Define options as data so the UI can be generated automatically:

```cpp
struct AppOption {
    QString settingsKey;       // QSettings key
    QString label;             // Display name for UI
    QString description;       // Tooltip / help text
    QVariant defaultValue;     // Type-aware default (bool, QString, QFont, etc.)
    bool requiresRestart;      // If true, show restart badge in UI
};
```

The dialog iterates the option list and renders appropriate widgets per QVariant type:
- `bool` -> checkbox
- `QFont` -> font preview label + "Change..." button -> `QFontDialog`
- Future types added by extending the renderer

Adding a new boolean toggle becomes a one-line registration -- no new dialog or menu
action required.

## Current Registry Options

| Setting | Type | Restart? | Notes |
|---------|------|----------|-------|
| LIF enabled | bool | Yes | Gates hardware registration, UI construction, experiment pages |
| Debug logging | bool | No | Signal already wired to `LogHandler` |
| Application font | QFont | No | Applied immediately via `QApplication::setFont()` |

CUDA is hidden and disabled for now -- the code needs significant work before revival.
It should not appear in the UI until it is ready.

## Save Path Panel

The save path configuration (currently `BCSavePathDialog`) is NOT part of the option
registry -- its setup logic (directory creation, experiment number scanning, path
validation) is too complex for the generic renderer. Instead, the existing save path
UI is embedded as a **separate panel** within the application config dialog, alongside
the auto-generated registry panel. This keeps the save path's custom logic intact
while consolidating everything into one dialog.

## Restart Handling

When the user applies changes that include restart-required options:
- Dialog shows a prompt: "Settings changed that require a restart. Restart now?"
- User can choose to restart immediately or defer
- Non-restart options take effect immediately regardless

## First-Run Onboarding

On first launch (no existing QSettings), the application configuration dialog opens
before the main UI is constructed. This allows the user to:
1. Set the data storage location (required before any experiments)
2. Enable/disable LIF
3. Optionally change the application font

Since the main UI has not been built yet, no restart is needed -- the configuration is
read during UI construction. This reuses the same dialog code as the ongoing settings
UI, avoiding a separate first-run wizard.

## ApplicationConfigManager Changes

`ApplicationConfigManager` becomes the backend for the option registry:
- Owns the `QVector<AppOption>` definitions
- Provides `getOptions()` for the UI to enumerate
- Handles persistence (already uses QSettings)
- Emits change signals (already has `configurationChanged()`)
- LIF/CUDA initialization switches from hardcoded values to QSettings-persisted values
  (with compile-time flags as defaults on first run only)
