# Widget Settings Cleanup on Profile Removal

## Current Gap

When a hardware profile is removed, `HardwareObject::purgeSettings()` purges the hardware
object's own SettingsStorage group (e.g., `PulseGenerator.main`). However, associated widget
settings live under separate keys (e.g., `PulseWidget.PulseGenerator.main`) and are NOT
cleaned up.

## Fix

When purging a hardware profile's settings, also purge all widget settings keyed to that
hardware. The widget key pattern is `<WidgetName>.<HardwareKey>` (constructed via
`BC::Key::widgetKey()`), so the cleanup can search for and remove all QSettings groups
that end with the hardware key suffix.

Implementation approach:
- `SettingsStorage::purgeGroup()` already supports removing arbitrary key paths
- On profile removal, iterate known widget key prefixes and call
  `purgeGroup({widgetKey + "." + hwKey})` for each
- Alternatively, scan QSettings top-level groups for any containing the hardware key

This prevents stale widget settings from being loaded if a new profile reuses the same label
with a different implementation.
