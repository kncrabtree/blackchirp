# Hardware Loadout System — Architecture Notes

## Future Loadout Data Structure

```cpp
struct HardwareLoadout {
    QString name;
    std::map<QString, QString> hardwareMap;   // identical to d_previewRuntimeConfig
    std::optional<RfConfigSnapshot> rfConfig; // optional; null = don't touch RF on load
};
```

## RfConfigSnapshot

Keys clock frequencies by clock **label** (not implementation), so shared hardware works correctly
when a clock appears in multiple loadouts with different frequencies.

## Loadout Storage

QSettings under `Loadouts/<name>/` group. `hardwareMap` serialized as subkeys. `rfConfig`
serialized in a `RfConfig/` subgroup if present.

## Dialog Integration

The dialog's `d_previewRuntimeConfig` becomes the "edit buffer" for the active loadout. A
`QComboBox` at the top of the Hardware Configuration tab selects the loadout; switching loadouts
replaces `d_previewRuntimeConfig` with the selected loadout's `hardwareMap` and optionally restores
the RF config. The enable/disable state of a hardware type is already implicit in the map (no entry
= disabled).

## Left Panel Navigation

Clicking a hardware type item in the left panel (the `QListWidget` / tree of hardware types)
should activate the corresponding category in the middle panel (radio button group) and update
the right panel to show that type's profiles.

Currently the left panel is display-only; selection has no effect on the middle/right panels.

**Implementation**: connect the left panel's `currentItemChanged` (or `itemClicked`) signal to
a slot that calls `selectHardwareType(type)` — the same function the middle panel radio buttons
already call to populate the right panel. The middle panel radio button for the clicked type
should also be checked programmatically so the two panels stay in sync.

---

## Constraints

- Do not embed RF config in `d_previewRuntimeConfig`
- Do not change the type of `d_previewRuntimeConfig`
- Do not store "enabled/disabled" as separate state from map presence
