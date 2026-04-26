# Hardware Loadout System — Feature Reference

## Concepts

A **loadout** captures the active hardware map for a given experimental setup
— which AWG, which digitizer, which clock implementations are bound to which
profile slots. An **FTMW preset** is a named operating point — RF chain
parameters, clock frequencies, chirp waveform, and digitizer configuration —
that lives inside a single loadout. Switching loadouts swaps the hardware map;
switching FTMW presets within a loadout swaps the FTMW operating point.

- **Loadout** — a hardware map (`Type.label` → implementation) plus the FTMW
  presets it owns.
- **FTMW Preset** — a named `FtmwPreset` (RF + chirp + digitizer) owned by a
  loadout. A preset cannot exist outside a loadout.
- **`__LastUsed__`** (`BC::Store::LM::lastUsedFtmwPresetName`) — a
  per-loadout sentinel preset updated automatically on dialog accept or
  experiment start. Hidden from all user-facing dropdowns; never user-deletable.
- **`currentFtmwPreset`** — per-loadout pointer to the most recently selected,
  applied, or accepted preset. Drives initial widget population. The active
  preset cannot be deleted; switch to a different preset first.

## Key Source Locations

| Component | Files |
|-----------|-------|
| Data model | `src/data/loadout/hardwareloadout.h` — `HardwareLoadout` struct and `FtmwPreset` struct |
| Persistence / CRUD | `src/data/loadout/loadoutmanager.{h,cpp}` — `LoadoutManager` singleton |
| FTMW config widget | `src/gui/widget/ftmwconfigwidget.{h,cpp}` — preset bar + dirty tracking |
| FTMW config dialog | `src/gui/dialog/ftmwconfigdialog.{h,cpp}` — thin shell around `FtmwConfigWidget` |
| Experiment setup | `src/gui/expsetup/experimentftmwconfigpage.{h,cpp}` — ESD FTMW page |
| Hardware config dialog | `src/gui/dialog/runtimehardwareconfigdialog.{h,cpp}` — loadout panel + drift detection |
| Hardware menu submenus | `src/gui/mainwindow.cpp` — `rebuildLoadoutMenu`, `rebuildFtmwPresetMenu` |
| Tests | `tests/tst_loadoutmanagertest.cpp` |

## Storage Layout

QSettings under `Loadouts/`:

```text
Loadouts/
  currentLoadout = "<name>"
  defaultLoadout = "Default"
  names/                               # array of {name}
  <name>/
    name = "<name>"
    hardwareMap/                       # array of {key, value}
    currentFtmwPreset = "<presetName>" # may be "__LastUsed__"
    ftmwPresetNames/                   # array of {name}
    lastModified = <ISO timestamp>
    ftmwPresets/
      <presetName>/
        rfScalars/
        rfClocks/
        chirpScalars/
        chirpSegments/
        chirpMarkers/
        digiScalars/
        digiAnalog/
        digiDigital/
        digiHwKey = "<hwKey>"
        lastModified = <ISO timestamp>
```

## Constraints

- FTMW presets cannot exist outside a loadout.
- AWG sample rate is hardware-derived; it is not stored in presets.
- Hardware map changes that invalidate named FTMW presets require explicit
  user confirmation (Discard / Save As / Cancel) in the Hardware Configuration
  dialog.
- `__LastUsed__` is updated only on `FtmwConfigDialog::accept` and on
  experiment start; not on Apply, not on Cancel.
- The active (`currentFtmwPreset`) preset cannot be deleted. Switch to a
  different preset before deleting. The Delete button in `FtmwConfigWidget`
  acts on the combo's selected preset and is disabled when that selection
  matches the active preset.
- Loadout and FTMW preset switching via the Hardware menu is gated to
  `Disconnected` / `Idle` states.
