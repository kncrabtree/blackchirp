# Hardware Threading Configuration

## Goal
Restore per-object threading support lost during the cmakemigration constructor refactor,
and make it user-configurable through RuntimeHardwareConfig so advanced users can control
which hardware objects run in their own QThread.

## Background
On the `devel` branch, certain hardware types defaulted to `d_threaded = true` and ran in
their own QThread. The cmakemigration refactor set `d_threaded(false)` unconditionally in
the `HardwareObject` constructor, so everything currently runs in the main thread.

Threading is important for hardware types that perform blocking I/O or lengthy operations
(e.g., digitizer readout, AWG waveform upload) — without it these block the GUI event loop.

## Design

### Type-level defaults
Each intermediate base class sets a sensible default in its constructor:

| Type | Default | Rationale |
|------|---------|-----------|
| FtmwScope | `true` | Digitizer readout can block for extended periods |
| AWG | `true` | Waveform uploads can be large/slow |
| IOBoard | `true` | Analog/digital reads may be slow on serial |
| LifScope | `true` | Same as FtmwScope |
| LifLaser | `true` | Position moves may block |
| GpibController | `true` | Serializes GPIB bus access; shouldn't block main thread |
| Clock | `false` | Typically fast operations; was threaded on devel but shouldn't have been |
| PulseGenerator | `false` | Fast command/response |
| FlowController | `false` | Fast command/response |
| PressureController | `false` | Fast command/response |
| TemperatureController | `false` | Fast command/response |

### RuntimeHardwareConfig integration
- Add a `bool threaded` field to `HardwareSelection` (persisted alongside `type` and `implementation`)
- When `HardwareManager` creates a hardware object, it reads the `threaded` value from
  `RuntimeHardwareConfig` and sets `d_threaded` on the object **before** the moveToThread
  decision at line ~1116 of `hardwaremanager.cpp`
- If no persisted value exists, fall back to the type-level default set in the constructor

### HardwareObject changes
- Make `d_threaded` mutable: `bool d_threaded{false};` (remove `const`)
- Remove the unused `BC::Key::HW::threaded` settings key from `hardwarekeys.h` (threading
  is owned by RuntimeHardwareConfig, not per-object SettingsStorage)
- Each intermediate class sets `d_threaded = true` or `false` in its constructor body

### Sequence
```
factory(label)
  → intermediate class constructor sets type-level default
  → HardwareManager reads threaded override from RuntimeHardwareConfig
  → sets obj->d_threaded
  → checks d_threaded → moveToThread() or setParent(this)
  → bcInitInstrument()
```

### Loadout integration
When the loadout system is implemented, `HardwareSelection::threaded` will be bundled into
loadouts alongside implementation and RF/chirp config. Switching loadouts will naturally
rebuild hardware objects with the correct threading setting.

### UI: Collapsible "Advanced" section in right panel

The Hardware Configuration dialog right panel currently shows an enable checkbox (for
single-instance types) and a profile list with add/remove buttons. A collapsible
"Advanced" section will be added at the bottom of the right panel, below the profile
list and above the validation status bar.

**Layout:**
```
┌─ Hardware Profiles ──────────────────────┐
│  ◉ main (VirtualFtmwScope) [system]      │
│  ○ backup (DSA71604C)                     │
│  [Add Profile] [Remove Profile]           │
└──────────────────────────────────────────-┘
▶ Advanced                          ← collapsed by default
┌──────────────────────────────────────────-┐
│  ☑ Run in own thread (recommended)        │
│  ... (space for future per-instance       │
│       advanced settings)                  │
└──────────────────────────────────────────-┘
[Validation status bar]
```

**Behavior:**
- Collapsed by default; remembers expand/collapse state for the session
- The checkbox label includes "(recommended)" when the type-level default is `true`,
  helping users understand the intended configuration
- Checkbox state comes from `HardwareSelection::threaded`; if not persisted yet, falls
  back to the type-level default
- Changes update `d_previewRuntimeConfig` like other settings, validated on accept
- For multi-instance types, the advanced section reflects the currently selected/checked
  profile; selecting a different profile updates the section

**Future use:** This section can host other per-instance advanced settings as they arise
(e.g., connection timeout, retry policy) without cluttering the main profile UI.

## Tasks
- [ ] Make `d_threaded` mutable in `HardwareObject`
- [ ] Set type-level defaults in intermediate class constructors
- [ ] Add `threaded` field to `HardwareSelection` and persist it
- [ ] Read/apply threading override in `HardwareManager` object creation flow
- [ ] Remove `BC::Key::HW::threaded` from `hardwarekeys.h`
- [ ] Add collapsible "Advanced" section to right panel of HW config dialog
- [ ] Wire threading checkbox to preview config and validation
- [ ] Update roadmap entry
