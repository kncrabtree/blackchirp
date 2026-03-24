# Hardware Loadout System

## Data Structure

```cpp
struct HardwareLoadout {
    QString name;
    std::map<QString, QString> hardwareMap;   // identical to d_previewRuntimeConfig
    std::optional<RfConfigSnapshot> rfConfig; // clock freqs + RF chain settings
    std::optional<ChirpConfig> chirpConfig;   // chirp segments, markers, timing
};
```

## RfConfigSnapshot

Keys clock frequencies by clock **label** (not implementation), so shared hardware works
correctly when a clock appears in multiple loadouts with different frequencies.

Also stores the RF chain parameters that are set-only (not read from hardware):
- AWG multiplication factor
- Chirp multiplication factor
- Upconversion/downconversion sideband selections
- Common LO flag

Clock frequencies should be **enforced at loadout activation time** (applied to hardware),
since they represent the intended operating state for this hardware configuration.

## ChirpConfig in Loadouts

ChirpConfig stores set-only parameters (never read back from hardware):
- Chirp segments (start/end frequencies, durations)
- Timing markers (pre/post protection, pre/post gate)
- Chirp interval (multi-chirp timing)
- Number of chirps

The AWG sample rate is derived from the active AWG hardware and should NOT be stored
in the loadout -- it will be populated from hardware settings when the loadout is applied.

## Loadout Storage

QSettings under `Loadouts/<name>/` group:
- `hardwareMap` serialized as subkeys
- `RfConfig/` subgroup for RF chain settings and clock frequencies
- `ChirpConfig/` subgroup for chirp waveform definition

## Hardware Configuration Dialog Changes

### RF/Chirp Config as Dialog Tabs

Move RF and chirp configuration into the Hardware Configuration dialog as additional tabs.
Currently:
- RF config: accessible via standalone dialog (Hardware menu) and experiment setup wizard
- Chirp config: only accessible in experiment setup wizard

Proposed:
- Hardware Configuration dialog gains **RF Configuration** and **Chirp Configuration** tabs
- These tabs use the existing `RfConfigWidget` and `ChirpConfigWidget`
- Changes made here are the "loadout-level" settings -- saved to the active loadout
- Changes made in the Experiment Setup Dialog are per-experiment overrides (not saved to loadout)

### Loadout Selection

The dialog's `d_previewRuntimeConfig` becomes the "edit buffer" for the active loadout. A
`QComboBox` at the top of the Hardware Configuration tab selects the loadout; switching loadouts
replaces `d_previewRuntimeConfig` with the selected loadout's `hardwareMap` and restores the
RF/chirp config. The enable/disable state of a hardware type is already implicit in the map
(no entry = disabled).

### Save Behavior

When the user modifies RF/chirp settings in the Hardware Configuration dialog and closes it:
- Prompt to save changes to the active loadout (similar to how hardware map changes work)
- This is the only path that updates the loadout's RF/chirp config
- Experiment Setup Dialog changes are temporary and do not propagate back to the loadout

## Experiment Setup Dialog Defaults

### Initialization Priority: Loadout as Baseline, Last-Used as Override

When opening the Experiment Setup Dialog:
1. **Default source**: Active loadout's RF/chirp config provides the baseline
2. **Override**: If the user previously ran an experiment with modified settings,
   those last-used settings take precedence (current SettingsStorage behavior)
3. **Reset button**: A "Reset to Loadout Defaults" button on the RF and Chirp pages
   restores the active loadout's config, discarding any per-experiment overrides

This preserves the current "repeat last experiment" workflow while giving users a clear
path back to the loadout baseline.

### Rationale

Most hardware widget settings (pulse channels, flow setpoints, etc.) are ephemeral and
read from their respective hardware -- they don't need loadout-level presets. The RF/chirp
config is different: these are set-only parameters that define the operating configuration,
and they are tightly coupled to which clock sources and AWG are active in the loadout.

## Constraints

- Do not embed RF/chirp config in `d_previewRuntimeConfig`
- Do not change the type of `d_previewRuntimeConfig`
- Do not store "enabled/disabled" as separate state from map presence
- Do not store AWG sample rate in loadout (derived from hardware)
- Loadout RF/chirp changes only through Hardware Configuration dialog, not experiment setup
