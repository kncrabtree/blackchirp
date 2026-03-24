# Remove HardwareObject d_name, Use hwKey for Display

## Motivation

`HardwareObject::d_name` is a mutable display string set in each hardware constructor
(e.g., "AWG70002A", "Prologix GPIB Controller"). With label-based hardware keys, the
`d_key` (e.g., `PulseGenerator.Default`) already provides a clear, unique identifier
suitable for UI display. Maintaining a separate `d_name` is redundant.

## Current State

- `d_name` is used in ~46 places across 16 files: log messages, UI labels, error
  messages, and a few hardware-specific formats (e.g., AWG SCPI instrument names)
- `d_key` has the form `HwType.label` which is already human-readable

## Changes

1. **Capitalize default labels**: Change the default label list in
   `BC::Key::generateDefaultLabel()` from `{"default", "main", "primary", ...}` to
   `{"Default", "Main", "Primary", ...}` so that keys like `PulseGenerator.Default`
   look presentable as UI titles
2. **Remove `d_name`** from `HardwareObject` and replace all UI/log usages with `d_key`
3. **Hardware-specific SCPI names**: A few implementations use `d_name` to construct
   SCPI instrument identifiers (AWG70002a, AWG7122b, AWG5204, AD9914). These should
   use `d_model` or a dedicated constant instead — `d_name` was never the right source
   for protocol-level identifiers

## Migration

- Existing profiles with lowercase labels continue to work (labels are user-editable)
- Only the default suggestions change; no data migration needed
- The numbered fallback (`Device1`, `Device2`, ...) should also be capitalized
