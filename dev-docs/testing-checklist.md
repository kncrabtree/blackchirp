# Future Testing: Items Requiring Real Hardware

Items marked [NT] from the cmakemigration branch testing that cannot be tested with virtual hardware alone.

---

## Vendor Library Configuration

- Hardware implementations requiring missing libraries show clear error messages (cannot test with system-installed libraries)
- Implementations gracefully degrade when their required library is unavailable (cannot test with system-installed libraries)

## Multi-Step Acquisitions

- Waveform is properly discarded between steps (stale waveform fix)
- Hardware synchronization is stable across steps

## LIF (when no LIF hardware configured)

- LIF UI elements are hidden/disabled
- Experiment loading handles experiments with LIF data even if LIF hardware not configured
- No crashes from LIF code paths when LIF hardware is absent

---

## Known Issues / Future Considerations

- Experiment wizard remembers last settings, but what if associated HW has changed? Invalidate/notify? Store per HW profile?

## Future UI Improvements

- Replace read-only QDoubleSpinBox with QLabel in status boxes (TemperatureStatusBox, GasFlowDisplayBox) for a more compact display
