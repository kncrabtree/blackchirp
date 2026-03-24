# Runtime Testing Checklist: cmakemigration Branch

Remaining manual testing scenarios for the cmakemigration branch.
Completed sections have been removed: Application Startup, Runtime HW Config
Dialog, Dynamic HW Sync, Communication Protocol, Library Status Tab, Data
Loading & Backward Compatibility, Hardware Keys, most of ClockManager,
Hardware Menu, and Settings/HeaderStorage.

---

## Known Issues / Future Considerations

- Experiment wizard remembers last settings, but what if associated HW has changed? Invalidate/notify? Store per HW profile?

---

## 1. Vendor Library Configuration

- [x] Custom library paths can be set
- [x] Additional search paths can be configured
- [x] "Test Load" button works with staged changes (temporary application + rollback)
- [x] Staging indicators (asterisks, colors) show when config has unsaved changes
- [x] Library changes applied before hardware synchronization on dialog close
- [NT] Hardware implementations requiring missing libraries show clear error messages (cannot test with system-installed libraries)
- [x] Application remains functional with only virtual hardware when no vendor libraries are installed
- [NT] Implementations gracefully degrade when their required library is unavailable (cannot test with system-installed libraries)

---

## 2. Experiment Lifecycle

### 2.1 Experiment Setup
- [x] Quick Experiment dialog opens and shows current hardware configuration

### 2.2 Experiment Execution (Virtual Hardware)
- [x] Target Duration experiment works
- [x] Peak Up mode starts (no experiment number assigned)
- [x] Forever mode starts and can be stopped
- [x] LO Scan experiment works (if applicable hardware configured)
- [x] DR Scan experiment works (if applicable hardware configured)

### 2.3 Data Recording
- [x] Aux data (flow, pressure, temperature) recorded correctly
- [x] Rolling data files created and populated

### 2.4 Multi-Step Acquisitions
- [x] LO Scan with multiple steps completes all steps
- [NT] Waveform is properly discarded between steps (stale waveform fix) (cannot test without real hardware)
- [NT] Hardware synchronization is stable across steps (cannot test without real hardware)

---

## 3. Viewer Application

- [x] blackchirp-viewer builds and launches
- [x] Viewer can open and display new-format experiments
- [x] Viewer can open and display legacy (devel branch) experiments
- [x] FT display works correctly
- [x] Hardware information displayed correctly in experiment summary

---

## 4. LIF Functionality

### 4.1 LIF UI
- [x] LIF-related UI elements visible when LIF is enabled (ApplicationConfigManager)
- [x] LIF display widget, control widget, and laser status box present
- [x] LIF tab/section in experiment setup dialog functional
- [x] LIF laser position controls work

### 4.2 LIF Experiment (if hardware available)
- [x] LIF experiment can be configured and started
- [x] LIF scope shots acquired and processed
- [x] LIF data saved to `lifparams.csv` and `lif/` directory
- [x] Laser units retrieved from hardware settings correctly

### 4.3 LIF Disabled
- [NT] When no LIF hardware configured, LIF UI elements are hidden/disabled
- [NT] Experiment loading handles experiments with LIF data even if LIF hardware not configured
- [NT] No crashes from LIF code paths when LIF hardware is absent

---

## 5. ClockManager

- [x] Multiple clocks can be configured with distinct labels

---

## 6. GUI Dynamic Updates

### 6.1 Status Boxes
- [x] Temperature controller status boxes update with temperature data

### 6.2 Overlays and Plots
- [ ] Overlay system works with label-based hardware keys
- [ ] Curve appearance settings persist correctly
- [ ] Plot widgets update correctly during acquisition

---

## 7. Edge Cases & Error Handling

- [x] Starting experiment with no hardware configured shows appropriate error
- [x] Removing hardware while experiment setup dialog is open (should be prevented)
- [x] Rapidly opening/closing hardware config dialog doesn't cause crashes
- [x] Application shutdown is clean (no thread cleanup warnings)
- [x] GPIB instrument without GPIB controller configured shows informative error
- [x] Hardware that fails connection test shows clear status in UI

---

## Future UI Improvements

- [ ] Replace read-only QDoubleSpinBox with QLabel in status boxes (TemperatureStatusBox, GasFlowDisplayBox) for a more compact display

---

## Test Environment Notes

- **Test with virtual hardware first** to verify all data paths
- **Test with real hardware** for communication protocol and vendor library scenarios
- **Test legacy data** by loading experiments from a devel-branch installation
- **Watch console output** for Qt warnings, especially about signal/slot connections and thread issues
