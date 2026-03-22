# Runtime Testing Checklist: cmakemigration Branch

This checklist covers manual testing scenarios for the major architectural changes
introduced in the cmakemigration branch. Assumes a successful build.

---

## 1. Application Startup & First Run

### 1.1 Clean Start (No Existing Settings)
- [x] Application launches without crash when no settings exist
- [x] Default virtual hardware is created for all required hardware types
- [x] Hardware connection tests complete (all virtual hardware connects)
- [x] MainWindow UI populates correctly (hardware menu, status boxes) - NOTE: Labels too large; overflow
- [x] No warnings/errors in console about missing profiles or configurations

### 1.2 Start with Existing devel-branch Settings
- [x] Application launches when settings from the devel branch already exist on disk
- [x] Old index-based hardware keys (e.g., `FlowController.0`) are handled gracefully
- [x] No crashes from settings format mismatches
- [x] `SettingsStorage` group values (`d_groupValues`) coexist with legacy key/array values

---

## 2. Runtime Hardware Configuration Dialog

### 2.1 Dialog Launch & Display
- [x] Hardware > Hardware Selection menu action opens the dialog
- [x] Dialog is only accessible when program is idle (not during experiment)
- [x] Configuration Overview (left panel) shows current hardware configuration
- [x] Hardware Browser (middle panel) lists all hardware types
- [x] Library Status tab displays correctly

### 2.2 Profile Management
- [x] Create new profile: select hardware type, enter label, choose implementation
- [x] Implementation combo box is sorted alphabetically
- [x] New profile appears in Configuration Overview immediately (preview)
- [x] Remove profile works via list selection
- [x] Single-instance hardware types (FtmwScope) show radio button behavior
- [x] Multi-instance types (Clock, FlowController) allow multiple profiles
- [x] Auto-activation works for first profile added to any type

### 2.3 Dialog Accept/Reject
- [x] Accept: changes are applied and persisted
- [x] Reject: changes are reverted, original configuration restored
- [x] Profile changes persist between application sessions (close and reopen)
- [x] Dialog does not crash on close (no QObject disconnect warnings)

### 2.4 Validation
- [x] Invalid configurations show real-time validation feedback
- [x] Apply button disabled when configuration is invalid
- [x] ThemeColors status feedback is visible and correct

---

## 3. Dynamic Hardware Synchronization

### 3.1 Hardware Addition
- [x] Adding a new hardware profile and accepting dialog creates the hardware object
- [x] New hardware appears in Hardware menu after dialog closes
- [x] New hardware gets a connection test after creation
- [x] Status box for new hardware appears in MainWindow

### 3.2 Hardware Removal
- [x] Removing a hardware profile and accepting dialog destroys the hardware object
- [x] Hardware menu item is removed after dialog closes
- [x] Status box for removed hardware disappears from MainWindow
- [x] No thread cleanup errors or crashes

### 3.3 Hardware Replacement
- [x] No settings migration occurs (clean slate for new implementation)
- [x] Connection test runs on new hardware

### 3.4 Connection Status Tracking
- [x] Per-hardware connection status shown individually in UI
- [x] Critical hardware failure disables experiment start
- [x] Non-critical hardware failure still allows experiment start
- [x] `allCriticalHardwareConnected()` works correctly with changing hardware sets
- [x] No deadlocks during connection testing with multiple hardware objects

-Experiment wizard remembers last settings, but what if associated HW has changed? Invalidate/notify? Store per HW profile?

---

## 4. Communication Protocol Configuration

### 4.1 Protocol Dialog
- [x] Right-click hardware menu item > Communication opens CommunicationDialog
- [x] Dialog shows current protocol (RS232, TCP, GPIB, Virtual, Custom)
- [x] Protocol switching works at runtime
- [x] GPIB controller combo box auto-populates with available controllers
- [-] Previously selected GPIB controller is restored when dialog opens FAIL: 2 problems: duplicate box. First has controllers listed but doesn't repopulate on opening. Second seems dead/unused.
- [x] Connection test from dialog returns result via HardwareManager signals

### 4.2 Protocol Settings Persistence
- [x] Protocol-specific settings (baud rate, IP address, GPIB address) persist
- [x] Switching protocol and restarting application preserves the selection
- [x] Group-based settings storage (`setGroupValue`/`getGroupValue`) works correctly

---

## 5. Vendor Library Management

### 5.1 Library Status Tab
- [x] Library Status tab shows all registered vendor libraries
- [x] Library availability status is correct (loaded/not found)
- [x] Version information displayed for loaded libraries (Spectrum driver/kernel versions)
- [x] Library details panel shows search paths and platform names

### 5.2 Library Configuration
- [ ] Custom library paths can be set
- [ ] Additional search paths can be configured
- [ ] "Test Load" button works with staged changes (temporary application + rollback)
- [ ] Staging indicators (asterisks, colors) show when config has unsaved changes
- [ ] Library changes applied before hardware synchronization on dialog close

### 5.3 Missing Library Handling
- [ ] Hardware implementations requiring missing libraries show clear error messages
- [ ] Application remains functional with only virtual hardware when no vendor libraries are installed
- [ ] Implementations gracefully degrade when their required library is unavailable

---

## 6. Experiment Lifecycle

### 6.1 Experiment Setup
- [ ] Quick Experiment dialog opens and shows current hardware configuration
- [ ] Experiment Setup dialog opens correctly with runtime hardware config
- [ ] FTMW digitizer configuration reflects actual configured scope
- [ ] Clock configuration shows all configured clock sources
- [ ] Pulse generator page shows all configured pulse generators
- [ ] Flow controller settings reflect configured flow controllers

### 6.2 Experiment Execution (Virtual Hardware)
- [ ] Target Shots experiment starts, acquires, and completes with virtual hardware
- [ ] Target Duration experiment works
- [ ] Peak Up mode starts (no experiment number assigned)
- [ ] Forever mode starts and can be stopped
- [ ] LO Scan experiment works (if applicable hardware configured)
- [ ] DR Scan experiment works (if applicable hardware configured)

### 6.3 Data Recording
- [ ] Experiment directory created with correct structure
- [ ] `hardware.csv` saved in NEW 3-column format: key, subKey, hardwareType
- [ ] Header file saved correctly with label-based hardware keys
- [ ] FID data saved and loadable
- [ ] Aux data (flow, pressure, temperature) recorded correctly
- [ ] Rolling data files created and populated
- [ ] Experiment number increments correctly

### 6.4 Multi-Step Acquisitions
- [ ] LO Scan with multiple steps completes all steps
- [ ] Waveform is properly discarded between steps (stale waveform fix)
- [ ] Hardware synchronization is stable across steps

---

## 7. Data Loading & Backward Compatibility

### 7.1 Loading New-Format Experiments
- [ ] Experiments saved by cmakemigration branch load correctly
- [ ] 3-column `hardware.csv` parsed correctly (key, subKey, hardwareType)
- [ ] Hardware type enum values correctly restored
- [ ] All optional hardware configs (flow, pulse gen, IO board, etc.) load from label-based keys
- [ ] FTMW data (FIDs, FT) loads and displays correctly
- [ ] Header display shows hardware with label-based keys

### 7.2 Loading Legacy Experiments (devel branch format)
- [ ] Experiments saved by devel branch load without crash
- [ ] 2-column `hardware.csv` (key, subKey) parsed via legacy path
- [ ] Index-based keys (e.g., `FlowController.0`) are handled by `legacyStringToHardwareType()`
- [ ] `FtmwDigitizer` legacy key maps correctly to `FtmwScope` type
- [ ] `GpibController` vs `GPIBController` case variations handled
- [ ] Optional hardware configs created from legacy keys
- [ ] FTMW data from legacy experiments loads and displays correctly
- [ ] Header storage keys (`headerKey()`) without `hwSubKey` work for old data

### 7.3 Very Old Experiments
- [ ] Pre-label-era experiments (if any exist) handle gracefully
- [ ] Single-column hardware format fallback works (if applicable)

---

## 8. Viewer Application

- [ ] blackchirp-viewer builds and launches
- [ ] Viewer can open and display new-format experiments
- [ ] Viewer can open and display legacy (devel branch) experiments
- [ ] FT display works correctly
- [ ] Hardware information displayed correctly in experiment summary

---

## 9. LIF Functionality

### 9.1 LIF UI
- [ ] LIF-related UI elements visible when LIF is enabled (ApplicationConfigManager)
- [ ] LIF display widget, control widget, and laser status box present
- [ ] LIF tab/section in experiment setup dialog functional
- [ ] LIF laser position controls work

### 9.2 LIF Experiment (if hardware available)
- [ ] LIF experiment can be configured and started
- [ ] LIF scope shots acquired and processed
- [ ] LIF data saved to `lifparams.csv` and `lif/` directory
- [ ] LIF data loads correctly when reopening experiment
- [ ] Laser units retrieved from hardware settings correctly

### 9.3 LIF Disabled
- [ ] When no LIF hardware configured, LIF UI elements are hidden/disabled
- [ ] Experiment loading handles experiments with LIF data even if LIF hardware not configured
- [ ] No crashes from LIF code paths when LIF hardware is absent

---

## 10. Settings & Storage Integrity

### 10.1 SettingsStorage Changes [AUTOMATED]
- [x] `groupKeys()` method returns correct group key list — covered by `tst_settingsstoragetest::testGroupKeys`
- [ ] Group values persist across application restart
- [x] Group values don't conflict with regular values or array values — covered by `tst_settingsstoragetest::testCrossContamination`
- [ ] `clearValue()` works for group values
- [ ] Default subKey changed from "virtual" to "invalid" doesn't break existing hardware settings

### 10.2 HeaderStorage Changes
- [ ] Removal of `hwSubKey` parameter doesn't break header saving
- [ ] Removal of `headerIndex()` doesn't break any dependent code paths
- [ ] Headers save and load with single `objKey` parameter

### 10.3 Hardware Keys [AUTOMATED]
- [x] Label-based keys (`FlowController.frontPanel`) used consistently throughout — covered by `tst_hardwarekeys`
- [ ] No residual index-based keys in new experiment data
- [x] `BC::Key::hwKey()` produces correct format keys — covered by `tst_hardwarekeys` (26 cases: hwKey, parseKey, parseIndexKey, isIndexKey, migrateIndexKey, generateDefaultLabel, widgetKey, HardwareDataContainer legacy mappings)

---

## 11. ClockManager

- [ ] ClockManager initializes correctly (no segfault from missing initialization)
- [ ] Clock roles (LO, UpConversion, DownConversion, etc.) assignable
- [ ] Multiple clocks can be configured with distinct labels
- [ ] RF configuration dialog accessible without crash
- [ ] Clock table model displays correctly in experiment setup
- [ ] Fixed clocks work as expected

---

## 12. GUI Dynamic Updates

### 12.1 Hardware Menu
- [ ] Hardware menu rebuilt when configuration changes
- [ ] Each hardware item has its own menu entry with communication dialog access
- [ ] Menu items enable/disable based on individual hardware connection status

### 12.2 Status Boxes
- [ ] Flow controller status boxes update with flow/pressure data
- [ ] Temperature controller status boxes update with temperature data
- [ ] Pressure controller status boxes update
- [ ] Status boxes appear/disappear when hardware is added/removed

### 12.3 Overlays and Plots
- [ ] Overlay system works with label-based hardware keys
- [ ] Curve appearance settings persist correctly
- [ ] Plot widgets update correctly during acquisition

---

## 13. Edge Cases & Error Handling

- [ ] Starting experiment with no hardware configured shows appropriate error
- [ ] Removing hardware while experiment setup dialog is open (should be prevented)
- [ ] Rapidly opening/closing hardware config dialog doesn't cause crashes
- [ ] Application shutdown is clean (no thread cleanup warnings)
- [ ] GPIB instrument without GPIB controller configured shows informative error
- [ ] Hardware that fails connection test shows clear status in UI

---

## Test Environment Notes

- **Test with virtual hardware first** to verify all data paths
- **Test with real hardware** for communication protocol and vendor library scenarios
- **Test legacy data** by loading experiments from a devel-branch installation
- **Watch console output** for Qt warnings, especially about signal/slot connections and thread issues
