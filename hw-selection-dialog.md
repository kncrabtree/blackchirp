# Runtime Hardware Configuration Dialog - Detailed UI/UX Design

## Overview

This document specifies the detailed design for BlackChirp's Runtime Hardware Configuration Dialog, implementing a hybrid tabbed/overview interface that provides intuitive hardware management while leveraging the existing runtime configuration infrastructure.

## Architecture Summary

**Design Philosophy**: Combine the benefits of overview-first navigation with tabbed organization to separate concerns. Users get immediate visibility into their current configuration while having dedicated spaces for different types of management tasks.

**Primary Goals**:
1. Provide at-a-glance view of current hardware configuration
2. Enable easy addition/removal/editing of hardware instances
3. Clear validation feedback with familiar UI patterns
4. Separate hardware configuration from library management concerns

## Dialog Structure

### Main Layout: Tabbed Interface

#### Tab 1: "Hardware Configuration" 
Primary interface for hardware selection and configuration management.

#### Tab 2: "Library Status"
Dedicated interface for vendor library management and diagnostics (future implementation).
- Tab header shows status icon indicating overall library health
- Content will include library detection, installation guidance, diagnostics

### Hardware Configuration Tab Layout

**Main Layout**: Horizontal 3-panel splitter with validation status bar at bottom

```
┌─────────────────────────────────────────────────────────────────┐
│ ┌─Configuration Overview─┐ ┌─Hardware Browser─┐ ┌─Configuration─┐ │
│ │                        │ │                  │ │               │ │
│ │ TreeWidget showing:    │ │ QListWidget:     │ │ Context-      │ │
│ │                        │ │                  │ │ sensitive     │ │
│ │ FtmwScope: mainScope   │ │ ☑ Clock (2)      │ │ controls for  │ │
│ │   (M4i2220x8) ✓        │ │ ☐ AWG (0)        │ │ selected      │ │
│ │ Clock                  │ │ ☑ FtmwScope (1)  │ │ hardware type │ │
│ │ ├─ rfSource (Valon)    │ │ ☐ FlowCtrl (0)   │ │               │ │
│ │ └─ loSource (HP) ⚠️    │ │ ...              │ │ [Add/Edit UI] │ │
│ │                        │ │                  │ │               │ │
│ └────────────(33%)───────┘ └───(33%)────────┘ └───(33%)──────┘ │
│ ┌─Validation Status──────────────────────────────────────────────┐ │
│ │ ✓ Configuration is valid                                      │ │
│ └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Panel Specifications

### Left Panel: Runtime Configuration Preview (33% width)

**Component**: `QTreeWidget` with hardware preset management buttons

**Purpose**: Show preview of runtime hardware configuration and manage hardware presets

**Display Format**:
- **Single Instance Hardware**: `"HardwareType: label (implementation)"`
  - Example: `"FtmwScope: mainScope (M4i2220x8)"`
- **Multiple Instance Hardware**: Parent-child hierarchy
  - Parent: `"HardwareType"` 
  - Children: `"label (implementation)"`
  - Example:
    ```
    Clock
    ├─ rfSource (Valon5009)
    └─ loSource (HP83712B)
    ```

**Hardware Preset Management**:
- **Save as Preset** button: Save current configuration as named preset
- **Load Preset** button: Load existing preset into preview

**Behavior**:
- **Initialization**: Shows current `RuntimeHardwareConfig` state
- **During Editing**: Shows preview of configuration changes from right panel
- **Real-time Updates**: Refreshes when profiles are checked/unchecked in right panel
- Always fully expanded for maximum visibility

**Backend Integration**:
- **Initialization**: `RuntimeHardwareConfig::getCurrentHardware()`
- **Preview Updates**: Internal dialog preview state
- **Preset Management**: `HardwareProfileManager` preset functionality (future implementation)

### Middle Panel: Hardware Registry Browser (33% width)

**Component**: `QListWidget` with custom item formatting

**Purpose**: Browse available hardware types from registry and see configuration status at-a-glance

**Display Format**: 
- `"HardwareType (count)"` where count shows active instances in preview
- Examples: `"Clock (2)"`, `"AWG (0)"`, `"FtmwScope (1)"`

**Visual Indicators**:
- **Bold text**: Hardware type has active instances in current preview
- **Normal text**: Hardware type available but not configured in preview

**Behavior**:
- Single selection triggers right panel update
- Selection persists during configuration changes
- Instance counts update as profiles are checked/unchecked in right panel

**Backend Integration**:
- **Hardware Types**: `HardwareRegistry::getHardwareTypes()`
- **Instance Counts**: Internal dialog preview state (updated from profile selections)

### Right Panel: Hardware Profile Management (33% width)

**Component**: Hardware profile management interface that updates based on middle panel selection

**Purpose**: Manage hardware profiles (type + label + implementation combinations) and control their activation in runtime configuration

#### Single Instance Hardware UI:
```
┌─FtmwScope Profiles─────────────┐
│ Available Profiles:            │
│ ◉ mainScope (M4i2220x8)        │
│ ○ backup (VirtualFtmwScope)    │
│                                │
│ [Add Profile] [Remove Profile] │
└────────────────────────────────┘
```

#### Multiple Instance Hardware UI:
```
┌─Clock Profiles─────────────────┐
│ Available Profiles:            │
│ ☑ rfSource (Valon5009)         │
│ ☐ backup (HP83712B)            │
│ ☑ loSource (FixedClock)        │
│                                │
│ [Add Profile] [Remove Profile] │
└────────────────────────────────┘
```

**Profile Selection Behavior**:
- **Single Instance**: Radio button behavior (only one profile active at a time)
- **Multiple Instance**: Checkbox behavior (multiple profiles can be active simultaneously)
- **Immediate Preview**: Checking/unchecking updates preview state and refreshes left panel

**Profile Operations**:
- **Add Profile**: Opens modal dialog with implementation selection and label validation
- **Remove Profile**: Confirms deletion and removes profile + associated settings permanently

**Backend Integration**:
- **Profile Storage**: `HardwareProfileManager` for persistent profile management
- **Profile List**: `HardwareProfileManager::getAllProfiles(hardwareType)`
- **Available Implementations**: `HardwareRegistry::getImplementations(hardwareType)`
- **Preview Updates**: Updates internal dialog preview state
- **Settings Management**: Profile deletion clears associated QSettings collections

**State Management**:
- **Profile Creation**: Immediate persistence via HardwareProfileManager
- **Profile Deletion**: Immediate deletion with settings cleanup + removal from both preview and original runtime configs
- **Activation Changes**: Update preview state only until dialog acceptance


### Bottom Panel: Validation Status Bar

**Component**: `QLabel` with ThemeColors styling (following UnifiedOverlayDialog pattern)

**Purpose**: Provide immediate feedback on configuration validity and changes

**Styling States**:
```cpp
// Success state
p_statusLabel->setText("Configuration is valid");
p_statusLabel->setStyleSheet(QString("QLabel { color: %1; }")
    .arg(ThemeColors::getCSSColor(ThemeColors::StatusSuccess, this)));

// Error state  
p_statusLabel->setText("Error: Duplicate label 'rfSource' for Clock hardware");
p_statusLabel->setStyleSheet(QString("QLabel { color: %1; }")
    .arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this)));

// Info/Processing state
p_statusLabel->setText("Checking configuration...");
p_statusLabel->setStyleSheet(QString("QLabel { color: %1; font-style: italic; }")
    .arg(ThemeColors::getCSSColor(ThemeColors::StatusInfo, this)));
```

**Message Types**:
- **Success**: "Configuration is valid" (green)
- **Validation Errors**: Specific error descriptions (red)
- **Processing**: "Checking configuration..." (blue, italic)
- **Library Issues**: "Warning: Missing libraries detected" (yellow)

**Integration**: 
- Updates in real-time as user makes changes
- Connected to `RuntimeHardwareConfig::validateConfiguration()`
- Integrates with Apply button enablement logic

## Data Flow and Integration

### Initialization Sequence:
1. **Load Current Configuration**: Query `RuntimeHardwareConfig::getCurrentHardware()`
2. **Populate Overview Tree**: Build hierarchical display with status indicators
3. **Populate Hardware Browser**: Get types from registry, count active instances
4. **Validate Configuration**: Run validation and update status label
5. **Set Initial Selection**: Select first hardware type in browser

### User Interaction Flows:

#### Adding Hardware Instance:
1. User selects hardware type in browser → Right panel updates
2. User configures implementation, label → Real-time validation
3. User clicks Add → `RuntimeHardwareConfig` updated
4. Overview tree refreshes → Browser counts update → Validation runs

#### Editing Hardware Instance:
1. User selects hardware type in browser → Right panel updates
2. User selects active implementation in right panel → UI updates implementation and label and Edit/Remove active state
3. User modifies settings → Real-time validation feedback
4. User clicks "Edit Selected" → Configuration updated, UI refreshes

#### Removing Hardware Instance:
1. User selects hardware type in browser → Right panel updates
2. User clicks active instance → Edit/Remove activated → User Clicks Remove → Confirmation dialog
3. Hardware removed from configuration → All panels refresh

### Validation Integration:
- **Real-time validation** when editing of label finishes or changes selections
- **Comprehensive validation** using existing `RuntimeHardwareConfig::validateConfiguration()`
- **Visual feedback** through status label and color
- **Blocking invalid configurations** via Apply button enablement

### Future Integration Points:
- **Configuration Persistence**: Ensure selected configuration is persisted as default hardware profile for next run
- **Library Status Tab**: Will integrate with dynamic library detection
- **Profile Management**: Save/load named hardware configurations for swapping between predefined HW sets.
- **Hardware Synchronization**: Integration with Phase 3.3 dynamic hardware sync system
- **Testing Integration**: Hardware connection testing and diagnostics

## Implementation Phases

### Phase 1: Dialog Structure ✅ **COMPLETED**
Complete dialog UI foundation with tabbed interface and 3-panel splitter layout. Ready for Phase 2.

### Phase 2: Configuration Overview ✅ **COMPLETED**
Left panel TreeWidget now integrates with RuntimeHardwareConfig to display actual hardware configuration with proper single/multi-instance formatting and sorting. Uses existing BC::Key::parseKey() API. Ready for Phase 3.

### Phase 3: Hardware Browser ✅ **COMPLETED**
Hardware Browser now integrates with HardwareRegistry and RuntimeHardwareConfig to display hardware types with instance counts. Selection flow to right panel established and verified. Ready for Phase 4.

### Phase 4: Context-Sensitive Configuration
#### Phase 4.1: Hardware Type Classification ✅ **COMPLETED**
HardwareRegistry now has static `isMultiInstanceType()` method using type-safe hardware class names via staticMetaObject.

#### Phase 4.2: Profile Management UI ✅ **COMPLETED**
- Replace right panel with Profile management interface
- Checkable QListWidget showing existing profiles as "label (implementation)"
- Single-instance: Radio button behavior (mutually exclusive selection)
- Multi-instance: Checkbox behavior (multiple selections allowed)
- Add/Remove buttons for profile creation/deletion
- Preview state management for runtime configuration changes

#### Phase 4.3: Profile Operations & State Management ✅ **COMPLETED**
- **Add Profile**: Modal dialog with implementation ComboBox + label validation
- **Remove Profile**: Immediate deletion with settings cleanup and confirmation dialog
- **Profile Deletion Edge Case**: Remove deleted profiles from both preview AND original runtime configs
- **Preview State**: Maintain separate preview vs original runtime configuration
- **Check/Uncheck**: Update preview state only, refresh left panel display
- Integration with HardwareProfileManager for persistent profile storage

### Phase 5: Validation Status Bar & Hardware Preset Management
#### Phase 5.1: Validation Status Bar ⚠️ **PENDING**
- **Status Bar Implementation**: Bottom panel validation status bar with ThemeColors styling
- **Real-time Validation**: Connect to `RuntimeHardwareConfig::validateConfiguration()` 
- **Validation Feedback**: Success/Error/Info states with specific error messages
- **Apply Button Logic**: Enable/disable Apply button based on validation state
- **User Feedback**: Clear indication of configuration validity and blocking issues

#### Phase 5.2: Hardware Preset Management ⚠️ **PENDING** 
- **Save as Preset**: Button in left panel to save current preview configuration as named preset
- **Load Preset**: Button in left panel to load existing preset into preview state
- **Preset Storage**: Integration with HardwareProfileManager preset functionality
- **Preset UI**: Modal dialogs for preset naming and selection
- **Preset Validation**: Ensure loaded presets are valid and handle conflicts

#### Phase 5.3: MainWindow Integration & Final Polish ⚠️ **PENDING**
- **Dialog Integration**: Ensure MainWindow properly instantiates and connects dialog  
- **Invalid Config Handling**: MainWindow handles invalid runtime configs after dialog completion
- **Error Recovery**: Graceful handling of hardware initialization failures
- **Final Testing**: Comprehensive validation of all dialog functionality

**Note**: Core dialog state management (Accept/Cancel/Preview) was completed in Phase 4.3

This design provides a comprehensive, user-friendly interface for runtime hardware configuration while maintaining clear separation of concerns and leveraging BlackChirp's existing architectural patterns.
