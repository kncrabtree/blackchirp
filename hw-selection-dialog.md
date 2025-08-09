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

### Left Panel: Configuration Overview (33% width)

**Component**: `QTreeWidget` with custom styling and context menu

**Purpose**: Provide hierarchical view of currently configured hardware

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


**Behavior**:
- Always fully expanded for maximum visibility

**Data Sources**:
- Populate from `RuntimeHardwareConfig::getCurrentHardware()`

### Middle Panel: Hardware Browser (33% width)

**Component**: `QListWidget` with custom item formatting

**Purpose**: Browse available hardware types and see configuration status at-a-glance

**Display Format**: 
- `"HardwareType (count)"` where count shows active instances
- Examples: `"Clock (2)"`, `"AWG (0)"`, `"FtmwScope (1)"`

**Visual Indicators**:
- ☑ (configured): Hardware type has active instances
- ☐ (unconfigured): Hardware type available but not configured
- Different styling (bold, color) for configured vs unconfigured

**Behavior**:
- Single selection triggers right panel update
- Selection persists during configuration changes

**Data Sources**:
- Hardware types from `HardwareRegistry::getHardwareTypes()`
- Instance counts from `RuntimeHardwareConfig::getActiveKeys<T>()`

### Right Panel: Context-Sensitive Configuration (33% width)

**Component**: Dynamic widget container that updates based on middle panel selection

**Purpose**: Provide appropriate interface for adding/editing hardware of the selected type

#### Single Instance Hardware UI:
```
┌─FtmwScope Configuration────────┐
│ Implementation: [ComboBox  v]  │
│ Label:         [LineEdit    ]  │
│                                │
│ Current: mainScope (M4i2220x8) │
└────────────────────────────────┘
```

#### Multiple Instance Hardware UI:
```
┌─Clock Configuration─────────────┐
│ Implementation: [ComboBox   v]  │
│ Label:         [LineEdit     ]  │
│                                 │
│ [Add Instance]                  │
│                                 │
│ Active Instances:               │
│ • rfSource (Valon5009)          │
│ • loSource (HP83712B)           │
│                                 │
│ [Edit Selected] [Remove]        │
└─────────────────────────────────┘
```

**Components**:
- **Implementation ComboBox**: Populated from `HardwareRegistry::getImplementations(hwType)`
- **Label LineEdit**: User-defined label for hardware identification
- **Add**: Adds new instance to runtime config
- **Active Instances List**: Show current instances with status (multiple instance hardware only)

**Behavior**:
- Real-time validation of label uniqueness
- Implementation selection enables/disables options based on library availability


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
- Implement dynamic right panel with interfaces for single vs. multi-instance HW (this determination should be the domain of the hardware registry; may need to add functionality there to look up whether selected is single or multi)
- Add real-time validation and user input handling
- Implement Add/Edit/Remove operations

### Phase 5: Validation Integration
- Connect all UI changes to validation system
- Implement comprehensive error messaging
- Add Apply button logic and configuration persistence

This design provides a comprehensive, user-friendly interface for runtime hardware configuration while maintaining clear separation of concerns and leveraging BlackChirp's existing architectural patterns.
