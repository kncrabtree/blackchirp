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

### Left Panel: Configuration Overview (40% width)

**Component**: `QTreeWidget` with custom styling and context menu

**Purpose**: Provide hierarchical view of currently configured hardware with status indicators

**Display Format**:
- **Single Instance Hardware**: `"HardwareType: label (implementation) status"`
  - Example: `"FtmwScope: mainScope (M4i2220x8) ✓"`
- **Multiple Instance Hardware**: Parent-child hierarchy
  - Parent: `"HardwareType"` 
  - Children: `"label (implementation) status"`
  - Example:
    ```
    Clock
    ├─ rfSource (Valon5009) ✓
    └─ loSource (HP83712B) ⚠️
    ```

**Status Icons**:
- ✓ (green): Hardware available and functional
- ⚠️ (yellow): Library missing or configuration issue  
- ❌ (red): Hardware error or validation failure

**Behavior**:
- Always fully expanded for maximum visibility
- Selection highlights current item for context menu
- Context menu options (context-sensitive):
  - **Add**: Available for hardware types supporting multiple instances
  - **Edit**: Available for all configured hardware
  - **Remove**: Available for all configured hardware
  - **Test Connection**: Available for all configured hardware

**Data Sources**:
- Populate from `RuntimeHardwareConfig::getCurrentHardware()`
- Status from `RuntimeHardwareConfig::validateConfiguration()`
- Hardware type information from `HardwareRegistry`

### Middle Panel: Hardware Browser (30% width)

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
- Keyboard navigation support

**Data Sources**:
- Hardware types from `HardwareRegistry::getHardwareTypes()`
- Instance counts from `RuntimeHardwareConfig::getActiveKeys<T>()`

### Right Panel: Context-Sensitive Configuration (30% width)

**Component**: Dynamic widget container that updates based on middle panel selection

**Purpose**: Provide appropriate interface for adding/editing hardware of the selected type

#### Single Instance Hardware UI:
```
┌─FtmwScope Configuration────────┐
│ Implementation: [ComboBox  v]  │
│ Label:         [LineEdit    ]  │
│ Protocol:      [ComboBox  v]   │
│                                │
│ [Test Connection] [Configure]  │
│                                │
│ Current: mainScope (M4i2220x8) │
│ Status: ✓ Available           │
└────────────────────────────────┘
```

#### Multiple Instance Hardware UI:
```
┌─Clock Configuration─────────────┐
│ Implementation: [ComboBox   v]  │
│ Label:         [LineEdit     ]  │
│ Protocol:      [ComboBox   v]   │
│                                 │
│ [Add Instance]                  │
│                                 │
│ Active Instances:               │
│ • rfSource (Valon5009) ✓        │
│ • loSource (HP83712B) ⚠️        │
│                                 │
│ [Edit Selected] [Remove]        │
└─────────────────────────────────┘
```

**Components**:
- **Implementation ComboBox**: Populated from `HardwareRegistry::getImplementations(hwType)`
- **Label LineEdit**: User-defined label for hardware identification
- **Protocol ComboBox**: Communication protocol selection (RS232, TCP, GPIB, Virtual)
- **Test Connection Button**: Validate hardware connectivity
- **Add/Configure Buttons**: Apply configuration changes
- **Active Instances List**: Show current instances with status (multiple instance hardware only)

**Behavior**:
- Real-time validation of label uniqueness
- Implementation selection enables/disables options based on library availability
- Test connection provides immediate feedback
- Changes update validation status immediately

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
2. User configures implementation, label, protocol → Real-time validation
3. User clicks Add/Configure → `RuntimeHardwareConfig` updated
4. Overview tree refreshes → Browser counts update → Validation runs

#### Editing Hardware Instance:
1. User right-clicks item in overview tree → Context menu appears
2. User selects "Edit" → Browser selects corresponding type, right panel loads settings
3. User modifies settings → Real-time validation feedback
4. User applies changes → Configuration updated, UI refreshes

#### Removing Hardware Instance:
1. User right-clicks item in overview tree → Context menu "Remove"
2. Confirmation dialog (if hardware is critical/required)
3. Hardware removed from configuration → All panels refresh

### Validation Integration:
- **Real-time validation** as user types or changes selections
- **Comprehensive validation** using existing `RuntimeHardwareConfig::validateConfiguration()`
- **Visual feedback** through status icons, colors, and status bar
- **Blocking invalid configurations** via Apply button enablement

### Future Integration Points:
- **Library Status Tab**: Will integrate with dynamic library detection
- **Profile Management**: Save/load named hardware configurations
- **Hardware Synchronization**: Integration with Phase 3.3 dynamic hardware sync system
- **Testing Integration**: Hardware connection testing and diagnostics

## Implementation Phases

### Phase 1: Dialog Structure ✅ **COMPLETED**
**Achievement**: Complete dialog UI foundation with clean tabbed interface and 3-panel splitter layout.

**Accomplishments**:
- ✅ **Tabbed Interface**: Created main tab widget with "Hardware Configuration" and "Library Status" tabs
- ✅ **3-Panel Splitter Layout**: Horizontal splitter with Configuration Overview (33%), Hardware Browser (33%), and Context-Sensitive Configuration (33%) panels
- ✅ **Validation Status Bar**: ThemeColors-styled status bar with success/error state management
- ✅ **MainWindow Integration**: Added "Hardware Selection" menu item and dialog launch functionality
- ✅ **Proper UI Architecture**: Clean separation of concerns with dedicated UI header file and structured layout
- ✅ **Placeholder Content**: Demonstrates proper panel behavior and proportions
- ✅ **Dialog Foundation**: Ready for Phase 2 implementation (configuration overview population)

### Phase 2: Configuration Overview
- Implement TreeWidget with hierarchical hardware display
- Add status icons and context menu system
- Integrate with RuntimeHardwareConfig for data population
- Implement real-time updates and selection handling

### Phase 3: Hardware Browser  
- Create QListWidget with hardware type display and counts
- Implement selection handling and visual state management
- Connect browser selection to right panel updates

### Phase 4: Context-Sensitive Configuration
- Implement dynamic right panel with hardware-type specific interfaces
- Create single vs multiple instance UI patterns
- Add real-time validation and user input handling
- Implement Add/Edit/Remove operations

### Phase 5: Validation Integration
- Connect all UI changes to validation system
- Implement comprehensive error messaging
- Add Apply button logic and configuration persistence
- Integrate with hardware testing capabilities

This design provides a comprehensive, user-friendly interface for runtime hardware configuration while maintaining clear separation of concerns and leveraging BlackChirp's existing architectural patterns.
