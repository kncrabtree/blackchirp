# Phase 3.4: GUI Dynamic Updates Implementation Plan

**Status**: 🔄 **IN PROGRESS** - Phase 1 Complete ✅  
**Goal**: Make MainWindow adapt to hardware configuration changes and provide fine-grained connection status feedback.

## Overview

Replace constructor-time hardware-dependent UI building with dynamic methods that rebuild UI when hardware configuration changes. Implement per-hardware connection status tracking instead of binary global state.

## Current Architecture Issues

### Constructor-Time Hardware Dependency (MainWindow.cpp:148-296) - ✅ RESOLVED
```cpp
// PROBLEMATIC: This runs at constructor time and cannot adapt to runtime changes
auto currentHardware = RuntimeHardwareConfig::constInstance().getCurrentHardware();
for(auto it = currentHardware.cbegin(); it != currentHardware.cend(); ++it) {
    auto key = it->first;
    auto hwType = ki.first;
    
    // Creates menu action
    auto act = ui->menuHardware->addAction(QString("%1: %2").arg(key, p_hwm->getHwName(key)));
    
    // Creates status widgets based on hardware type
    if(hwType == QString(FlowController::staticMetaObject.className())) {
        auto w = new GasFlowDisplayBox(key);
        ui->hwStatusLayout->addWidget(w);
        // ... complex signal connections ...
    }
    // ... similar blocks for PressureController, PulseGenerator, TemperatureController, LifLaser
}
```

### Binary Connection State Logic (MainWindow.cpp:1132-1133)
```cpp
// PROBLEMATIC: Disables ALL hardware when ANY hardware fails
if(!d_hardwareConnected)
    s = Disconnected;
```

## Implementation Architecture

### Core Data Structures

**File**: `src/gui/mainwindow.h`

Add to MainWindow private section:
```cpp
struct HardwareUIElements {
    QAction* menuAction;
    QWidget* statusWidget;  // GasFlowDisplayBox, PressureStatusBox, etc.
    QVector<QMetaObject::Connection> connections;
};
std::map<QString, HardwareUIElements> d_hardwareUI;  // key.label -> UI elements
std::map<QString, bool> d_hardwareConnectionState;  // key.label -> connection status
```

### Core Methods to Implement

**File**: `src/gui/mainwindow.h` (declarations)
```cpp
// Add to private section
void buildHardwareUI();
void clearHardwareUI();
void updateHardwareConnectionState(const QString& hwKey, bool connected);
void configureUiForHardwareState();
bool isCriticalHardwareConnected() const;
```

**File**: `src/gui/mainwindow.cpp` (implementations)

#### 1. Extract Constructor Logic → `buildHardwareUI()`
```cpp
void MainWindow::buildHardwareUI()
{
    // Move ENTIRE block from constructor (lines 148-296) here
    // Modify to store UI elements in d_hardwareUI map
    auto currentHardware = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    for(auto it = currentHardware.cbegin(); it != currentHardware.cend(); ++it) {
        auto key = it->first;
        auto ki = BC::Key::parseKey(key);
        auto hwType = ki.first;
        
        HardwareUIElements elements;
        
        // Create menu action
        elements.menuAction = ui->menuHardware->addAction(QString("%1: %2").arg(key, p_hwm->getHwName(key)));
        elements.menuAction->setObjectName(Ui::actionStr+key);
        
        // Create status widget and connections based on hardware type
        if(hwType == QString(FlowController::staticMetaObject.className())) {
            auto w = new GasFlowDisplayBox(key);
            w->setObjectName(key+Ui::sbStr);
            ui->hwStatusLayout->addWidget(w);
            elements.statusWidget = w;
            
            // Store all connections in elements.connections vector
            elements.connections.append(connect(p_hwm,&HardwareManager::flowUpdate,w,&GasFlowDisplayBox::updateFlow));
            // ... store ALL signal connections
        }
        // ... repeat for all hardware types
        
        d_hardwareUI[key] = elements;
        d_hardwareConnectionState[key] = false; // Initialize as disconnected
    }
    
    ui->hwStatusLayout->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::MinimumExpanding));
}
```

#### 2. Implement `clearHardwareUI()`
```cpp
void MainWindow::clearHardwareUI()
{
    for(auto& [hwKey, elements] : d_hardwareUI) {
        // Disconnect all signals
        for(const auto& connection : elements.connections) {
            disconnect(connection);
        }
        
        // Remove and delete widgets
        if(elements.statusWidget) {
            ui->hwStatusLayout->removeWidget(elements.statusWidget);
            delete elements.statusWidget;
        }
        
        // Remove menu action
        if(elements.menuAction) {
            ui->menuHardware->removeAction(elements.menuAction);
            delete elements.menuAction;
        }
    }
    
    d_hardwareUI.clear();
    d_hardwareConnectionState.clear();
    
    // Remove the spacer item too
    QLayoutItem* spacer = ui->hwStatusLayout->takeAt(ui->hwStatusLayout->count()-1);
    delete spacer;
}
```

#### 3. Update Connection Status Tracking
```cpp
void MainWindow::updateHardwareConnectionState(const QString& hwKey, bool connected)
{
    d_hardwareConnectionState[hwKey] = connected;
    
    // Update individual UI element state
    if(d_hardwareUI.contains(hwKey)) {
        auto& elements = d_hardwareUI[hwKey];
        elements.menuAction->setEnabled(connected);
        elements.statusWidget->setEnabled(connected);
        // Could add visual feedback (grayed out, different styling, etc.)
    }
    
    // Update overall UI state
    configureUiForHardwareState();
}
```

#### 4. Replace Binary Logic in `configureUi()`
**File**: `src/gui/mainwindow.cpp:1129` (modify existing method)
```cpp
void MainWindow::configureUi(MainWindow::ProgramState s)
{
    d_state = s;
    
    // REPLACE this binary logic:
    // if(!d_hardwareConnected)
    //     s = Disconnected;
    
    // WITH fine-grained critical hardware checking:
    if(!isCriticalHardwareConnected())
        s = Disconnected;
    
    // ... rest of existing method unchanged
    // Individual hardware menu actions/widgets managed separately in updateHardwareConnectionState()
}

bool MainWindow::isCriticalHardwareConnected() const
{
    // Check only critical hardware - implementation depends on how criticality is determined
    // Could read from settings, check HardwareObject::d_critical, or use hardcoded list
    for(const auto& [hwKey, connected] : d_hardwareConnectionState) {
        // Query if this hardware is critical and if it's disconnected
        // Return false if any critical hardware is disconnected
    }
    return true;
}
```

### Integration Points

#### 1. Constructor Modification
**File**: `src/gui/mainwindow.cpp:68` (constructor)
```cpp
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    // ... existing initialization code ...
    
    // REMOVE lines 148-296 (the hardware UI building loop)
    
    // REPLACE with:
    buildHardwareUI();
    
    // ADD per-hardware connection tracking:
    connect(p_hwm, &HardwareManager::connectionResult, 
            this, &MainWindow::updateHardwareConnectionState);
    
    // ... rest of existing constructor ...
}
```

#### 2. Runtime Hardware Config Dialog Integration
**File**: `src/gui/mainwindow.cpp` (find `launchRuntimeHardwareConfigDialog()` method)
```cpp
void MainWindow::launchRuntimeHardwareConfigDialog()
{
    RuntimeHardwareConfigDialog d(this);
    
    if(d.exec() == QDialog::Accepted) {
        // UI must be rebuilt BEFORE hardware synchronization
        clearHardwareUI();
        buildHardwareUI();
        
        // NOW trigger hardware synchronization - UI elements ready to receive signals
        QMetaObject::invokeMethod(p_hwm, &HardwareManager::syncWithRuntimeConfig);
    }
}
```

### Files to Modify

#### Primary Files:
- **`src/gui/mainwindow.h`** - Add data structures and method declarations
- **`src/gui/mainwindow.cpp`** - Implement all methods, modify constructor and dialog integration

#### Key Methods/Areas to Locate:
- **Constructor** (line 68): Remove hardware UI building, add `buildHardwareUI()` call
- **Hardware UI Loop** (lines 148-296): Extract to `buildHardwareUI()` method
- **`configureUi()`** (line 1129): Replace binary logic with `isCriticalHardwareConnected()`
- **`hardwareInitialized()`** (line 643): Modify to work with per-hardware tracking
- **`launchRuntimeHardwareConfigDialog()`**: Add UI rebuild before hardware sync

### Signal Integration

#### Existing Signals to Leverage:
- **`HardwareManager::connectionResult(QString hwKey, bool success, QString msg)`** - Per-hardware status
- **`HardwareManager::allHardwareConnected(bool)`** - Keep for overall status logging

#### Connection Pattern:
```cpp
// In constructor - persistent connection for all hardware
connect(p_hwm, &HardwareManager::connectionResult, 
        this, &MainWindow::updateHardwareConnectionState);
```

## Implementation Phases

### Phase 1: Extract and Restructure ✅ COMPLETED
1. ✅ Add data structures to `mainwindow.h` - `HardwareUIElements` struct and maps added
2. ✅ Implement `buildHardwareUI()` by extracting constructor logic - Complete extraction from lines 148-296
3. ✅ Implement `clearHardwareUI()` with proper cleanup - Handles widget deletion and signal disconnection
4. ✅ Update constructor to use `buildHardwareUI()` - Constructor now calls `buildHardwareUI()` instead of inline building

**Implementation Notes**:
- All hardware UI building logic successfully extracted from constructor to `buildHardwareUI()` method
- `HardwareUIElements` struct stores menu actions, status widgets, and signal connections per hardware
- `clearHardwareUI()` properly cleans up widgets, menu actions, and disconnects all signals
- Constructor simplified with clean call to `buildHardwareUI()` at line 145
- Placeholder implementations added for connection status methods (to be completed in Phase 2)

### Phase 2: Connection Status Tracking
5. Implement `updateHardwareConnectionState()` 
6. Connect to `HardwareManager::connectionResult` signal
7. Implement `isCriticalHardwareConnected()` logic
8. Update `configureUi()` to use fine-grained checking

### Phase 3: Runtime Integration
9. Update `launchRuntimeHardwareConfigDialog()` for proper timing
10. Test UI rebuild on hardware configuration changes
11. Verify connection status updates work correctly

## Critical Success Factors

1. **Timing**: UI elements must exist BEFORE `HardwareManager::syncWithRuntimeConfig()` begins
2. **Signal Management**: All connections must be properly stored and disconnected during UI rebuilding
3. **Memory Management**: Widgets and actions must be properly deleted in `clearHardwareUI()`
4. **State Consistency**: `d_hardwareConnectionState` must accurately reflect actual hardware status

## Testing Scenarios

1. **Application Startup**: UI builds correctly with initial hardware configuration
2. **Hardware Configuration Changes**: UI rebuilds when runtime config dialog closes
3. **Individual Hardware Failures**: Only affected hardware UI elements are disabled
4. **Critical Hardware Failures**: Experiment controls are disabled, non-critical hardware remains accessible
5. **Hardware Connection Recovery**: UI elements re-enable when hardware reconnects

## Expected Benefits

- **User Experience**: Access working hardware even when other hardware fails
- **Real-time Feedback**: Individual status boxes show per-hardware connection state  
- **Safe Operations**: Still prevents experiments when critical hardware unavailable
- **Dynamic Compatibility**: Seamless runtime hardware configuration changes
- **Clean Architecture**: Clear separation between UI state and hardware state