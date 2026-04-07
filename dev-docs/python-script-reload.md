# Python Script Hot-Reload

## Problem

When a user edits their Python hardware script, there is no way to reload
it from within Blackchirp. The user must restart the entire application to
pick up changes. There is also no way to open the script in an editor from
within Blackchirp, and no feedback when a script fails to load (syntax
errors, import errors, etc.).

## Key Insight: Reload Is Lightweight

A script reload does **not** require destroying or rebuilding the C++
`HardwareObject`. The Python subprocess is cleanly separated by IPC:

1. `PythonProcess::stop()` — kills the subprocess
2. `PythonHardwareBase::startPythonProcess()` — launches a new subprocess
   with the same script path and class name
3. `testPythonConnection()` — sends `test_connection` to verify

The C++ object's state (settings, comm protocol, thread, signal
connections) is unaffected. `PythonProcess` already handles clean
stop/start cycles — `testPythonConnection()` calls `startPythonProcess()`
automatically if the process isn't running.

## Error Feedback

The error path already works end-to-end:

- `python_hw_host.py`'s `load_user_class()` catches `SyntaxError`,
  `ImportError`, `AttributeError` with full tracebacks
- Process exits with code 1 on failure
- `PythonProcess::start()` captures the error via `processError` signal
  and `_init` response
- `PythonHardwareBase::pythonErrorString()` exposes the error message

We just need to surface this in the UI.

## UI Design

### PythonHardwareControlWidget

A new widget in `src/gui/widget/` (not in `src/hardware/`) to avoid
GUI dependencies in the hardware library. Contains:

- **Script path label** (read-only, shows current script path)
- **"Open in Editor" button** — `QDesktopServices::openUrl(QUrl::fromLocalFile(path))`
  to launch the user's default editor
- **"Reload Script" button** — triggers stop + restart of the Python process
- **Status label** — shows "Running", "Stopped", or error message

### Integration with HWDialog

`MainWindow::createHWDialog()` currently creates type-specific control
widgets based on hardware type. To add the Python widget without
introducing a `HardwareObject` GUI dependency:

1. Check if the hardware implementation class name contains "Python"
   (using the model/implementation string from `SettingsStorage`)
2. If so, create a composite `QWidget` with a `QVBoxLayout`:
   - Add `PythonHardwareControlWidget` at the top
   - Add the type-specific control widget below (if one exists for
     that hardware type, e.g., `GasControlWidget` for a
     `PythonFlowController`)
3. Pass the composite widget to `HWDialog` as the `controlWidget`

```cpp
// Sketch of the composite widget creation in MainWindow
QWidget* MainWindow::createPythonCompositeWidget(const QString& key,
                                                  QWidget* typeWidget)
{
    auto* composite = new QWidget;
    auto* layout = new QVBoxLayout(composite);
    layout->setContentsMargins(0, 0, 0, 0);

    auto* pyWidget = new PythonHardwareControlWidget(key, composite);
    layout->addWidget(pyWidget);

    if (typeWidget) {
        typeWidget->setParent(composite);
        layout->addWidget(typeWidget);
    }

    return composite;
}
```

This means a `PythonFlowController` would get both the reload UI and
the gas control widget, while a `PythonAwg` would get just the reload UI.

### Thread Safety

Python hardware objects run on their own threads (`d_threaded = true`).
The "Reload" button is in the GUI thread. The reload action must be
dispatched to the hardware object's thread, following the same pattern
used by `updateObjectSettings`:

```cpp
// In PythonHardwareControlWidget, on reload button click:
QMetaObject::invokeMethod(p_hwm, [hwm, key]() {
    hwm->reloadPythonScript(key);
});
```

`HardwareManager::reloadPythonScript()` would:
1. Look up the `HardwareObject` by key
2. Call stop/start on the `PythonProcess` (which runs on the object's thread)
3. Call `testConnection` to verify
4. Emit a signal with success/failure + error string for the UI to display

### Files Modified

| File | Change |
|---|---|
| `src/gui/widget/pythonhardwarecontrolwidget.h/.cpp` | New widget (script path, open, reload, status) |
| `src/gui/mainwindow.cpp` | Detect Python implementations, create composite widget |
| `src/hardware/core/hardwaremanager.h/.cpp` | Add `reloadPythonScript()` slot |
| `CMakeLists.txt` (gui) | Add new widget source files |

### Not Modified

| File | Reason |
|---|---|
| `PythonProcess` | stop/start already works correctly |
| `PythonHardwareBase` | startPythonProcess/testPythonConnection already exist |
| `python_hw_host.py` | Error reporting already works |
| `HWDialog` | No changes needed — composite widget is passed in |
