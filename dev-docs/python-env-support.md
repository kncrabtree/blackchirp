# Python Environment Support (venv/conda per-profile) **COMPLETE**

## Problem

Python hardware processes are launched with a hardcoded `python3` executable
(`pythonprocess.cpp:41`). Users whose hardware scripts depend on packages
installed in a venv or conda environment must activate the environment
system-wide before launching Blackchirp. There is no way to configure a
per-profile Python environment.

## Goal

Each Python hardware profile can optionally specify a venv or conda
environment directory. When set, the process launches using that
environment's Python interpreter. When empty, the system `python3` is
used (current behavior).

## Design

### Data Layer

Add a `pythonEnvPath` field to `HardwareProfileManager::ProfileInfo`,
alongside the existing `pythonScriptPath` and `pythonClassName`:

```cpp
// In ProfileInfo struct
QString pythonEnvPath;  // Path to venv/conda env dir (empty = system python3)
```

Add corresponding key constant, getter/setter, and load/save persistence
in `HardwareProfileManager`, following the same pattern as
`pythonScriptPath`.

### Launch Logic

Add a static helper to `PythonHardwareBase`:

```cpp
static QString resolvePythonExecutable(const QString &envPath)
{
    if (envPath.isEmpty())
        return QStringLiteral("python3");

    // venv/conda layout: bin/python3 (Unix), Scripts/python.exe (Windows)
    QStringList candidates = {
        envPath + QStringLiteral("/bin/python3"),
        envPath + QStringLiteral("/bin/python"),
        envPath + QStringLiteral("/Scripts/python.exe"),
    };

    for (const auto &path : candidates) {
        if (QFile::exists(path))
            return path;
    }

    return QStringLiteral("python3");  // fallback
}
```

Update `PythonHardwareBase::startPythonProcess()` to look up the env
path from `HardwareProfileManager` and resolve the executable before
passing it to `PythonProcess::start()`.

### PythonProcess::start Signature

Change from:

```cpp
bool start(const QString &hostScriptPath,
           const QString &userScriptPath,
           const QString &className);
```

To:

```cpp
bool start(const QString &pythonExe,
           const QString &hostScriptPath,
           const QString &userScriptPath,
           const QString &className);
```

The hardcoded `QStringLiteral("python3")` in `start()` is replaced by
the caller-provided executable path.

### UI

In `RuntimeHardwareConfigDialog`, the Python profile section already has
fields for script path and class name. Add a third field:

- **Label:** "Python Environment"
- **Widget:** `QLineEdit` + Browse button (directory picker)
- **Tooltip:** "Path to a venv or conda environment directory. Leave
  empty to use the system Python."
- **Persistence:** via `HardwareProfileManager::setPythonEnvPath()`

This slots in naturally next to the existing Python fields with no new
UI infrastructure required.

### Files Modified

| File | Change |
|---|---|
| `hardwareprofilemanager.h/.cpp` | Add `pythonEnvPath` field, key, getter/setter, load/save |
| `pythonhardwarebase.h/.cpp` | Add `resolvePythonExecutable()`, update `startPythonProcess()` |
| `pythonprocess.h/.cpp` | Add `pythonExe` parameter to `start()` |
| `runtimehardwareconfigdialog.cpp` | Add env path field in Python profile section |
