# LabJack Cross-Platform Support

Reference document for the `BC::Labjack` facade and its two platform backends.

## Architecture Overview

LabJack hardware access is split into three layers:

1. **`LabjackLibrary`** (`hardware/library/labjacklibrary.{h,cpp}`) — dynamic
   loader. Finds and loads the vendor library at runtime. Symbol set is
   conditionally compiled per platform:
   - Linux/macOS: `LJUSB_*` transport symbols from `liblabjackusb.so`/`.dylib`
   - Windows: `OpenLabJack`, `eAIN/eDAC/eDI/eDO`, `eTCConfig/eTCValues`,
     `ErrorToString`, `GetDriverVersion` from `LabJackUD.dll`

2. **`BC::Labjack` facade** (`hardware/library/labjackdriver.h`) — thin,
   platform-neutral interface. Exposes:
   - `isAvailable()` / `errorString()` — delegates to `LabjackLibrary`
   - `openU3(serialOrLocalId)` — returns an opaque `HandlePtr`; pass `-1` to
     open the first found device
   - `readAnalog`, `readDigital`, `writeAnalog`, `writeDigital`,
     `configureTimers`, `readTimers` — all take a raw `DeviceHandle*`

3. **Backend translation units** — implement the facade; selected by CMake:
   - `labjackdriver_exo.cpp` (Linux/macOS, `NOT WIN32`) — wraps `u3.cpp`
     helper functions; `DeviceHandle` carries a `void* h` (LJUSB handle) and
     a `u3CalibrationInfo` union member
   - `labjackdriver_ud.cpp` (Windows, `WIN32`) — calls UD easy functions
     directly via `LabjackLibrary`; `DeviceHandle` carries a `long h`
     (LJ_HANDLE)

`u3.cpp` (vendored LabJack helper) is also gated to `NOT WIN32` in CMake and
is the sole consumer of the LJUSB transport symbols.

## DeviceHandle Layout

```cpp
// exo backend (labjackdriver_exo.cpp)
struct DeviceHandle {
    enum class Kind { U3 } kind;
    void* h;                        // LJUSB device handle
    union { u3CalibrationInfo u3Cal; };
};

// UD backend (labjackdriver_ud.cpp)
struct DeviceHandle {
    enum class Kind { U3 } kind;
    long h;                         // LJ_HANDLE
};
```

Both structs are defined inside their respective translation units and are
opaque to all callers. Only `labjacku3.cpp` (and future device files) interact
with `DeviceHandle` through the facade API.

## Caller Pattern

```cpp
// labjacku3.cpp
d_handle = BC::Labjack::openU3(d_serialNo);   // d_serialNo = -1 → first found
BC::Labjack::configureTimers(d_handle.get(), {0L,0L}, {0L,0L}, 4L,
                             BC::Labjack::Const::tc48MHZ, 0L,
                             {0L,0L}, {0.0,0.0});
BC::Labjack::readAnalog (d_handle.get(), channel, voltage);
BC::Labjack::readDigital(d_handle.get(), channel, state);
```

`HandlePtr` is `std::unique_ptr<DeviceHandle, void(*)(DeviceHandle*)>`. The
deleter calls the appropriate close function (`LJUSB_CloseDevice` or UD
`Close`) and frees the struct. A null `HandlePtr` (pointer and deleter both
null) is safe because `unique_ptr` does not invoke the deleter on a null
managed pointer.

## Platform Notes

- **UD calling convention**: `__stdcall` is a no-op on x86-64 Windows; bare
  symbol names resolve via `QLibrary::resolve`. 32-bit Windows is not
  supported (decorated names would need a fallback).
- **`LJ_HANDLE`** is `long` per UD documentation. `LJ_ERROR` is also `long`.
- **`openU3` on Windows**: passes `serialOrLocalId` as a string address to
  `OpenLabJack`; if `serialOrLocalId < 0`, sets `FirstFound=1` instead.
- **Error reporting**: the exo backend reports errors via `bcError` with the
  numeric return code; the UD backend calls `ErrorToString` first to produce
  a human-readable message.

## Pending Tests

- **Windows compile test** — no Windows build has been run yet. The UD backend
  (`labjackdriver_ud.cpp`) and the platform-split `LabjackLibrary` are
  compile-only validated on Linux at this point.
- **Linux/macOS hardware test** — manual regression against a real U3 to
  confirm the facade wrapper introduces no behavioral change.
- **Windows hardware test** — manual test against a real U3 with
  `LabJackUD.dll` installed.

## Adding U6 Support

The facade is shaped to accept U6 without restructuring:

1. Add `LJ_dtU6 = 6` (or equivalent) to `BC::Labjack::Const` in
   `labjackdriver.h`. Declare `openU6(int serialOrLocalId)`.
2. **exo backend**: vendor `u6.h`/`u6.cpp` from LabJack (analogous to
   `u3.cpp`). Add `u6CalibrationInfo` to the `DeviceHandle` union. Add a
   `Kind::U6` arm to each `switch` in `labjackdriver_exo.cpp`. Gate
   `u6.cpp` to `NOT WIN32` in CMake.
3. **UD backend**: add a `Kind::U6` arm to `openU6`'s `OpenLabJack` call
   with `DeviceType = LJ_dtU6`. The operational functions (`eAIN`, etc.) are
   device-agnostic in the UD library and need no changes.
4. Add `LabjackU6` hardware class in `hardware/optional/ioboard/`, following
   `LabjackU3` as a template.

No changes to `LabjackLibrary`, `VendorLibrary`, or the facade's operational
API are required.
