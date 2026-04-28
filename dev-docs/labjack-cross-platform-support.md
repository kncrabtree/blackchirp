# LabJack Cross-Platform Support

Implementation plan for adding Windows support to the LabJack U3 ioboard
driver. Phase 1 (dynamic loading of the Linux/macOS exodriver) is already
complete; this document covers Phase 2 (Windows via the UD library).

## Background

Blackchirp's LabJack U3 ioboard implementation depends on two distinct
vendor APIs depending on platform:

- **Linux/macOS** — the **exodriver** (`liblabjackusb.so` / `.dylib`),
  which exposes a low-level USB transport (`LJUSB_GetDevCount`,
  `LJUSB_OpenDevice`, `LJUSB_Write`, `LJUSB_Read`, ...). The "easy"
  functions (`eAIN`, `eDI`, `eDAC`, `eTCConfig`, ...) are *not* in the
  driver — they are implemented in user space by LabJack-supplied
  example code that ships as `u3.h` / `u3.cpp` (1567 lines, vendored
  into Blackchirp). Calibration is exposed as a caller-managed
  `u3CalibrationInfo` struct that must be passed to every conversion
  call.
- **Windows** — the **UD library** (`LabJackUD.dll`), which exposes
  the easy functions and request/response API (`OpenLabJack`, `eAIN`,
  `eDI`, `eDO`, `eDAC`, `eTCConfig`, `eTCValues`, `AddRequest`, `Go`,
  `GetResult`, `ErrorToString`, `GetDriverVersion`) directly. Calibration
  is read and applied internally; it never crosses the API boundary.

The two ABIs differ in handle type (`void*` vs `LJ_HANDLE`/`long`),
calling convention (cdecl vs `__stdcall`), parameter sets (UD `eAIN`
omits `*CalibrationInfo`, `ConfigIO`, and `*DAC1Enable`), error reporting
(`-1` + `printf` vs `LJ_ERROR` + `ErrorToString`), and discovery
(LJUSB enumeration + `ConfigU3` walk vs `OpenLabJack` with serial in
`pAddress`).

The only honest place to merge is **at the operations Blackchirp actually
performs**, not at the LabJack-supplied helper layer.

## What Blackchirp Currently Uses

`labjacku3.cpp` calls only six entry points across `u3.cpp`:
`openUSBConnection`, `closeUSBConnection`, `getCalibrationInfo`,
`eTCConfig`, `eAIN`, `eDI`. None of `eDAC`, `eDO`, `eTCValues`, the
LJTDAC/I2C helpers, or the calibration-conversion functions are
exercised. The remaining ~90% of `u3.cpp` exists but is unused.

The `LabjackLibrary` class loads the exodriver dynamically and exposes
LJUSB symbols. `u3.cpp` is the sole consumer of those symbols.

`librarystatuswidget.cpp` and `hardwaremanager.cpp` reference
`LabjackLibrary` via its singleton accessor. One spot
(`librarystatuswidget.cpp:547`) reaches directly into
`LabjackLibrary::LJUSB_GetLibraryVersion`; everywhere else uses
`VendorLibrary` virtual methods.

## Proposed Architecture: Thin Operational Facade Over Two Backends

Lift the operations Blackchirp uses (or will plausibly use soon) into a
small, platform-neutral facade. Keep `LabjackLibrary` as a library
loader, but conditionally resolve a different symbol set per platform.
The bulk of `u3.cpp` stays as the implementation detail of *one*
backend — it does not get reorganized.

### File Layout

```
hardware/library/
  vendorlibrary.{h,cpp}          unchanged
  labjackconstants.h             grow: LJ_dtU3, LJ_ctUSB, LJ_tc48MHZ, etc.
                                 (LJ_dtU6 added later when U6 support lands)
  labjacklibrary.{h,cpp}         CHANGED: per-platform symbol set behind #ifdef
                                 - Linux/macOS: existing LJUSB_* members + load
                                 - Windows: OpenLabJack/Close/eAIN/eDI/eDO/eDAC/
                                            eTCConfig/eTCValues/ErrorToString/
                                            GetDriverVersion members + load
                                 (Same symbol set covers U3, U6, UE9 — no
                                 changes needed when adding more devices.)
  labjackdriver.h                NEW: facade interface, opaque DeviceHandle,
                                 per-device factories (openU3 today; openU6
                                 added later)
  labjackdriver_exo.cpp          NEW: Linux/macOS impl. Initially wraps
                                 u3.cpp's openUSBConnection/eAIN/eDI/eDO/eDAC/
                                 eTCConfig/eTCValues. Later: also wraps u6.cpp.
  labjackdriver_ud.cpp           NEW: Windows impl, dispatches on the device
                                 tag stored in DeviceHandle. UD's eAIN/eDI/
                                 eDO/eDAC/eTCConfig/eTCValues serve all
                                 devices; only OpenLabJack's DeviceType
                                 constant and a few default-parameter rules
                                 are device-specific.

hardware/optional/ioboard/
  ioboard.{h,cpp}                unchanged
  labjacku3.{h,cpp}              CHANGED: drops #include "u3.h" and the
                                 d_calInfo member; uses labjackdriver.h and
                                 calls BC::Labjack::openU3(...).
  u3.{h,cpp}                     CHANGED: printf -> bcDebug only; build-gated
                                 to NOT WIN32 in CMake
  labjacku6.{h,cpp}              FUTURE: sibling of labjacku3.{h,cpp}; calls
                                 BC::Labjack::openU6(...).
  u6.{h,cpp}                     FUTURE: vendored from LabJack; analogous to
                                 u3.{h,cpp}, build-gated to NOT WIN32.
```

### Facade Surface

The namespace is `BC::Labjack` (not `BC::Labjack::U3`) so the same
abstraction can host additional LabJack devices later. Per-device
factory functions (`openU3`, `openU6`, ...) keep callers explicit about
which hardware they expect, while the rest of the operational API takes
an opaque `DeviceHandle*` and is device-agnostic.

```cpp
// labjackdriver.h
namespace BC::Labjack {
    struct DeviceHandle;                          // opaque; layout per backend
    using HandlePtr = std::unique_ptr<DeviceHandle, void(*)(DeviceHandle*)>;

    bool      isAvailable();
    QString   errorString();

    HandlePtr openU3(int serialOrLocalId);        // null on failure
    // HandlePtr openU6(int serialOrLocalId);     // added when U6 support lands

    // Operational primitives — device-agnostic; backend dispatches on the
    // device tag stored inside DeviceHandle.
    bool readAnalog  (DeviceHandle*, int channel, double &out);
    bool readDigital (DeviceHandle*, int channel, bool   &out);
    bool writeAnalog (DeviceHandle*, int channel, double  voltage);
    bool writeDigital(DeviceHandle*, int channel, bool    state);

    bool configureTimers(DeviceHandle*,
                         std::array<long,2>   enableTimers,
                         std::array<long,2>   enableCounters,
                         long                 pinOffset,
                         long                 timerClockBaseIdx,
                         long                 timerClockDivisor,
                         std::array<long,2>   timerModes,
                         std::array<double,2> timerValues);

    bool readTimers(DeviceHandle*,
                    std::array<long,2>   readTimers,
                    std::array<long,2>   updateResetTimers,
                    std::array<long,2>   readCounters,
                    std::array<long,2>   resetCounters,
                    std::array<double,2> &timerValues,
                    std::array<double,2> &counterValues);
}
```

The facade exposes all six operational primitives Blackchirp may need:
analog read/write, digital read/write, timer configuration, and timer
value read. eDAC, eDO, and eTCValues are surfaced now (not strictly
required by the current `labjacku3.cpp`) so that future ioboard features
can use them without revisiting the abstraction.

`DeviceHandle` carries a device tag plus backend-specific state. The
struct is opaque to callers; only the backend translation units see its
layout. Sketch:

- **exo** — tagged variant:
  ```cpp
  struct DeviceHandle {
      enum class Kind { U3 /*, U6 */ } kind;
      void* h;                            // LJUSB device handle
      union {
          u3CalibrationInfo u3Cal;
          // u6CalibrationInfo u6Cal;     // when U6 support lands
      };
  };
  ```
- **ud** — single layout, device tag drives parameter defaults:
  ```cpp
  struct DeviceHandle {
      enum class Kind { U3 /*, U6 */ } kind;
      long h;                             // LJ_HANDLE
  };
  ```

`labjacku3.cpp` becomes:

```cpp
d_handle = BC::Labjack::openU3(d_serialNo);
BC::Labjack::configureTimers(d_handle.get(), {0,0}, {0,0}, 4,
                             BC::Labjack::Const::tc48MHZ, 0,
                             {0,0}, {0.0,0.0});
BC::Labjack::readAnalog (d_handle.get(), k, v);
BC::Labjack::readDigital(d_handle.get(), k, b);
```

A future `labjacku6.cpp` looks identical apart from `openU6(...)`.

### Why This Is Better

- **No code reshuffling for the exodriver path.** `u3.cpp` keeps working
  as-is; the facade's exo backend wraps existing functions.
- **Per-device state lives in `DeviceHandle`.** Calibration never
  appears at the boundary or on Windows.
- **Two distinct ABI surfaces stay in their own translation units**,
  gated by CMake. No `#ifdef` hair in `labjacku3.cpp`.
- **Existing `LabjackLibrary` consumers do not move.**
  `librarystatuswidget`, `hardwaremanager`, and `REGISTER_LIBRARY` all
  continue to refer to `LabjackLibrary` by name; only the symbol set
  inside it changes per platform.
- **Future expansion is a single-method addition per backend.**
- **U6 (and other LabJack USB devices) drop into the same abstraction.**
  Adding `openU6` plus a U6 case in each backend's switch is the entire
  facade change; the operational API does not grow.

## Implementation Details and Risks

### Header Hygiene

- Drop the `typedef void* HANDLE;` from `labjacklibrary.h` — it would
  collide with `<windows.h>` on Windows. Move it into `u3.h`
  (exodriver-only) and use bare `void*` in the LJUSB function-pointer
  typedefs.
- After the facade lands, `labjacku3.h` no longer includes `u3.h`, so
  no caller is exposed to LabJack vendor types.

### Type Definitions

- `LJ_HANDLE` is `long` per the UD documentation. Confirm against
  `LabJackUD.h` from the SDK or `LabJackPython` bindings before
  finalizing typedefs.
- `LJ_ERROR` is `long`.
- Calling convention: `__stdcall` on Windows. On x86-64 Windows
  `__stdcall` is a no-op and bare names resolve via `QLibrary::resolve`.
  On x86 Windows the exports are decorated as `_OpenLabJack@N`. **Plan
  supports 64-bit Windows only initially**; document this. Add a
  decorated-name fallback later if 32-bit Windows becomes a target.

### UD Symbols To Resolve

Required for the facade:

- `OpenLabJack`
- `eAIN`, `eDAC`, `eDI`, `eDO`
- `eTCConfig`, `eTCValues`
- `ErrorToString`, `GetDriverVersion`
- `Close` if present in `LabJackUD.h` (verify; many Windows examples
  omit it and rely on process exit). Resolve optionally; treat absence
  as no-op.

### Default UD Easy-Function Parameters

Match the implicit behavior of the exo path:

- `eAIN`: `Range=0` (driver default — ±10 V on U3-HV, ±2.4 V on
  U3-LV), `Resolution=0`, `Settling=0`, `Binary=0`, `ChannelN=31` for
  single-ended. Surface through `IOBoardConfig` later only if a user
  request justifies it.
- `eTCConfig`: pass through facade arguments unchanged. The single
  current call site supplies `pinOffset=4`, `clockBase=LJ_tc48MHZ`,
  `divisor=0`, all others zero.

### Error Reporting Cleanup (in scope)

`u3.cpp` reports errors via `printf(...)`, which on the GUI binary goes
nowhere useful. While we're touching this code, replace those calls
with `bcDebug` / `bcError` (matching the rest of Blackchirp). The
facade's UD backend wraps `ErrorToString` into a small helper:

```cpp
char buf[256];
LabjackLibrary::instance().ErrorToString(rc, buf);
bcError(QString::fromLatin1(buf));
```

Both backends route errors back through the facade's `errorString()`
accessor so `LabjackU3::testConnection()` can populate `d_errorString`
without backend-specific knowledge.

### `librarystatuswidget` Cleanup

`librarystatuswidget.cpp:547` reaches directly into
`LabjackLibrary::LJUSB_GetLibraryVersion`. Replace with a call to
`library.getVersionInfo()` (already a `VendorLibrary` virtual). Each
platform's `LabjackLibrary::getVersionInfo()` resolves to either
`LJUSB_GetLibraryVersion` (Linux/macOS) or `GetDriverVersion`
(Windows). Title in the UI changes from "LabJack USB" to "LabJack U3
Driver" since on Windows the library is not strictly USB-specific.

### Testing

- **Linux/macOS regression** — manual run against a real U3, since
  Blackchirp tests do not exercise hardware.
- **Windows integration** — manual run against a real U3 on a Windows
  host. Compile-only validation in CI is the practical automated check;
  hardware loopback is operator-only.

## Phased Implementation (Build-Green at Every Step)

| Step | Scope | Outcome |
|------|-------|---------|
| 1 | Add `labjackdriver.h` + `labjackdriver_exo.cpp` (Linux/macOS impl wrapping existing `u3.cpp` calls, including eDAC/eDO/eTCValues paths). Namespace is `BC::Labjack` and `DeviceHandle` carries a device-kind tag (only `U3` populated). Switch `labjacku3.cpp` to use `BC::Labjack::openU3(...)`. Drop `#include "u3.h"` and `d_calInfo` from `labjacku3.h`. CMake: keep `u3.cpp` building. | Linux/macOS unchanged in observable behavior; boundary established with U6 hooks already in the type shape. |
| 2 | Refactor `LabjackLibrary` so the LJUSB symbol set is `#ifndef Q_OS_WIN` and the UD symbol set is `#ifdef Q_OS_WIN`. Update `loadFunctions`, `platformLibraryNames`, `getVersionInfo`. Replace `librarystatuswidget.cpp:547` LJUSB call with `getVersionInfo()`. | Linux/macOS still works; Windows builds and successfully loads `LabJackUD.dll` (no functional change yet). |
| 3 | Add `labjackdriver_ud.cpp`. The UD backend dispatches on `DeviceHandle::Kind` for `OpenLabJack`'s `DeviceType` constant and any device-specific defaults; the `eAIN`/`eDI`/`eDO`/`eDAC`/`eTCConfig`/`eTCValues` symbols are shared across U3 and (future) U6. CMake: gate `u3.cpp` and `labjackdriver_exo.cpp` to `NOT WIN32`; gate `labjackdriver_ud.cpp` to `WIN32`. | Windows build links cleanly; runtime requires UD-installed system. |
| 4 | Replace `printf` error reporting in `u3.cpp` with `bcDebug` / `bcError`. | Errors surface to the GUI log. |
| 5 | Update `cmake/HardwareConfig.cmake.template` (drop the obsolete LabJack-SDK-include suggestion). Update `BlackchirpApplication.cmake` comment. Adjust `getInstallationInstructions()` if needed. | Build configuration accurate. |
| 6 | Manual test on Linux against a real U3 (regression). Manual test on Windows against a real U3 (new). Update `dev-docs/devel-roadmap.md` to mark Phase 2 complete and remove this document. | Done. |

## Out of Scope (Explicit Non-Goals)

- Surfacing UD-only knobs (`Range`, `Resolution`, `Settling`) in
  `IOBoardConfig`. Defer until a user asks.
- LJTDAC / I2C helpers from `u3.cpp`. Not used; not portable to UD via
  the easy-function set.
- 32-bit Windows. Add later via a decorated-name fallback in the
  loader if needed.
- CI hardware loopback.
- LabJack U6 support. Out of scope for Phase 2, but the facade is
  shaped to accept it without restructuring (see below).

## Future: LabJack U6 Support

The U6 shares Blackchirp's LabJack infrastructure with the U3 — same
loader, same library, same facade — and would slot in as additional
hardware classes alongside the U3.

**Free reuse:**

- `VendorLibrary` / `LabjackLibrary` loader. The UD library on Windows
  serves U3, U6, and UE9 through `LabJackUD.dll` and the same symbol
  set. The exodriver on Linux/macOS likewise uses the same `LJUSB_*`
  transport for all LabJack USB devices. **No loader changes needed.**
- `IOBoard` base class, settings UI, hardware registration macros, and
  the `BC::Labjack` facade's operational API. All device-agnostic by
  construction.

**U6-specific work, when the time comes:**

- **Linux/macOS exo backend.** Vendor LabJack's `u6.h` / `u6.cpp` (the
  U6 analog of `u3.cpp` — different USB packets, different calibration
  block, 14 analog inputs at 16 bits, no `ConfigIO` AIN/digital
  reconfiguration). Wire the U6 case into `labjackdriver_exo.cpp` and
  add `u6CalibrationInfo` to the exo `DeviceHandle` variant. Replace
  `printf`s with `bcDebug` to match the U3 cleanup. Estimated 2–4 days.
- **Windows UD backend.** Add `LJ_dtU6 = 6` to `labjackconstants.h`,
  add a `Kind::U6` arm in the UD backend's `OpenLabJack` call, and
  apply U6-correct defaults in `eAIN` (differential `ChannelN` rules,
  range/gain constants). Estimated ~1 day.
- **Hardware classes.** New `LabjackU6` (and any U6 variants) sibling
  to `LabjackU3`, with U6-correct settings registration (channel
  count, ADC depth, range options). Estimated ~½ day.
- **Constants.** Extend `labjackconstants.h` with U6 device-type and
  range constants only when U6 work begins.

Total estimate: roughly a week of focused work, most of it on the
Linux/macOS side because that path requires LabJack's vendor helper
file.
