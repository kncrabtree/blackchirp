# Bundle 12k — Developer Guide: Vendor Libraries

**Status:** complete

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-05-03: not started → complete. Authored
  doc/source/developer_guide/vendor_libraries.rst covering the
  VendorLibrary contract, staged-configuration model and HardwareManager
  reload coordination, REGISTER_LIBRARY linkage, the LabjackLibrary and
  SpectrumLibrary subclasses, the LabJack exo/UD case study with a
  Mermaid diagram of the three-layer split, and the 8-step recipe for
  adding a new VendorLibrary subclass. No source-tree changes; no
  warnings tied to the new page. Content commit
  341d1aa18fb3b2a092c8f25dd83e87894baa4f0e.
-->

Sub-page of the Developer Guide chapter. Documents how Blackchirp
integrates closed-source vendor SDKs at runtime via the
`VendorLibrary` base class, the `REGISTER_LIBRARY` linkage between
hardware implementations and libraries, the staged-configuration UI
pattern, and the recipe for adding a new vendor library subclass.
The LabJack `exo`/`UD` cross-platform split is the worked example.

## Scope

Single Sphinx file:
`doc/source/developer_guide/vendor_libraries.rst`.

The page should answer the following for a contributor:

1. **Why dynamic loading.** One-paragraph rationale:

   - Blackchirp must run on machines that lack any given
     vendor SDK. Linking vendor libraries at compile time
     would force a rebuild per deployment and would prevent
     the binary distribution from supporting hardware whose
     drivers are not installed on the build machine.
   - Each `VendorLibrary` subclass uses `QLibrary` to locate
     and load the vendor library at runtime. If the library is
     absent or fails to load, Blackchirp starts normally and
     the dependent hardware implementations report themselves
     as unavailable — they do not prevent application launch.
   - The result: one binary, runtime-discovered hardware
     support.

2. **`VendorLibrary` contract.**

   - Singleton. Each subclass exposes `instance()` that
     returns a reference to the per-process instance.
   - The loading sequence (cross-link to
     `:doc:`/classes/vendorlibrary`` for the API):
     1. Try the user-provided path
        (`setUserProvidedPath`), if set.
     2. Try the user-specified search directories
        (`setUserSearchPaths`), then the platform-specific
        defaults (`defaultSearchPaths`), if automatic
        discovery is enabled.
     3. Resolve function pointers via the subclass's
        `loadFunctions()` implementation.
     4. Validate that all required symbols were found.
   - Lifecycle: load on first `isAvailable()` query (or on
     explicit reload). Unload on application exit.
   - The subclass owns the typed function pointers as
     private members and exposes them through inline
     accessors so hardware code can call them directly.

3. **Staged-configuration UI pattern.**

   - The `LibraryStatusWidget` in the Application Settings
     dialog must let the user point at a library path, set
     custom search directories, or toggle automatic discovery
     — without affecting running hardware until the user
     clicks Apply.
   - `VendorLibrary` exposes paired *staged* and *applied*
     state. UI code calls `setStagedUserProvidedPath`,
     `setStagedSearchPaths`, `setStagedAutoDiscoveryEnabled`
     to accumulate changes; `applyChanges()` promotes the
     staged values, persists them, and reloads the library;
     `revertChanges()` discards staged edits;
     `hasUnstagedChanges()` enables/disables the Apply button.
   - Reload coordination: when a library reload affects
     existing hardware, `HardwareManager` consults
     `HardwareRegistry::getLibraryDependencies` for the
     dependent hardware implementations and tears them down
     before the reload, recreates them after.
   - Cross-link to `:doc:`/classes/vendorlibrary``,
     `:doc:`/classes/hardwareregistry``, and the user-guide
     page `:doc:`/user_guide/library_status``.

4. **`REGISTER_LIBRARY` linkage.**

   - A hardware implementation that depends on a vendor
     library declares the dependency via
     `REGISTER_LIBRARY(MyHardware, MyLibrary)` after
     `REGISTER_HARDWARE_META`.
   - `HardwareRegistry` records the dependency so:
     - The `Hardware Registry` panel of the Hardware
       Configuration dialog can show "library X is missing"
       next to implementations that need it.
     - `HardwareManager` can destroy and recreate affected
       hardware around a library reload.
   - The macro is in `hardware/core/hardwareregistration.h`;
     it is documented at the registry level on
     `:doc:`/classes/hardwareregistry``.

5. **Concrete subclasses.** Two ship with Blackchirp:

   - **`LabjackLibrary`** — wraps the LabJack U3 driver. The
     LabJack case has additional structure because the
     vendor's library and ABI differ between platforms; see
     the *Case study* section below.
   - **`SpectrumLibrary`** — wraps the Spectrum
     Instrumentation SDK
     (`spcm_linux` / `spcm64.dll`), used by the M4i digitizer
     for FTMW acquisition. Singleton-enforced because the
     Spectrum library maintains global state and cannot be
     loaded twice in one process.

   Cross-link both to their API pages.

6. **Case study: LabJack `exo` / `UD` cross-platform split.**

   The LabJack integration illustrates a non-trivial
   pattern that future cross-platform vendor libraries can
   follow.

   - **Three layers:**
     - `LabjackLibrary`
       (`hardware/library/labjacklibrary.{cpp,h}`) — the
       dynamic loader. Symbol set is conditionally compiled
       per platform:
       - Linux/macOS: `LJUSB_*` transport symbols from
         `liblabjackusb.so` / `.dylib` (the vendor's
         "exodriver" transport library).
       - Windows: high-level UD symbols (`OpenLabJack`,
         `eAIN/eDAC/eDI/eDO`, `eTCConfig/eTCValues`,
         `ErrorToString`, `GetDriverVersion`) from
         `LabJackUD.dll`.
     - `BC::Labjack` facade
       (`hardware/library/labjackdriver.h`) — a thin,
       platform-neutral interface. Exposes
       `isAvailable()`, `errorString()`, `openU3(serialOrLocalId)`
       (returning an opaque `HandlePtr`), and
       per-operation functions
       (`readAnalog`, `writeAnalog`, `readDigital`,
       `writeDigital`, `configureTimers`, `readTimers`).
     - **Backend translation units** — implement the
       facade; selected by CMake:
       - `labjackdriver_exo.cpp` (Linux/macOS, gated by
         `NOT WIN32`) — wraps `u3.cpp` helper functions.
         `DeviceHandle` carries a `void* h` (LJUSB
         handle) and a `u3CalibrationInfo` union member.
       - `labjackdriver_ud.cpp` (Windows, gated by
         `WIN32`) — calls UD easy functions directly via
         `LabjackLibrary`. `DeviceHandle` carries a
         `long h` (`LJ_HANDLE`).
     - The vendor's `u3.cpp` helper is also gated to
       `NOT WIN32` and is the sole consumer of the LJUSB
       transport symbols.
   - **Caller pattern:** `labjacku3.cpp` uses the
     `BC::Labjack::*` facade exclusively. It never sees a
     raw `LJUSB_*` symbol or a raw UD function. The
     `HandlePtr` is `std::unique_ptr<DeviceHandle,
     void(*)(DeviceHandle*)>`; the deleter calls the
     appropriate close function (`LJUSB_CloseDevice` or
     UD `Close`) and frees the struct.
   - **Why the split exists:** the LJUSB transport library
     and the UD high-level library have *different ABIs*
     and *different vendor licenses*. Wrapping both behind
     a facade lets `labjacku3.cpp` (the hardware class) be
     platform-agnostic, while keeping the platform-specific
     logic confined to two translation units chosen at
     CMake configure time.
   - **Adding a new LabJack model (e.g., U6):** documented
     as a recipe in the original case-study source. The
     facade is shaped so a U6 addition does not require
     restructuring — add `openU6()` to the facade, vendor
     `u6.cpp` (analogous to `u3.cpp`) on the exo side, add
     a `Kind::U6` arm to each `switch` in the backend
     translation units, gate `u6.cpp` to `NOT WIN32` in
     CMake. The `LabjackLibrary` and the operational facade
     API do not change.

   This case study is the page's worked example for the
   broader pattern: when a vendor library has a multi-
   platform reality (different ABI, different driver name,
   different calling convention), wrap the differences in a
   facade and select the backend at CMake time. Do not bake
   the platform conditionals into the hardware class.

7. **Recipe: adding a new `VendorLibrary` subclass.** A
   numbered list:

   1. Create `hardware/library/<name>library.{cpp,h}`
      defining a singleton subclass of `VendorLibrary`.
   2. Override `loadFunctions()` to resolve the vendor
      symbols via `QLibrary::resolve` and store them in
      typed function-pointer members. Mark required symbols;
      a missing required symbol means the library failed to
      load.
   3. Define inline accessors that hardware code uses to
      call the resolved symbols (typed wrappers).
   4. Override `defaultSearchPaths()` to return the
      conventional install locations on each platform.
   5. Add `#include` of the new library to
      `BlackchirpHardware.cmake`'s `HARDWARE_SYSTEM_SOURCES`
      list (the `cmake/Blackchirp*.cmake` glob does **not**
      pick up `hardware/library/` automatically — it is
      enumerated explicitly).
   6. In each hardware implementation that depends on the
      library, add `REGISTER_LIBRARY(YourHwClass,
      YourLibraryClass)` after `REGISTER_HARDWARE_META`.
   7. If the vendor SDK has a multi-platform ABI split,
      consider the LabJack pattern: provide a thin facade
      header and select the backend `.cpp` at CMake time.
   8. If the library has global mutable state and cannot be
      reloaded safely while in use, follow the
      `SpectrumLibrary` pattern and document the
      restriction in the class-level Doxygen.

## Out of scope

- The user-facing Application Settings → Library Status
  workflow — `:doc:`/user_guide/library_status``.
- `HardwareRegistry` itself — bundle 12d.
- Per-class documentation of `VendorLibrary`,
  `LabjackLibrary`, `SpectrumLibrary` — covered on their
  API pages.
- Vendor library authoring (writing the vendor library) —
  not Blackchirp's domain.

## Sources

### Related source files

- `src/hardware/library/vendorlibrary.{cpp,h}` — base.
- `src/hardware/library/labjacklibrary.{cpp,h}` — concrete
  subclass; the `#ifdef WIN32` symbol-set split.
- `src/hardware/library/labjackdriver.h` — the `BC::Labjack`
  facade.
- `src/hardware/library/labjackdriver_exo.cpp` — Linux/macOS
  backend.
- `src/hardware/library/labjackdriver_ud.cpp` — Windows
  backend.
- `src/hardware/library/labjackconstants.h`,
  `src/hardware/library/spectrumconstants.h` — vendor
  constants.
- `src/hardware/library/spectrumlibrary.{cpp,h}` — concrete
  subclass.
- `src/hardware/core/hardwareregistration.h` —
  `REGISTER_LIBRARY` macro.
- `src/hardware/core/hardwareregistry.{cpp,h}` —
  `getLibraryDependencies`.
- `src/gui/widget/librarystatuswidget.{cpp,h}` —
  Application Settings UI.
- `src/cmake/BlackchirpHardware.cmake` — to confirm the
  conditional-compile gating (`if(WIN32)` vs.
  `else()`) and the explicit listing of vendor-library
  sources.
- `src/hardware/optional/ioboard/labjacku3.cpp` — the
  caller-side use of the `BC::Labjack` facade.
- A representative hardware implementation that uses
  `REGISTER_LIBRARY` for cross-reference (e.g., a Spectrum-
  using digitizer like `m4i2220x8.cpp`).

### Related dev-docs

- `dev-docs/labjack-cross-platform-support.md` — research
  material for the case study. Do not link.

### Related user-guide pages

- `doc/source/user_guide/library_status.rst` — the user-
  facing surface; cross-link.

### Related API reference pages

- `doc/source/classes/vendorlibrary.rst`
- `doc/source/classes/hardwareregistry.rst` (for
  `REGISTER_LIBRARY` and `getLibraryDependencies`)
- `doc/source/classes/hardwaremanager.rst` (for the
  reload coordination)

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/vendor_libraries.rst`.

## Page structure

H1 intro: 1–2 paragraphs framing the why and the pattern.

H2 sections (`-` underlines):

- *Why dynamic loading*
- *VendorLibrary contract*
- *Staged configuration*
- *REGISTER_LIBRARY linkage*
- *Concrete subclasses* — `LabjackLibrary`, `SpectrumLibrary`.
- *Case study: LabJack exo/UD split*
- *Recipe: adding a new VendorLibrary subclass* — numbered
  list.

## Acceptance criteria

- The dynamic-loading rationale is one paragraph, timeless.
- The `VendorLibrary` four-step loading sequence is
  documented.
- The staged-configuration model is documented at the
  staged/applied/persist/reload level.
- `REGISTER_LIBRARY` is documented as the way a hardware
  implementation declares its library dependency, with a
  one-line forward-pointer to the registry mechanism.
- Both concrete subclasses (`LabjackLibrary`,
  `SpectrumLibrary`) are named with their domain.
- The LabJack case study covers the three layers (library,
  facade, backends) and explicitly documents why the split
  exists (different ABI, different vendor licenses).
- The recipe for adding a new `VendorLibrary` subclass is
  an 8-step numbered list.
- The CMake-explicit-listing point is made (vendor-library
  sources are not in the glob).
- No duplication of per-class API content; cross-links cover
  per-class detail.
- No rendered link points into `dev-docs/`.
