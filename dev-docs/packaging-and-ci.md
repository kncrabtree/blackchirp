# Packaging — Alpha Prep TODOs

Ephemeral scratchpad for the v2.0.0-alpha release. The durable
architecture reference (CPack/linuxdeploy strategy, Qt/Qwt sourcing
matrix, signing and provenance, AppImage glibc floor, non-intuitive
constructions, etc.) lives in the developer guide at
`doc/source/developer_guide/packaging.rst` — read that first if you
need the "how it works." This file is just the running checklist of
work items that block the alpha tag, and gets purged once the alpha
ships.

## Next steps before alpha tag

1. **Manual clean-VM smoke test on macOS and Windows.** The Linux
   platforms (DEB on Ubuntu LTS, RPM on openSUSE Tumbleweed and
   Fedora, AppImage on a non-Qt6-distro host) have all been
   manually verified; macOS and Windows are the remaining gap. The
   in-CI `--version` exercises only `main()` entry and the dynamic
   loader; the entire Qt runtime + `MainWindow` constructor is
   uncovered. Per-platform: launch the installed .app/.exe, see the
   main window draw, dismiss it. ~10 minutes per platform. What the
   manual pass catches that CI does not:
   - Qt platform plugin load (`qwindows` / cocoa) inside the
     `QApplication` ctor.
   - Deferred plugin loads on first paint (`imageformats/`,
     `iconengines/qsvgicon`, `tls/qschannelbackend`,
     `styles/qmodernwindowsstyle`).
   - OpenGL context creation when qwt's plot is first
     instantiated (linking `Qt6OpenGL.dll` ≠ initializing it).
   - Release-build (-O3) codepath through GUI init — local builds
     here are Debug, CI Release-builds but doesn't exercise GUI;
     any UB-at-O3 bug is unobserved.
   - First-run dialog (`savePath` empty → `ApplicationConfigDialog`
     + `RuntimeHardwareConfigDialog`); `.ui` forms loaded by uic.
   - Gatekeeper / SmartScreen UX on unsigned binaries; what the
     user has to click through after downloading.
   - `QSettings` against a fresh user profile (HKCU / plist
     creation).
   - The crash handler itself (`CrashHandler::install`,
     `MiniDumpWriteDump`, signal handlers); never run in Release
     because `--version` early-returns before it.

   Acceptance criteria per platform:
   - `.dmg` (arm64) on a clean Apple Silicon Mac — both `.app`s launch
     and find bundled libqwt (`otool -L` shows `@executable_path/...`).
   - `.dmg` (x86_64) on a clean Intel Mac — same checks; this one
     covers Mac users still on pre-2022 hardware.
   - NSIS `.exe` on a clean Windows install — main window opens
     without `STATUS_DLL_NOT_FOUND` from a transitive Qt6OpenGL
     load.

2. **Cut the alpha release.** Tag `v2.0.0-alpha`, push; the
   workflow's `release: published` trigger fires across all five
   platforms and uploads artifacts to the GitHub release. The three
   pre-release notices added in `f0a8596a` (README,
   `doc/source/index.rst`, `doc/source/python.rst`) stay through
   alpha / beta / rc — they go away only at the `v2.0.0` release.

3. **Follow-up not blocking the alpha** (deferred):
   - VC++ Redistributable distribution for end users on a stock
     Windows install. Smoke-runners-with-VS-installed happens to
     mask this; the clean-VM Windows test in item 1 will surface
     it. Options: NSIS auto-run of `vc_redist.x64.exe`, or document
     it as a prerequisite. The clean-VM pass decides which.
   - Apple Developer ID + notarization and Windows Authenticode
     signing. Both cost money (~$99/yr Apple, $200–$500/yr code
     signing CA). Defer until budget exists or a sponsor offers a
     cert; Gatekeeper / SmartScreen click-through is the stand-in
     for alpha.

## Windows hardware-registry truncation (blocks alpha)

**Status:** Part 1 (`/WHOLEARCHIVE` / `--whole-archive` / `-force_load`
INTERFACE link options on `blackchirp-hardware`) and Part 2
(`testAllExpectedImplementationsRegistered` in
`tests/tst_hardwareregistrytest.cpp`) implemented and passing on
Linux. Awaiting Windows CI verification per the flow below.

### Symptom

Field report from the manual Windows clean-VM smoke pass (NSIS installer
on Windows 10/11, the artifact built by `release.yml`'s `windows-nsis`
job): the Add Profile dialog in the hardware browser is missing most
hardware types. Only `FtmwDigitizer`, `LifDigitizer`, and `TemperatureController`
appear, and within each only a subset of implementations is offered.
**No `virtual*` implementation is present for any hardware type**, which
means a fresh install cannot satisfy the "every required hardware type
needs at least one active profile" precondition that
`HardwareManager::initialize()` enforces via
`HardwareProfileManager::ensureSystemProfiles()` →
`RuntimeHardwareConfig::activateMissingSystemProfiles()`. The app comes
up but the user has no way to construct a valid configuration without
hand-editing settings; LIF mode is unreachable; even the FTMW path is
crippled because `Clock` (→ `FixedClock`) is absent.

Linux DEB/RPM/AppImage and (presumably) macOS DMG do not exhibit this.

### Root cause

`cmake/BlackchirpHardware.cmake` builds the hardware tree as
`add_library(blackchirp-hardware STATIC ${BLACKCHIRP_HARDWARE_SOURCES})`
and `cmake/BlackchirpApplication.cmake` consumes it via
`target_link_libraries(blackchirp PRIVATE Blackchirp::Hardware)`. Each
hardware implementation file (e.g.,
`src/hardware/core/ftmwdigitizer/virtualftmwdigitizer.cpp`,
`src/hardware/core/clock/fixedclock.cpp`,
`src/hardware/optional/chirpsource/virtualawg.cpp`, …) registers itself
with `HardwareRegistry` exclusively through **namespace-scope static
initializers** produced by the macros in
`src/hardware/core/hardwareregistration.h` — `REGISTER_HARDWARE_META`,
`REGISTER_HARDWARE_PROTOCOLS`, `REGISTER_HARDWARE_SETTINGS`, and the
array/library/custom-comm variants. There is no `extern` symbol in
those `.cpp` files that any other translation unit references; the
implementation classes are constructed only through the factory lambdas
captured by `HardwareAutoRegistration`, which themselves are reachable
only after the static initializers have run.

MSVC's `link.exe` does not preserve static-initializer side effects
when resolving a static library:

- For each `.obj` member of a `.lib`, the linker pulls it in **only**
  if some unresolved external in the link refers to a symbol the
  `.obj` defines.
- Static-initializer objects (the `register_<CLASS>` `HardwareAutoRegistration`
  instances and the `settings_registered_<CLASS>` / `protocols_registered_<CLASS>`
  / `BC_ARRDEF_VAR(...)` bools) are private to the `.obj` and have no
  outside-the-TU references, so they do not anchor the file.
- Result: every `virtualftmwdigitizer.obj`,
  `fixedclock.obj`, `valon5009.obj`, `virtualawg.obj`,
  `virtualpulsegenerator.obj`, `virtualflowcontroller.obj`,
  `virtualgpibcontroller.obj`, `virtualioboard.obj`,
  `virtualpressurecontroller.obj`, `virtualtempcontroller.obj`,
  `virtuallifdigitizer.obj`, `virtualliflaser.obj`, etc. that **does not
  happen to also define a symbol referenced from the executable** is
  silently dropped, and its registration constructors never run.

The handful of implementations that do show up
(`FtmwDigitizer`/`LifDigitizer`/`TemperatureController` types appearing at
all means at least one of each type linked in) survive only by
coincidence: e.g.,
`src/hardware/core/hardwareprofilemanager.cpp` references
`VirtualFtmwDigitizer::staticMetaObject.className()`,
`FixedClock::staticMetaObject.className()`,
`VirtualLifDigitizer::staticMetaObject.className()`, and
`VirtualLifLaser::staticMetaObject.className()` from
`ensureSystemProfiles()`, but `<Class>::staticMetaObject` is defined
in `moc_<class>.cpp` (separate `.obj`), not in the implementation
`.cpp` that holds the static initializers — so the moc object gets
pulled in while the implementation object stays dropped. A handful of
files happen to define non-registration symbols (e.g., out-of-line
vtable anchors referenced through the moc) that incidentally pull them
into the link, which is why a "subset" — not zero — non-virtual
implementations appear. The pattern is not deliberate, just whatever
the linker happens to chase.

GNU `ld` / Mach-O `ld64` are subject to the same archive-extraction
rule in principle, but their default behavior around `.init_array` /
`.mod_init_func` sections and weak symbols ends up keeping these
`.o` files alive in practice (which is why Linux and macOS builds
appear fine without intervention). MSVC has no equivalent
heuristic — its archive extraction is purely symbol-driven.

The author was already aware of this pattern in one specific case:
`cmake/BlackchirpHardware.cmake` appends `PYTHON_HARDWARE_HEADERS` to
`HARDWARE_IMPLEMENTATION_HEADERS` with the comment "Without this, the
static registration initializers are silently dropped." That fix
relied on AUTOMOC emitting moc TUs that reference the implementation
symbols, but AUTOMOC operates on a target's listed sources, not on
indirectly-included headers, so the trick was load-bearing only in the
narrow case the author tested (and only on Linux); the same drop
silently affects every other `virtual*.cpp` on Windows.

### Proposed fix

Two parts: a linker-flag change that forces full inclusion of
`blackchirp-hardware` members on MSVC, and a new ctest case that
fails fast on the GitHub runner if the registration set regresses
again (so the next regression surfaces in `ctest` output rather
than during a manual clean-VM pass).

#### Part 1 — `/WHOLEARCHIVE` on `blackchirp-hardware` (INTERFACE)

Place the link-option in `cmake/BlackchirpHardware.cmake` next to
`add_library(blackchirp-hardware STATIC …)` and attach it to the
library's **`INTERFACE`** so every consumer (the `blackchirp`
executable and the four tests that link the library:
`tst_hardwareregistrytest`, `tst_runtimehardwareconfigtest`,
`tst_hardwareprofilemanagertest`, `tst_experimentloading`)
automatically inherits it. Putting it on `blackchirp` directly
would leave the test binaries with the original drop behavior,
and the registry regression test (Part 2) would either fail
spuriously or mask the very bug it is supposed to detect.

```cmake
add_library(blackchirp-hardware STATIC ${BLACKCHIRP_HARDWARE_SOURCES})
add_library(Blackchirp::Hardware ALIAS blackchirp-hardware)

# Hardware implementations register themselves via namespace-scope
# static initializers (see src/hardware/core/hardwareregistration.h
# REGISTER_HARDWARE_META and friends). MSVC link.exe drops static-
# library .obj files whose only "use" is a static-init side effect,
# silently disabling most of the hardware registry on Windows.
# Force the linker to keep every .obj in blackchirp-hardware. GNU ld
# and ld64 happen to keep these alive through .init_array handling
# today, so the equivalent flags there are harmless-but-unneeded;
# wire them up anyway so a future -Wl,--gc-sections (or a switch to
# lld with archive-pruning enabled) does not silently regress us.
# INTERFACE so every consumer of blackchirp-hardware — executable
# *and* tests — gets the flag, otherwise the registry regression
# test below would not actually exercise the production behavior.
if(MSVC)
    target_link_options(blackchirp-hardware INTERFACE
        "/WHOLEARCHIVE:$<TARGET_FILE_NAME:blackchirp-hardware>")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$" AND NOT APPLE)
    target_link_options(blackchirp-hardware INTERFACE
        "LINKER:--whole-archive"
        "$<TARGET_FILE:blackchirp-hardware>"
        "LINKER:--no-whole-archive")
elseif(APPLE)
    target_link_options(blackchirp-hardware INTERFACE
        "LINKER:-force_load,$<TARGET_FILE:blackchirp-hardware>")
endif()
```

The MSVC form uses `$<TARGET_FILE_NAME:>` (filename only) because
`/WHOLEARCHIVE:` matches against the library's basename in the link
line, not its full path. The two POSIX forms use `$<TARGET_FILE:>`
(full path) because `--whole-archive` / `-force_load` consume the
library path positionally.

CMake ≥ 3.24 supports
`$<LINK_LIBRARY:WHOLE_ARCHIVE,blackchirp-hardware>` as a portable
single-form replacement, but `CMakeLists.txt` currently pins
`cmake_minimum_required(VERSION 3.16)`. Bumping the floor is its own
discussion (Ubuntu 22.04 ships 3.22; openSUSE Leap 16 / Tumbleweed
both ship ≥ 3.27) — out of scope for this fix.

Alternative, more invasive: convert `blackchirp-hardware` from
`STATIC` to `OBJECT`. Object libraries embed their `.obj` files into
the consumer rather than going through an archive, sidestepping the
extraction rule entirely. Downsides: every test target that
`target_link_libraries(... blackchirp-hardware)` would embed a
private copy of the hardware tree, inflating link time and test
binary size more than `/WHOLEARCHIVE` does (which only inflates the
binaries that actually pull the library through the link). The
`/WHOLEARCHIVE` approach also keeps the existing shape of the build
graph; `OBJECT` would require auditing every consumer's transitive
link.

#### Part 2 — registry regression test

Add a new test case to `tests/tst_hardwareregistrytest.cpp` that
calls `HardwareRegistry::instance().getHardwareTypes()` and
`getImplementations(type)` and asserts a hard-coded baseline of
`(type, implementation)` pairs. The test binary already links
`blackchirp-hardware`, so with Part 1's INTERFACE flag in place it
will exercise the same linker behavior the production `blackchirp`
binary does: if the drop reappears, this test fails in `ctest`
on the `windows-nsis` job before any artifact is even uploaded.

Baseline pairs to assert (one row per required type, one row per
virtual for each optional type — kept hard-coded for a
self-documenting failure message; new drivers need to add a row
here when they are wired in):

```text
FtmwDigitizer             ↔ VirtualFtmwDigitizer         (required type)
Clock                 ↔ FixedClock                (required type)
AWG                   ↔ VirtualAwg
PulseGenerator        ↔ VirtualPulseGenerator
FlowController        ↔ VirtualFlowController
GpibController        ↔ VirtualGpibController
IOBoard               ↔ VirtualIOBoard
PressureController    ↔ VirtualPressureController
TemperatureController ↔ VirtualTempController
LifDigitizer              ↔ VirtualLifDigitizer           (required when LIF on)
LifLaser              ↔ VirtualLifLaser           (required when LIF on)
```

The LIF rows go under the same gate the application uses
(`ApplicationConfigManager::instance().isLifEnabled()`); the
simpler form is to assert them unconditionally — the LIF
sources compile into `blackchirp-hardware` in every configuration
the CI builds, so the registry should contain them regardless of
the app-level enable flag. The runtime gate only controls whether
the app *uses* them, not whether they register.

Why hard-coded instead of derived from `hw_impl.h`: parsing the
generated header to discover what "should" be present would mirror
whatever subset survived the linker pass, defeating the test.
The expected list has to be authored independently of the build
output to anchor the assertion.

Why not derive from
`RuntimeHardwareConfig::isHardwareRequired()`: that function tells
us which *types* must have an active profile, but says nothing
about which *implementations* of each type need to register. The
bug under test is "the virtual is missing" — type presence alone
is insufficient.

Sketch:

```cpp
void HardwareRegistryTest::testAllExpectedImplementationsRegistered()
{
    struct Expect { const char *type; const char *impl; };
    static constexpr Expect required[] = {
        {"FtmwDigitizer",             "VirtualFtmwDigitizer"},
        {"Clock",                 "FixedClock"},
        {"AWG",                   "VirtualAwg"},
        {"PulseGenerator",        "VirtualPulseGenerator"},
        {"FlowController",        "VirtualFlowController"},
        {"GpibController",        "VirtualGpibController"},
        {"IOBoard",               "VirtualIOBoard"},
        {"PressureController",    "VirtualPressureController"},
        {"TemperatureController", "VirtualTempController"},
        {"LifDigitizer",              "VirtualLifDigitizer"},
        {"LifLaser",              "VirtualLifLaser"},
    };

    const QStringList types =
        HardwareRegistry::instance().getHardwareTypes();
    for (const auto& e : required) {
        QVERIFY2(types.contains(QLatin1String(e.type)),
                 qPrintable(QStringLiteral(
                     "Hardware type '%1' missing from registry — "
                     "almost certainly a static-init drop in the "
                     "blackchirp-hardware static library. See "
                     "dev-docs/packaging-and-ci.md → \"Windows "
                     "hardware-registry truncation\".")
                     .arg(QLatin1String(e.type))));

        const QStringList impls =
            HardwareRegistry::instance().getImplementations(
                QLatin1String(e.type));
        QVERIFY2(impls.contains(QLatin1String(e.impl)),
                 qPrintable(QStringLiteral(
                     "Hardware implementation '%2' missing from "
                     "registry under type '%1'.")
                     .arg(QLatin1String(e.type),
                          QLatin1String(e.impl))));
    }
}
```

Verify the actual class names against the registered subKeys
(`<Class>::staticMetaObject.className()` for each
`REGISTER_HARDWARE_META` site) before committing the table —
e.g., `VirtualTempController` vs. `VirtualTemperatureController`,
`VirtualAwg` vs. `VirtualAWG`. Source of truth is the class name
in each `virtual*.h` header under `src/hardware/`.

Size cost: with `/WHOLEARCHIVE` and the registry test in place,
`tst_hardwareregistrytest.exe` grows to embed the full hardware
tree (a few MB extra). Acceptable for a single test binary in
exchange for catching the regression in `ctest` instead of after
artifact upload. The other three tests that link
`blackchirp-hardware` already do so today and pick up the same
inflation; size is not load-bearing on CI runtime.

### Verification once fixed

Cannot run on the dev box (no Windows host). Verification flow:

1. Apply Part 1 and Part 2. Push a non-tag commit to `master`,
   trigger `release.yml` via `workflow_dispatch` with only
   `run_windows` enabled.
2. Watch the `windows-nsis` job's `Test` step. The new
   `HardwareRegistryTest::testAllExpectedImplementationsRegistered`
   case must pass. If it fails on Windows but passes on Linux,
   the `/WHOLEARCHIVE` flag is not reaching the test binary —
   most likely the `INTERFACE` attachment is wrong or
   `target_link_libraries(tst_hardwareregistrytest blackchirp-hardware …)`
   has been replaced with a direct source list somewhere.
3. Once `ctest` passes, download `blackchirp-windows.zip`,
   install on a clean Windows VM, launch.
4. Open the hardware browser → Add Profile dialog. Confirm:
   - All hardware types appear (`FtmwDigitizer`, `Clock`, `AWG`,
     `PulseGenerator`, `FlowController`, `GpibController`,
     `IOBoard`, `PressureController`, `TemperatureController`,
     plus `LifDigitizer` and `LifLaser` if LIF is enabled in the
     application config).
   - Each type lists its `virtual*` (or `FixedClock`)
     implementation among the options.
5. Cancel out, confirm
   `RuntimeHardwareConfig::activateMissingSystemProfiles()` has
   produced a workable default config (Settings file should
   contain `<type>.virtual` entries for every
   `isHardwareRequired(type)` type).

With Part 2 in place, step 2 is the gate; steps 3–5 become a
confirmation pass rather than the primary detector.

## Legacy LIF experiment viewer regression (blocks alpha)

**Status:** Resolved on Linux; Windows clean-VM viewer pass still
required. Part A (digitizer rename, commit `9291461f`) and Part B
(`loadLegacyLif883_*` regression cases in
`tests/tst_experimentloading.cpp` against
`tests/testdata/legacy_lif/883/`) are landed. Settings-vs-data
violations resolved by reading laser units from column 6 of the
`LaserStart` header row and inferring fractional digits from the
on-disk value strings (no schema field added) — see
`LifConfig::retrieveValues` and the matching
`LifDisplayWidget::prepareForExperiment` rewire. The "Root cause"
section below describes the pre-rename code path; left in place
as historical context for the regression cases' assertions.

### Symptom

Field report from a Windows user loading a pre-v2 LIF experiment
(reference fixture: `~/Downloads/883`, `BCBuildVersion=v0.1-496`) in
`blackchirp-viewer`: the experiment opens but the LIF panes are
broken — spectrogram looks garbage, slice plots empty, and
`LifProcessingWidget`'s gate spin-boxes are constrained to a
near-zero range so the user cannot push them to the values shown in
the `processing.csv` from the original capture
(`LifGateStartPoint=290`, `LifGateEndPoint=6130`). Hypothesised
candidates from the bug report were: (a) the Windows hardware-
registry truncation above, (b) a side effect of
`f0c972311ed20d308606f8e64b8155afe590afda`, or (c) a missed legacy
read-compat path. (c) is correct; (a) and (b) are unrelated.

### Not the Windows hardware registry

The hardware-registry bug above is about which implementations
register at startup. The viewer never instantiates hardware — it
reads an on-disk experiment. None of the load path touches
`HardwareRegistry::createHardware` or any constructor anchored by
the dropped initializers. Same failure reproduces on Linux/macOS
with the same fixture.

### Not commit `f0c97231`

`f0c972311ed20d308606f8e64b8155afe590afda` ("Harden CSV enum-cell
reads and simplify hardware.csv schema") changes:

- `hardware.csv` from a 3-column to a 2-column schema, with reader
  backward-compat for the 1-, 2-, and 3-column forms — the legacy
  fixture's 2-column form (`key;subKey` header, `Clock.0;fixed`
  rows etc.) loads cleanly under the new reader; verified by
  inspecting `HardwareDataContainer::loadFromFile` against
  `~/Downloads/883/hardware.csv`.
- Enum-cell reads (`FtmwConfig::FtmwType`, `RfConfig::Sideband`,
  `FtWorker::FtUnits`, `FtWorker::FtWindowFunction`, `ClockType`,
  `MultOperation`) routed through `BC::CSV::enumFromVariant`.
  None of these enums are read by the LIF load path —
  `LifConfig::retrieveValues()` and `LifDigitizerConfig::retrieveValues()`
  only pull primitives (`int`, `double`, `bool`) and the
  `ScanOrder` / `CompleteMode` enums which were already in
  name-form on the legacy disk.

So the post-`f0c97231` reader handles this fixture's bytes
correctly. The problem is upstream of the row-level read.

### Pre-rename root cause — missing `"LifDigitizer"` in the legacy type map

`src/data/experiment/hardwaredatacontainer.h:158-175` defines
`legacyStringToHardwareType()`, which the loader uses (via
`extractHardwareType()`) to recover a `HardwareType` enum from the
key prefix of every row in `hardware.csv`. As originally written
the map covered every historical hardware-type alias except one:
the **`LifDigitizer`** root key was absent, even though its sibling
`FtmwDigitizer` was covered.

The `883` fixture's `hardware.csv` has `LifDigitizer.0;m4i2211x8`.
The loader runs `legacyStringToHardwareType("LifDigitizer")` →
`HardwareType::Unknown`, and the row lands in
`hardwareMap["LifDigitizer.0"] = {"m4i2211x8", HardwareType::Unknown}`.

`Experiment::enableLif()`
(`src/data/experiment/experiment.cpp:474-502`) then searches
`hardwareMap` for any entry whose `.type == HardwareType::LifDigitizer`
to recover the digitizer hardware key. Since the only candidate is
tagged `Unknown`, the search produces an empty `digitizerHwKey`. The
constructed `LifConfig` wraps a `LifDigitizerConfig("")` whose
`HeaderStorage::d_headerKey` is the empty string.

`HeaderStorage::storeLine`
(`src/data/storage/headerstorage.cpp:96-135`) dispatches every
header row to the child whose `d_headerKey` equals the row's
`ObjKey` cell. Every `LifDigitizer.0;…;RecordLength;8192;` row
(and every other `LifDigitizer.0` line — sample rate, bytes per
point, analog-channel array, etc.) fails to match an empty
`d_headerKey`, so all the digitizer's stored values are silently
dropped on load. `LifDigitizerConfig::retrieveValues()` then falls
through to its `retrieve(recLen,0)` default, leaving
`d_recordLength = 0`.

`LifDisplayWidget::prepareForExperiment`
(`src/gui/lif/gui/lifdisplaywidget.cpp:126`) calls
`p_procWidget->initialize(e.lifConfig()->digitizerConfig().d_recordLength, …)`
with the zero `recordLength`.
`LifProcessingWidget::initialize`
(`src/gui/lif/gui/lifprocessingwidget.cpp:147-156`) sets the
spin-box ranges to `[0, recLen-2]` for the start gates and
`[1, recLen-1]` for the end gates. With `recLen=0` those are
`[0, -2]` and `[1, -1]`, which QSpinBox clamps to effectively
`[0, 0]` and `[1, 1]`. `LifDisplayWidget::prepareForExperiment`
then calls `setAll(e.lifConfig()->d_procSettings)` immediately
afterward, and `QSpinBox::setValue(290)` / `setValue(6130)` clamp
to the broken range — silently overwriting the persisted gate
positions with the spin-box maxima. From then on, every
`LifTrace::integrate(getSettings())` call in `updatePoint` /
`updatePlots` integrates over an essentially zero-width window
and the spectrogram fills with garbage.

This was a single one-line omission cascading through six layers,
which the rename has obviated — `LifDigitizer` is now the canonical
key and resolves directly without going through the legacy alias.

### Proposed fix — Part A (resolved by digitizer rename)

The bug is resolved by the `FtmwScope`/`LifScope` → `FtmwDigitizer`/
`LifDigitizer` class rename. With `LifDigitizer` now the canonical
class name, the legacy type map in
`src/data/experiment/hardwaredatacontainer.h:158-175` carries it as
the canonical row and adds `FtmwScope`/`LifScope` as pre-rename
aliases for pre-1.0.0 and devel-era data:

```cpp
static HardwareType legacyStringToHardwareType(const QString& legacyTypeString) {
    static const QHash<QString, HardwareType> legacyTypeMap = {
        ...
        {"FtmwDigitizer", HardwareType::FtmwDigitizer},   // canonical
        {"FtmwScope", HardwareType::FtmwDigitizer},       // pre-rename alias
        ...
        {"LifDigitizer", HardwareType::LifDigitizer},     // canonical
        {"LifScope", HardwareType::LifDigitizer},         // pre-rename alias
        {"LifLaser", HardwareType::LifLaser}
    };
    return legacyTypeMap.value(legacyTypeString, HardwareType::Unknown);
}
```

The `883` fixture's `LifDigitizer.0` row now tags directly as
`HardwareType::LifDigitizer` (canonical, not aliased), `enableLif()`
finds it, `LifDigitizerConfig` is constructed with `d_headerKey =
"LifDigitizer.0"`, dispatch matches the header rows, the record
length loads as 8192, the spin-box ranges open up to `[0, 8190]` /
`[1, 8191]`, and the persisted gate values (290 / 6130) survive
the `setAll()` clamp.

### Proposed fix — Part B (regression test)

Add an `ExperimentLoading` test case that loads a pruned copy of
`883` from `tests/testdata/legacy_lif/` (header + objectives +
hardware + the lifparams.csv + a single non-empty trace .csv —
keeping the fixture size in the tens of KB) and asserts:

- `exp.lifEnabled() == true`
- `exp.lifConfig()->digitizerConfig().d_recordLength == 8192`
- `exp.lifConfig()->d_delayPoints == 1`,
  `exp.lifConfig()->d_laserPosPoints == 201`
- `exp.lifConfig()->d_procSettings.lifGateStart == 290`,
  `lifGateEnd == 6130`

`tests/tst_experimentloading.cpp` already exists and is the
natural home — it links `blackchirp-test-hardware`, which is fine
since the test exercises the on-disk loader, not the hardware
registry. The CSV-side legacy compat for `LifDigitizer` is the
narrow thing being tested; an inline-temp-file case
(`hardwareReaderRecognisesLegacyTypeAliases`) added in the rename
pass already covers the `FtmwScope`/`LifScope` aliases pointing to
the new canonical types, so the `883` fixture test only needs to
assert the end-to-end recordLength / gate-value chain.

### Settings-vs-data violations in display widgets

The user also asked to flag any place a viewer widget reads
hardware metadata (units, decimals, ranges, etc.) from the local
machine's `SettingsStorage::Hardware` group instead of from the
on-disk experiment. These are portability bugs even when not
triggering visible breakage: the same `883` directory opened on
a workstation whose `BC::Key::LifLaser` group says
`units="cm-1"` instead of `nm` would silently mislabel the laser
axis without any obvious failure.

**Violations under `FtmwViewWidget` / its children**: none found.
The FTMW view path threads experiment data through end to end —
verified by scanning every `.cpp` reachable from
`gui/widget/ftmwviewwidget.cpp` for `SettingsStorage::Hardware`,
`BC::Key::FtmwDigitizer`, `BC::Key::AWG`, `BC::Key::Clock`, etc.; the
only hits are inside acquisition-only widgets
(`ChirpConfigWidget` is reached only from `FtmwConfigWidget`,
which is reached only from `ExperimentFtmwConfigPage` and
`FtmwConfigDialog` — neither used by the viewer).

**Violations under `LifDisplayWidget` / its children**:

| File : line | What it reads | What it should read |
| --- | --- | --- |
| `src/gui/lif/gui/lifdisplaywidget.cpp:117-119` | `BC::Key::LifLaser::units` and `BC::Key::LifLaser::decimals` from `SettingsStorage(lifLaserKey, Hardware)` — drives the laser-axis suffix string and decimal places on every spectrogram / delay-slice update | Persist these on the `LifConfig` (or, better, on the `HardwareDataContainer` row for the laser entry) at save time and retrieve them from the loaded `Experiment` here. |
| `src/data/experiment/experiment.cpp:493-495` | Same — reads `"units"` from the laser hardware group at `enableLif()` time and stuffs it into `LifConfig::d_laserUnits`, which then feeds `storeValues()`'s `Unit` cells in `header.csv`. The viewer ends up with whatever the *local* machine has, not what was recorded. | `LifConfig::retrieveValues()` should pull the unit string from the loaded header.csv (`lStart`'s Units column already carries it on disk — the legacy `883` file's `LifConfig;;;LaserStart;280;nm` records it), instead of having `enableLif()` re-derive it from local hardware. |

Worth doing alongside Part B so legacy and current LIF experiments
both display the unit string the user actually recorded, not the
local default.

**Out of scope for the viewer audit but flagged for awareness**:
the laser-position decimal count (`LifLaser::decimals`) is not
persisted at all today — it's a local convention. If we go ahead
with persisting the laser units on `LifConfig`, decimals should
ride along.

### Acquisition-side hardware reads (not violations)

For completeness, the following `SettingsStorage::Hardware` reads
are acquisition-time only and correctly read from local hardware:

- `src/gui/lif/gui/liflaserwidget.cpp:20-26,39` — live laser
  control panel.
- `src/gui/lif/gui/liflaserstatusbox.cpp:26,43` — live laser
  status box on the main acquisition window.
- `src/gui/widget/pulsestatusbox.cpp:19,67`,
  `src/gui/widget/temperaturestatusbox.cpp:18,54`,
  `src/gui/widget/pressurestatusbox.cpp:56`,
  `src/gui/widget/gasflowdisplaywidget.cpp:32,87,117` — live
  status boxes (none of these are placed in
  `LifDisplayWidget`/`FtmwViewWidget`; they sit on the main
  acquisition window).
- All hardware reads in `gui/dialog/`, `gui/expsetup/`,
  `gui/widget/*protocolwidget.cpp`, `gui/widget/customprotocolwidget.cpp`,
  `gui/widget/hwsettingswidget.cpp`,
  `gui/widget/chirpconfigwidget.cpp`,
  `gui/widget/pulseconfigwidget.cpp`,
  `gui/widget/pressurecontrolwidget.cpp`,
  `gui/mainwindow.cpp:863-877` — all acquisition setup or live
  control; correct to use local hardware state.

### Verification

1. Drop the pruned `883` fixture into `tests/testdata/legacy_lif/`
   and apply Part B. Re-run `tst_experimentloading`; the new test
   case must pass.
2. Open `~/Downloads/883` in `blackchirp-viewer` and verify the
   LIF panes populate: spectrogram is non-zero, slice plots show
   the recorded LIF/ref traces, the gate spin-boxes are usable
   (`[0, 8190]` / `[1, 8191]`) and pick up the persisted
   `290`/`6130` values from `processing.csv`.
3. With the settings-vs-data fix applied, also check that flipping
   the local machine's `BC::Key::LifLaser::units` to something
   nonsensical (e.g., `"foo"`) does **not** change the axis suffix
   when reopening `883` — the suffix should stay `nm` because
   that's what the header.csv recorded.
4. Smoke the Windows binary on the same fixture once the
   hardware-registry fix from the previous section also lands —
   if the Windows-only registry truncation is preventing
   `blackchirp` (the acquisition app) from creating new LIF
   experiments at all, the legacy load path needs to be exercised
   through `blackchirp-viewer` rather than the main acquisition
   app to isolate the fix.
