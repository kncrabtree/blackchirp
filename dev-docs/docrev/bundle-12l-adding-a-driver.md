# Bundle 12l — Developer Guide: Adding a New Hardware Driver

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. doc/source/developer_guide/adding_a_driver.rst
  drafted directly per the orchestrator-direct workflow. Eleven hardware-type
  interface classes enumerated with domains and source directories; five-file
  driver layout with the no-CMake-edits glob convention; constructor plus
  REGISTER_HARDWARE_META / _PROTOCOLS / _SETTINGS skeleton with the
  threaded-hardware constructor restriction; three state-management patterns
  mapped onto C++ overrides with one worked example per pattern (IOBoard for
  Pattern A, FlowController for Pattern B, AWG for Pattern C); initialize()
  vs testConnection() split, aux/validation data, REGISTER_CUSTOM_COMM and
  REGISTER_LIBRARY, smoke-testing checklist. Per-driver Virtual<Driver>
  pushed out of scope here and routed to bundle 12m. Smoke-testing section
  reframed to acknowledge that virtual-protocol profiles only verify the
  registration is intact; recommends hwDebug + runtime debug logging + the
  Python trampoline path for actual protocol bring-up. Build clean except
  for three forward :doc: references to /developer_guide/adding_a_hardware_type
  (bundle 12m, not started). Content commit 96a5c34598d59d04a043b86ef20b4245f39c291e.
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Developer Guide chapter. The most common contributor
task: adding a new *driver* (a new implementation of an existing
hardware type). For example, adding a new AWG model, a new flow
controller, a new GPIB instrument. Step-by-step recipe with multiple
worked examples covering the three state-management patterns.

Adding a new hardware *type* (a new abstract base class that no
existing implementation matches) is a separate, rarer task — bundle
12m. This page assumes the type already exists.

## Scope

Single Sphinx file:
`doc/source/developer_guide/adding_a_driver.rst`.

The page should answer the following for a contributor:

1. **Pre-flight: pick the right base class.**

   - List the available hardware-type interface classes and
     their domains:
     `AWG`, `Clock`, `FtmwScope`, `LifScope`, `LifLaser`,
     `FlowController`, `GpibController`, `IOBoard`,
     `PressureController`, `PulseGenerator`,
     `TemperatureController`.
   - Decide which one the new driver implements. If none fit
     and a new abstract type is needed, jump to bundle 12m.
   - For each hardware type, name the directory the driver
     should live in:
     - core types in `src/hardware/core/<type>/`,
     - optional types in `src/hardware/optional/<type>/`.
     The split is historical and corresponds to whether the
     type is required to run an FTMW experiment (core) or
     genuinely optional (optional). Cross-link to the
     `:doc:`/developer_guide/architecture`` source-tree
     section for the directory layout.

2. **The five canonical files.** Every driver consists of
   the same five touches:

   1. `src/hardware/<core|optional>/<type>/<driver>.cpp` — the
      implementation.
   2. `src/hardware/<core|optional>/<type>/<driver>.h` — the
      class declaration.
   3. `src/data/settings/hardwarekeys.h` — additional setting
      keys, if the driver introduces new ones beyond the
      base class. Add them in the appropriate
      `BC::Key::<DriverName>` sub-namespace.
   4. (Optional) registration of any new vendor library
     dependency via `REGISTER_LIBRARY` — see bundle 12k.
   5. (Optional) a virtual sibling
      (`Virtual<Driver>`) — for tests; see *Virtual sibling*
      below.

   No CMakeLists edits are required: the
   `BlackchirpHardware.cmake` glob picks up files matching
   the recognized name patterns. Bundle 12a documents the
   patterns and the AUTOMOC linkage requirement.

3. **The constructor and registration macros.** The boilerplate:

   ```cpp
   #include <hardware/optional/chirpsource/awg.h>
   #include <hardware/core/hardwareregistration.h>
   #include <data/settings/hardwarekeys.h>

   class MyAwg : public AWG
   {
       Q_OBJECT
   public:
       explicit MyAwg(const QString& label, QObject *parent = nullptr);
       ~MyAwg() override = default;
   private:
       // hw* virtuals + helpers
   };
   ```

   ```cpp
   REGISTER_HARDWARE_META(MyAwg,
                          "Vendor Model 1234 — short description")
   REGISTER_HARDWARE_PROTOCOLS(MyAwg, CommunicationProtocol::Tcp,
                                       CommunicationProtocol::Rs232)
   REGISTER_HARDWARE_SETTINGS(MyAwg,
       { BC::Key::AWG::rate, "Sample Rate (Hz)", "DAC sample rate",
         8e9, 1e6, 1000e9, HwSettingPriority::Important },
       // … only settings that *differ* from the base class default;
       //   inherited settings are picked up automatically.
   )

   MyAwg::MyAwg(const QString& label, QObject *parent)
       : AWG(QString(MyAwg::staticMetaObject.className()),
             label,
             parent)
   {
       // Set d_threaded / d_critical here if defaults from AWG do not fit.
       // For most drivers the base class's defaults are correct.
   }
   ```

   - The base class (`AWG`) constructor takes
     `(impl, label, parent)`. `impl` is supplied via
     `staticMetaObject.className()` so the driver class name
     becomes the implementation key.
   - `REGISTER_HARDWARE_META` / `_PROTOCOLS` / `_SETTINGS`
     run at static-init time. Cross-link to
     `:doc:`/developer_guide/hardware_configuration`` for the
     macro family and the base/impl override pattern.
   - The constructor body should **not** create child
     `QObject`s if the base class sets `d_threaded = true`
     (most do). Construct children in `initialize()`
     instead. Cross-link to
     `:doc:`/developer_guide/hardware_runtime``.

4. **Choose the right state-management pattern.** Bundle
   12j defines three patterns; they apply equally to C++
   drivers, not just Python trampolines. Re-frame for a
   C++ author:

   - **Pattern A (bulk configure)** — applies when the
     hardware type expects a `configure(config&)` virtual
     (e.g., `IOBoard`, `LifScope`). Override `configure`
     to apply the experiment-supplied config to the device,
     read back actuals, and update the config reference.
   - **Pattern B (granular)** — applies when the hardware
     type exposes per-channel/per-parameter `hw*` pure
     virtuals (e.g., `FlowController`, `PulseGenerator`,
     `TemperatureController`, `PressureController`).
     Implement each `hw*` virtual; the base class owns
     sequencing.
   - **Pattern C (stateless / pass-through)** — applies
     when the hardware type expects per-experiment data at
     `prepareForExperiment` time (`AWG`, `Clock`).
     Override `prepareForExperiment` to program the
     hardware with the supplied chirp / clock config.

   Pick the pattern by reading the type's interface header.
   Each `hw*`-style virtual on the type is a Pattern B
   hint; a `configure(...)` virtual is a Pattern A hint;
   neither but with a non-trivial `prepareForExperiment`
   override expectation is Pattern C.

5. **Worked example A — Pattern B (FlowController).**

   - Walk through a hypothetical `MyFlowController`
     implementing the eight `hw*` pure virtuals
     (`hwReadFlow(ch)`, `hwSetFlow(ch, val)`,
     `hwReadFlowSetpoint(ch)`,
     `hwReadPressure()`,
     `hwReadPressureSetpoint()`, `hwSetPressureSetpoint(val)`,
     `hwReadPressureControlMode()`,
     `hwSetPressureControlMode(b)`).
   - Each virtual issues a few SCPI / serial commands via
     `p_comm->queryCmd` or `p_comm->writeCmd`, parses the
     response, returns the value.
   - The base class handles polling order, `readAll()`, and
     the per-experiment validation/aux-data calls.
   - Cross-reference an existing implementation
     (`MksMfc647c` is a representative example; the drafter
     should pick whichever is currently in
     `src/hardware/optional/flowcontroller/`).

6. **Worked example B — Pattern C (AWG).**

   - Walk through a hypothetical `MyAwg` overriding
     `prepareForExperiment(Experiment&)`. Inside the
     override:
     - Read `exp.ftmwConfig()->d_rfConfig` and
       `chirpConfig()` to obtain the chirp parameters.
     - Compute the waveform samples (or upload a
       precomputed pattern).
     - Compute the marker bytes via
       `ChirpConfig::getPackedMarkerData()` and apply the
       per-implementation bit remapping (see bundle 12n's
       worked example, or the existing AWG implementations
       in `src/hardware/optional/chirpsource/`).
     - Send `*WAI`/equivalent and verify the upload via
       any vendor-specific status query.
     - Return true on success; on failure, set
       `d_errorString` and return false.
   - Show how to declare optional `markerCount` overrides
     via `REGISTER_HARDWARE_SETTINGS` (rare; usually the
     `markerCount` is fixed for a model).
   - Cross-reference an existing AWG implementation as the
     ground truth for the pattern (`Awg70002a`, `Awg5204`,
     `Awg7122b`, `M8195a`, `M8190`, `Ad9914`,
     `VirtualAwg` — drafter picks whichever best
     illustrates the pattern).

7. **Worked example C — Pattern A (IOBoard).**

   - Walk through a hypothetical `MyIOBoard` overriding
     `bool configure(IOBoardConfig&)`. Inside:
     - Apply each enabled analog channel's range,
       coupling, sample rate to the hardware.
     - Apply each enabled digital channel.
     - Read back actuals and update the config reference.
     - Per-call channel selection on each
       `readAnalogChannels(...)` etc. (the base class hands
       the trampoline / driver the enabled-channel list so
       it does not need to remember between calls).
   - Cross-reference `LabjackU3` or `VirtualIOBoard` as
     the pattern reference.

8. **`testConnection()` and `initialize()`.**

   - `initialize()` is called once after construction and
     thread move. Build any persistent helpers, register
     timers, allocate device-side state. **Do not attempt
     vendor I/O here**; the comm protocol may fail to
     connect.
   - `testConnection()` is the place for the cheap
     interaction with the device — typically `*IDN?` or
     vendor equivalent — plus an assertion that the
     responding device is the expected model. On failure,
     set `d_errorString` and return false. May be called
     many times.

9. **Aux/validation/rolling data.**

   - `readAuxData()` returns an `AuxDataMap` of per-experiment
     readings. Override when the device produces values worth
     plotting on the Aux/Rolling tabs and persisting in
     `AuxDataStorage`.
   - `readValidationData()` returns the subset of
     `readAuxData` values that should be range-checked during
     acquisition. The base class decides which keys are
     validated via `validationKeys()`.
   - Both are optional; default returns an empty map. See
     bundle 12e for the fan-out plumbing.

10. **Virtual sibling.**

    - For every concrete driver, prefer to also provide a
      `Virtual<Driver>` (or contribute fixtures to the
      existing `Virtual<Type>` if a per-type virtual is more
      appropriate). The virtual implementation:
      - Lets users without the real hardware run experiments
        end-to-end (synthesizes plausible aux data, generates
        synthetic chirps, etc.).
      - Is the canonical fixture for `blackchirp-test-hardware`
        unit tests (cross-link to bundle 12a).
    - Conventionally registered with `Virtual` in the
      protocols list and an in-source-tree settings template.

11. **Custom communication protocol parameters.**

    - If the driver uses `CommunicationProtocol::Custom`
      (vendor SDK, USB-HID, memory-mapped, etc.), declare
      the user-facing connection parameters via
      `REGISTER_CUSTOM_COMM`:
      ```cpp
      REGISTER_CUSTOM_COMM(MyDriver,
          { BC::Key::MyDriver::devicePath, "Device Path",
            "Path to the device node",
            CustomCommType::FilePath, {}, {}, {} },
      )
      ```
    - The descriptors drive `CustomProtocolWidget` to
      render the right input widgets at profile creation.
      Cross-link to bundle 12e for the runtime side.

12. **Vendor library dependency.** If the driver depends
    on a vendor SDK loaded by a `VendorLibrary` subclass,
    add `REGISTER_LIBRARY(MyDriver, MyVendorLibrary)`
    after the META macro. Bundle 12k covers the vendor-
    library side.

13. **Smoke testing.**

    - Build with `BC_BUILD_TESTS=ON` and run the existing
      hardware tests
      (`tst_hardwareregistrytest`,
      `tst_runtimehardwareconfigtest`,
      `tst_hardwareprofilemanagertest`,
      `tst_hardwarekeys`) to catch macro misuse and
      hardware-key collisions.
    - Run the application; the new driver should appear in
      the Add Profile dialog under its hardware type.
    - Create a profile with the Virtual protocol (or
      `Virtual<Driver>` if you provided one), verify the
      hardware connects.
    - Run a short experiment exercising the relevant code
      paths.

## Out of scope

- Adding a new hardware *type* — bundle 12m.
- Adding a new Python *trampoline* — bundle 12j (which
  cross-references this page for the C++ side).
- Adding a new experiment mode (FtmwType / BatchManager) —
  bundle 12n.
- Vendor-library subclass authoring — bundle 12k.
- The full registration-macro parameter list — bundle 12d
  (which forward-links to
  `:doc:`/classes/hardwareregistry`` for the macro
  reference).

## Sources

### Related source files

- A representative spread of existing implementations:
  - `src/hardware/optional/chirpsource/awg70002a.cpp` and
    similar AWG files — Pattern C.
  - `src/hardware/optional/flowcontroller/mks*.cpp` —
    Pattern B.
  - `src/hardware/optional/ioboard/labjacku3.cpp` —
    Pattern A.
  - `src/hardware/core/clock/valon5009.cpp` — Clock
    (Pattern C).
  - `src/hardware/core/ftmwdigitizer/m4i2220x8.cpp` —
    `FtmwScope` (overrides `prepareForExperiment` directly,
    "Pattern A-ish" with the WaveformBuffer story).
  - `src/hardware/optional/chirpsource/virtualawg.cpp` and
    other `virtual*.cpp` siblings — Virtual patterns.
- `src/hardware/core/hardwareregistration.h` — macros.
- `src/hardware/core/hardwareobject.{cpp,h}` — base class
  contract and the `bcInit*` / `bcTest*` wrappers.
- `src/data/settings/hardwarekeys.h` — namespace conventions.
- The base classes for each hardware type:
  `src/hardware/optional/chirpsource/awg.{cpp,h}`,
  `src/hardware/core/clock/clock.{cpp,h}`,
  `src/hardware/core/ftmwdigitizer/ftmwscope.{cpp,h}`,
  `src/hardware/optional/flowcontroller/flowcontroller.{cpp,h}`,
  etc.

### Related dev-docs

None directly. (The hardware addition workflow is documented
in code patterns, not a dedicated dev-doc.)

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/hardware_config.rst` — the user-
  facing profile-creation flow.

### Related API reference pages

- `doc/source/classes/hardwareobject.rst`
- `doc/source/classes/hardwareregistry.rst`
- `doc/source/classes/communicationprotocol.rst`
- `doc/source/classes/custominstrument.rst`
- `doc/source/classes/auxdatastorage.rst`

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/adding_a_driver.rst`.

## Page structure

H1 intro: 1–2 paragraphs framing the page as the canonical
recipe for the most common contributor task; explicitly
distinguish "new driver" from "new type" and "new Python
trampoline" with forward-links.

H2 sections (`-` underlines):

- *Picking the right base class*
- *Files you will create*
- *Constructor and registration macros*
- *State-management patterns* — re-frame Patterns A/B/C for C++
  drivers; cross-link to 12j for the parallel Python view.
- *Worked example A: Pattern B (FlowController)*
- *Worked example B: Pattern C (AWG)*
- *Worked example C: Pattern A (IOBoard)*
- *initialize() and testConnection()*
- *Auxiliary and validation data*
- *Virtual sibling*
- *Custom protocol and vendor libraries*
- *Smoke testing*

The worked examples are the page's most important content;
they should be concrete enough that a contributor can copy
them and adapt without reading another page.

## Acceptance criteria

- The eleven-or-so hardware-type interface classes are
  enumerated with one-line domains.
- The five canonical files are listed; the no-CMake-edit
  point is made (rely on the `BlackchirpHardware.cmake`
  glob).
- The constructor + three registration macros are shown as
  a working code skeleton.
- The three state-management patterns are mapped explicitly
  to C++ overrides (`configure` / `hw*` / `prepareForExperiment`).
- Each pattern has at least one worked example referencing
  a real existing implementation as ground truth.
- `initialize()`-vs-`testConnection()` split is documented.
- Aux / validation virtuals are documented.
- The virtual-sibling convention is documented.
- Custom protocol descriptor declaration via
  `REGISTER_CUSTOM_COMM` is documented.
- Vendor library dependency declaration via
  `REGISTER_LIBRARY` is documented.
- Smoke-testing checklist names the relevant existing tests.
- No duplication of per-class API content; cross-links cover
  per-class detail.
- No rendered link points into `dev-docs/`.
