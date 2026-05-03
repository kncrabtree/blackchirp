# Bundle 12m — Developer Guide: Adding a New Hardware Type

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Sub-page of the Developer Guide chapter. The rarer and broader
contributor task: adding a new abstract *hardware type* — a new
interface class that no existing driver matches. This involves
defining the interface, choosing the state-management pattern,
adding optional config and storage classes, integrating with the
GUI (status box, control widget, experiment-setup page), and
optionally providing a Python trampoline.

This is sibling content to bundle 12l (adding a driver). The
recipes overlap, but adding a *type* requires considerably more
coordination across `data/`, `gui/`, and `hardware/`.

## Scope

Single Sphinx file:
`doc/source/developer_guide/adding_a_hardware_type.rst`.

The page should answer the following for a contributor:

1. **When this applies.** Open by drawing the line:

   - If the new device is an instance of an existing hardware
     type (a new AWG model, a new flow controller), this is
     the wrong page — go to bundle 12l.
   - If the new device implements a *category* Blackchirp does
     not yet have (e.g., a new "BeamBlocker" or a "MagneticField
     Coil"), this page applies.
   - The threshold for "new type" vs. "new role on existing
     type" is whether the existing types' interfaces could be
     adapted with reasonable effort. Adding a new type is real
     work; prefer fitting an existing type when possible.

2. **Pre-flight: design the interface.** The interface class
   defines the slot/signal API the rest of Blackchirp uses.
   Decisions to make before writing code:

   - **Base class.** Always
     `public HardwareObject` (and possibly other auxiliary
     bases like `HeaderStorage` if the type carries
     experiment-time config).
   - **`d_threaded` default.** Most hardware types are
     threaded (set in the type's constructor body), so vendor
     I/O does not block the GUI. Reserve non-threaded for
     hardware that must run on the manager thread (rare).
   - **`d_critical` default.** Critical = a connection
     failure aborts experiments; non-critical = experiment
     continues. The default in the constructor is one
     decision; users can override it per-profile.
   - **Supported communication protocols.** The set of
     protocols the type can support across implementations.
     The base class declares them via
     `REGISTER_HARDWARE_BASE` (for shared protocols) and
     each implementation refines via
     `REGISTER_HARDWARE_PROTOCOLS`.
   - **State-management pattern.** Pick A, B, or C
     (cross-link to bundles 12j and 12l for the patterns):
     - **A — Bulk Configure** when the type owns a complex
       config object (typical for digitizers and IO boards):
       expose a pure-virtual `configure(config&)` that
       implementations override.
     - **B — Granular Methods** when the type's interaction
       is per-channel/per-parameter (typical for flow
       controllers, pulse generators, temperature
       controllers): expose a set of `hw*` pure virtuals,
       and have the base class own polling sequencing.
     - **C — Stateless / Pass-Through** when the type is
       configured exclusively at experiment time (typical
       for AWG, Clock): expose a pure-virtual
       `prepareForExperiment(Experiment&)` that
       implementations override.
   - **Validation keys.** What aux-data keys this type
     produces that should be range-checked during
     acquisition (see `validationKeys()`).
   - **Optional config object.** Whether the type needs a
     dedicated `HeaderStorage` subclass for its
     experiment-time configuration (e.g., `IOBoardConfig`,
     `LifDigitizerConfig`). If yes, this is registered with
     `Experiment::addOptHwConfig` so it serializes alongside
     the experiment.

3. **The interface class.** Skeleton for a new type
   `MyType`:

   ```cpp
   #include <hardware/core/hardwareobject.h>
   #include <data/settings/hardwarekeys.h>

   class MyType : public HardwareObject
   {
       Q_OBJECT
   public:
       MyType(const QString& impl, const QString& label,
              QObject *parent = nullptr);
       virtual ~MyType();

   public slots:
       // Pattern-A example:
       virtual bool configure(MyTypeConfig &cfg) = 0;
       // OR Pattern-B example:
       virtual double hwReadFoo(int channel) = 0;
       virtual bool   hwWriteFoo(int channel, double v) = 0;
       // OR Pattern-C example: nothing extra; the base class's
       //   hwPrepareForExperiment() forwards directly.

       // Public slot for the rest of Blackchirp:
       double readFoo(int channel) {
           auto v = hwReadFoo(channel);
           if (qIsNaN(v)) emit hardwareFailure();
           emit fooUpdated(channel, v, QPrivateSignal());
           return v;
       }

   signals:
       void fooUpdated(int channel, double v, QPrivateSignal);

   public:
       virtual QStringList validationKeys() const override;
   };
   ```

   ```cpp
   REGISTER_HARDWARE_BASE(MyType,
       { BC::Key::MyType::numChannels, "Channels",
         "Number of supported channels", 4, 1, 32,
         HwSettingPriority::Required },
       // … other shared settings
   )

   MyType::MyType(const QString& impl, const QString& label,
                  QObject *parent)
       : HardwareObject(QString(MyType::staticMetaObject.className()),
                        impl, label, parent)
   {
       d_threaded = true;       // typical default
       // d_critical defaults to true; override here only if appropriate.
   }
   ```

   The implementation file owns the `REGISTER_HARDWARE_BASE`
   for shared settings; concrete drivers inherit those and
   add or override their own via `REGISTER_HARDWARE_SETTINGS`.

4. **Wire the type into the build.**

   - Add the `.cpp` and `.h` to the type's directory under
     `src/hardware/core/<type>/` or
     `src/hardware/optional/<type>/`. The
     `BlackchirpHardware.cmake` glob picks them up.
   - Add the type's interface header to
     `HARDWARE_TYPE_HEADERS` in
     `BlackchirpHardware.cmake`. This is **not** a glob —
     interface headers are enumerated explicitly. Bundle
     12a documents the `hw_base.h` aggregator.
   - If the type lives in a *new* subdirectory, add the
     directory's glob patterns to the
     `HARDWARE_IMPLEMENTATIONS_SOURCES` and
     `HARDWARE_IMPLEMENTATION_HEADERS` lists in
     `BlackchirpHardware.cmake`. Use the existing patterns
     as templates.

5. **Optional config object.** When the type needs a
   `HeaderStorage`-derived config:

   - Create
     `src/data/experiment/hardware/<core|optional>/<type>/<typename>config.{cpp,h}`.
     The convention is one config class per hardware type
     that owns experiment-time state.
   - Inherit `HeaderStorage` and implement the
     `storeValues`/`retrieveValues`/`prepareChildren`
     virtuals. Cross-link to bundle 12i for the
     `HeaderStorage` tree.
   - Register with `Experiment::addOptHwConfig` from the
     hardware object's `hwPrepareForExperiment`. The
     experiment owns the config copy via
     `std::shared_ptr<HeaderStorage>` keyed by header key.
   - If the config also needs UI representation in the
     experiment-setup wizard, see step 7 below.

6. **`hwPrepareForExperiment` and lifecycle hooks.**

   - The base wrapper
     `HardwareObject::hwPrepareForExperiment` reattempts a
     connection if disconnected and dispatches to the
     virtual `prepareForExperiment`. Pattern A types
     additionally have a base-level
     `hwPrepareForExperiment` that calls the virtual
     `configure(config&)` after pulling the right config
     from `Experiment` and writes the validated config
     back; Pattern B/C types have the base wrapper forward
     directly.
   - `beginAcquisition` / `endAcquisition` are broadcast
     hooks: implement when the type needs to start/stop
     hardware actions at experiment boundaries (e.g., start
     a continuous data stream).
   - `sleep(bool)` puts the device into a low-power state
     when no experiment is running. Override when the
     hardware supports it.
   - `readSettings()` is called when the user accepts the
     hardware settings dialog so the driver can refresh
     cached state.

7. **GUI integration.** A new hardware type usually needs
   three GUI surfaces; each is optional but each makes the
   device usable.

   - **Status box** — a small widget shown on the
     Hardware Status panel for live device state. Inherits
     `HardwareStatusBox` (see `src/gui/widget/`); subscribes
     to the relevant `HardwareManager` signals
     (`fooUpdate(hwKey, ...)` etc.). The GUI auto-discovers
     status boxes for known types; new types must be added
     to the type-dispatch in
     `MainWindow::createStatusBox` (or equivalent — confirm
     the exact site name in source).
   - **Control widget** — the Control tab of `HwDialog`
     (cross-link to bundle 12e). Inherits `QWidget`; communicates
     with the live hardware object via `HardwareManager`-mediated
     queued slots. New types must be added to the
     type-dispatch in
     `MainWindow::createHWDialog` (or equivalent — confirm
     in source).
   - **Experiment-setup page** — a wizard page in
     `gui/expsetup/` for any experiment-time configuration
     the type contributes (for example,
     `ExperimentFlowConfigPage` for `FlowController`).
     Inherits `ExperimentConfigPage`; reads/writes the
     optional config object from step 5. New types must be
     added to the experiment-setup wizard's page-construction
     list in `ExperimentSetupDialog`.

8. **Auxiliary and validation data.**

   - Override `readAuxData()` to publish per-experiment
     readings — they appear on the Aux/Rolling tabs and
     persist in `AuxDataStorage`.
   - Override `readValidationData()` if the type produces
     values that should range-check during acquisition.
     `validationKeys()` declares the keys; the experiment-
     validation page in the setup wizard lets the user set
     ranges.
   - Cross-link to bundle 12e (the fan-out plumbing) and
     bundle 12f (the validation abort path).

9. **HardwareManager fan-out.** `HardwareManager` already
   handles the generic `auxData` / `validationData` /
   `rollingData` signals with `hwKey` prefixing — no per-
   type code needed there. For type-specific signals
   (e.g., per-channel updates), add a signal pattern
   `<typename>Update(QString hwKey, …)` to
   `HardwareManager` and forward from the hardware
   object's signals via queued connection. Cross-link to
   the existing flow-controller / pressure-controller
   patterns in `HardwareManager` for the convention.

10. **Optional Python trampoline.** Most new hardware
    types should have a Python trampoline so users can write
    drivers without recompiling Blackchirp:

    - Subclass both the new type and `PythonHardwareBase`.
      Initialize `PythonHardwareBase(d_key, d_model)` in the
      constructor.
    - Pick the matching state-management pattern (the same
      A/B/C decision as for C++ drivers; bundle 12j covers
      the trampoline contract).
    - Provide `python_<typename>_template.py` with the
      canonical class name (`<TypeName>Driver`).
    - Cross-link to bundle 12j for the trampoline recipe.

11. **Tests.** A new hardware type touches enough surface
    area that smoke tests are insufficient. Add:

    - A unit test for the optional config object's
      round-trip serialization (analogous to
      `tst_headerstoragetest`).
    - A virtual implementation
      (`Virtual<TypeName>`) and a fixture in
      `blackchirp-test-hardware` (cross-link to bundle 12a).
    - An entry in `tst_hardwareregistrytest` if appropriate
      to verify the registration pipeline picks up the new
      type.

12. **Document.** A new hardware type triggers documentation
    work in three places:

    - **API ref:** add a class page under
      `doc/source/classes/<typename>.rst` following
      `:doc:`/developer_guide/api_style``.
    - **User guide:** add a per-device page under
      `doc/source/user_guide/hw/` (the hardware-details
      collection) when at least one concrete driver exists.
    - **Developer guide:** if the new type introduces a new
      pattern not covered in this guide, flag it for a
      developer-guide refresh in the orchestrator's
      hand-off.

## Out of scope

- The simpler "new driver" recipe — bundle 12l.
- Python-trampoline implementation detail — bundle 12j.
- New experiment mode (FtmwType / BatchManager) — bundle
  12n.
- Vendor library subclass authoring — bundle 12k.

## Sources

### Related source files

- The interface headers for existing hardware types — read
  several to cement the conventions:
  - `src/hardware/optional/chirpsource/awg.{cpp,h}` — Pattern C.
  - `src/hardware/optional/flowcontroller/flowcontroller.{cpp,h}` —
    Pattern B with an extensive `hw*` surface.
  - `src/hardware/optional/ioboard/ioboard.{cpp,h}` — Pattern A
    with a config class.
  - `src/hardware/core/clock/clock.{cpp,h}` — Pattern C with
    role assignments.
  - `src/hardware/core/ftmwdigitizer/ftmwscope.{cpp,h}` —
    the WaveformBuffer-aware Pattern A variant.
- `src/hardware/core/hardwareobject.{cpp,h}` — base
  contract.
- `src/hardware/core/hardwareregistration.h` — macros.
- `src/data/experiment/hardware/...` — existing optional
  hardware config classes; pick representative examples.
- `src/gui/widget/hardwarestatusbox.{cpp,h}` and the
  per-type subclasses — status-box pattern.
- `src/gui/widget/gascontrolwidget.{cpp,h}` (or another
  representative) — control-widget pattern.
- `src/gui/expsetup/experimentflowconfigpage.{cpp,h}` (or
  another) — experiment-setup-page pattern.
- `src/gui/mainwindow.{cpp,h}` — the dispatch sites that
  need updating for a new type.
- `src/cmake/BlackchirpHardware.cmake` — the explicit
  `HARDWARE_TYPE_HEADERS` list and the per-type globs.

### Related dev-docs

None directly. Bundles 12d, 12e, 12i provide the
conceptual context.

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/hardware_details.rst` and the
  `hw/` per-device sub-pages.
- `doc/source/user_guide/experiment_setup.rst`.

### Related API reference pages

- `doc/source/classes/hardwareobject.rst`
- `doc/source/classes/hardwareregistry.rst`
- `doc/source/classes/headerstorage.rst`
- `doc/source/classes/experiment.rst`
- `doc/source/classes/auxdatastorage.rst`
- `doc/source/classes/communicationprotocol.rst`
- `doc/source/developer_guide/api_style.rst`

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/adding_a_hardware_type.rst`.

## Page structure

H1 intro: 2 paragraphs framing this as a rare-but-foundational
task; explicit pointer to bundle 12l for the more common
"new driver" task.

H2 sections (`-` underlines):

- *When this applies vs. adding a driver*
- *Designing the interface* — pattern selection, threading,
  protocols, validation keys, optional config object.
- *The interface class* — skeleton.
- *Wiring into the build*
- *Optional config object*
- *Lifecycle hooks* — `hwPrepareForExperiment`,
  `beginAcquisition`/`endAcquisition`, `sleep`,
  `readSettings`.
- *GUI integration* — status box, control widget,
  experiment-setup page.
- *Auxiliary and validation data*
- *HardwareManager fan-out*
- *Optional Python trampoline*
- *Tests*
- *Documentation follow-up*

## Acceptance criteria

- The "new type vs. new driver" decision is stated up
  front.
- Pattern selection (A/B/C) is documented for new types
  with the same vocabulary as bundles 12j and 12l.
- Skeleton code shows the interface class, the
  `REGISTER_HARDWARE_BASE` block, and the constructor.
- The build-system integration step covers the
  `HARDWARE_TYPE_HEADERS` explicit list as well as the
  glob pickup for implementations.
- The optional config object's role and registration
  through `Experiment::addOptHwConfig` is documented.
- The three GUI surfaces (status box, control widget,
  experiment-setup page) are each described with their
  base class and dispatch site in `MainWindow` /
  `ExperimentSetupDialog`.
- The aux/validation virtuals and the `validationKeys()`
  override are documented.
- The Python trampoline is recommended (not mandated) and
  forward-links to bundle 12j.
- Test recommendations name specific existing tests as
  templates.
- Documentation follow-up is enumerated (API page, user
  guide per-device, possible developer-guide refresh).
- No duplication of per-class API content; cross-links
  cover per-class detail.
- No rendered link points into `dev-docs/`.
