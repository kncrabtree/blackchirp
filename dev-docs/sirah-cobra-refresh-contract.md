# Sirah Cobra refresh — frozen interface contract

This document freezes the cross-task interface seams for the
[Sirah Cobra refresh](sirah-cobra-refresh.md). It is **authored once,
consumed by every task**. The keystone (Task 1) authors §0 and §A; Task 3
authors §B and §C; Tasks 2/4/5/6 consume them. Do **not** redesign these
signatures, key names, enum names, or semantics without escalating to the
orchestrator — divergence here breaks downstream tasks.

Everything not fixed here is left to the implementing task, which should
follow the plan doc and existing codebase conventions (`src/AGENTS.md`).

## Internal representation — vacuum wavenumber (cm⁻¹)

**The entire LIF laser/axis pipeline works internally in vacuum
wavenumber (cm⁻¹).** This is the single canonical unit for every
`double` position/frequency that crosses a LIF interface: `LifLaser`
positions and `minPos`/`maxPos`, `LifFreqConversionStage` local
setpoints, `LifConversion` inputs/outputs, `LifConfig` axis values, and
the values `HardwareManager` dispatches. cm⁻¹ is proportional to
frequency, so every conversion op (§A) is affine and exactly invertible.

Two unit boundaries, and only two:

- **Driver boundary:** a concrete driver converts between its hardware's
  native unit and cm⁻¹ (e.g. Sirah's grating math yields nm →
  driver converts nm↔cm⁻¹). Above the driver, everything is cm⁻¹.
- **Display boundary:** the GUI / on-disk axis label converts cm⁻¹ → the
  user-selected `LaserUnit` (§0) for presentation and for the
  `header.csv` value+label.

No backward-compatibility path (Harvey-Mudd instrument retired); on-disk
LIF axis values change unit with no migration shim.

---

## 0. Unit foundation — `LaserUnit` enum + conversion utility (Task 1)

**Home:** a new header in the **Data** library (e.g.
`src/data/lif/lifunits.h` / `.cpp`), added to `cmake/BlackchirpData.cmake`.
Data is linked by both the hardware library and the GUI.

Declare a `Q_NAMESPACE` (so the enum is `Q_ENUM_NS`-registered and works
with `BC::CSV::enumFromVariant` and `EnumComboBox<T>`):

```cpp
namespace BC::LifConv {
Q_NAMESPACE

// User-facing display units. Internal representation is always Cm1.
enum class LaserUnit { Cm1, Nm, GHz, eV };
Q_ENUM_NS(LaserUnit)

// Convert a value expressed in unit u to internal vacuum wavenumber (cm⁻¹).
double toCm1(double value, LaserUnit u);
// Convert internal vacuum wavenumber (cm⁻¹) to a value in unit u.
double fromCm1(double cm1, LaserUnit u);
// Short display suffix for a unit ("cm⁻¹", "nm", "GHz", "eV").
QString unitLabel(LaserUnit u);
}
```

Conversion relations (define the constants in this TU; CODATA values,
`c = 2.99792458e10 cm/s`):

- `Nm`  : `λ_nm = 1e7 / ṽ`,  `ṽ = 1e7 / λ_nm`  (vacuum wavelength)
- `GHz` : `f_GHz = 29.9792458 · ṽ`,  `ṽ = f_GHz / 29.9792458`
- `eV`  : `E_eV = 1.239841984e-4 · ṽ`,  `ṽ = E_eV · 8065.543937`
- `Cm1` : identity

`Nm`/`eV`/`GHz` are reciprocal in wavelength: guard against `ṽ <= 0` (and
`λ <= 0`) — return a sentinel `< 0` rather than dividing by zero.

Enum settings persistence uses the enum **key name** via
`BC::CSV::enumFromVariant<LaserUnit>` on read; store the key name string
on write (the established pattern — see `data/experiment/digitizerconfig`
and `LifConfig::LifScanOrder`).

---

## A. `LifConversion` value type (Task 1)

**Home:** `src/data/lif/lifconversion.h` / `.cpp`, added to
`cmake/BlackchirpData.cmake` (alongside `lifconfig.{h,cpp}`). Pure value
type: **no `HardwareObject`, no `SettingsStorage`, no hardware
dependency**. Must be unit-testable from hand-built node lists alone.

**All values are vacuum wavenumber (cm⁻¹).** Use `BC::LifConv::toCm1`/
`fromCm1` (§0) only at call sites that present to the user; the
`LifConversion` interface itself is cm⁻¹ end to end.

`Op` and `RefType` are `Q_ENUM`s persisted by name (no raw string
sentinels anywhere — the enum key names are the canonical tokens). They
are declared in **`lifunits.h`** alongside `LaserUnit`, because a single
`Q_NAMESPACE` can have only one moc-owning header; `lifconversion.h`
includes `lifunits.h`. Downstream tasks needing `Op`/`RefType` should
include `data/lif/lifunits.h` (or `lifconversion.h`, which re-exports it).
Logical shape:

```cpp
namespace BC::LifConv {
// (Q_NAMESPACE from §0)

enum class Op { NHG, SFG, DFG };        // N-th harmonic; sum-freq; diff-freq
Q_ENUM_NS(Op)
enum class RefType { Laser, Stage, Fixed };
Q_ENUM_NS(RefType)

// One ordered input to a node. inputs[0] is the PRIMARY beam.
struct InputRef {
    RefType type{RefType::Laser};
    QString stageKey;        // target stage's hwKey; valid iff type==Stage
    double  fixedCm1{0.0};   // fixed mixing beam (cm⁻¹); valid iff type==Fixed
};

// One crystal/stage node. The tunable LASER source is implicit (not a
// Node); Nodes are contributed by conversion stages.
struct Node {
    QString stageKey;        // owning stage's hwKey (unique within a graph)
    Op      op{Op::NHG};
    int     n{2};            // harmonic order for NHG (>=1); ignored otherwise
    std::vector<InputRef> inputs;  // NHG -> exactly 1; SFG/DFG -> exactly 2
    bool    isFinal{false};  // marks the beam that is the LIF output axis
};
}

class LifConversion
{
public:
    struct AssemblyResult {
        bool ok{false};
        QString errorString;   // populated iff !ok
        LifConversion conversion;
    };

    // Identity conversion: output == fundamental (the zero-stage case).
    LifConversion();

    // Assemble + validate (plan §7). On any failure returns {false,msg,{}}.
    // Validation: every InputRef of type Stage resolves to a Node in
    // `nodes`; input arity matches op (NHG=1, SFG/DFG=2); exactly one
    // isFinal across all nodes when nodes is non-empty (empty -> identity,
    // no FINAL required); no cycles; exactly one tunable source (reject any
    // topology implying a second tunable input — defer multi-tunable-source
    // per plan §1, with a comment marking the omission deliberate).
    static AssemblyResult assemble(const std::vector<BC::LifConv::Node> &nodes);

    // Output (FINAL beam) wavenumber for a grating fundamental (all cm⁻¹).
    double laserToOutput(double fundamentalCm1) const;

    // Analytic inverse: grating fundamental for a given output-beam
    // wavenumber. Exact (affine), no numerics.
    double outputToLaser(double outputCm1) const;

    // Local PRIMARY-input-beam wavenumber (inputs[0]) seen by the named
    // stage for a given grating fundamental — what that FCU calibrates its
    // phase-match motion against. Unknown stageKey -> return < 0.
    double stageInput(const QString &stageKey, double fundamentalCm1) const;

    // Output-axis bounds for the grating's native [laserMin,laserMax]
    // (cm⁻¹), returned sorted ascending (topology may reverse direction).
    std::pair<double,double> outputRange(double laserMinCm1, double laserMaxCm1) const;

    bool isIdentity() const;   // true when there are no stages
};
```

Op math (cm⁻¹, exact/affine): `NHG` → `ṽ_out = n · ṽ_in`; `SFG` →
`ṽ_out = ṽ_primary + ṽ_secondary`; `DFG` → `ṽ_out = ṽ_primary −
ṽ_secondary`. For SFG/DFG exactly one input is (transitively) the tunable
source; the other is `Fixed` or a fixed-derived beam.

Consumers: `assemble` is the **only** validating path; callers (Task 4
`HardwareManager` at prep, Task 5 GUI seeding) build the `Node` list from
settings snapshots via `LifFreqConversionStage::conversionNode()` (§C),
call `assemble`, and treat `!ok` as a prep-time error. `stageInput` keys
on the same value `getActiveKeys<LifFreqConversionStage>()` yields, which
equals `HardwareObject::d_key` (`hwType + hwIndexSep + label`).

---

## B. Node-descriptor settings schema (Task 3, on the base class)

`LifFreqConversionStage` (base) registers these via `REGISTER_HARDWARE_BASE`
(scalars) and `REGISTER_HARDWARE_BASE_ARRAY` / `_ENTRY` (the inputs
array). Keys in a new `namespace BC::Key::LifConvStage` in
`liffreqconversionstage.h`, frozen so Tasks 4/5/6 read the same keys:

```cpp
namespace BC::Key::LifConvStage {
inline constexpr QLatin1StringView op{"conversionOp"};        // BC::LifConv::Op key name
inline constexpr QLatin1StringView harmonic{"harmonicOrder"}; // int N, NHG only
inline constexpr QLatin1StringView isFinal{"finalBeam"};      // bool
inline constexpr QLatin1StringView verify{"verifyMove"};      // bool, default true
inline constexpr QLatin1StringView inputs{"conversionInputs"};// array
// inputs[] entry subkeys:
inline constexpr QLatin1StringView refType{"refType"};        // RefType key name
inline constexpr QLatin1StringView refKey{"refStageKey"};     // stage hwKey when Stage
inline constexpr QLatin1StringView refFixedCm1{"fixedCm1"};   // when Fixed
}
```

- `op` / `refType` are stored as their `Q_ENUM` key-name strings and read
  back with `BC::CSV::enumFromVariant` (§0/§A). They are **enum-valued
  hardware settings**, so they depend on the `HwSettingsWidget`
  enum-combobox rendering added in Task 2 (§E) for authoring.
- The base provides a helper that reads these settings into a
  `BC::LifConv::Node` (its `stageKey` = `d_key`), so Tasks 4/5 assemble a
  graph without re-parsing raw keys:

```cpp
// On LifFreqConversionStage (public):
BC::LifConv::Node conversionNode() const;
```

---

## C. `LifFreqConversionStage` base class interface (Task 3)

**Home:** `src/hardware/core/liflaser/liffreqconversionstage.h` / `.cpp`.
Direct child of `HardwareObject` (sibling of `LifLaser`), so it earns its
own `hwType`. Register in `cmake/BlackchirpHardware.cmake`: add the `.cpp`
to `HARDWARE_TYPES_SOURCES` and the `.h` to `HARDWARE_TYPE_HEADERS` (both
near the `liflaser.*` entries).

```cpp
class LifFreqConversionStage : public HardwareObject
{
    Q_OBJECT
public:
    LifFreqConversionStage(const QString& impl, const QString& label,
                           QObject *parent = nullptr);
    ~LifFreqConversionStage() override;

    BC::LifConv::Node conversionNode() const;   // §B

public slots:
    // Dispatch target (Task 4 calls this). localCm1 is the stage's PRIMARY
    // input-beam wavenumber (cm⁻¹), already computed by the caller from the
    // assembled topology (LifConversion::stageInput). Maps to a phase-match
    // motor position via the driver calibration and moves. Returns success.
    // When the verify flag is off, returns true best-effort (logs a warning
    // on mismatch instead of failing).
    bool setPosition(double localCm1);

    double readPosition();   // verify hook; achieved position (cm⁻¹) or <0

private:
    // Driver hooks, mirroring LifLaser::setPos/readPos shape (cm⁻¹).
    virtual void   setPos(double localCm1) = 0;
    virtual double readPos() = 0;

protected:
    // Base owns ONLY the generic contract (node descriptor + verify flag).
    // No structured calibration settings on the base — those belong to the
    // driver (plan §2).
};
```

Semantics:

- A stage emits **no** output-position update to the display; only
  `LifLaser::laserPosUpdate` drives the axis (plan §5).
- `d_threaded = true` in the constructor (like `LifLaser`).

**Virtual + Fixed implementations (also Task 3):**

- `VirtualLifFreqConversionStage` on `CommunicationProtocol::Virtual` —
  CI/testing vehicle (covered by the existing `liflaser/virtual*` cmake
  globs).
- `FixedLifFreqConversionStage` on `CommunicationProtocol::Virtual`,
  `FixedClock` motif (plan §7): a real, user-selectable device whose
  `setPosition` is a no-op returning `true`, serving as the `Fixed<Type>`
  system implementation for uncontrolled stages. A `fixed*` file under
  `liflaser/` is **not** matched by an existing glob — add
  `liflaser/fixed*.cpp` to `HARDWARE_IMPLEMENTATIONS_SOURCES` and
  `liflaser/fixed*.h` to `HARDWARE_IMPLEMENTATION_HEADERS` in
  `cmake/BlackchirpHardware.cmake`.

---

## D. `HardwareManager` fan-out contract (Task 4)

- Add `bool setLifConversionStages(double outputCm1)` mirroring
  `setPGenLifDelay`'s active-keys pattern but **non-blocking parallel**
  dispatch + AND-join (plan §5): resolve
  `getActiveKeys<LifFreqConversionStage>()` (empty → return `true`), and
  for each active stage compute
  `w = conversion.stageInput(stageKey, fundamental)` where
  `fundamental = conversion.outputToLaser(outputCm1)`, post
  `stage->setPosition(w)` to the stage thread non-blocking, collect
  futures/promises, then AND all results.
- Hold a **cached `LifConversion`** assembled at experiment prep from the
  active laser + active stage `conversionNode()`s; treat `assemble`
  failure as a prep error that aborts the experiment before acquisition.
- `setLifParameters` join order: `setLifLaserPos(pos)` → (if ok)
  `setLifConversionStages(pos)` → (if ok) `setPGenLifDelay(delay)`. Do
  **not** bake a two-tier (laser, stages) assumption into callers —
  priority-tiered dispatch is a future extension (plan §5).
- Add a `qobject_cast<LifFreqConversionStage*>(obj)` branch to the connect
  ladder (`hardwaremanager.cpp` ~line 1049) for lifecycle/failure-signal
  parity. The stage emits no competing position signal.

---

## E. `LifLaser` output-unit semantics + enum settings UI (Task 2)

- `setPosition(outputCm1)` / `readPosition()` operate in the **output-beam
  wavenumber** (cm⁻¹). `setPosition` converts output→fundamental via the
  active `LifConversion`, range-checks the **fundamental** against
  `minPos`/`maxPos` (now cm⁻¹, still the grating's native range), tunes,
  then returns `readPosition()` which reads the grating and converts back
  to the output beam.
- `laserPosUpdate(double)` emits **output-beam cm⁻¹** values.
- Migrate `BC::Key::LifLaser::units` from a `QString` setting to a
  `LaserUnit` (§0) enum setting (default display unit `Nm` — Sirah users
  think in nm — while the internal value is cm⁻¹). Range-check error
  messages and any base-level display format via `fromCm1`/`unitLabel`.
- Migrate `minPos`/`maxPos` defaults to cm⁻¹ on the base
  (`REGISTER_HARDWARE_BASE`). Per-driver overrides move with their driver
  (Sirah in Task 6).
- **Teach `HwSettingsWidget` to render enum-valued settings as a
  combobox.** `makeScalarWidget` (`gui/widget/hwsettingswidget.cpp:241`)
  dispatches on `QVariant` type and today falls through to a free-text
  `QLineEdit` for enums. Add a branch that detects a `Q_ENUM`/`Q_ENUM_NS`
  default value (via `QMetaType(typeId).flags() & QMetaType::IsEnumeration`
  and `QMetaType::metaObject()`/`QMetaEnum`) and builds a combobox of the
  enum keys, with matching read-back in `readWidget`/`values`. This is the
  shared enabler for the `units` setting here and the `op`/`refType`
  settings in Task 3 (§B).
- Conversion seam (frozen so Tasks 2 and 4 agree): `LifLaser` gains
  `public slots: void setConversion(const LifConversion &c);` (defaults to
  identity in the constructor). Task 4 calls it at prep on the active laser
  with the assembled conversion, dispatched the same thread-aware way it
  dispatches `setPosition` (direct call when on the laser thread, else
  `QMetaObject::invokeMethod(..., Qt::BlockingQueuedConnection)`).
- Because Task 2 lands before Task 4's prep-time cache, the day-1 laser
  with no stages is the **identity** conversion (output == fundamental);
  ship the identity path and let Task 4 supply the assembled conversion.
  Do not have `LifLaser` read other devices' hardware settings directly.

---

## F. `LifConfig` / axis migration (Task 5)

**The scan axis is built and stored in the display `LaserUnit`, not cm⁻¹**
— a deliberate revision of the earlier "cm⁻¹ internal" framing for this one
seam. Rationale: some lasers actuate only in a native unit (e.g. the
Opolette, minimum step 0.01 nm); a uniform-cm⁻¹ grid rounds to an *uneven*
step sequence in that native unit at the hardware resolution limit —
strictly worse than a uniform grid in the display unit. Building the scan
in the display unit lets the user pick the unit natural to their laser, and
keeps the whole axis pipeline in display units with cm⁻¹ crossing only the
single hardware-dispatch boundary. (This supersedes the "§0 internal
representation is cm⁻¹" rule *for the `LifConfig` scan-axis scalars only*;
`LifConversion`, laser/stage setpoints, and `HardwareManager` dispatch are
unchanged and remain cm⁻¹.)

- `LifConfig::d_laserUnits` migrates from `QString` to `BC::LifConv::LaserUnit`
  (with `setLaserUnits`/`laserUnits` accessors updated). Axis scalars
  (`d_laserPosStart`/`Step`, and the derived range) are in that display
  `LaserUnit`; the grid is uniform in the display unit.
- `currentLaserPos()` converts the display-unit setpoint to **output-beam
  cm⁻¹** at the dispatch boundary: `toCm1(start + i*step, laserUnits)`. This
  is the one place the axis leaves display units; the returned value flows to
  `HardwareManager::setLifParameters` as the output cm⁻¹ setpoint (§D/§E),
  and the laser converts output→fundamental internally.
- Config-page range **bounds** seed from the **output** view: assemble a
  `LifConversion` from active laser + stage settings snapshots (GUI-thread,
  snapshot-only) and use `outputRange(minPos,maxPos)` (cm⁻¹) for the box
  limits, converting both endpoints via `fromCm1` and sorting (a reciprocal
  unit reverses direction) to get the display-unit range. Box **values** and
  suffix are in the display `LaserUnit`; on accept, the box values are stored
  directly as the display-unit axis scalars (no conversion).
- `header.csv` `LaserStart`/`LaserStep` are written **directly** as the
  display-unit value + `unitLabel` (the internal scalars are already display
  units — no conversion), so the analysis x-axis is a physical excitation
  unit. Read parses the unit cell back to `LaserUnit`; decimals inference is
  unchanged.
- LIF status boxes / widgets that read the old string `units` as a suffix
  (`gui/lif/gui/liflaserstatusbox.cpp:45`, `liflaserwidget.cpp:25`,
  `experimenttypepage.cpp:252`) update to the enum + `unitLabel`, and convert
  the cm⁻¹ position delivered by `LifLaser::laserPosUpdate` to the display
  unit via `fromCm1`.
