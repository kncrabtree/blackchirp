# Sirah Cobra refresh — LIF frequency conversion & multi-port drivers

Full implementation plan for the Sirah Cobra integration refresh. The
[roadmap entry](devel-roadmap.md#sirah-cobra-integration-refresh) points
here. This plan **supersedes the earlier "Approach A" direction**
(single `HardwareObject` owning multiple `CommunicationProtocol`
objects, formalized into reusable multi-port infrastructure). The
approach below is cleaner: it needs no change to the `HardwareObject`
comm model or the comm-config dialog, and it solves a problem Approach A
did not address at all — the LIF spectrum axis for a frequency-converted
laser.

## Implementation status (resume here)

Work is proceeding on branch **`feature/sirah-cobra-refresh`** as a
sequence of tasks (see [Sequencing](#sequencing)), each dispatched to a
single Sonnet subagent, reviewed, then committed. The frozen cross-task
interface — exact signatures, key names, enum names — lives in
[`sirah-cobra-refresh-contract.md`](sirah-cobra-refresh-contract.md) and
is authoritative; read it before resuming.

**Done (committed):**

- **Task 1 — keystone** (`fafa44de`): `BC::LifConv::LaserUnit`/`Op`/
  `RefType` `Q_ENUM`s + cm⁻¹↔nm/GHz/eV utility (`data/lif/lifunits.*`) and
  the `LifConversion` value type (`data/lif/lifconversion.*`), with
  `tst_lifconversion`. Contract §0/§A.
- **Task 2 — `LifLaser` cm⁻¹ + enum UI** (`5dd2ce57`): `LifLaser`
  driver-hooks work in the grating fundamental (cm⁻¹); public
  `setPosition`/`readPosition` work in output-beam cm⁻¹ and bridge via a
  held `LifConversion` (identity until `setConversion`); `units` migrated
  to the `LaserUnit` enum; `minPos`/`maxPos` defaults now cm⁻¹
  (5000–40000); `HwSettingsWidget` renders enum settings as comboboxes;
  Opolette + virtual laser updated to the cm⁻¹ boundary. Contract §E.
- **Enum-reflection consolidation** (`a7e796ae`): shared
  `BC::CSV::metaEnumFromType`/`enumKeyName`; `EnumComboBoxBase` reused by
  both `EnumComboBox<T>` and `HwSettingsWidget`; construction-path test
  that a built `LifLaser` seeds `units` as the `"Nm"` key string.
- **Task 3 — `LifFreqConversionStage` base + Virtual/Fixed impls**:
  new base `LifFreqConversionStage : HardwareObject`
  (`hardware/core/liflaser/liffreqconversionstage.*`), a `d_threaded`
  sibling of `LifLaser` earning its own `hwType`. Owns only the generic
  node-descriptor contract — `BC::Key::LifConvStage` scalars
  (`op`/`harmonic`/`isFinal`/`verify`/`tolerance`) + the `conversionInputs`
  array via `REGISTER_HARDWARE_BASE`/`_BASE_ARRAY`, `conversionNode()`
  reading them into a `BC::LifConv::Node` (`stageKey = d_key`), and
  `setPosition`/`readPosition` with a verify/best-effort split. The
  move-verification window is the registered `tolerance` setting (default
  1.0 cm⁻¹, min 0), not a hardcoded constant, so a coarse mount can widen
  it. `VirtualLifFreqConversionStage` (CI vehicle) and
  `FixedLifFreqConversionStage` (`FixedClock` motif; the `Fixed<Type>`
  system-profile device for uncontrolled stages) added; both registered in
  `cmake/BlackchirpHardware.cmake` (base type + `fixed*` globs).
  `tst_hardwareregistrytest` gains a construction-path test and both impls
  in the whole-archive guard table. Contract §B/§C. Neither new impl calls
  `save()` in its constructor — the `HardwareObject` base ctor already
  persists the seeded defaults and the impls mutate no settings;
  deliberately dropped rather than mirroring the vestigial
  `VirtualLifLaser`/`FixedClock` calls.
- **Task 4 — `HardwareManager` conversion-stage fan-out**: `HardwareManager`
  holds a cached `d_lifConversion` assembled at `initializeExperiment`
  prep from the active laser + every active `LifFreqConversionStage`'s
  `conversionNode()`; `LifConversion::assemble` failure is surfaced as a
  prep-time error that aborts before `experimentInitialized` (empty stage
  set → identity, not special-cased), and the assembled conversion is
  pushed to the laser via `setConversion` (thread-aware). New
  `setLifConversionStages(outputCm1)` dispatches each stage's local setpoint
  (`stageInput(key, outputToLaser(outputCm1))`) **non-blocking** via a
  per-stage `std::promise`/`future` + `Qt::QueuedConnection` so moves run
  concurrently, then AND-joins; empty stage set returns `true`. Spliced into
  `setLifParameters` between the laser and pulse-generator legs (flat join,
  no two-tier assumption). Connect-ladder gains a documented no-op
  `LifFreqConversionStage` branch — the stage forwards no type-specific
  signal and `hardwareFailure` is already wired generically in
  `handleConnectionResult`. Contract §D. **Integration test deferred**: no
  existing harness constructs a `HardwareManager`, and a real
  virtual-laser + virtual-stage `setLifParameters` test needs a friend-class
  seam to author stage node-descriptor settings plus a multi-device async
  harness (pulse generator, `Experiment`/`LifConfig`, event-loop pumping) —
  a deliberate follow-up, not built here.

- **Task 5 — LIF axis migration to display units**: `LifConfig::d_laserUnits`
  is now a `BC::LifConv::LaserUnit` (accessors updated); the scan scalars
  `d_laserPosStart/Step` are stored in the **display unit** and the grid is
  uniform in that unit (contract §F decision — a uniform-cm⁻¹ grid rounds to
  uneven steps on a native-unit-limited laser like the Opolette).
  `currentLaserPos()` is the sole display→output-cm⁻¹ boundary
  (`toCm1(start + i*step, unit)`); `header.csv` stores display-unit values +
  `unitLabel` directly; `retrieveValues` parses the unit cell back by
  `unitLabel` comparison (legacy `"nm"` → `Nm`). Config page and the live
  laser widget seed their box ranges from `outputRange(minPos,maxPos)`
  converted through `fromCm1` (reciprocal-sorted) and emit/store in the
  display unit; the status box stores raw cm⁻¹ and renders via `fromCm1`.
  Two shared GUI-thread helpers on `LifFreqConversionStage`:
  `nodeFromSettings(SettingsStorage&, key)` (also now backs `conversionNode()`)
  and free `assembleActiveLifConversion()` (identity fallback on mid-edit
  topology; `HardwareManager` prep-time `assemble` stays the authoritative
  validator). Minor known quirk: before the first `laserPosUpdate`, the
  status box renders the `fromCm1(0, Nm)` sentinel rather than a placeholder.
- **Task 6 — `SirahFcu` driver + `SirahCobra` cleanup**: the decommissioned
  Harvey-Mudd external-doubling-stage rig (second serial port, `"%1ma%2"`
  ASCII commands, `"in"` query, crystal/compensator polynomials, all
  `extStage*`/`poly*` keys, `p_extStagePort`) was **deleted** from
  `SirahCobra`, not carried forward — the plan §3's compensator-preservation
  direction was wrong. `SirahCobra` is now a pure grating driver: its `stages`
  geometry array + scalars migrated from the imperative constructor to
  `REGISTER_HARDWARE_ARRAY`/`_SETTINGS`, and `minPos`/`maxPos` re-expressed in
  cm⁻¹ (14285.7/22222.2 = 700/450 nm) with `setPos`/`readPos` converting
  nm↔cm⁻¹ at the sine-bar boundary. New `SirahFcu : LifFreqConversionStage`
  models the doubler as an `NHG` node (harmonic `N` Required, default 2;
  `isFinal` default true) on its own RS232 port, reusing the grating's sine-bar
  tuning + binary protocol. The pure wire-format layer (`BC::Sirah::Status`,
  `buildCommand`, `parseStatus`) is shared via `sirahprotocol.{h,cpp}`; the
  comm-driving loops and tuning math are deliberately duplicated per driver
  (flagged for consolidation once the two units are bench-confirmed identical).
  Placeholder FCU crystal geometry copied from the grating pending real
  calibration.

**Sequence complete.** All six tasks are committed on
`feature/sirah-cobra-refresh`. What remains is **manual bench validation**
(no virtual Sirah / hardware in CI): drive a virtual laser first, then a test
deployment against the live Opolette, then grating + doubling-stage co-tuning
across the OH band on the new Sirah, verifying the LIF axis reads in the
doubled excitation wavelength. See [Testing](#testing).

**Working notes for future work on this area:**

- `Op`/`RefType`/`LaserUnit` are declared in `data/lif/lifunits.h` (one
  `Q_NAMESPACE` = one moc owner); include that (or `lifconversion.h`, which
  re-exports it) to use them. Enum-valued hardware settings persist as their
  `Q_ENUM` key-name string and read back via `BC::CSV::enumFromVariant`;
  `HwSettingsWidget` renders them as comboboxes.
- `minPos`/`maxPos` render as raw cm⁻¹ spin boxes in the profile-creation
  dialog (not display-unit-aware) — an unaddressed UX rough edge.
- The `setLifParameters` fan-out blocks the `HardwareManager` thread on the
  parallel-move join (consistent with the pre-existing laser/pgen dispatch);
  revisit if/when the roadmap's manager-wide async delivery lands.
- No automated `HardwareManager` integration test exists for the LIF fan-out;
  a harness would need a friend-class seam to author stage node-descriptor
  settings plus a multi-device async setup.

## Problems being solved

1. **Second serial port lives outside the comm system.** `SirahCobra`
   drives its external doubling stage through a hand-instantiated
   `Rs232Instrument *p_extStagePort` (`sirahcobra.h:103`), built in
   `initialize()` and opened via `Rs232Instrument::testManual(...)` with
   hard-coded 8N1/no-flow-control. It bypasses `buildCommunication()`,
   is invisible to the comm-config dialog, and carries the standing TODO
   at `sirahcobra.cpp` (external-stage comm settings — baud and read
   terminator — need a real home, distinct from the laser comm port).

2. **FCU settings never migrated to the registry.** The external-stage
   scalars (`hasExtStage`, `extStagePort`, `extStageBaud`, crystal/
   compensator addresses, `theta0`/`slope`) and the `stages`,
   `extStageCrystalPoly`, `extStageCompPoly` arrays are created
   imperatively in the `SirahCobra` constructor with `setDefault` /
   `appendArrayMap` / `setArray` + `save()` — the exact anti-pattern
   `src/AGENTS.md` forbids. They persist but carry no registry metadata
   and never surface in the profile-creation UI.

3. **The `stages` array conflates three distinct concerns.** It was
   designed to hold grating geometry in slot 0 and optional Frequency
   Conversion Unit (FCU) stages in slots 1..N. But each FCU needs its
   own serial port, which the laser object cannot own — a real design
   fork. The reference (Harvey-Mudd) instrument had zero FCU slots, so
   this was never exercised.

4. **No output-wavelength axis for frequency-converted lasers.** An FCU
   can double, triple, or perform SFG/DFG. When scanning (e.g.) the OH
   band at 282 nm with a frequency-doubled dye laser, the user must
   currently reason in the dye fundamental (~564 nm) and the LIF x-axis
   is labelled in the fundamental, not the light that actually excites
   the sample. Every other Blackchirp subsystem that involves a
   multiplier (e.g. a multiplied `Clock`) lets the user enter the final
   value and does the conversion internally; LIF should match.

## Chosen architecture

Keep the existing **1 `HardwareObject` : 1 `CommunicationProtocol`**
model. Model each externally-controlled FCU as its own first-class
hardware device, and add a small, hardware-independent *conversion
topology* to the `LifLaser` base so the acquisition axis can be expressed
in the final (output) wavelength/frequency.

Two independent concerns that meet only through a cached, per-point
setpoint computation:

- **Conversion topology** (optical/math: "this stage doubles", "this
  stage sums with a fixed 1064 nm beam") — small, generic, lives on
  `LifLaser`. Exists *whether or not* any FCU is under Blackchirp
  control, because it defines the axis.
- **FCU hardware control** (crystal/compensator motion, phase-match
  calibration keyed on the stage's *local* input wavelength) — lives in a
  new `LifFreqConversionStage` driver family, each stage its own
  `HardwareObject` with its own comm port and its own registry-migrated
  settings, and each carrying the node descriptor that places it in the
  topology.

The contract between them: **the acquisition axis is the output
wavelength; a cached `LifConversion` (a pure value type assembled from
the active laser + stage node descriptors) turns each output setpoint
into the grating fundamental *and* each stage's local input wavelength,
and
`HardwareManager` dispatches those computed setpoints in parallel and
joins the success flags.** For every realistic single-tunable-source
setup the whole chain is affine in frequency, so all beam wavelengths —
and therefore every stage's target — are a closed-form function of the
one scan variable *f*; no per-device topology and no numerics are needed.
The join reuses the `getActiveKeys<T>()` + AND-join pattern already used
for multi-instance pulse generators in
`HardwareManager::setPGenLifDelay` (`hardwaremanager.cpp:646-667`),
extended to non-blocking parallel dispatch (§5).

Calibrating each FCU in its *local* input wavelength (rather than
broadcasting the single scalar *f* to every stage) is a modularity
choice, not a correctness requirement for single-source setups: it makes
each crystal's angle-vs-wavelength polynomial intrinsic to the crystal
and reusable if the topology is rewired, and it is the seam that makes
genuinely multi-tunable-source conversion (§1) possible later. The
cheaper fallback — broadcast *f*, calibrate FCUs in *f* — is correct for
the near-term doubler but forgoes both properties.

### Rejected alternatives

- **Approach A — generalize `HardwareObject` to N comm protocols +
  extend the comm-config dialog.** The correct model *only* for a device
  whose ports are genuinely inseparable (one logical device, channels
  that cannot be independently activated). The Sirah doubling stage does
  not meet that bar: it is a physically separate unit, optional, and
  independently connectable. Approach A is also the heaviest change —
  `d_commType`/`p_comm` become vectors (`hardwareobject.h:117,400`),
  `buildCommunication` (`hardwareobject.cpp:219-274`) and the whole
  single-protocol comm-config dialog change — for a benefit the approach
  here delivers without touching the base class. Keep A in reserve for a
  future device that truly is one box with two inseparable ports.

- **Approach B — broadcast every action to all `HardwareObject`s with
  virtual per-type filtering (message bus).** Justified only by *several*
  subset-broadcast commands; we have one (set wavelength). It also does
  not model success-joining (today `lifSettingsComplete` is a single
  bool), so it would require building both a bus and a result-aggregation
  layer. The `beginAcquisition`/`endAcquisition` fan-out
  (`hardwaremanager.cpp:985-986`) is real precedent, but it is
  fire-and-forget with no joined return — exactly the hard part B would
  have to invent.

## Design detail

> **Interface contract & unit standardization (supersedes the framing
> below on units).** The frozen cross-task interface is in
> [`sirah-cobra-refresh-contract.md`](sirah-cobra-refresh-contract.md);
> consult it for exact signatures, key names, and enum names. Two
> decisions made during sequencing refine this plan:
>
> - **Internal representation is vacuum wavenumber (cm⁻¹)** everywhere in
>   the LIF laser/axis pipeline (laser positions, `minPos`/`maxPos`,
>   stage setpoints, `LifConversion`, `LifConfig` axis, `HardwareManager`
>   dispatch). cm⁻¹ is proportional to frequency, so the affine/exact
>   inversion property below holds unchanged. There are exactly two unit
>   boundaries — the driver (native hardware unit ↔ cm⁻¹) and the display
>   (cm⁻¹ → the user-selected unit). Where the text below says "frequency"
>   or "wavelength", read "vacuum wavenumber (cm⁻¹)"; wavelength/eV/GHz are
>   display conversions via a `toCm1`/`fromCm1` utility.
> - **`LaserUnit` is a `Q_ENUM`** (`{ Cm1, Nm, GHz, eV }`), not the old
>   free-text `units` string, so the display unit is type-safe and renders
>   as a combobox. The conversion-topology `Op` (`NHG`/`SFG`/`DFG`) and
>   input `RefType` (`Laser`/`Stage`/`Fixed`) are likewise `Q_ENUM`s
>   persisted by name — no raw string sentinels. Migrating `units` to an
>   enum hardware setting also requires teaching `HwSettingsWidget` to
>   render enum-valued settings as comboboxes.

### 1. Conversion topology — `LifConversion` value type

The conversion chain is a small DAG of beams and operations from the dye
fundamental (the one tunable "scan" source, always the active `LifLaser`)
to the final output beam, plus any fixed mixing sources. The DAG is **not**
stored as a laser-owned array; it is **assembled at experiment prep from
the node descriptors carried by the active loadout** — the `LifLaser`
contributes the tunable source node, and each active
`LifFreqConversionStage` contributes one crystal node (§2, §7). This
keeps each node authored on the device it describes and removes the
laser↔stage binding a central array would need.

Node operations:

- operation ∈ { `NHG` (N-th harmonic, integer `N` ≥ 1; `N`=1 is the
  identity), `SFG`, `DFG` } (4WM is a future op);
- `NHG` takes one input; `SFG`/`DFG` take two — one may be a fixed
  mixing source (a constant frequency), the other another beam node,
  which is how a physical tripler is expressed (an `NHG` `N`=2 doubler
  feeding an `SFG` node whose second input references the fundamental).

Nodes are at **physical-crystal granularity** — one node per crystal — so
every controllable crystal is its own device/node. A loadout with no
conversion stages is a bare `LASER` source whose output *is* the
fundamental (today's behaviour).

All math is done **in frequency**: `NHG` scales (`f → N·f`), `SFG`/`DFG`
shift by a fixed mix (`f ± f_mix`). For a single tunable source every
beam wavelength is therefore **affine in *f***, so the whole topology is
affine and the output↔*f* inversion is **closed-form and exact — no
numerics**. Values are presented in the laser's configured
`units`/`decimals` for display only (wavelength input is converted to
frequency at the boundary via `c/λ`).

`LifConversion` is a small free-standing value type assembled from the
node descriptors (a `LifLaser` snapshot plus zero or more stage
snapshots), providing:

- `double laserToOutput(double fundamental) const;`
- `double outputToLaser(double output) const;`  (analytic inverse)
- `double stageInput(const QString &stageKey, double fundamental) const;`
  — the local input wavelength at a given stage node, keyed by the
  stage's hwKey, which is what that FCU calibrates against (§2)
- `std::pair<double,double> outputRange(double laserMin, double laserMax) const;`
  (topology applied to the native bounds; may invert direction —
  doubling maps a max wavelength to a min wavelength)

Because it is assembled from **settings snapshots**, both the GUI/config
layer (§6) and `HardwareManager` (§5) build and use it without touching
any threaded hardware object. Assembly performs the topology validation
in §7 and reports failures as prep-time errors.

**Semantics change:** `LifLaser::setPosition`/`readPosition` now work in
**output** units. `setPosition(output)` converts to the fundamental,
range-checks the fundamental against `minPos`/`maxPos`
(`liflaser.cpp:42-50`, unchanged bounds — still the grating's native
range), tunes the grating, then returns `readPosition()` which reads the
grating and converts back to output. `laserPosUpdate(double)` therefore
emits output values, which is what the display wants. Because we are
dropping backward compatibility (the Harvey-Mudd instrument is retired),
the semantics flip is acceptable; no on-disk migration is required.

**Scope boundary — single tunable source.** The day-1 solver supports
exactly one tunable scan source (the grating fundamental) plus fixed
mixing sources; `SFG`/`DFG` mix only against a **fixed** frequency. The
topology type can *represent* a second tunable source, but its inversion
is then underdetermined without a coordination constraint, and it also
requires a multi-axis acquisition model in `LifConfig`. Reject that
combination in the solver with a clear error and defer it as a unit —
this is the real complexity boundary, and it is not a `LifConversion`-only
change. Note the deferral in a comment so the omission reads as
deliberate.

### 2. New `LifFreqConversionStage` hardware type

A new base `LifFreqConversionStage : HardwareObject`, a **direct child
of `HardwareObject`** (sibling of `LifLaser`) so it earns its own
`hwType` (per `findHardwareBaseType`, hwType is the direct child of
`HardwareObject`). Lives in `src/hardware/core/liflaser/`.

Set `d_threaded = true` in the constructor (like `LifLaser`), so stages
always run on their own threads — the normal path for the parallel
dispatch in §5.

Base responsibilities (thin):

- Public slot `bool setPosition(double localWavelength)` — the dispatch
  target. Maps this stage's local input wavelength → phase-match motor
  position via the driver's calibration and moves. Returns success. The
  caller (`HardwareManager`, §5) computes `localWavelength` from the
  assembled topology; the stage itself needs no topology.
- A registered **node descriptor** — op type (`NHG(N)`/`SFG`/`DFG`),
  ordered input refs, and a `FINAL` marker — that places this device in
  the DAG (§1, §7). This *is* the stage's identity in the graph; there is
  no separate central binding. Validated at experiment prep.
- Public slot `double readPosition()` / verify hook — confirms the move.
  A stage reports success/failure only; it does **not** emit a competing
  output-wavelength update (the display axis comes solely from
  `LifLaser::laserPosUpdate`, §5).
- A per-device **verify flag** (registered setting, default *verify*).
  When verify is off, `setPosition` returns `true` best-effort (logs a
  warning on mismatch instead of failing). This lets a non-critical
  compensator tolerate an unverified move while the grating still aborts
  the step. (Distinct from `HardwareObject::d_critical`, which governs
  *connection* criticality, not per-move verification.)
- Pure-virtual driver hooks for the actual calibration + motion,
  mirroring `LifLaser`'s `setPos`/`readPos` shape.

The base declares **no** structured calibration settings — those differ
by mount type (sine bar, piezo rotation stage, dual-channel doubler/
compensator) and belong entirely to the driver (per §3 and roadmap
point 1). Because each FCU calibrates against its **local input
wavelength** (§1), that polynomial is intrinsic to the crystal and
independent of what sits upstream in the topology. The base owns only
the generic contract: local setpoint in, success out, node descriptor,
verify flag.

Because each FCU is a normal `HardwareObject`, it gets its own comm
protocol through the standard `buildCommunication()` path and its own
comm-config dialog entry **for free** — no dialog changes.

### 3. Sirah external doubling-stage driver

New concrete driver (e.g. `SirahDoublingStage : LifFreqConversionStage`)
that absorbs the FCU logic currently buried in `SirahCobra`:

- The crystal/compensator polynomial evaluation and `"%1ma%2"` move
  commands (`sirahcobra.cpp:193-240`), the poly-coefficient loading
  (`sirahcobra.cpp:272-319`), and the `"in"` steps-per-degree query.
- `REGISTER_HARDWARE_PROTOCOLS(..., Rs232)` +
  `REGISTER_COMM_DEFAULTS` carrying the doubling stage's **own** baud and
  read terminator — this is where the standing `sirahcobra.cpp` TODO is
  finally resolved: the external port's comm settings live in its own
  device settings group, configured through the normal dialog.
- `REGISTER_HARDWARE_SETTINGS` / `REGISTER_HARDWARE_ARRAY` for the
  crystal/compensator addresses and the position-vs-local-wavelength
  polynomials — migrated, not constructor-set. (The compensator move is
  currently commented out at `sirahcobra.cpp:219-220`; carry that state
  forward explicitly rather than silently.) Its node descriptor (§2) is
  a single `NHG` `N`=2 node with input `LASER` and output `FINAL`, so its
  local input wavelength equals the dye fundamental.

### 4. `SirahCobra` cleanup

- Delete `p_extStagePort`, the `initialize()` block that builds it
  (`sirahcobra.cpp:91-99`), the `testManual` path
  (`sirahcobra.cpp:111-133`), and all `extStage*` keys
  (`sirahcobra.h:23-33`).
- Migrate the surviving grating settings — the `stages` array
  (`sirahcobra.h:10-22`) and the four scalars (`minPos`/`maxPos`/
  `decimals`/`hasFl` are already registered) — from the constructor's
  imperative block to `REGISTER_HARDWARE_ARRAY` /
  `REGISTER_HARDWARE_ARRAY_ENTRY`, satisfying the `src/AGENTS.md` rule
  the constructor currently violates.
- `stages` reduces to grating geometry only (its FCU-slot ambition moves
  to §2/§3), so `posToWavelength`/`wavelengthToPos` keep their existing
  `stage=0` call sites without the dormant multi-stage fork.

### 5. `HardwareManager` fan-out

`setLifParameters` (`hardwaremanager.cpp:608-644`) receives the output
setpoint (`LifConfig::currentLaserPos()`, now in output units). It holds
a **cached `LifConversion`** assembled at experiment prep from the active
laser + active stage settings snapshots (the topology is static during an
experiment, and the value type is thread-safe to copy — no call into the
threaded devices). Per LIF point it computes, from that one output value:
the grating fundamental (`outputToLaser`) and each active stage's local
input wavelength (`stageInput(stageKey, fundamental)`), then dispatches
those computed setpoints to the laser and every active stage **in
parallel** and joins:

```cpp
bool success = true;
success &= setLifLaserPos(pos);             // laser: setPosition(output), converts internally
if(success)
    success &= setLifConversionStages(pos); // stages: parallel dispatch of computed local setpoints
if(success)
    success &= setPGenLifDelay(delay);      // existing
```

`setLifConversionStages(pos)` resolves
`getActiveKeys<LifFreqConversionStage>()` (empty → returns `true`, the
zero-FCU case), and for each active stage computes
`w = conversion.stageInput(stageKey, fundamental)`, then posts
`stage->setPosition(w)` **non-blocking** to the stage's own thread
(each stage is `d_threaded`), collecting an `std::future<bool>` (or a
per-stage promise fulfilled inside a `QueuedConnection` lambda). After
all are launched it joins by AND-ing every result. The
`HardwareManager` thread blocks on the join but is not any stage's
thread, so the moves run concurrently with no deadlock. Best-effort
stages fold in naturally because their `setPosition` returns `true`
(§2). Topology assembly and its validation (§7) happen at prep, so a
malformed graph fails before acquisition starts. Add the matching
`qobject_cast<LifFreqConversionStage*>` branch to the connect ladder
(`hardwaremanager.cpp:1049-1052`) — the FCU emits no competing position
update; only `LifLaser::laserPosUpdate` drives the display axis.

**Ordering — parallel from day 1.** The laser and all stages are
dispatched concurrently and joined; no ordering is imposed. Serial or
**tiered** dispatch is the future extension: a per-device
**priority-integer** setting (higher priority first; equal priorities
dispatched together and joined — roadmap point 4) turns the single join
into a sorted sequence of parallel-join phases. The day-1 parallel path
is the degenerate single-tier case, so do not bake a two-element (laser,
stages) assumption into callers that the tiered scheduler would have to
unwind.

### 6. Acquisition / config axis migration to output units

`LifConfig` already stores `d_laserUnits`/`d_laserDecimals` and a
`(start, step, points)` axis (`lifconfig.h:90-92,212-213`) seeded from
the laser settings at `experiment.cpp:518-519`. Shift these to the
**output view, expressed in the display `LaserUnit`** (see the frozen
[contract §F](sirah-cobra-refresh-contract.md); the exact seams live
there):

- **The scan grid is built and stored in the display unit, not cm⁻¹.**
  This revises the "cm⁻¹ internal" framing for the `LifConfig` scan
  scalars only. A uniform-cm⁻¹ grid rounds to an *uneven* step sequence
  on a laser that actuates in a native unit at a fixed resolution (the
  Opolette: 0.01 nm minimum step) — strictly worse than a uniform grid in
  the display unit. So the axis stays in the user's unit end to end, and
  cm⁻¹ crosses only the single hardware-dispatch boundary.
- Seed the config-page range bounds (`experimenttypepage.cpp:564`,
  `p_lStartBox` et al.) from the **output** view: assemble a
  `LifConversion` from the active laser + stage settings snapshots and use
  `outputRange(minPos, maxPos)` (cm⁻¹) for the box limits, converted to the
  display unit via `fromCm1` (sorting, since a reciprocal unit reverses
  direction). Box values/suffix are the display unit. GUI-thread
  computation over settings snapshots — no call into a threaded device.
- `currentLaserPos()` (`lifconfig.cpp:55-57`) converts the display-unit
  setpoint to output cm⁻¹ at the dispatch boundary
  (`toCm1(start + i*step, laserUnits)`) and flows to
  `HardwareManager::setLifParameters` via `AcquisitionManager::nextLifPoint`
  (`acquisitionmanager.cpp:217`,`76`). The output→fundamental conversion
  then happens inside the laser.
- On-disk `header.csv` `LaserStart`/`LaserStep`
  (`lifconfig.cpp:121-122,150`) record the display-unit values with the
  `unitLabel` (no conversion — the scalars are already display units),
  correct for the Python/analysis side whose x-axis is the excitation
  unit.

### 7. Topology configuration & UX

The DAG is authored **per device**, in the existing hardware-settings
panels — no new configuration UI for day 1. Each `LifFreqConversionStage`
(§2) carries its node descriptor as ordinary registered settings:

- **op type** — `NHG` (+ integer `N`), `SFG`, or `DFG`;
- **input refs** — ordered, one per input the op needs (`NHG` → 1,
  `SFG`/`DFG` → 2). Each ref is one of: `LASER` (the single tunable
  source, the active `LifLaser` by construction), another stage's
  `hwType.label`, or `FIXED` paired with a companion mixing-frequency
  value;
- **`FINAL` marker** — a boolean; exactly one node in the graph sets it,
  identifying the beam that is the LIF axis.

Edges are authored **only at the inputs** — the input refs fully
determine the DAG, so a node does not also name its consumer. (An
explicit symmetric child ref would double-author each edge and invite
inconsistency; if a cross-check is ever wanted, derive it rather than
require it.)

**Uncontrolled / manual stages** (a doubler Blackchirp does not drive)
are represented by a purpose-built logical driver following the
**`FixedClock` motif** — `FixedClock` (`clock/fixedclock.{h,cpp}`) is a
real, user-selectable `Clock` on `CommunicationProtocol::Virtual` with a
trivially-passing `testClockConnection()` and no-op init, representing
fixed/logical frequency inputs to the clock topology. Mirror it with a
`FixedLifFreqConversionStage : LifFreqConversionStage` on the `Virtual`
protocol: it carries a node descriptor and participates in the axis math
like any stage, but its `setPosition` is a no-op that returns success.
Because it is a legitimate first-class device (not a stand-in for absent
hardware), it does not read as a virtual placeholder and can serve as the
type's `Fixed<Type>` system implementation that
`HardwareProfileManager::ensureSystemProfiles`
(`hardwareprofilemanager.cpp:928-943`) already looks for by the
`Virtual<Type>`/`Fixed<Type>` naming convention. This preserves the rule
that the topology — and thus the axis — exists whether or not a stage is
under Blackchirp control, with no new concept and no second authoring
location. The rejected alternative (a math-only node array kept on the
laser for uncontrolled stages) reintroduces exactly the dual authoring
location that per-device descriptors eliminate.

**Validation at experiment prep** (during `LifConversion` assembly, §1),
reported as clear prep-time errors because the config is stringly-typed
and cross-device: every input ref resolves to an active device (or
`LASER`/`FIXED`); input arity matches the op; exactly one `FINAL`; no
cycles; exactly one tunable source. A malformed graph aborts prep, not
the acquisition.

**Future — a dedicated "LIF Conversion" tab.** A graphical editor that
lays out stages and edges is the better UX once chains grow beyond a
single stage or multi-tunable-source (§1) lands. Because the per-device
settings remain the source of truth, that editor is a *view* over the
same data — layerable later with no data-model change.

## Settings layout summary

| Concern | Owner | Mechanism |
|---|---|---|
| Grating native range (`minPos`/`maxPos`), output units, decimals, flashlamp | `LifLaser` base | `REGISTER_HARDWARE_BASE` (exists) |
| Grating geometry (`stages`) | `SirahCobra` driver | `REGISTER_HARDWARE_ARRAY` (migrated) |
| Node descriptor (op `NHG(N)`/`SFG`/`DFG`, input refs, `FINAL` marker) | `LifFreqConversionStage` base | `REGISTER_HARDWARE_SETTINGS`/`_ARRAY` (new) |
| FCU verify flag | `LifFreqConversionStage` base | `REGISTER_HARDWARE_SETTINGS` (new) |
| FCU comm port (baud, terminator) | `LifFreqConversionStage` driver | `REGISTER_COMM_DEFAULTS` (new; resolves TODO) |
| FCU phase-match calibration (local-wavelength polynomials, addresses) | driver | `REGISTER_HARDWARE_SETTINGS`/`_ARRAY` (migrated) |

The conversion topology is **not** a laser-owned setting — it is
assembled at prep from the per-stage node descriptors above (§1, §7).

## Sequencing

1. `LifConversion` value type — assembly from node descriptors +
   validation (§7) + unit tests (pure math, no hardware):
   `laserToOutput`/`outputToLaser`/`stageInput` round-trips for
   `NHG(N)`/`SFG`/`DFG`, range inversion, a two-node tripler DAG,
   uncontrolled (`Virtual`) nodes, and the malformed-graph /
   multi-tunable-source rejection paths.
2. `LifLaser` `setPosition`/`readPosition` output-unit semantics; the
   laser as the `LASER` source node; virtual laser drivers still pass
   with a bare source (no stages → output = fundamental).
3. `LifFreqConversionStage` base (`d_threaded`, node descriptor, verify
   flag) + a virtual FCU implementation for CI (also the uncontrolled-
   stage vehicle, §7).
4. `HardwareManager`: cached `LifConversion` assembled at prep, parallel
   `setLifConversionStages` fan-out/join, connect-ladder branch,
   `getActiveKeys<LifFreqConversionStage>` plumbing, prep-time topology
   validation.
5. `LifConfig`/config-page/`experiment.cpp` axis migration to output
   units.
6. `SirahDoublingStage` driver; strip `SirahCobra` of the external stage
   and migrate its remaining settings.

## Testing

- Unit: `LifConversion` round-trips and range inversion (Qt-Test, no
  hardware — passes in CI on virtual implementations per `src/AGENTS.md`).
- Integration: virtual laser + virtual conversion stage through
  `setLifParameters`, asserting the joined success flag (including the
  best-effort path) and that `currentLaserPos()` values survive the
  output→fundamental→output round-trip within `decimals`.
- Manual, on the bench when the new Sirah is up: grating + doubling-stage
  co-tuning across the OH band, verifying the axis reads in the doubled
  wavelength.

## Open decisions

- **Topology model vs implicit linear chain** — resolved toward the
  DAG-capable `LifConversion` with local-wavelength calibration and
  per-device node descriptors (this plan). It is a modularity/
  extensibility choice, not a correctness requirement: for the near-term
  single-tunable-source doubler, broadcasting the single scalar *f* to
  every stage (calibrating FCUs in *f*, no topology, no descriptors)
  would also be correct and smaller. Revisit only if day-1 surface needs
  to shrink; the fallback forgoes intrinsic per-crystal calibration and
  the multi-source seam.
- **Uncontrolled stages as logical devices** — resolved toward a
  purpose-built `FixedLifFreqConversionStage` following the `FixedClock`
  motif (§7), so the axis topology is complete with no second authoring
  location and no virtual-placeholder warning. Veto in favour of a
  laser-side math-only node array if a logical device in the loadout
  proves confusing in practice.
- **Edge authoring** — resolved toward input-only refs + a single
  `FINAL` marker (§7), not the symmetric parent+child form, to avoid
  double-authoring each edge.
- **Verify-flag granularity** — per-device (proposed) vs a single
  laser-level flag. Per-device is only marginally more code and expresses
  "abort if the grating misses, tolerate an unverified compensator".
- **Multi-tunable-source conversion (deferred, not decided)** — the
  topology type can represent it, but the solver and a multi-axis
  `LifConfig` acquisition model are out of scope. Picking it up is a
  separate project, not a tweak to this one.
- **When to pick this up** — gated on the new Sirah reaching the bench
  and the 2.0.0-alpha packaging work landing, same as the original
  roadmap entry.

## Related future work — RF configuration as a DAG

The per-device conversion-topology model here (nodes = signal-processing
elements, edges = beams, one closed-form solve from a target output back
to source setpoints, `Fixed<Type>` logical nodes for un-owned elements)
is a candidate pattern for a future refactor of the **RF configuration**,
which currently uses a *fixed* topology. The RF chain has directly
analogous elements — `ChirpSource`/AWG, `Clock` (with internal
multipliers), and would-be new `Multiplier`, `Mixer`, and `Divider`
nodes — where the user cares about the final RF/output frequency and
Blackchirp should back-solve the source and clock setpoints. This is
flagged as a design consideration, not committed work: it adds real
complexity (the RF graph is richer than the LIF one), and the open
question is how much, and how robust a shared DAG abstraction could be
made across both subsystems. Scoped separately in the roadmap.
