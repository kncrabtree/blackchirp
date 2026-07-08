# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Medium

### Chirp jitter monitor (sub-sample trigger-timing diagnostics)

The phase-correction hill-climb in `FtmwConfig::preprocessChirp`
already evaluates the chirp-correlation FOM at three adjacent lags and
discards the sub-sample information they contain. Parabolic
interpolation of those three values gives each shot's trigger-timing
offset at picosecond resolution — and the per-shot offsets `dt_k`
determine, exactly, the filter `Phi(f) = (1/N) sum exp(-i 2 pi f dt_k)`
that the averaging process applies to the stored spectrum. Recording
them turns an invisible band-dependent intensity attenuation (10% at
14.5 GHz baseband for 5 ps rms jitter, vs 2% at 5 GHz) into a
deconvolvable, exactly-known correction, and answers per-acquisition
whether deep averages are jitter-limited.

Plan: monitor-only mode decoupled from the correction (no shift
applied, no shot rejection, clip-tolerant `sign()` FOM option), aux-data
aggregates per tick (mean/rms of the offset), and an accumulated
`Phi(f)` on a coarse frequency grid written as `jitterphi.csv` at
experiment completion for the analysis pipeline to divide out. Full
per-shot series deferred. ~150–300 LOC + tests (virtual digitizer
gains a configurable per-shot delay). Details:
[`chirp-jitter-monitor.md`](chirp-jitter-monitor.md).

Trigger: the first campaign where cross-band relative intensities
matter, or when the companion `ftmwpipeline` timebase/deconvolution
analysis is ready to consume `jitterphi.csv`.

### Sirah Cobra integration refresh

A new Sirah Cobra dye laser coming online triggers a rework of the
`SirahCobra` driver: move its hand-rolled second serial port and
frequency-conversion logic into a first-class `LifFreqConversionStage`
hardware type, add a hardware-independent conversion topology to the
`LifLaser` base so the LIF axis reads in the final (converted)
wavelength, and migrate the driver's ad-hoc settings to the registry.
Full plan (which supersedes the earlier "Approach A" multi-port
direction) in [sirah-cobra-refresh.md](sirah-cobra-refresh.md); pick it
up once the new instrument is on the bench and the 2.0.0-alpha packaging
work is finished.

### Sirah FCU calibration schemes

The `SirahFcu` doubling stage from the refresh above still tunes with
`SirahCobra`'s **grating** diffraction math as a placeholder — wrong for
a phase-matched doubling crystal. Replace it with a user-selectable
calibration scheme: a best-effort BBO/KDP Type-I SHG physical model
(cut angle, temperature, sine-bar offsets, screw pitch — the parameter
set confirmed by the Sirah Autotracker service manual §6), plus
polynomial and spline fallbacks. Blackchirp evaluates only; the fits are
produced offline and imported (a generic "Import CSV…" button is added
to `HwArrayEditDialog`). The tuning law moves to a hardware-free
`FcuCalibration` value type (sibling of `LifConversion`), unit-tested in
CI. Full plan in
[sirah-fcu-calibration.md](sirah-fcu-calibration.md). Being picked up on
`feature/sirah-cobra-refresh`, starting with the value type and tests.

## Large

### RF configuration as a flexible frequency-conversion DAG

Generalize the RF signal-chain configuration from its current **fixed
topology** to a flexible DAG, reusing the frequency-conversion topology
model designed for the LIF laser
([sirah-cobra-refresh.md](sirah-cobra-refresh.md)). Today the chain is a
single hardcoded 3-stage formula — `chirpFreq = (awgFreq × awgMult ±
upLO) × chirpMult` in `RfConfig::calculateChirpFreq`/`calculateAwgFreq`
(`rfconfig.cpp:204-228`) — over a closed six-value role enum
(`RfConfig::ClockType`) used as `QHash` keys. A DAG would model nodes
(sources, `Multiplier`, `Divider`, `Mixer`), let the user enter the
final RF frequency, and back-solve the AWG/clock setpoints.

Feasibility (from an architecture map): the deep, risky assumptions are
just two — (1) the `ClockType` `Q_ENUM` consumed reflectively and as
hash keys across ~4 layers, and (2) the linear 3-stage formula treated
as a pure `double→double` at ~8 call sites. Everything around them is
already node-shaped and generalizes cheaply: per-output ×/÷ exists
(`Clock::d_multFactors` + `MultOperation`), logical/un-owned nodes exist
(`FixedClock`), the `header.csv` RfConfig scalars are additive/default-
tolerant, and — critically — the **Python analyzer is insulated**: it
consumes only the collapsed per-FID `probefreq`+`sideband` from
`fidparams.csv` (`bcfid.py:40-41,165-202`), never the upconversion
topology.

Why it can be robust: every RF element is **affine in frequency** and
each acquisition point has **one tunable variable** (the AWG chirp; LO/DR
scans re-parameterize fixed LOs between steps), so the solve is
closed-form and unit-testable — the same complexity class as the LIF
topology. Robustness levers: keep the collapsed per-FID `probeFreq`+
`sideband` as the acquisition/analysis invariant (bounds blast radius,
leaves `Fid` and Python untouched); model transmit and receive as two
DAGs sharing source nodes (the one step beyond LIF, which has a single
chain); and migrate the current fixed chain into a canonical graph so
old experiments load losslessly.

Suggested sequencing: (1) build the affine single-tunable-source solver
as the shared `FreqConversion` abstraction while doing the LIF work — the
lower-stakes proving ground; (2) drop the graph in *behind* the existing
`RfConfig` API so `calculateChirpFreq` delegates to `graph.solve()` with
byte-identical output — a pure refactor of the deepest spot, zero
behavior change; (3) only then open the role enum, GUI
(`RfConfigWidget` + `ClockTableModel`), scan builders, and the
`clocks.csv` 7-column schema to arbitrary graphs.

Rough scope: step 2 is a contained refactor; step 3 is a new DAG-editor
GUI plus a `clocks.csv` schema version bump and generalized LO/DR scan
builders — a multi-week effort. Trigger to pick it up: a real
instrument whose RF topology the fixed 3-stage model cannot express
(e.g. a second up-mixer stage, an IF divider, or a non-`UpLO`/`DownLO`
mixing role), or the LIF conversion work landing and proving the shared
abstraction. Not release-blocking.

### Async PythonProcess + hardware base contracts

Refactor `PythonProcess::sendRequest` from its current
synchronous-with-nested-`QEventLoop` shape into a true async API
(`QFuture<QJsonObject>` or callback-style), and propagate the change
through every Python-driver-facing hardware base class
(`FlowController`, `PressureController`, `TemperatureController`,
`IOBoard`, `LifLaser`, `LifDigitizer`, `FtmwDigitizer`, `Clock`,
`ChirpSource`/AWG, `PulseGenerator`, `GpibController`).

Motivation: the nested event loop in `sendRequest` is the structural
source of a destruction race observed at app shutdown — the loop
processes events that can free `this` mid-call, so the post-loop
member accesses dereference a corpse. The shutdown ordering fix in
`python-process-shutdown-fix.md` (B + QPointer guard) treats the
observed trigger and one defensive case but does not eliminate the
class of bugs. Any new caller that initiates a `sendRequest` during
a destructible sequence is still re-entrant.

Why it cannot be hidden inside `PythonProcess`: relay requests from
the Python script (`self.comm.write`, `self.settings.set`,
`self.log`, scope waveform pushes) need to be serviced *on the
hardware thread* while a Python method is in flight. Blocking the
hardware thread on a semaphore while a separate dispatcher services
relays deadlocks on the `BlockingQueuedConnection` back into the
blocked thread. So either the nested loop stays (current) or the
contract changes all the way out to the per-driver virtual.

Rough scope: ~500–1000 LOC across ~50 files; 2–3 days of focused
work plus a per-driver-type testing pass and updates to
`developer_guide/adding_a_driver.rst`. Trigger for picking it up:
the next major hardware-contract change (e.g., the Sirah aux-port
work, or a new "remote hardware proxy" driver type that genuinely
needs async), or evidence in production that the QPointer guard in
`sendRequest` is being hit.

### Cross-experiment memory budget

In-memory data caches in the viewer (and the acquisition app) are
currently unbounded and owned per-experiment, with no awareness of
each other. Two concrete instances today:

- `LifStorage::d_data` retains every `LifTrace` ever loaded
  (raw `qint64` samples per cell). For a 100×100 grid with 8192
  samples per cell on two channels, that is ~130 MB just for the
  raw data of one open experiment.
- A filtered-trace cache adjacent to `d_data` would roughly
  double the per-cell cost (8192 × 2 channels × `double` =
  128 KB filtered, vs 128 KB raw `qint64`). An attempt at a
  per-`LifStorage` byte budget was prototyped and reverted
  because the worst-case sample count is a guess that papers over
  the absence of a real policy — see the `lif-progress-dialog`
  / async-reprocess thread for context.

FTMW has its own caching semantics that aren't accounted for in
either of the above and would need to participate in any
process-wide budget.

Direction (not chosen):

1. **Static registry**: each cache class registers itself (with
   accessors for current bytes used, an `evict(bytes)` callback,
   and a category tag — "lif raw", "lif filtered", "ftmw …") at
   construction. A central `MemoryBudget` singleton tracks
   totals, applies a configurable cap, and drives LRU eviction
   across all registered caches when the cap is exceeded.
2. **Settings**: one user-facing knob (`BC::Key::MemoryBudgetBytes`,
   say), with sensible default sized to typical workstations.
3. **Instrumentation**: a status-bar widget or dev menu surface
   showing per-category usage and giving the user a "purge"
   button.

Rough scope: a few hundred LOC for the registry + a per-cache
adapter for each existing in-memory cache, plus the policy and
UI. Trigger to pick it up: actual user reports of memory
pressure with multiple open experiments, or a deliberate move to
keep filtered traces / spectrogram bitmaps / FTMW intermediates
warm across reprocesses.

In the meantime: the LIF viewer's reprocess path runs on a worker
thread with a cancelable progress dialog
(`LifDisplayWidget::reprocess`), which is enough to keep the UI
responsive even when the SG-filter convolution runs from scratch
on every gate adjustment.

## Cleanups

Low-priority code-debt items, none release-blocking. Each is gated on
an external trigger; revisit when the trigger fires.

### Drop the `QAnyStringView` -> `QString` workaround in `hwLog/hwWarn/hwError/hwDebug`

`src/hardware/core/hardwareobject.h` calls `text.toString()` on the
`QAnyStringView` parameter before passing it to
`QString::arg(d_key, ...)` (the four `hwLog`-family one-liners around
line 403). Qt 6.4's `QString::arg` variadic-template trait
(`is_convertible_to_view_or_qstring`) does not accept
`QAnyStringView`; Qt 6.5 added it. The deb job's Ubuntu runner pins
to apt's `qt6-base-dev`, which on the current `ubuntu-latest`
(noble, 6.4.2) is the version that forces the workaround.

When the GitHub-hosted `ubuntu-latest` image rolls forward to an
Ubuntu release whose `qt6-base-dev` is >= 6.5, drop the
`.toString()` calls in those four lines and remove the explanatory
comment. No other call site is affected — `loghandler.cpp` already
pins its `QAnyStringView` entry point through a
`text.toString()` conversion before any `arg()` reaches it, and no
multi-arg `arg()` elsewhere in the tree consumes a `QAnyStringView`.

### Drop the `QStringView` -> `QString` workaround in `lifdisplaywidget.cpp`

`src/gui/lif/gui/lifdisplaywidget.cpp:102` concatenates `QString` with
`BC::Unit::us` (a `QStringView`) via `operator+`. Qt 6.4's `QString`
has no `operator+` overload accepting `QStringView`; Qt 6.5 added one.
The same Ubuntu-noble apt-Qt 6.4.2 ceiling that forces the
`hwLog`-family workaround is what forces this one. When the deb-job
Qt rolls forward to >= 6.5, drop the `.toString()` call and remove
the inline comment.

### Long-tail symbol storage

GitHub workflow artifacts cap at ~90 days. Crashes against older
releases lose easy symbol access once that window closes. If
long-tail support matters post-alpha, publish the symbol artifacts
to a private S3 bucket on every release-tag run, or attach them to
the release as password-protected ZIPs. Tracked here so the
decision surfaces if a triager hits a stale-symbols wall.

### MSVC cosmetic warnings

The Windows release build emits ~50 unique non-vendor warnings.
The three warnings with cross-platform behavior or correctness implications
(C4701 uninitialized `ChirpSegment`, C4702 dead `return`, C4804
`bool > 0`) are fixed. The remaining warnings produce identical,
well-defined behavior on MSVC and GCC; they're left as a future
clean-up pass when there's appetite for `-Wall`/MSVC-W4 hygiene work.

Categories, all platform-consistent (no Linux-vs-Windows divergence):

- **C4267** (size_t -> int truncation) — ~11 sites in
  `digitizerconfig.cpp`, `overlaystorage.cpp`, `settingsstorage.cpp`,
  `clock.cpp`, the AWG drivers, `temperaturecontrollerconfig.h`,
  `pulsestatusbox.cpp`. Channel/segment/array sizes that comfortably
  fit in `int`. Fix by switching the receiving locals to
  `qsizetype` / `std::size_t` or adding a `static_cast<int>` at the
  use site.
- **C4456 / C4457 / C4458 / C4459** (name shadowing) — ~17 sites,
  mostly inner `key` / `i` / `obj` locals shadowing globals or outer
  scopes. Inner scope wins identically on both compilers; rename
  the inner locals to silence.
- **C4101** (unused `e` in `catch`) — `xiamparser.cpp:371,497`,
  `catalogoverlaywidget.cpp:547`. Drop the binding (`catch (...)`)
  or `[[maybe_unused]]` it.
- **C4334** `ftworker.cpp:475` — `1 << zeroPadFactor` where
  `zeroPadFactor <= 2`; the int-shift result is then promoted to
  `size_t` for the surrounding multiplication. Cast the `1` to
  `size_t` to silence.
- **C4305** `ftmwconfig.cpp:429` — `float thresh = 1.15;` literal is
  `double`. Append `f` (`1.15f`) to silence.
- **C4309** `tst_waveformbuffertest.cpp:708` — `QByteArray(size, 0xAB)`
  truncates to `signed char`. `static_cast<char>(0xAB)` silences.
- **C4005** `crashhandler_win.cpp:10` —
  `WIN32_LEAN_AND_MEAN` already defined by Qt headers; guard the
  redefinition with `#ifndef`.
- **C4996** `main.cpp` `sprintf(mem->name, "Blackchirp")` — MSVC's
  "use `sprintf_s`" deprecation notice (and AppleClang's matching
  `-Wdeprecated-declarations`; see the AppleClang subsection
  below). Switch to `snprintf` (available on every platform) to
  silence both with one edit.

### AppleClang cosmetic warnings

The macOS release build emits ~59 unique non-vendor warnings, all
pre-existing and platform-discovery rather than regression-driven —
the same code is silent under GCC because GCC's analogous flags
aren't on by default with `-Wall -Wextra`. Each is platform-
consistent (no macOS-vs-Linux behavior divergence); fix when there's
appetite for a `-Wall` hygiene pass.

The one **non-cosmetic** warning from this set —
`-Wdelete-non-abstract-non-virtual-dtor` /
`-Wdelete-abstract-non-virtual-dtor` on `OverlayBase` and its three
derived types — was fixed at point-of-discovery (added
`virtual ~OverlayBase() = default;`) because all owning sites use
`std::shared_ptr<OverlayBase>` and the type-erased deleter happened
to dispatch correctly, but the moment anyone introduced a
`unique_ptr<OverlayBase>` or a `delete bp;` it would have been UB.
Not in this list.

Categories of cosmetic warnings:

- **`-Winconsistent-missing-override`** — ~32 sites across the
  hardware-driver and overlay/operation hierarchies. `sizeHint`,
  `loadDifferentialFidList`, `beginAcquisition`, `endAcquisition`,
  `configure`, etc. Add the `override` keyword; behavior unchanged.
  A bulk regex pass over `src/hardware/` and `src/data/processing/`
  catches most of them.
- **`-Wunused-lambda-capture`** — ~4 sites. Drop the unused capture
  from the lambda's capture list. Includes the `[this]` capture in
  `hardwaremanager.cpp:1485` (also flagged by the IDE's clang
  diagnostic).
- **`-Wunused-but-set-variable`** — 2 sites (`textColumns`, `i`).
  Either remove the dead store or `[[maybe_unused]]` if the
  side-effecting RHS is intentional.

### Bundle license texts inside the packages (beta prep)

The `release-assets` job in `.github/workflows/release.yml` attaches
`COPYING` and a `blackchirp-licenses.zip` (the `licenses/` directory:
Qwt, LGPL-3.0, GPL-3.0, MPL-2.0, Heroicons) as standalone GitHub
release assets. That covers users who land on the release page, but a
user who installs only the `.deb`/`.rpm`/`.dmg`/`.zip` still does not
get the third-party texts on disk — and the binary packages bundle Qwt
(`BC_BUNDLE_QWT=ON` on deb/rpm) and ship Heroicons, whose terms
(LGPL/Qwt license) are meant to travel with the binary.

As part of beta preparation, add an `install(DIRECTORY licenses/ ...)`
rule (and `COPYING`) into the package payload — e.g. under
`share/doc/blackchirp/` for deb/rpm, the `.app` `Resources` for macOS,
and the install root for the Windows zip/NSIS — so the texts ship
inside every artifact. `CPACK_RESOURCE_FILE_LICENSE` already places
`COPYING` as the deb copyright / NSIS license page / DMG SLA; this is
specifically about the bundled-dependency texts in `licenses/`. Touches
every CPack generator in `cmake/Packaging.cmake`, so verify each
package layout before the beta tag. Once bundled, the standalone
`release-assets` upload can stay as a convenience or be retired.

### Windows linker: `ignoring duplicate libraries`

3 sites on the macOS link line (`-lm`, `libblackchirp-data.a`).
The CMake `target_link_libraries` graph lists the static archive
both as a transitive dep (via `Blackchirp::Data`) and as an explicit
dep on the executable target; ld dedupes. Cosmetic, but
disentangling the duplicate path would also clean up the warning on
any future linker that doesn't dedupe. Not urgent.
