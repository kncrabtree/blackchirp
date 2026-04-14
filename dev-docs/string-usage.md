# String Usage and Logging Cleanup

Policy and migration guidance for `QString`, string literals,
string-keyed containers, and the diagnostic log system in Blackchirp.

This document covers two related projects that share a critical
interface — `LogHandler` and its ~445 call sites — and are therefore
best handled as a single staged plan.

## Scope

**String usage** — eliminate inefficient `QString` patterns (per-TU
duplication of `static const QString` keys, by-value `QString`
parameters), standardize on `"..."_s` literals, and migrate hot APIs
to `QAnyStringView`.

**Logging cleanup** — replace the current signal-chain `LogHandler`
with a thread-safe global singleton, eliminate `qDebug()` in favor of
the unified log system, and rationalize message severity before the
2.0.0 release.

The two projects intersect at the `LogHandler` interface: string
usage identifies `LogHandler::logMessage(const QString text, ...)`
as one of the worst by-value `QString` offenders, and logging cleanup
wants to redesign `LogHandler` entirely. Doing these separately would
touch all ~445 call sites twice. A single staged plan touches them
once, with the new API taking `QAnyStringView` and call-site literals
standardized on `"..."_s`.

## Context and Motivation

### String usage

Blackchirp uses `QString` pervasively, but the current patterns
predate Qt 6 idioms and waste both startup time and memory. A walk of
the codebase finds:

- **Central key headers use `static const QString` at namespace
  scope**:
  - `src/data/settings/hardwarekeys.h` — 127 declarations across
    nested namespaces (`HW`, `Comm`, `RS232`, `TCP`, `GPIB`,
    `FtmwScope`, `Digi`, `Clock`, `AWG`, `PGen`, `Flow`, `PController`,
    `TC`, …).
  - `src/data/bcglobals.h` — a handful more in `BC::Key` and
    `BC::Unit`.
  - `src/data/storage/settingsstorage.h` — ~8 more near lines 23–32.
  - Per-subsystem: `src/data/experiment/auxdatakeys.h`,
    `src/data/lif/lifstorage.h`, `src/data/lif/lifconfig.h`,
    `src/hardware/core/hardwaremanager.h`,
    `src/hardware/core/runtimehardwareconfig.h`.

  `static` at namespace scope in a header gives the declaration
  **internal linkage**, so every translation unit that includes the
  header gets its own copy of each `QString`. With ~130 keys in
  `hardwarekeys.h` and dozens of hardware translation units, that is
  a meaningful multiplier on both heap-allocation cost at startup and
  binary bloat. The cost is per-TU, not per-process.

- **Beyond the central headers, ~747 `static const QString`
  declarations exist across ~105 files**, roughly 40% in headers and
  the rest in .cpp files. Migration scope is the whole codebase, not
  just the key headers.

- **String literal style is mid-migration without a policy**: ~1432
  occurrences of `"..."_s` (Qt::StringLiterals) coexist with ~559
  occurrences of `QStringLiteral(...)`. Zero header-scope `inline` or
  `constexpr` string constants exist — every `_s` usage is local
  inside a function body.

- **~282 `std::map<QString, T>` sites exist**, none using
  `std::less<>` for heterogeneous lookup.

- **By-value `QString` parameters are common in hot-call APIs**:
  - `LogHandler::logMessage(const QString text, ...)` —
    `src/data/loghandler.h:34`, ~445 call sites.
  - `CommunicationProtocol::writeCmd(QString cmd)` and
    `queryCmd(QString cmd, bool)` —
    `src/hardware/core/communication/communicationprotocol.h:93,119`.
  - `SettingsStorage::get(const QString key, ...)` and its overloads —
    `src/data/storage/settingsstorage.h:332` and peers.

  `const QString` taken by value (not by reference) is strictly worse
  than `const QString &`: it bumps the implicit-sharing refcount on
  every call with no upside.

### Logging cleanup

During development of the `cmakemigration` branch, diagnostic output
accumulated in two forms: `qDebug()` calls (which bypass the log
system) and `emit logMessage()` calls (which go to the UI log tab).
The recent addition of debug logging to the application configuration
gives a proper channel for diagnostic output, but most messages have
not been reclassified to use it.

Survey results:

**`qDebug()`** — 41 calls across 8 files
- Hardware registration/initialization tracing (19 calls in
  `hardware/core/`)
- Overlay system diagnostics (18 calls in `data/processing/` and
  `gui/overlay/`)
- Vendor library loading (2 calls in `hardware/library/`)
- None of these go through the log system.

**`emit logMessage()`** — ~445 calls across 46 files
- ~380 Error (~85%) — but many are configuration verification traces,
  not true errors
- ~31 Normal (~7%) — many are internal lifecycle tracing, not
  user-facing status
- ~25 Warning (~6%) — generally appropriate
- ~6 Debug (~1%) — drastically underused
- 0 Highlight — never used

Every message should be intentionally categorized: shown to the user,
sent only to the debug log, or removed entirely. This is pre-release
polish — one of the last things before beginning documentation
revision for 2.0.0.

### What is *not* the motivation

- **Hot-path data flow is already QString-free.** `FtmwScope::emitShot()`
  and its callers operate on `QVector<qint64>` waveforms. Settings are
  read once at construction and in `hwPrepareForExperiment()`, never
  in per-shot loops. String-level optimization does not affect
  acquisition throughput.
- **Translation / i18n is not a concern.** There are no `.ts` / `.qm`
  files, no `QT_TR_NOOP`, and no translation pipeline. The ~59 `tr()`
  calls are inherited from `QObject` with no translations behind them.
  i18n is not on the roadmap; see "Out of Scope" below.

The motivation is therefore **startup cost, binary size, API
hygiene, log consistency, and pre-release polish** — not hot-path
throughput and not i18n compatibility.

## House Style for String Literals

For new code:

- **Use `"..."_s`** from `Qt::StringLiterals` for `QString` literals.
  This is the house style going forward and is already the dominant
  form in the codebase.
- **Use `u"..."_s`** (UTF-16 prefix) for literals containing non-ASCII
  characters. Example: `BC::Unit::us` currently uses
  `QString::fromUtf8("μs")`; its modern form is `u"μs"_s`.
- **`QStringLiteral(...)` is not used in new code.** Existing sites
  are left alone — a mechanical sweep is possible but not required.
- **`"..."_L1`** produces a `QLatin1StringView` for ASCII-only
  constants where a non-`QString` view is desired (see Pattern B
  under Key Declaration).

The C++23 standard (`CMAKE_CXX_STANDARD 23` in `CMakeLists.txt`) and
Qt 6 minimum guarantee that `_s`, `_L1`, `QAnyStringView`, and
`inline constexpr QStringView` are all available.

## Key Declaration Idiom

Three portable patterns are documented here. The implementation task
picks one per header based on how the keys are consumed. **No single
pattern wins in all cases.**

All three replace the current `static const QString k{"key"};` idiom,
and all three eliminate the per-TU duplication.

### Pattern A — `inline const QString`

```cpp
inline const QString trigCh = u"trigCh"_s;
```

- ODR-safe via `inline`; single definition across all translation
  units.
- Not `constexpr` — `QString` is not a C++ literal type — but the
  `_s` literal stores the UTF-16 data in `.rodata`, and the `QString`
  constructor performs no heap allocation.
- One constructor call per process, not per TU.
- Stores directly into `std::map<QString, T>` and `QHash<QString, T>`
  with zero conversion.
- Best when the consumers of these keys already take `QString` /
  `const QString &` and signatures are not changing in the same
  pass.

This is the lowest-risk migration target for the existing key
headers.

### Pattern B — `inline constexpr QLatin1StringView`

```cpp
inline constexpr QLatin1StringView trigCh = "trigCh"_L1;
```

- True `constexpr`; zero runtime cost at the constant's own expense.
- Consumers must accept `QAnyStringView`, or the containing container
  must use heterogeneous lookup
  (`std::map<QString, T, std::less<>>`), or the call site must pay
  the `QString` construction that pattern A avoids.
- Best when the consumer API is migrating to `QAnyStringView` in the
  same change — the container / signature migration and the key
  declaration migration land together.
- ASCII-only. Use pattern C for non-ASCII keys.

### Pattern C — `inline constexpr QStringView`

```cpp
inline constexpr QStringView us = u"μs";
```

- Same tradeoff as pattern B but UTF-16, so non-ASCII keys are safe.
- Useful for `BC::Unit` and any other namespace with non-ASCII
  content.

### Do not use

```cpp
// Does NOT compile portably — QString is not a literal type.
inline constexpr auto trigCh = "trigCh"_s;
```

`QString`'s internal `QArrayDataPointer` has a non-trivial destructor
and non-`constexpr` members, so `constexpr QString` is not valid in
the Qt versions this project targets. `constexpr auto` with `_s`
deduces to `QString` and fails for the same reason. Guidance that
recommends this pattern predates the Qt/C++ reality — do not use it
in Blackchirp.

## Function Signature Policy

Rules of thumb, in order of preference:

1. **Never pass `QString` by value** unless the callee is going to
   take ownership and move. The current offenders are listed in
   [Context and Motivation](#string-usage). All should be corrected.
2. **`const QString &`** is the conservative default for parameters
   when the callee needs a `QString` (to pass to another `QString`
   API, to store, or to do `QString`-specific operations like
   `arg()`).
3. **`QAnyStringView`** is appropriate when the function is a pure
   lookup, comparison, or passthrough — it lets callers pass
   `QString`, `QStringView`, `QLatin1StringView`, `const char *`, or
   a `"..."_s` literal without any temporary `QString`. Good
   candidates:
   - `SettingsStorage::get` / `set` / `containsValue`
   - `HeaderStorage` lookup methods
   - `HardwareObject::validationKeys` consumers
   - The new `LogHandler::log` / `bcLog` API (see redesign below)
4. **`QStringView`** when you need `constexpr`-eligibility on the
   parameter and the function is truly view-only. Less useful than
   `QAnyStringView` in Blackchirp because of the Latin-1 /
   `const char *` paths.

### Priority

Signature migration is **deprioritized relative to key consolidation
and the `LogHandler` redesign**. Blackchirp's hot paths are
QString-free, so the performance upside is modest. The real upside is
API hygiene: fixing the by-value parameters and removing temporary
`QString` allocations at call sites. Do this incrementally as
specific APIs are touched for other reasons, not as a codebase-wide
sweep — with one major exception: the `LogHandler` interface is
being redesigned anyway, and the new API should take `QAnyStringView`
from the start so that the ~445 call sites are touched only once.

## Virtual Cascade Hotspots

Any signature change on a virtual function forces every override to
change in the same commit. The following base classes are the
coordination points — scope any signature migration accordingly:

- **`CommunicationProtocol::writeCmd` / `queryCmd`** —
  `src/hardware/core/communication/communicationprotocol.h:93,119`.
  Taken by value today. Overridden by `Rs232Instrument`,
  `TcpInstrument`, `GpibInstrument`, `CustomInstrument`,
  `VirtualInstrument`, and the Python-hardware wrappers (~8
  overrides total). Fixing these to `const QString &` is a
  straightforward cleanup; going further to `QAnyStringView` requires
  coordinated subclass edits.
- **`HardwareObject` QString/QStringList virtuals** —
  `src/hardware/core/hardwareobject.h:232,404` and peers
  (`validationKeys`, `forbiddenKeys`, …). ~100 subclass overrides
  across `src/hardware/core/` and `src/hardware/optional/`. Any
  signature migration here is a heavy coordinated refactor.
- **`DataStorageBase`** —
  `src/data/storage/datastoragebase.h:18`. Constructor takes
  `QString path = ""`.
- **`FileParser::canParse`** —
  `src/data/processing/parsers/fileparser.h:26`. Takes
  `const QString &`, already reasonable.

Migrations touching virtual functions should be done as single
commits per base class, with all overrides updated together.

## Container Policy

- **`std::map<QString, T, std::less<>>`** is the default declaration
  for new maps. The `std::less<>` transparent comparator enables
  heterogeneous lookup, so callers can `find("..."_s)`,
  `find(QStringView(...))`, or `find(QLatin1StringView(...))` without
  allocating a temporary `QString`.
- Blanket retrofitting `std::less<>` onto the ~282 existing
  `std::map<QString, T>` declarations is **cheap and safe**, but the
  benefit is *deferred* — it only materializes once lookups start
  passing non-`QString` keys. Practically, the retrofit is worth
  doing wherever Pattern B / C keys or `QAnyStringView` signatures
  are being rolled out in the same change.
- **`SettingsStorage::SettingsMap`** —
  `src/data/storage/settingsstorage.h` line 24
  (`using SettingsMap = std::map<QString, QVariant>;`) is the central
  typedef for the class of map most affected by this policy.
  Updating it updates all downstream sites uniformly.
- **`QHash<QString, T>`** is rare in Blackchirp (~4 sites). Qt 6
  supports `QHash` heterogeneous lookup via `qHash(QStringView)`; no
  special declaration is needed.

## LogHandler Redesign

The current `LogHandler` design routes messages through Qt signal
chains — hardware objects `emit logMessage()`, which propagates
through `HardwareManager` to `LogHandler`. This creates signal
duplication and complex connection management, especially across
threads. It also locks the API into a by-value `QString` parameter
that cannot be changed without coordinated updates across every
subclass.

**New design: thread-safe global singleton**, similar to
`RuntimeHardwareConfig` and `HardwareRegistry`. Any code — `QObject`
or not, any thread — can call:

```cpp
LogHandler::instance().log(u"message"_s, LogHandler::Debug);
// or a convenience free function:
bcLog(u"message"_s, LogHandler::Debug);
```

**API adopts string-usage policy from day one:**

- The `log` method takes `QAnyStringView`, not `const QString &` or
  by-value `QString`. `QAnyStringView` lets callers pass `QString`,
  `QStringView`, `QLatin1StringView`, `const char *`, or a `"..."_s`
  literal without any temporary `QString`.
- The existing `emit logMessage(const QString text, ...)` signature
  on `HardwareObject` is kept temporarily as a shim that forwards to
  `LogHandler::instance().log(...)`, so existing call sites continue
  to compile during the migration, and is removed once migration is
  complete.
- All new `log(...)` call sites use `"..."_s` (or `u"..."_s` for
  non-ASCII) for literal messages.

**Benefits:**

- Eliminates the `emit logMessage()` → signal chain → `LogHandler`
  relay pattern.
- Non-`QObject` and static contexts (registration, factories) can
  log directly — removes the need for `qDebug()` entirely.
- Simplifies `HardwareManager` connection setup (no `logMessage`
  forwarding).
- Fixes the worst by-value `QString` offender in the codebase (~445
  call sites) as a natural consequence of the redesign.
- Thread safety via internal mutex or queued dispatch to the UI
  thread.

**Considerations:**

- `LogHandler` still needs to update the UI log widget, which must
  happen on the main thread. The singleton can use
  `QMetaObject::invokeMethod` with `Qt::QueuedConnection` internally
  to dispatch to the UI, or use a lock-free queue polled by a timer.
  Pick during implementation.
- The `d_startLogMessage` / `d_endLogMessageCode` pattern in
  `Experiment` already calls `LogHandler` directly — this becomes
  the standard pattern.

## Logging Principles

Once the global logger is available, every existing message is
triaged into one of these categories.

### What users should see (Normal / Warning / Error)

- Connection success/failure outcomes
- Experiment progress milestones (start, completion, abort)
- Hardware state changes the user initiated or needs to act on
- Errors that require user intervention or indicate data loss risk

### What goes to debug log only (Debug)

- Hardware lifecycle tracing (creation, thread assignment,
  destruction)
- Configuration loading/syncing progress
- Protocol-level command/response details
- Parameter verification traces (digitizer scale parsing, etc.)
- Registration and initialization diagnostics

### What gets removed

- Development-time scaffolding (`qDebug` calls added during feature
  development)
- Redundant messages (e.g., logging the same event at multiple
  levels)
- Messages that duplicate information already visible in the UI

### `qDebug()` policy going forward

All `qDebug()` calls should be replaced with
`bcLog(..., LogHandler::Debug)` or removed if no longer useful. Once
the global logger exists, there is no remaining reason to use
`qDebug()` directly — non-`QObject` and static contexts can call
`bcLog` the same way `QObject` subclasses can.

## Logging Cleanup Work Areas

Ordered by volume; each is a focused pass once the global logger is
in place.

### 1. FTMW Digitizer Files (~285 logMessage calls, 7 files)

**Largest volume by far.** These files contain extensive
command/response verification that was written as Error but is really
diagnostic tracing.

Files: `mso72004c.cpp`, `dpo71254b.cpp`, `mso64b.cpp`,
`dsa71604c.cpp`, `dsov204a.cpp`, `dsox92004a.cpp`, `m4i2220x8.cpp`

**Approach:** Per-message triage.

- **Keep as Error:** Failures that prevent acquisition (can't
  configure, can't read waveform)
- **Downgrade to Debug:** Response parsing details, parameter
  comparison traces, hex dumps
- **Downgrade to Warning:** Parameter mismatches that are corrected
  automatically

Most labor-intensive area; requires discussion on individual
messages since the line between "real error" and "diagnostic trace"
is not always obvious in digitizer configuration.

### 2. HardwareManager (~74 logMessage calls)

**31 Normal messages** that are mostly initialization lifecycle
tracing:

- "Loading hardware configuration from runtime profiles..."
- "Started thread for hardware: [key]"
- "Hardware created and initialized..."
- "Updating ClockManager with N clock(s)"

**Approach:** Bulk reclassification. Most Normal → Debug. Keep Error
and Warning as-is. User-facing Normal messages to retain: connection
test results emitted to the log tab.

### 3. `qDebug()` Elimination (41 calls, 8 files)

- `runtimehardwareconfig.cpp` (8): Initialization/sync tracing →
  Debug `bcLog` or remove
- `hardwareregistration.cpp` (4): Registration enumeration → remove
  (startup-only)
- `hardwareregistry.cpp` (4): Instance creation tracing → Debug
  `bcLog` or remove
- `hardwareprofilemanager.cpp` (1): System profile creation → remove
  or Debug
- `overlaymanagerwidget.cpp` (15): Widget state diagnostics → Debug
  `bcLog` or remove
- `overlaystorage.cpp` (4): File operation diagnostics → Debug
  `bcLog` or remove
- `overlayprocessmanager.cpp` (3): Operation state → Debug `bcLog`
  or remove
- `labjacklibrary.cpp` (2): Library loading → Debug `bcLog`

With the global logger in place, the "no access to `logMessage`"
problem disappears and this becomes straightforward mechanical work.

### 4. Communication Protocol Files (~15 calls)

- `communicationprotocol.cpp`: Error messages are appropriate
  (connection failures)
- `tcpinstrument.cpp`: 2 Normal messages about socket state → Debug
- `gpibinstrument.cpp`: 2 Debug messages already correctly
  categorized

### 5. Optional Hardware Implementations (~110 calls)

- Flow controllers, pulse generators, AWGs, pressure/temp
  controllers
- Most are Error on communication failures — likely appropriate
- Quick pass to verify no diagnostic traces masquerading as errors

### 6. LIF Components (~36 calls)

- Laser and digitizer operation errors — likely appropriate
- Quick pass to verify severity levels

### 7. `LogHandler::Highlight` Usage

Currently used for experiment start (`mainwindow.cpp:717`) and
normal experiment completion (`experiment.cpp:403,409`). These go
through direct `p_lh->logMessage()` calls rather than
`emit logMessage()`, which is why they were missed in the
signal-based survey. Consider whether other milestone events warrant
Highlight.

## Out of Scope

- **Internationalization / `tr()` / `.ts` files.** Blackchirp has no
  translation infrastructure and none is planned. Existing `tr()`
  calls are cosmetic. If i18n is ever added, UI strings will need a
  separate audit; this document does not reserve the `tr()` path and
  does not introduce `QT_TR_NOOP`.
- **Hot-path optimization of digitizer data flow.** FTMW acquisition,
  accumulation, and shot handling do not touch `QString`. See the
  [Digitizer Data Flow Optimization](digitizer-data-flow.md) task
  for the relevant concerns in that area (complete).
- **`QStringLiteral(...)` mass rewrite.** Existing occurrences can
  stay. The policy applies to new code and to sites that are already
  being edited for another reason.
- **UI log-level filtering.** Adding a show/hide filter on the UI
  log tab (e.g., Debug/Warning toggles) would complement the
  severity cleanup but is not part of this task.

## Staged Migration Plan

Each step is independently valuable and can be committed on its own.
Steps are ordered so that earlier steps unlock or simplify later
steps.

Steps 1–2 and 9–11 are the "string usage" core; steps 3–8 and 12
are the "logging cleanup" core; step 3 is the integration point that
makes the rest of the plan coherent.

1. **Standardize new code on `"..."_s`.** No sweep; just policy.
   This is the baseline for every subsequent step.

2. **Consolidate and migrate central key headers.** Target list:
   - `src/data/settings/hardwarekeys.h`
   - `src/data/bcglobals.h`
   - `src/data/storage/settingsstorage.h` (top-of-file constants)
   - `src/data/experiment/auxdatakeys.h`
   - `src/data/lif/lifstorage.h`
   - `src/data/lif/lifconfig.h`
   - `src/hardware/core/hardwaremanager.h`
   - `src/hardware/core/runtimehardwareconfig.h`

   For each header, pick **Pattern A** (default), **Pattern B**, or
   **Pattern C** based on how its keys are consumed:
   - Pattern A if consumers take `QString` / `const QString &` and
     are not being migrated.
   - Pattern B if the container or consumer API is migrating to
     `QAnyStringView` / heterogeneous lookup in the same change
     (ASCII keys only).
   - Pattern C for non-ASCII keys in the same situation.

3. **Redesign `LogHandler` as a thread-safe global singleton with a
   `QAnyStringView` API.** Introduce
   `LogHandler::instance().log(...)` and `bcLog(...)`. Keep the
   existing `emit logMessage()` signal in place as a shim that
   forwards to the new API, so the existing ~445 call sites keep
   compiling unchanged. This is the integration point between the
   two projects.

4. **Migrate `logMessage` call sites to `bcLog`.** Bulk mechanical
   pass across all ~445 sites. Where the call site uses a literal,
   adopt `"..."_s` at the same time. Once this pass completes,
   remove the forwarding `emit logMessage()` shim and the signal
   connection scaffolding from `HardwareManager`.

5. **`qDebug()` elimination pass.** With the global logger in place
   and the signal cascade removed, every `qDebug()` call site can
   call `bcLog(..., LogHandler::Debug)` directly regardless of
   context. Straightforward mechanical replacement or deletion.

6. **`HardwareManager` severity reclassification.** Bulk Normal →
   Debug, keeping Error/Warning as-is and keeping user-facing
   connection-test Normal messages.

7. **FTMW digitizer message triage.** Per-file, per-message review
   of the ~285 calls. Most labor-intensive step; separate commits
   per file make review tractable.

8. **Quick pass on remaining logging files.** Communication
   protocols, optional hardware, LIF components, Highlight usage.

9. **Fix remaining by-value `QString` parameters.**
   `CommunicationProtocol::writeCmd`,
   `CommunicationProtocol::queryCmd`, `SettingsStorage::get` and
   peers. `const QString &` is the conservative fix;
   `QAnyStringView` is the aggressive one. Match the choice to what
   the callee actually does with the parameter. Virtual cascades
   (see above) must be updated in a single commit per base class.

10. **Retrofit `std::less<>` on `std::map<QString, T>`** where
    pattern B or C keys land, or where signature migrations create
    non-`QString` lookup callers. A codebase-wide sweep is optional.

11. **Long-tail signature migration to `QAnyStringView`.** Only as
    specific APIs are touched for other reasons. Not a sweep.

12. **Final review.** Read through the log output of a typical
    startup + experiment cycle to verify the user sees a clean,
    informative log without noise. This is the pre-release
    checkpoint before documentation revision for 2.0.0.

## Open Questions

- Are there digitizer error messages that hardware vendors or
  support staff rely on seeing? If so, those should stay at Error
  even if they look diagnostic.
- Exact threading model for the `LogHandler` singleton: internal
  mutex with direct writes, or queued dispatch to the UI thread via
  `QMetaObject::invokeMethod`? Both are viable; pick during step 3.

## Verification

Any change informed by this document should be validated with a
debug build and the existing test suite:

```
cmake . -B build/Desktop-Debug/
make -C build/Desktop-Debug/ -j$(nproc)
cmake . -B build/tests
make -C build/tests tests -j$(nproc)
ctest --test-dir build/tests
```

Settings-adjacent tests (`tst_settingsstoragetest`,
`tst_headerstoragetest`) are the primary regression surface for key
declaration changes. For logging changes, the regression surface is
manual: start the application, run a typical experiment cycle, and
review the log tab output for noise or missing information.
