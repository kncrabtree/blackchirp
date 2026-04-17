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
   lookup, comparison, or passthrough — it accepts `QString`,
   `QStringView`, `QLatin1StringView`, and `const char *` directly.
   At call sites with string literals, use **`"..."_L1`** for ASCII
   content and **`u"..."_s`** for non-ASCII content; both reach the
   parameter as a zero-allocation view. Do *not* use `"..."_s` at
   `QAnyStringView` call sites — `"..."_s` constructs a temporary
   `QString`, defeating the purpose of the `QAnyStringView` parameter.
   Good candidates:
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

1. ✅ **Standardize new code on `"..."_s`.** No sweep; just policy.
   This is the baseline for every subsequent step.
   *Completed: policy documented in `CLAUDE.md`.*

2. ✅ **Consolidate and migrate all key headers, coordinated with API
   migration of the primary consumer classes.**

   **Consumer API migration (prerequisite for Pattern B/C):**
   The three classes that consume nearly all string keys are
   `SettingsStorage`, `HeaderStorage`, and `BlackchirpCSV`. Migrating
   them unlocks Pattern B/C for the key declarations:
   - **`SettingsStorage`**: change `SettingsMap` typedef to
     `std::map<QString, QVariant, std::less<>>` and migrate all
     key-taking methods (`get`, `set`, `getOrSetDefault`, `getArray`,
     `getArrayValue`, `getArrayMap`, `getArraySize`, `getMultiple`,
     `getGroup`, `containsValue`, `registerGetter`, `registerSetter`)
     to accept `QAnyStringView` for the key parameter. No virtual
     cascade — none of these are virtual.
   - **`HeaderStorage`**: change `HeaderMap` typedef to
     `std::map<QString, ValueUnit, std::less<>>` and migrate
     `store`, `storeArrayValue`, `retrieve`, `retrieveArrayValue`,
     `arrayStoreSize` to accept `QAnyStringView` for key and unit
     parameters.
   - **`BlackchirpCSV`**: no API surface needs changing — its
     constants are used directly with Qt string operations
     (`QTextStream <<`, `QString::split`, etc.) that already accept
     `QLatin1StringView`; Pattern B works without any API change.
   - **`BC::Key` functions in `bcglobals.cpp`** (`hwKey`, `parseKey`,
     `parseIndexKey`, `widgetKey`, `migrateIndexKey`,
     `generateDefaultLabel`): migrate `const QString &` parameters to
     `QAnyStringView` where the body only reads the string (lookup,
     comparison, `split`, `arg`). Return types stay `QString`.

   **Key declaration migration:**
   Apply to *all* headers with `static const QString` key declarations
   (a full list is derivable from `grep -r "static const QString" src/`):

   - **Pattern B** (`inline constexpr QLatin1StringView k = "..."_L1`)
     for all ASCII keys whose consumers have been migrated above.
     This covers the original eight headers plus all data experiment
     configs (`chirpconfig.h`, `ftmwconfig.h`, `digitizerconfig.h`,
     `rfconfig.h`, `ftmwconfigtypes.h`, `flowconfig.h`,
     `pulsegenconfig.h`, `pressurecontrollerconfig.h`,
     `temperaturecontrollerconfig.h`, `ioboardconfig.h`,
     `lifdigitizerconfig.h`, `markertablemodel.h`, `validationmodel.h`,
     `clocktablemodel.h`, `chirptablemodel.h`, `fidstoragebase.h`,
     `curveappearance.h`, `overlaystorage.h`,
     `applicationconfigmanager.h`), hardware headers
     (`hardwareprofilemanager.h`, `vendorlibrary.h`, `liflaser.h`,
     `sirahcobra.h`, `fixedclock.h`, and implementation-specific
     headers in `src/hardware/optional/` and `src/hardware/core/`),
     and GUI headers (widgets, dialogs, plots, expsetup pages in
     `src/gui/`).
   - **Pattern C** (`inline constexpr QStringView k = u"..."`) for
     non-ASCII keys. The only known case is `BC::Unit::us`; its value
     changes from `QString::fromUtf8("μs")` to `u"μs"`.
   - **Pattern A** (`inline const QString k = "..."_s`) for any key
     on which `QString`-specific member functions are called directly
     (e.g., `.arg()`). Known exceptions:
     - `BC::Aux::Flow::flow` (`"Flow%1"`) — `.arg(i+1)` at call site
     - `AuxDataStorage::keyTemplate` (`"%1.%2"`) — `.arg()` at call
       site
     - `TemperatureController::temperature` (`"Temperature%1"`) —
       `.arg()` at call site
     If any other template strings are found during the sweep, apply
     Pattern A and note them.
   - For **class-member** statics (e.g., in `RuntimeHardwareConfig`,
     `HardwareProfileManager`, `ApplicationConfigManager`,
     `VendorLibrary`, `SpectrumLibrary`, `LabJackLibrary`): the
     syntax is `inline static constexpr QLatin1StringView k = "..."_L1`
     (Pattern B) rather than namespace-scope `inline constexpr`.

   **Note on Steps 9–10:** The `SettingsStorage` and `HeaderStorage`
   API migrations above subsume the SettingsStorage portion of Step 9
   and the map-typedef portion of Step 10. Step 9 retains only the
   `CommunicationProtocol` virtual cascade fixes; Step 10 retains only
   the optional codebase-wide sweep of remaining `std::map<QString, T>`
   sites not touched here.

   *Completed: all consumer APIs migrated; 722 `static const QString`
   declarations across 102 headers converted. Implementation notes:*
   - *Pattern B uses constructor form `{"..."}` rather than `"..."_L1`
     to avoid requiring `using namespace Qt::Literals::StringLiterals`
     in headers.*
   - *`BC::Unit::us` is Pattern C (`inline constexpr QStringView`).*
   - *Pattern A exceptions confirmed: `BC::Aux::Flow::flow`,
     `BC::Aux::keyTemplate`, `TemperatureController::temperature`.*
   - *One call-site fix required: `exptsummarymodel.cpp` uses
     `BC::Unit::us.toString()` where a `QStringList` initializer
     required `QString`, not `QStringView`.*
   - *Hardware AWG/pulse-generator display-name strings (e.g.,
     `"Arbitrary Waveform Generator AWG70002A"`) were kept as
     `inline const QString` rather than `QLatin1StringView`; these
     are correct but suboptimal — can be swept in Step 10.*

3. ✅ **Redesign `LogHandler` as a thread-safe global singleton with a
   `QAnyStringView` API.** Introduce
   `LogHandler::instance().log(...)` and `bcLog(...)`. Keep the
   existing `emit logMessage()` signal in place as a shim that
   forwards to the new API, so the existing ~445 call sites keep
   compiling unchanged. This is the integration point between the
   two projects. Add `bcWarn`, `bcError`, `bcDebug`, and `bcHighlight`
   convenience methods. `bcLog` can retain the second argument 
   (defaults to LogHandler::Normal) for use in situations in which
   the MessageCode is conditional.

   *Completed: `LogHandler` is now a singleton accessed via
   `instance()`. `log(QAnyStringView, MessageCode)` is the primary
   API. `std::atomic` members (`d_currentExperimentNum`,
   `d_logToFile`, `d_debugLogging`) provide thread-safe reads from
   worker threads; `QMutex d_fileMutex` serializes file I/O.
   `logMessage` and `logMessageWithTime` slots forward to the private
   `doLog()` helper, preserving all existing signal connections
   without change. `MainWindow` now uses `&LogHandler::instance()`
   instead of `new LogHandler`. `ExperimentViewWidget` continues to
   create a local `LogHandler(false, parent)` for log replay — its
   constructor remains public for this purpose.*

4. ✅ **Migrate `logMessage` call sites to `bcLog`, `bcWarn`, etc.**
   Bulk mechanical pass across all ~445 sites. Where the call site
   uses a string literal, use `"..."_L1` for ASCII content and
   `u"..."_s` for non-ASCII content — not `"..."_s`, which would
   construct a temporary `QString` against the intent of the
   `QAnyStringView` parameter. Once this pass
   completes, remove the forwarding `emit logMessage()` shim and
   the signal connection scaffolding from `HardwareManager`.

   *In progress: all call sites outside `src/hardware/` have been
   migrated (9 calls across `gui/mainwindow.cpp`,
   `acquisition/acquisitionmanager.cpp`,
   `acquisition/batch/batchmanager.cpp`). All remaining
   `emit logMessage` calls are in files under `src/hardware/` and
   are covered by steps 6–8.*

5. ✅ **`qDebug()` elimination pass.** With the global logger in place
   and the signal cascade removed, every `qDebug()` call site can
   call `bcDebug` directly regardless of context. Straightforward
   mechanical replacement or deletion.

6. **Hardware severity reclassification and `emit logMessage`
   migration.** All `src/hardware/` files except the FTMW digitizers
   (deferred to step 7). Migrates every `emit logMessage` call to the
   appropriate `bcLog`/`bcWarn`/`bcError`/`bcDebug` free function and
   applies the severity policy below. Once all files in this step are
   complete, the `logMessage` shim slot and its forwarding signal
   connections are removed (commit H).

   #### Severity policy

   **Error** — operation failed; hardware state may be unknown.
   - Hardware read/write failures ("Could not read X", "No response
     to Y query")
   - Invalid parameters that prevent the operation from completing
   - Communication timeout or port-closed failures

   **Warning** — unexpected but recoverable; no data loss.
   - Parameter clamped to nearest valid hardware value
   - Command echo not received
   - Unsupported device configuration detected (operation continues
     with fallback)
   - Virtual instrument in use (user needs to know)

   **Normal** — significant event worth a log timestamp; few per
   session during routine use.
   - Hardware connected or failed (connection test outcome)
   - Hardware added or removed from configuration
   - Top-level operation completions (sync complete, library apply
     summary)
   - Results of explicit user actions (Python script reloaded)

   **Debug** — technical detail; shown only when debug logging is
   enabled.
   - ID response strings emitted during `testConnection()`
   - Internal lifecycle sub-steps (threads started, GPIB controllers
     resolved, etc.)
   - Raw command/response content and hex dumps
   - Convergence loop internal diagnostics

   #### Split pattern for error + diagnostic messages

   When a single call combines a user-facing error with a raw
   technical detail (response bytes, hex), split it into two calls:

   ```cpp
   // Before
   emit logMessage(QString("Could not read %1 frequency. Response: %2 (Hex: %3)")
                   .arg(ch).arg(resp).arg(resp.toHex()), LogHandler::Error);

   // After
   bcError(u"Could not read %1 frequency."_s.arg(ch));
   bcDebug(u"%1: Could not read %2 frequency. Response = %3 (Hex: %4)"_s
           .arg(d_key, ch, resp, resp.toHex()));
   ```

   The debug line must be self-contained: the debug log file contains
   only Debug-level messages, so a reader cannot cross-reference it
   against the normal log. The debug line therefore repeats the
   user-facing error text *and* appends the technical detail, prefixed
   with the device/object identifier (`d_key` for communication
   protocol and hardware object subclasses).

   For already-separate supplementary calls (e.g., a Normal hex dump
   that immediately follows an Error), merge the error text and the
   supplementary detail into a single bcDebug call using the same
   pattern, and remove the now-redundant supplementary call.

   #### Commented-out calls

   Update the severity in the commented text to the appropriate
   level (usually Debug) but leave the call commented out.

   #### `HardwareObject` subclass call sites

   `HardwareObject` provides four protected inline helpers that
   prepend `d_key` to the message automatically:

   ```cpp
   hwLog(text)   // Normal — replaces bcLog   at HardwareObject call sites
   hwWarn(text)  // Warning — replaces bcWarn
   hwError(text) // Error   — replaces bcError
   hwDebug(text) // Debug   — replaces bcDebug
   ```

   Use these instead of the bare `bc*` free functions inside any
   class that inherits `HardwareObject` (Clock subclasses, AWG,
   pulse generators, etc.).  `ClockManager` is **not** a
   `HardwareObject`; it continues to use the bare `bc*` free
   functions directly.

   #### String literal form at call sites

   - Pure literal (no `.arg()`): `"..."_L1` — zero allocation at the
     call site.
   - Formatted literal (with `.arg()`): `u"..."_s.arg(...)`.
   - Existing `QStringLiteral(...)` at a site being edited: update
     to `"..."_L1` or `u"..."_s.arg(...)` as appropriate.
   - `using namespace Qt::StringLiterals` is available at all call
     sites via the `loghandler.h` include — no extra `using`
     declaration is needed.

   #### File checklist

   Commits are grouped logically; each group is one commit. Check off
   files as they are completed.

   **Commit A — Communication layer**
   - [x] `src/hardware/core/communication/communicationprotocol.cpp`
   - [x] `src/hardware/core/communication/gpibinstrument.cpp`
   - [x] `src/hardware/core/communication/tcpinstrument.cpp`

   **Commit B — Core hardware objects**
   - [x] `src/hardware/core/hardwareobject.cpp`
   - [x] `src/hardware/core/hardwaremanager.cpp`

   **Commit C — Clock subsystem**
   - [x] `src/hardware/core/clock/clock.cpp`
   - [x] `src/hardware/core/clock/clockmanager.cpp`
   - [x] `src/hardware/core/clock/hp83712b.cpp`
   - [x] `src/hardware/core/clock/valon5009.cpp`
   - [x] `src/hardware/core/clock/valon5015.cpp`

   **Commit D — LIF hardware**
   - [x] `src/hardware/core/lifdigitizer/m4i2211x8.cpp`
   - [x] `src/hardware/core/lifdigitizer/rigolds2302a.cpp`
   - [x] `src/hardware/core/liflaser/liflaser.cpp`
   - [x] `src/hardware/core/liflaser/opolette.cpp`
   - [x] `src/hardware/core/liflaser/sirahcobra.cpp`

   **Commit E — AWG/RF sources, GPIB controller, I/O, pressure,
   temperature**
   - [x] `src/hardware/optional/chirpsource/ad9914.cpp`
   - [x] `src/hardware/optional/chirpsource/awg5204.cpp`
   - [x] `src/hardware/optional/chirpsource/awg70002a.cpp`
   - [x] `src/hardware/optional/chirpsource/awg7122b.cpp`
   - [x] `src/hardware/optional/chirpsource/m8190.cpp`
   - [x] `src/hardware/optional/chirpsource/m8195a.cpp`
   - [x] `src/hardware/optional/gpibcontroller/prologixgpibcontroller.cpp`
   - [x] `src/hardware/optional/ioboard/labjacku3.cpp`
   - [x] `src/hardware/optional/pressurecontroller/intellisysiqplus.cpp`
   - [x] `src/hardware/optional/tempcontroller/lakeshore218.cpp`

   **Commit F — Flow controllers and pulse generators**
   - [ ] `src/hardware/optional/flowcontroller/flowcontroller.cpp`
   - [ ] `src/hardware/optional/flowcontroller/mks647c.cpp`
   - [ ] `src/hardware/optional/flowcontroller/mks946.cpp`
   - [ ] `src/hardware/optional/pulsegenerator/bnc577.cpp`
   - [ ] `src/hardware/optional/pulsegenerator/pulsegenerator.cpp`
   - [ ] `src/hardware/optional/pulsegenerator/qc9210series.cpp`
   - [ ] `src/hardware/optional/pulsegenerator/qc9510series.cpp`
   - [ ] `src/hardware/optional/pulsegenerator/qc9520series.cpp`
   - [ ] `src/hardware/optional/pulsegenerator/qcpulsegenerator.cpp`
   - [ ] `src/hardware/optional/pulsegenerator/srsdg645.cpp`

   **Commit G — Python hardware**
   - [ ] `src/hardware/python/pythonawg.cpp`
   - [ ] `src/hardware/python/pythonftmwscope.cpp`
   - [ ] `src/hardware/python/pythonioboard.cpp`
   - [ ] `src/hardware/python/pythonlifscope.cpp`
   - [ ] `src/hardware/python/pythonprocess.cpp`

   **Commit H — Shim removal** (after all files above are complete)
   - [ ] Remove `logMessage` slot and shim body from `HardwareObject`
   - [ ] Remove `logMessage` signal from `HardwareObject`
   - [ ] Remove `logMessage` forwarding connections from
         `HardwareManager`

7. **FTMW digitizer message triage.** Per-file, per-message review
   of the ~285 calls. Most labor-intensive step; separate commits
   per file make review tractable.

8. **Fix remaining by-value `QString` parameters.**
   `CommunicationProtocol::writeCmd` and `queryCmd` are the primary
   remaining targets (`SettingsStorage` and `HeaderStorage` are
   handled in step 2). `const QString &` is the conservative fix;
   `QAnyStringView` is the aggressive one. Match the choice to what
   the callee actually does with the parameter. Virtual cascades
   (see [Virtual Cascade Hotspots](#virtual-cascade-hotspots)) must
   be updated in a single commit per base class.

9. **Retrofit `std::less<>` on remaining `std::map<QString, T>`**
   sites not already updated in step 2. The `SettingsStorage` and
   `HeaderStorage` maps are handled there. A codebase-wide sweep of
   remaining sites is optional but cheap.

10. **Long-tail signature migration to `QAnyStringView`.** Only as
    specific APIs are touched for other reasons. Not a sweep.

11. **Final review.** Read through the log output of a typical
    startup + experiment cycle to verify the user sees a clean,
    informative log without noise. This is the pre-release
    checkpoint before documentation revision for 2.0.0.

## Open Questions

- Are there digitizer error messages that hardware vendors or
  support staff rely on seeing? If so, those should stay at Error
  even if they look diagnostic.

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
