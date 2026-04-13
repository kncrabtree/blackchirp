# String Usage

Policy and migration guidance for `QString`, string literals, and
string-keyed containers in Blackchirp.

## Context and Motivation

Blackchirp uses `QString` pervasively, but the current patterns predate Qt 6
idioms and waste both startup time and memory. A walk of the codebase finds:

- **Central key headers use `static const QString` at namespace scope**:
  - `src/data/settings/hardwarekeys.h` — 127 declarations across nested
    namespaces (`HW`, `Comm`, `RS232`, `TCP`, `GPIB`, `FtmwScope`, `Digi`,
    `Clock`, `AWG`, `PGen`, `Flow`, `PController`, `TC`, …).
  - `src/data/bcglobals.h` — a handful more in `BC::Key` and `BC::Unit`.
  - `src/data/storage/settingsstorage.h` — ~8 more near lines 23–32.
  - Per-subsystem: `src/data/experiment/auxdatakeys.h`,
    `src/data/lif/lifstorage.h`, `src/data/lif/lifconfig.h`,
    `src/hardware/core/hardwaremanager.h`,
    `src/hardware/core/runtimehardwareconfig.h`.

  `static` at namespace scope in a header gives the declaration **internal
  linkage**, so every translation unit that includes the header gets its own
  copy of each `QString`. With ~130 keys in `hardwarekeys.h` and dozens of
  hardware translation units, that is a meaningful multiplier on both
  heap-allocation cost at startup and binary bloat. The cost is per-TU, not
  per-process.

- **Beyond the central headers, ~747 `static const QString` declarations
  exist across ~105 files**, roughly 40% in headers and the rest in .cpp
  files. Migration scope is the whole codebase, not just the key headers.

- **String literal style is mid-migration without a policy**: ~1432
  occurrences of `"..."_s` (Qt::StringLiterals) coexist with ~559
  occurrences of `QStringLiteral(...)`. Zero header-scope `inline` or
  `constexpr` string constants exist — every `_s` usage is local inside a
  function body.

- **~282 `std::map<QString, T>` sites exist**, none using `std::less<>`
  for heterogeneous lookup.

- **By-value `QString` parameters are common in hot-call APIs**:
  - `LogHandler::logMessage(const QString text, ...)` —
    `src/data/loghandler.h:34`, ~445 call sites.
  - `CommunicationProtocol::writeCmd(QString cmd)` and
    `queryCmd(QString cmd, bool)` —
    `src/hardware/core/communication/communicationprotocol.h:93,119`.
  - `SettingsStorage::get(const QString key, ...)` and its overloads —
    `src/data/storage/settingsstorage.h:332` and peers.

  `const QString` taken by value (not by reference) is strictly worse than
  `const QString &`: it bumps the implicit-sharing refcount on every call
  with no upside.

### What is *not* the motivation

- **Hot-path data flow is already QString-free.** `FtmwScope::emitShot()`
  and its callers operate on `QVector<qint64>` waveforms. Settings are
  read once at construction and in `hwPrepareForExperiment()`, never in
  per-shot loops. String-level optimization does not affect acquisition
  throughput.
- **Translation / i18n is not a concern.** There are no `.ts` / `.qm`
  files, no `QT_TR_NOOP`, and no translation pipeline. The ~59 `tr()` calls
  are inherited from `QObject` with no translations behind them. i18n is
  not on the roadmap; see "Out of Scope" below.

The motivation is therefore **startup cost, binary size, API hygiene, and
consistency** — not hot-path throughput and not i18n compatibility.

## House Style for String Literals

For new code:

- **Use `"..."_s`** from `Qt::StringLiterals` for `QString` literals. This
  is the house style going forward and is already the dominant form in the
  codebase.
- **Use `u"..."_s`** (UTF-16 prefix) for literals containing non-ASCII
  characters. Example: `BC::Unit::us` currently uses
  `QString::fromUtf8("μs")`; its modern form is `u"μs"_s`.
- **`QStringLiteral(...)` is not used in new code.** Existing sites are
  left alone — a mechanical sweep is possible but not required.
- **`"..."_L1`** produces a `QLatin1StringView` for ASCII-only constants
  where a non-`QString` view is desired (see Pattern B under Key
  Declaration).

The C++23 standard (`CMAKE_CXX_STANDARD 23` in `CMakeLists.txt`) and Qt 6
minimum guarantee that `_s`, `_L1`, `QAnyStringView`, and
`inline constexpr QStringView` are all available.

## Key Declaration Idiom

Three portable patterns are documented here. The implementation task picks
one per header based on how the keys are consumed. **No single pattern
wins in all cases.**

All three replace the current `static const QString k{"key"};` idiom, and
all three eliminate the per-TU duplication.

### Pattern A — `inline const QString`

```cpp
inline const QString trigCh = u"trigCh"_s;
```

- ODR-safe via `inline`; single definition across all translation units.
- Not `constexpr` — `QString` is not a C++ literal type — but the `_s`
  literal stores the UTF-16 data in `.rodata`, and the `QString`
  constructor performs no heap allocation.
- One constructor call per process, not per TU.
- Stores directly into `std::map<QString, T>` and `QHash<QString, T>`
  with zero conversion.
- Best when the consumers of these keys already take `QString` /
  `const QString &` and signatures are not changing in the same pass.

This is the lowest-risk migration target for the existing key headers.

### Pattern B — `inline constexpr QLatin1StringView`

```cpp
inline constexpr QLatin1StringView trigCh = "trigCh"_L1;
```

- True `constexpr`; zero runtime cost at the constant's own expense.
- Consumers must accept `QAnyStringView`, or the containing container
  must use heterogeneous lookup
  (`std::map<QString, T, std::less<>>`), or the call site must pay the
  `QString` construction that pattern A avoids.
- Best when the consumer API is migrating to `QAnyStringView` in the same
  change — the container / signature migration and the key declaration
  migration land together.
- ASCII-only. Use pattern C for non-ASCII keys.

### Pattern C — `inline constexpr QStringView`

```cpp
inline constexpr QStringView us = u"μs";
```

- Same tradeoff as pattern B but UTF-16, so non-ASCII keys are safe.
- Useful for `BC::Unit` and any other namespace with non-ASCII content.

### Do not use

```cpp
// Does NOT compile portably — QString is not a literal type.
inline constexpr auto trigCh = "trigCh"_s;
```

`QString`'s internal `QArrayDataPointer` has a non-trivial destructor and
non-`constexpr` members, so `constexpr QString` is not valid in the Qt
versions this project targets. `constexpr auto` with `_s` deduces to
`QString` and fails for the same reason. Guidance that recommends this
pattern predates the Qt/C++ reality — do not use it in Blackchirp.

## Function Signature Policy

Rules of thumb, in order of preference:

1. **Never pass `QString` by value** unless the callee is going to take
   ownership and move. The current offenders are listed in
   [Context and Motivation](#context-and-motivation). All should be
   corrected.
2. **`const QString &`** is the conservative default for parameters when
   the callee needs a `QString` (to pass to another `QString` API, to
   store, or to do `QString`-specific operations like `arg()`).
3. **`QAnyStringView`** is appropriate when the function is a pure
   lookup, comparison, or passthrough — it lets callers pass `QString`,
   `QStringView`, `QLatin1StringView`, `const char *`, or a `"..."_s`
   literal without any temporary `QString`. Good candidates:
   - `SettingsStorage::get` / `set` / `containsValue`
   - `HeaderStorage` lookup methods
   - `HardwareObject::validationKeys` consumers
4. **`QStringView`** when you need `constexpr`-eligibility on the
   parameter and the function is truly view-only. Less useful than
   `QAnyStringView` in Blackchirp because of the Latin-1 /  `const char *`
   paths.

### Priority

Signature migration is **deprioritized relative to key consolidation**.
Blackchirp's hot paths are QString-free, so the performance upside is
modest. The real upside is API hygiene: fixing the by-value parameters
and removing temporary `QString` allocations at call sites. Do this
incrementally as specific APIs are touched for other reasons, not as a
codebase-wide sweep.

## Virtual Cascade Hotspots

Any signature change on a virtual function forces every override to
change in the same commit. The following base classes are the
coordination points — scope any signature migration accordingly:

- **`CommunicationProtocol::writeCmd` / `queryCmd`** —
  `src/hardware/core/communication/communicationprotocol.h:93,119`. Taken
  by value today. Overridden by `Rs232Instrument`, `TcpInstrument`,
  `GpibInstrument`, `CustomInstrument`, `VirtualInstrument`, and the
  Python-hardware wrappers (~8 overrides total). Fixing these to
  `const QString &` is a straightforward cleanup; going further to
  `QAnyStringView` requires coordinated subclass edits.
- **`HardwareObject` QString/QStringList virtuals** —
  `src/hardware/core/hardwareobject.h:232,404` and peers
  (`validationKeys`, `forbiddenKeys`, …). ~100 subclass overrides across
  `src/hardware/core/` and `src/hardware/optional/`. Any signature
  migration here is a heavy coordinated refactor.
- **`DataStorageBase`** —
  `src/data/storage/datastoragebase.h:18`. Constructor takes
  `QString path = ""`.
- **`FileParser::canParse`** —
  `src/data/processing/parsers/fileparser.h:26`. Takes `const QString &`,
  already reasonable.

Migrations touching virtual functions should be done as single commits
per base class, with all overrides updated together.

## Container Policy

- **`std::map<QString, T, std::less<>>`** is the default declaration for
  new maps. The `std::less<>` transparent comparator enables
  heterogeneous lookup, so callers can `find("..."_s)`,
  `find(QStringView(...))`, or `find(QLatin1StringView(...))` without
  allocating a temporary `QString`.
- Blanket retrofitting `std::less<>` onto the ~282 existing
  `std::map<QString, T>` declarations is **cheap and safe**, but the
  benefit is *deferred* — it only materializes once lookups start
  passing non-`QString` keys. Practically, the retrofit is worth doing
  wherever Pattern B / C keys or `QAnyStringView` signatures are being
  rolled out in the same change.
- **`SettingsStorage::SettingsMap`** — `src/data/storage/settingsstorage.h`
  line 24 (`using SettingsMap = std::map<QString, QVariant>;`) is the
  central typedef for the class of map most affected by this policy.
  Updating it updates all downstream sites uniformly.
- **`QHash<QString, T>`** is rare in Blackchirp (~4 sites). Qt 6 supports
  `QHash` heterogeneous lookup via `qHash(QStringView)`; no special
  declaration is needed.

## Out of Scope

- **Internationalization / `tr()` / `.ts` files.** Blackchirp has no
  translation infrastructure and none is planned. Existing `tr()` calls
  are cosmetic. If i18n is ever added, UI strings will need a separate
  audit; this document does not reserve the `tr()` path and does not
  introduce `QT_TR_NOOP`.
- **Hot-path optimization of digitizer data flow.** FTMW acquisition,
  accumulation, and shot handling do not touch `QString`. See the
  [Digitizer Data Flow Optimization](digitizer-data-flow.md) task for
  the relevant concerns in that area.
- **`QStringLiteral(...)` mass rewrite.** Existing occurrences can stay.
  The policy applies to new code and to sites that are already being
  edited for another reason.

## Migration Sequencing

When the implementation task is picked up, follow this order. Each step
is independently valuable and can be committed on its own.

1. **Standardize new code on `"..."_s`.** No sweep; just policy.
2. **Consolidate and migrate central key headers.** Target list:
   - `src/data/settings/hardwarekeys.h`
   - `src/data/bcglobals.h`
   - `src/data/storage/settingsstorage.h` (the top-of-file constants)
   - `src/data/experiment/auxdatakeys.h`
   - `src/data/lif/lifstorage.h`
   - `src/data/lif/lifconfig.h`
   - `src/hardware/core/hardwaremanager.h`
   - `src/hardware/core/runtimehardwareconfig.h`

   For each header, pick **Pattern A** (default), **Pattern B**, or
   **Pattern C** based on how its keys are consumed:
   - Pattern A if consumers take `QString` / `const QString &` and are
     not being migrated.
   - Pattern B if the container or consumer API is migrating to
     `QAnyStringView` / heterogeneous lookup in the same change (ASCII
     keys only).
   - Pattern C for non-ASCII keys in the same situation.
3. **Fix by-value `QString` parameters.** Minimum target:
   `LogHandler::logMessage`, `CommunicationProtocol::writeCmd`,
   `CommunicationProtocol::queryCmd`, `SettingsStorage::get` and peers.
   `const QString &` is the conservative fix; `QAnyStringView` is the
   aggressive one. Match the choice to what the callee actually does
   with the parameter.
4. **Retrofit `std::less<>` on `std::map<QString, T>`** where pattern B
   or C keys land, or where signature migrations create
   non-`QString` lookup callers. A codebase-wide sweep is optional.
5. **Long-tail signature migration to `QAnyStringView`.** Only as
   specific APIs are touched for other reasons. Not a sweep.

## Verification

Any change informed by this document should be validated with a debug
build and the existing test suite:

```
cmake . -B build/Desktop-Debug/
make -C build/Desktop-Debug/ -j$(nproc)
cmake . -B build/tests
make -C build/tests tests -j$(nproc)
ctest --test-dir build/tests
```

Settings-adjacent tests (`tst_settingsstoragetest`,
`tst_headerstoragetest`) are the primary regression surface for key
declaration changes.
