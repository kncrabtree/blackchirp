# Bundle 01 — C++ Enum-String Migration and Reader Hardening

**Status:** in progress
**Depends on:** —
**Blocks:** 03, 04, 05
**Effort:** M (2 sessions)

## Status notes

Decisions captured before implementation:

- **`hardware.csv` `hardwareType` column → drop.** The cell is fully
  redundant with the key prefix. `RuntimeHardwareConfig::createHardware
  DataContainer` derives the type from the key prefix when populating
  the writer-side container, no downstream consumer reads the stored
  type field independently of the key, and the legacy 2-column reader
  path already proves the key prefix alone is enough. The reader keeps
  accepting historical 3-column files (silently ignoring the third
  column) and three on-disk key shapes: bare `"hwType"` (oldest),
  `"hwType.index"` (mid-life), and `"hwType.label"` (current). Dropping
  the column lets us also remove `toCsvTuples()` /
  `fromCsvTuples()` and the `isNewFormat` branch in `loadFromFile`.
- **`subKey` → `driver` rename.** Verified positional in the C++
  reader; `liflasercontroldoublespinbox.cpp:16` and two
  `tst_settingsstoragetest.cpp` matches are unrelated QSettings keys
  and have been logged as latent-bug follow-ups in
  `dev-docs/devel-roadmap.md` for a separate session.
- **Overlay storage writer flip.** The audit turned up
  `overlaystorage.cpp:111` writing
  `static_cast<int>(overlay->type())`. Pulled into scope under the
  bundle's "fix any inconsistent writer the audit turns up" provision.
- **`CatalogOverlay::LineshapeType` promotion.** Surfaced by a
  follow-up scan of the data-storage user-guide page: the
  `catalogLineshapeType` cell in overlay metadata files was being
  written as a raw integer. The enum is now `Q_ENUM`-registered (with
  `Q_GADGET` on `CatalogOverlay`); writer wraps with
  `QVariant::fromValue`, reader hardened with `enumFromVariant`. The
  sibling `GenericXYOverlay` also gains an explicit `Q_GADGET` so its
  pre-existing `Q_ENUM(DelimiterType)` continues to moc cleanly.
- **Dual-form helper.** Added as `BlackchirpCSV::enumFromVariant<E>`
  (free function in `blackchirpcsv.h`). `HeaderStorage::retrieve<T>`
  and `retrieveArrayValue<T>` route through it for `Q_ENUM`-bearing
  types via `if constexpr`, so existing call sites get backward-
  compatible numeric-string parsing without per-site changes.

## Why this is a prerequisite

The Python module refresh assumes that every enum cell in a Blackchirp
CSV file can be parsed by a single name-or-int helper. That invariant
only holds if the C++ writer is consistent (we know which form to
expect) and the C++ reader is defensive (Blackchirp can read its own
prior output after the writer flips). This bundle establishes both
sides of that invariant before any Python work assumes it.

It also addresses two specific writers that are currently inconsistent
with the rest of the codebase: `FtUnits` in `processing.csv` (still a
raw integer) and the `hardwareType` column in `hardware.csv` (also a
raw integer, and possibly redundant — see decision point below).

## Scope

1. Decide the fate of `hardware.csv`'s `hardwareType` column (see the
   decision point below) and implement the chosen path. In every
   path, the C++ reader must remain defensive against the historical
   numeric form so that older fixtures still load.
2. Rename the `hardware.csv` second column header from `subKey` to
   `driver`. Verify no Blackchirp code parses the header row by name
   (the column is positional in the current writer/reader); fix any
   call sites the verification turns up.
3. Switch `FidStorageBase::writeProcessingSettings` to wrap `FtUnits`
   with `QVariant::fromValue(e)` so it serializes as the enum name.
4. Audit every C++ read site that consumes a Blackchirp-written CSV
   cell whose value originates from a `Q_ENUM` field, and confirm it
   accepts both numeric and name forms. Where it does not, fix it.
5. Add a tiny shared helper (or document the canonical idiom) so that
   future enum reads use one consistent dual-form parse.
6. Add unit-test coverage for the dual-form read path.

## Decision point: `hardware.csv` `hardwareType` column

The `hardwareType` column was added for programmatic convenience: it
encodes the numeric value of the driver-type enum so the loader can
route settings without having to map the `driver` (formerly `subKey`)
column back to an enum value. The bundle owner chooses one of:

- **Drop the column.** Cleanest long-term result if the `driver`
  column is sufficient on its own. Writer omits the cell; reader
  silently ignores the column when present in older fixtures.
- **Promote to string-form enum.** Add `Q_ENUM` / `Q_ENUM_NS` to
  `HardwareType`, wrap the writer cell in `QVariant::fromValue`,
  reader accepts numeric or name.
- **Leave numeric and scope-exclude.** Keep the writer as it is;
  ensure the reader is defensive (handles numeric); explicitly carve
  this enum out of the broader string-migration audit and document
  the carve-out here so future audits do not reopen the question.

Pick the path during the audit — the determining question is whether
any code path currently consumes `hardwareType` independently of the
`driver` column. If yes, promote; if no, drop or leave-numeric per
maintainer preference. Document the choice in the bundle's status
header when the decision lands.

## `subKey` → `driver` rename

The second column of `hardware.csv` is currently labeled `subKey`,
which describes how the writer assembles the value (a sub-key
appended to the hardware-type root key) rather than what the cell
contains (the driver class identifier). Rename to `driver`.

Implementation notes:

- The `hardware.csv` writer composes the header row from a string
  literal — change that literal.
- The reader skips the header row (per inspection at the time of
  this writing) and consumes columns positionally. Verify by grep
  for `subKey` across `src/`. If any consumer parses by name, fix it
  to use the new label.
- v1 fixtures (e.g. `python/example-data/mtbe/`) and any existing v2
  fixtures contain `subKey` in their header rows. Reader must accept
  either header label so historical files continue to load. The
  Python module (bundle 03) likewise must accept both header labels.

## Inputs to mine

- `src/data/storage/blackchirpcsv.{h,cpp}` — `writeLine` + `readLine`
  semantics. Confirm that `QVariant::toString()` on a metatype-tagged
  enum returns the name.
- `src/data/storage/fidstoragebase.cpp:276` — `writeProcessingSettings`
  (the `FtUnits` site).
- `src/data/storage/headerstorage.{h,cpp}` — the path that already
  uses `QVariant::fromValue` for header enums (template / canonical
  example for the writer side).
- `hardware.csv` writer — locate via `grep -nR "hwFile\|hardware\.csv"
  src/`.
- `src/data/bcglobals.h` (and any other header defining `HardwareType`)
  — promotion target.
- All `loadFromSettings` / `readLine` / `value<T>()` / `toInt()` /
  `static_cast<EnumT>` call sites in `src/`. Use the
  `codebase-memory-mcp` tools to enumerate.

## C++ writer changes

For each site, the change is the same shape:

```cpp
// before
m.emplace(units, c.units);
// after
m.emplace(units, QVariant::fromValue(c.units));
```

Sites known to need this change:

- `FidStorageBase::writeProcessingSettings` — `FtUnits` field (one
  line).
- `hardware.csv` writer — `hardwareType` column (one line, after
  `Q_ENUM` promotion below).

If the decision is to promote `HardwareType` to `Q_ENUM`, also grep
the codebase for existing writers that emit an unwrapped
`HardwareType` value and fix each. If the decision is to drop or
leave-numeric, no additional writer changes are required for that
enum.

## C++ reader hardening

The audit must cover every enum field that reaches disk. The general
pattern:

```cpp
// canonical dual-form read
template <typename E>
E readEnumField(const QVariant &v, E defaultValue) {
    if (v.canConvert<E>()) return v.value<E>();   // direct metatype hit
    bool ok = false;
    int n = v.toInt(&ok);
    if (ok) return static_cast<E>(n);             // numeric-form fallback
    auto meta = QMetaEnum::fromType<E>();
    auto name = v.toString().toUtf8();
    int idx = meta.keyToValue(name.constData(), &ok);
    if (ok) return static_cast<E>(idx);           // name-form fallback
    return defaultValue;
}
```

Decide during the bundle whether to add this as a free helper in
`src/data/storage/blackchirpcsv.h` or to inline the idiom. Either is
fine; what matters is that every read site is covered.

Audit checklist:

- [ ] Every `loadFromSettings` override that reads a key whose source
      enum has `Q_ENUM` annotated.
- [ ] Every direct `BlackchirpCSV::readLine` consumer in
      `src/data/experiment/`, `src/data/lif/`, `src/data/storage/`.
- [ ] Every place a value is pulled out of `processing.csv`,
      `fidparams.csv`, `lifparams.csv`, `hardware.csv`, `markers.csv`,
      or `clocks.csv`.
- [ ] HeaderStorage round-trip through `QVariant::value<E>()` —
      confirm via test that a header line with the *name* form parses
      back to the enum.
- [ ] HardwareManager / loadout reload — the typed reader currently
      built around the integer must accept the name as well.

## Tests

Add coverage in the existing `tests/` tree:

- `tst_headerstoragetest.cpp` — extend with a round-trip case that
  writes via the *name* form and reads via the typed accessor for at
  least one enum (e.g. `FtUnits`).
- A new dual-form fixture or extension to `tst_blackchirpcsvtest.cpp`
  exercising both numeric and name input strings against the audited
  reader sites.
- Where reasonable, smoke-test by constructing a hand-rolled CSV
  string with the *old* numeric form and confirming the loader still
  parses it (this is the live backward-compat guarantee).

## Files touched

- `src/data/bcglobals.h` (or wherever `HardwareType` is defined) —
  `Q_ENUM`/`Q_ENUM_NS` promotion *only if* the chosen path for
  `hardwareType` is "promote".
- `src/data/storage/fidstoragebase.cpp` — `writeProcessingSettings`.
- The hardware-loadout writer (TBD by the audit) — `hardware.csv`
  header rename (`subKey` → `driver`) and, depending on chosen path,
  either dropping the `hardwareType` cell or wrapping it.
- `src/data/storage/blackchirpcsv.{h,cpp}` — optional shared helper.
- `src/data/experiment/*.cpp`, `src/data/lif/*.cpp`,
  `src/hardware/core/**/*.cpp` — read-site fixes as the audit turns
  them up; any caller that referenced the `subKey` column header by
  name must be updated.
- `tests/tst_headerstoragetest.cpp`, possibly
  `tests/tst_blackchirpcsvtest.cpp` — coverage.

## Acceptance criteria

- All known enum-bearing CSV writers within scope wrap their values in
  `QVariant::fromValue` so `QVariant::toString()` emits the name.
  (Any enum explicitly carved out by the decision point above is
  documented in the bundle's status header along with the rationale.)
- Every audited read site accepts both the numeric and the name form
  without warnings or fallbacks to defaults.
- The `hardware.csv` header row uses `driver` in place of `subKey`.
  Reader accepts either header label so historical files load
  unchanged.
- New unit tests cover the dual-form read for at least one enum from
  each of `processing.csv`, `fidparams.csv`, and `header.csv` (and
  `hardware.csv` if the chosen path involves an enum on that file).
- A v1 fixture (e.g. the `python/example-data/mtbe/` tree) loads
  cleanly via Blackchirp's own loader after the change. This is the
  live regression check for backward compatibility.
- `ctest --test-dir build/tests` passes; `cmake --build
  build/Desktop-Debug/ -j$(nproc)` succeeds with no new warnings.

## Out of scope

- Any Python-side change.
- Migration of additional enum fields beyond `FtUnits` and the
  `hardwareType` decision. (If the audit turns up additional
  inconsistent writers, fix them; do not opportunistically widen the
  migration to fields that already work.)
- Wire-format changes other than the writer-side flip and the
  `subKey` → `driver` header rename.
