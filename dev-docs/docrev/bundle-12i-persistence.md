# Bundle 12i — Developer Guide: Persistence

**Status:** complete

<!--
Status log:
- 2026-05-03: not started → complete. doc/source/developer_guide/persistence.rst
  landed covering the two-layer model, SettingsStorage, the BC::Key/Store/CSV
  namespaces, BlackchirpCSV, the HeaderStorage tree, the DataStorageBase
  lifecycle and its subclasses, the experiment-directory layout, the
  auxiliary on-disk streams (rolling data, log files, text exports — added
  by reference beyond the bundle's stated scope so the persistence chapter
  has a complete picture of every CSV stream Blackchirp writes), and the
  FileParser/FileParserRegistry ecosystem. Corrected the bundle file's
  static-init claim about parser registration (parsers are registered from
  main()). Confirmed overlay annotations live under <experiment>/overlays/.
  No new doc-build warnings or unresolved cross-references. Content commit
  093cb2bb.
-->

Sub-page of the Developer Guide chapter. Documents the two-layer
persistence model: `QSettings`-via-`SettingsStorage` for
configuration, semicolon-delimited CSV via `BlackchirpCSV` for
per-experiment data. Explains the `HeaderStorage` tree, the
`DataStorageBase` lifecycle and its concrete subclasses, and the
file-parser ecosystem used for overlay imports and external data.

## Scope

Single Sphinx file: `doc/source/developer_guide/persistence.rst`.

The page should answer the following for a contributor:

1. **Two-layer persistence model.** Open by stating the split
   plainly:

   - **Configuration / preferences** — anything that survives
     across application invocations and is not part of an
     experiment's data record. Stored in `QSettings` via
     `SettingsStorage`. Examples: per-profile hardware settings,
     loadout definitions, application-wide settings (data path,
     debug logging, vendor library paths), GUI preferences.
   - **Experiment data** — the per-acquisition record. Written
     to a per-experiment directory as a tree of semicolon-
     delimited CSV files via `BlackchirpCSV`. Driven by the
     `DataStorageBase` lifecycle (start/advance/save/finish) and
     the `HeaderStorage` tree.

   These two layers do not share keys, classes, or files. A
   contributor adding new persistent state should consciously
   pick one. The default for "remember between sessions" is
   `SettingsStorage`; the default for "save with the
   experiment" is `HeaderStorage` + (if there's bulk data) a
   new `DataStorageBase` subclass.

2. **`SettingsStorage` quick reference.**

   - Constructed over a `QSettings` group; reads and writes
     keep an in-memory copy and call into `QSettings` directly.
     The cache is per-instance; multiple `SettingsStorage`
     instances over the same group see the same `QSettings`
     state but maintain separate caches.
   - The `set` family is **protected**; only the owning class
     (or a declared `friend`) can mutate. Read access is
     unrestricted via `get`/`getArray`/`getGroupValue`.
   - Three extensions:
     - **Array values** — list of `SettingsMap` entries (used
       for pulse-generator channels, chirp segments, marker
       channels).
     - **Group values** — nested map (used for protocol-
       specific configuration blocks).
     - **Getter registration** — bind a key to a member
       function so the value is computed on demand from the
       owning object's live state and re-saved automatically
       on destruction.
   - Defaults for hardware settings come from the registry
     (`REGISTER_HARDWARE_*`) and are applied by
     `HardwareObject::applyRegisteredSettings()`. Subclass
     constructors should not call `setDefault` for registry-
     declared keys.

   Cross-link to `:doc:`/classes/settingsstorage`` for the
   class-level API.

3. **Key namespaces.** Centralized under `BC::` to avoid raw
   string literals at call sites:

   - `BC::Key::` — hardware-related setting keys (declared in
     `data/settings/hardwarekeys.h`). Per-type sub-namespaces
     (`BC::Key::AWG`, `BC::Key::Flow`, `BC::Key::FidStorage`,
     etc.).
   - `BC::Store::` — non-hardware persistent-storage keys,
     scattered through data-class headers
     (`BC::Store::LM` for LoadoutManager,
     `BC::Store::FtmwLO` for LO-scan parameters, etc.).
   - `BC::CSV::` — canonical experiment-directory filenames
     (`versionFile`, `headerFile`, `validationFile`,
     `objectivesFile`, `hwFile`, `chirpFile`, `markersFile`,
     `clockFile`, `auxFile`, `fidparams`, `fidDir`,
     `lifparams`, `lifDir`).

   Bundle 12b documents the three key declaration patterns
   (Pattern A/B/C) for declaring these constants without
   per-TU duplication.

4. **`BlackchirpCSV` and the experiment directory.**

   - `BlackchirpCSV` (`data/storage/blackchirpcsv.{cpp,h}`) is
     the workhorse persistence class. It owns the canonical
     experiment-directory layout and provides static write
     helpers, directory helpers, and format utilities.
   - All CSV files use `;` (`BC::CSV::del`) as the cell
     delimiter. The pipe character `|` (`BC::CSV::altDel`)
     is reserved for `QStringList` values embedded within a
     single cell.
   - Two construction paths:
     - Default constructor — for static-method use and writing
       new experiments.
     - `(num, path)` constructor — reads `version.csv` from
       the existing experiment directory and populates an
       internal configuration map so subsequent `readLine` /
       `readFidLine` calls tokenize with the correct delimiter
       (delimiter has historically varied; the
       `version.csv`-keyed config is the source of truth for
       a loaded experiment).

   Cross-link to `:doc:`/classes/blackchirpcsv``.

5. **`HeaderStorage`: the experiment-config tree.**

   - `HeaderStorage` (`data/storage/headerstorage.{cpp,h}`) is
     the base class for any object that contributes fields to
     an experiment's `header.csv`. The header file is the
     human-readable record of every parameter that defined an
     acquisition.
   - The CSV uses six columns:
     `objectKey, arrayKey, arrayIndex, key, value, unit`.
     `HeaderStorage` packs values into that schema on write
     and unpacks them on read.
   - `Experiment` is the root of the tree. Children:
     `FtmwConfig` (which itself owns `RfConfig`, the digitizer
     config), `LifConfig` (which owns `LifDigitizerConfig`),
     the validator, the optional hardware configs (pulse gen,
     flow, IO board, pressure, temperature). Each may add
     grandchildren.
   - The framework dispatches incoming rows to the correct
     subtree by matching the object key in column 0; on
     write, it walks the tree depth-first to produce the full
     row set.
   - Two virtuals to override in a subclass:
     `storeValues()` (write side) and `retrieveValues()`
     (read side). Plus `prepareChildren()` to register
     `HeaderStorage` children. The class-level Doxygen on
     `HeaderStorage` carries the full how-to; cross-link to
     `:doc:`/classes/headerstorage``.
   - Adding a new `HeaderStorage` child to `Experiment`: register
     it in `Experiment::prepareChildren` (or the parent's
     equivalent), implement the two virtuals on the new class,
     pick a unique object key (typically a class-name-derived
     constant in a `BC::Store::` namespace).

6. **`DataStorageBase` lifecycle and concrete subclasses.**

   - `DataStorageBase` (`data/storage/datastoragebase.{cpp,h}`)
     is the abstract root for objects that persist *bulk*
     experiment data (waveforms, traces, overlay annotations).
     Identified by `(d_number, d_path)`; `d_number == -1`
     marks a transient (peak-up / dummy) instance for which
     all disk I/O is silently skipped.
   - Four pure virtuals define the acquisition-lifecycle
     contract:
     - `start()` — acquisition begins; arm internal state.
     - `advance()` — segment boundary; flush current
       in-progress accumulation, prepare next.
     - `save()` — persist current in-memory state.
     - `finish()` — acquisition ends; clear acquiring flag.
   - Direct subclasses:
     - `FidStorageBase` (`data/storage/fidstoragebase.{cpp,h}`)
       and its concretes
       (`FidSingleStorage`, `FidMultiStorage`,
       `FidPeakUpStorage`) — FTMW. Bundle 12g covers the
       cache and processing-settings persistence.
     - `LifStorage` — LIF traces on the 2D grid. Bundle 12h
       covers the grid flattening.
     - `OverlayStorage` — plot overlay annotations.
   - `AuxDataStorage` plays a similar role (collecting auxiliary
     time-series readings) but does **not** inherit
     `DataStorageBase`. It is owned and driven directly by
     `Experiment`. The split is deliberate: aux data is
     accumulated time-series with its own lifecycle, not the
     start/advance/save/finish pattern of bulk data.

   Cross-link to `:doc:`/classes/datastoragebase``,
   `:doc:`/classes/fidstoragebase``,
   `:doc:`/classes/lifstorage``,
   `:doc:`/classes/overlaybase``,
   `:doc:`/classes/auxdatastorage``.

7. **The experiment-directory layout.**

   - One directory per experiment, named by the experiment
     number, under the user's data path. Top-level files:
     `version.csv` (BC version + delimiter),
     `header.csv` (the `HeaderStorage` tree),
     `hardware.csv` (the active hardware map at acquisition
     time),
     `objectives.csv` (the active objectives — FTMW/LIF),
     `validation.csv` (validation map),
     `chirps.csv`, `markers.csv`, `clocks.csv` (RF/chirp
     setup),
     `aux.csv` (time-series auxiliary readings).
   - Subdirectories:
     `fid/` — FTMW FID data (`fidparams.csv` plus per-segment
     `<i>.csv`),
     `lif/` — LIF data (`lifparams.csv` plus per-trace files),
     `backup/` — periodic FTMW backups,
     overlay annotations (location depends on storage class —
     confirm).
   - `BC::CSV` filename constants are the canonical reference
     for these names; never hard-code them.

8. **File-parser ecosystem.** External-data parsers used for
   overlay imports and catalog data:

   - `FileParser` (`data/processing/parsers/fileparser.{cpp,h}`)
     — abstract base.
   - `FileParserRegistry`
     (`data/processing/parsers/fileparserregistry.{cpp,h}`) —
     registry singleton; parsers register themselves at
     static-init time so the GUI can discover supported file
     types without hard-coding.
   - Concrete subclasses:
     - `GenericXyParser` — generic X/Y CSV/TSV import.
     - `SpcatParser` — SPCAT catalog format (computed
       rotational spectra).
     - `XiamParser` — XIAM catalog format.
     - `CatalogParser` — internal catalog format
       (Blackchirp's own).
   - The parser registry is consumed by overlay-creation
     widgets (`gui/overlay/`) to populate file-format pickers
     and dispatch parsing.
   - Cross-link to `:doc:`/classes/fileparser``,
     `:doc:`/classes/fileparserregistry``,
     `:doc:`/classes/genericxyparser``,
     `:doc:`/classes/spcatparser``,
     `:doc:`/classes/xiamparser``,
     `:doc:`/classes/catalogparser``.

9. **Adding new persistent state.** A short decision-tree:

   - Per-profile hardware setting → register via
     `REGISTER_HARDWARE_SETTINGS` (or `_BASE`); the registry +
     `applyRegisteredSettings` does the rest. See bundle 12d.
   - Other configuration setting → declare a key in the
     appropriate `BC::Store::` namespace, expose a
     `SettingsStorage`-derived owner with friend-restricted
     `set` access.
   - New experiment-level header field → add to `Experiment`'s
     `storeValues`/`retrieveValues` (or to an existing
     child's). Choose an existing key namespace
     (`BC::Store::Exp` for experiment-level) or add a new
     sub-namespace.
   - New experiment data file → if it's bulk data with the
     start/advance/save/finish lifecycle, subclass
     `DataStorageBase`. If it's a one-shot config blob, treat
     it as a `HeaderStorage` child instead.
   - New file parser → subclass `FileParser`, declare the
     supported extensions, register with `FileParserRegistry`.

## Out of scope

- `SettingsStorage` API surface — already on
  `:doc:`/classes/settingsstorage``.
- Hardware-settings registry mechanism — bundle 12d.
- Per-class CSV layouts — already documented on the relevant
  API pages and the user-guide
  `:doc:`/user_guide/data_storage`` page.
- The FTMW caching model — bundle 12g (mentioned here as a
  one-paragraph forward).
- The LIF grid flattening — bundle 12h (one-paragraph
  forward).
- Adding new persistent state in detail — the *Adding new
  persistent state* section is a decision-tree, not a
  step-by-step recipe.

## Sources

### Related source files

- `src/data/storage/settingsstorage.{cpp,h}`.
- `src/data/storage/blackchirpcsv.{cpp,h}` — confirm the
  `BC::CSV` filename constants and the column constants.
- `src/data/storage/headerstorage.{cpp,h}` — base class.
- `src/data/storage/datastoragebase.{cpp,h}` — base class.
- `src/data/storage/auxdatastorage.{cpp,h}`.
- `src/data/storage/overlaystorage.{cpp,h}`.
- `src/data/storage/fidstoragebase.{cpp,h}` and the three
  concretes.
- `src/data/storage/applicationconfigmanager.{cpp,h}` — for
  the application-wide settings store.
- `src/data/lif/lifstorage.{cpp,h}`.
- `src/data/experiment/experiment.{cpp,h}` — to confirm the
  `prepareChildren` registrations and the experiment-
  directory write sequence in `initialize` and `finalSave`.
- `src/data/processing/parsers/*.{cpp,h}` — every parser.
- `src/data/settings/hardwarekeys.h` — for the `BC::Key`
  namespace examples.
- `src/data/bcglobals.{cpp,h}` — for the `BC::Store` /
  `BC::CSV` namespace examples.

### Related dev-docs

None directly. (Persistence has been stable enough that the
dev-docs material is in code comments and API pages.)

### Related user-guide pages

Forward-link, do not duplicate:

- `doc/source/user_guide/data_storage.rst` — the user-facing
  view of the experiment directory and CSV formats.
- `doc/source/user_guide/overlays.rst` — overlay creation
  workflow.

### Related API reference pages

- `doc/source/classes/settingsstorage.rst`
- `doc/source/classes/blackchirpcsv.rst`
- `doc/source/classes/headerstorage.rst`
- `doc/source/classes/datastoragebase.rst`
- `doc/source/classes/fidstoragebase.rst`
- `doc/source/classes/lifstorage.rst`
- `doc/source/classes/overlaybase.rst`
- `doc/source/classes/auxdatastorage.rst`
- `doc/source/classes/applicationconfigmanager.rst`
- `doc/source/classes/fileparser.rst`
- `doc/source/classes/fileparserregistry.rst`
- `doc/source/classes/genericxyparser.rst`
- `doc/source/classes/spcatparser.rst`
- `doc/source/classes/xiamparser.rst`
- `doc/source/classes/catalogparser.rst`

## Sphinx file deltas

**Created:**

- `doc/source/developer_guide/persistence.rst`.

## Page structure

H1 intro: 2 paragraphs framing the two-layer model.

H2 sections (`-` underlines):

- *Two layers: configuration vs. experiment data*
- *SettingsStorage*
- *Key namespaces*
- *BlackchirpCSV and the experiment directory*
- *HeaderStorage: the configuration tree*
- *DataStorageBase: the bulk-data lifecycle*
- *Experiment-directory layout* — table or bulleted file map.
- *File parsers* — overlay imports and catalog data.
- *Adding new persistent state* — decision tree.

## Acceptance criteria

- The two-layer model (QSettings vs. CSV) is the page's
  organizing frame and is stated explicitly.
- The protected-`set` policy on `SettingsStorage` is
  documented.
- The three key namespaces (`BC::Key`, `BC::Store`, `BC::CSV`)
  are each named with their domain.
- The `BlackchirpCSV` two-construction-paths model is
  documented (default for new writes, `(num, path)` for
  reads).
- The `HeaderStorage` tree's six-column schema is stated.
- The four `DataStorageBase` virtuals (`start`/`advance`/
  `save`/`finish`) are listed with their meaning, and the
  three direct subclasses (`FidStorageBase`, `LifStorage`,
  `OverlayStorage`) are named.
- `AuxDataStorage`'s "similar role but not a subclass"
  position is documented.
- The experiment-directory layout names every top-level CSV
  file and the two main subdirectories.
- The file-parser registry is documented at the architecture
  level with all four concrete subclasses named.
- The decision-tree section gives a contributor a clear
  starting point for adding new persistent state.
- No duplication of per-class API content; cross-links cover
  per-class detail.
- No rendered link points into `dev-docs/`.
