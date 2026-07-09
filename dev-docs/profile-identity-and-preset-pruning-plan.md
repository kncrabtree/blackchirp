# Plan — stable profile identity + preset lifecycle hardening

Post-merge work item (targeted after `feature/sirah-cobra-refresh` lands).
Covers both FTMW and LIF presets. Goal: a preset can never silently bind to
a *different* physical profile than the one it was captured against, and
stale presets are cleaned up at the moment their hardware goes away.

## Motivation

A preset stores hardware references by **hwKey** (`"<Type>.<label>"`). That
string is the only identity Blackchirp has for a profile, and it is not
unique over time: deleting a profile and recreating one with the same label
(even the same implementation) yields the same hwKey. Every consumer then
treats the new profile as the old one.

Two concrete failure modes:

1. **Deleted profile, lingering preset.** A profile is deleted; presets in
   any loadout that reference its hwKey are left in place and only fail later
   — at apply/prep for FTMW, or (now) at load for LIF via the at-load reject
   added on this branch.
2. **Recreate-same-label.** A profile is deleted and recreated at the same
   `type.label`. The hwKey matches, so a preset re-binds to the new profile.
   If the new profile has a different op/arity (LIF) the at-load reject
   catches it; if it has the *same* structure but is physically a different
   device, nothing catches it — the preset applies against the wrong
   hardware.

## Current state (what already exists)

- **Profiles** (`HardwareProfileData`, `hardwareprofilemanager.*`): carry a
  persisted `created` and `modified` `QDateTime` (serialized via
  `QDataStream` on save/load). Identity is `type` + `label`; `implementation`
  is immutable (changing it requires a new profile). `created` is stable
  across restarts and is a natural identity anchor. `deleteHardwareProfile()`
  has a single caller: `runtimehardwareconfigdialog.cpp`.
- **Loadouts** (`HardwareLoadout`, `loadoutmanager.*`): `hardwareMap` is
  `type.label -> implementation` (a denormalized field the docs already call
  out as "used for validation and drift detection"). Each loadout owns
  `ftmwPresets` / `lifPresets` maps plus a `current{Ftmw,Lif}PresetName` and a
  `__LastUsed__` sentinel.
- **Drift handling** (`runtimehardwareconfigdialog.cpp` ~1437–1510): on
  applying a runtime-config change, compares `ftmwRelevantHwKeys()` /
  `lifRelevantHwKeys()` of the new vs stored `hardwareMap`; on drift with
  named presets, prompts Preserve / Discard / SaveAs / Cancel; on drift with
  no named presets, discards `__LastUsed__`.
- **At-load LIF reject** (this branch, `lifconversiontablemodel.cpp`): a LIF
  preset whose wired hwKey is missing, or whose arity no longer matches, is
  rejected wholesale and the configuration cleared, surfaced in the
  conversion footer.

### Gaps

- **G1** Drift detection compares hwKey *sets* (+impl). A recreate-same-label
  (same impl) is not a set difference, so it is invisible.
- **G2** Deletion does not prune presets in other loadouts; drift handling
  only runs for the *active* loadout at config-apply time.
- **G3** FTMW has no structural at-load/at-prep validation equivalent to the
  LIF reject; it leans entirely on the drift dialog.

## Design

### 1. Stable profile identity (closes G1)

Give every profile a stable identity token independent of relabeling reuse:

```
identity = hash(type + "\x1f" + label + "\x1f" + implementation
                     + "\x1f" + created.toMSecsSinceEpoch())
```

`implementation` is immutable for a profile's life, so it does not change the
token's *uniqueness* — but including it makes the token fully self-describing
and guards the (unlikely) `type+label+created` collision from scripted
same-millisecond creation.

- Add `HardwareProfileManager::getProfileCreated(type,label)` and
  `getProfileIdentity(type,label)` accessors (mirroring the existing
  `getProfileLastModified`). Compute the hash from persisted fields only.
- Extend the loadout membership value from `implementation` to
  `{implementation, identity}`:
  - Option A (minimal format change): keep `hardwareMap` as `hwKey -> impl`
    and add a parallel `hwKey -> identity` array to the loadout record.
  - Option B (cleaner): change `hardwareMap`'s value to a small struct
    `{QString impl; QString identity;}` and update `hardwareMapArray` /
    `hardwareMapFromArray` (hardwareloadout.cpp) plus the QSettings schema.
  Prefer **B**; it keeps the identity next to the impl it validates.
- `ftmwRelevantHwKeys()` / `lifRelevantHwKeys()` comparisons switch to
  comparing `(hwKey, identity)` pairs, so recreate-same-label now registers as
  drift and flows through the existing Preserve/Discard/SaveAs dialog.

**Migration.** Loadouts written before this change have no identity field.
On read, a missing identity is a wildcard: it must not force a false-positive
drift on first upgrade. Backfill the identity from the current profile the
next time the loadout is saved. Document the one-time "old loadouts trust the
hwKey until first re-save" behavior.

### 2. Deletion-time pruning + informed confirmation (closes G2)

Deleting a profile has two distinct consequences, both of which must be shown
to the user **before** anything is executed:

- **Presets lost.** Presets in any loadout that reference the deleted hwKey
  are removed.
- **Loadouts modified (not deleted).** A loadout whose member profile is
  deleted is *not* deleted. For an **optional** hardware type the member is
  simply dropped from the loadout. For a **required** hardware type the
  loadout cannot be left with that type empty, so the member is replaced with
  the **system fallback** profile (via the same `ensureSystemProfiles` /
  required-type handling the runtime-config dialog already uses).

#### Dry-run + confirmation

Add a pure query that computes these consequences without mutating anything:

```
struct PruneConsequences {
    std::vector<std::pair<QString,QString>> lostPresets;   // (loadout, preset)
    std::vector<QString> modifiedLoadouts;                 // member dropped
    std::vector<std::pair<QString,QString>> fallbackSubs;  // (loadout, required type -> fallback)
};
PruneConsequences LoadoutManager::previewPruneReferencing(const QString &hwKey) const;
```

The deletion confirmation dialog (currently a plain confirm at
`runtimehardwareconfigdialog.cpp:~1085`) is upgraded to enumerate
`lostPresets`, `modifiedLoadouts`, and `fallbackSubs` explicitly — named
loadout→preset pairs, not a count — so the user sees exactly what will be lost
or substituted. Deletion + pruning execute only on confirmation.

#### Execution

```
int LoadoutManager::prunePresetsReferencing(const QString &hwKey);   // # presets removed
```

For every loadout:
- For each FTMW/LIF preset, if `referencesHardware(hwKey)` → remove it.
- If a removed preset was `current{Ftmw,Lif}PresetName` → set current to
  `__LastUsed__`.
- If `__LastUsed__` itself references `hwKey` → **remove it** and clear the
  current pointer (widgets then start from defaults; the at-load reject
  backstops any residue). *(Decision recorded: remove, not tombstone.)*
- Apply the required-type fallback substitution to `hardwareMap`.
- Emit the existing `*PresetChanged` / loadout-changed signals so open widgets
  refresh.

Orchestrated from `runtimehardwareconfigdialog.cpp` — the single deletion call
site — around `deleteHardwareProfile(type,label)` with `hwKey = type + "." +
label`: preview → confirm → delete → prune. (`HardwareProfileManager` is a
`SettingsStorage`, not a `QObject`, so there is no signal to hang this on; the
dialog already holds both managers.)

#### `referencesHardware(hwKey)` predicates

- **LIF** (`LifPreset`): any `conversion.wiring[].stageKey == hwKey`, or
  `conversion.laserKey == hwKey`. (Stage-typed `InputRef`s are redundant with
  the referenced node's own wiring entry.)
- **FTMW** (`FtmwPreset`): `digiHwKey == hwKey`, plus the hardware referenced
  by `rfConfig` (`RfConfigSnapshot`) — clock hwKeys and the upconversion
  source. This enumeration does not exist yet and is the larger part of the
  work; add an `RfConfigSnapshot::referencedHwKeys()` helper so the predicate
  has a single source of truth.

### 3. Identity-aware validation on apply (uses 1)

With identity in `hardwareMap`, the existing drift dialog already becomes the
apply-time guard for recreate-same-label (G1). No new dialog needed — just the
comparison-key change in §1. Keep the LIF at-load reject as defense-in-depth;
optionally add an FTMW structural check at prep (G3) if pruning + identity
drift prove insufficient in practice (likely unnecessary once §1/§2 land).

### 4. Relationship to the interim at-load LIF reject

The at-load reject shipped on this branch stays. Once §1/§2 land it demotes
from primary protection to a cheap backstop (covers the window between
deletion and pruning, pruning bugs, and cross-version driver redefinition).
No removal planned.

## Sequencing

1. Profile identity accessors + hash (§1, profile manager).
2. Loadout format: identity alongside impl + migration (§1, hardwareloadout /
   loadoutmanager + QSettings schema).
3. Switch drift comparisons to identity pairs (§1, dialog).
4. `referencesHardware` predicates: LIF, then `RfConfigSnapshot` refs for FTMW
   (§2).
5. `previewPruneReferencing` (dry-run) + `prunePresetsReferencing` execution +
   `__LastUsed__` handling + required-type fallback substitution (§2).
6. Wire preview → confirmation dialog (enumerating consequences) → delete →
   prune into the deletion call site (§2, dialog).

## Test plan

The consequence computation is the reliability-critical piece, and it is pure
(no GUI): `previewPruneReferencing` and `prunePresetsReferencing` live on
`LoadoutManager`, and the deletion dialog is a thin renderer over the
preview's result. This must be the **primary unit-tested surface** (extend
`tests/tst_loadoutmanagertest.cpp`) — driving the dry run over an in-test
loadout/preset fixture is far more reliable than manual walkthroughs over live
profile creations.

### Core fixture (dry-run driven)

Build a multi-loadout fixture in-test and drive it through mutations,
asserting the dry run returns the exact expected operations before any
execution:

- Two or more loadouts sharing some member profiles and differing in others,
  spanning both **required** and **optional** hardware types.
- Each loadout carries several FTMW and LIF presets: some referencing the
  soon-to-be-deleted hwKey (via `digiHwKey` / rfConfig refs / wiring
  `stageKey` / `laserKey`), some not — plus a `__LastUsed__` in both
  referencing and non-referencing variants.
- For each mutation — delete an optional-type member; delete a required-type
  member; delete a member referenced only by `__LastUsed__`; delete a member
  no preset references — assert the returned `PruneConsequences`:
  - `lostPresets` = exactly the `(loadout, preset)` pairs that reference the
    hwKey, and nothing else;
  - `modifiedLoadouts` = the loadouts losing an optional member;
  - `fallbackSubs` = the loadouts whose required-type member is replaced, each
    with the correct fallback profile.
- Then run `prunePresetsReferencing` and assert the post-state matches the
  preview exactly: referencing presets gone; `current` retargeted to
  `__LastUsed__` where the current preset was removed; a stale `__LastUsed__`
  removed; required-type members substituted; unrelated presets and loadouts
  untouched.
- Cancel path: preview computed, execution skipped → every loadout and preset
  intact.

### Identity + drift

- Profile identity hash is stable across save/load; differs after
  delete+recreate-same-label (persisted `created` differs); independent of
  relabeling reuse.
- Drift comparison triggers on recreate-same-label (identity differs though
  hwKey and impl are unchanged); does not trigger when hardware is truly
  unchanged.
- Migration: a pre-identity loadout does not report false drift on first load;
  identity is backfilled on next save.

### Predicates + regression

- LIF `referencesHardware` true for a wired `stageKey` and for `laserKey`;
  FTMW true for `digiHwKey` and each rfConfig clock/upconversion ref; false
  otherwise.
- Regression: existing `LoadoutManagerTest` preserve/discard/clear behavior is
  unchanged where hardware is unchanged.

## Out of scope

- Detecting an arity-preserving op swap (SFG<->DFG) purely from a stored
  snapshot — the snapshot deliberately omits op. Identity drift (recreate) and
  the at-load reject cover the realistic routes to this.
- Reworking the immutable-implementation rule.
