# Bundle 02 — First Run, Application Configuration, Hardware Onboarding

**Status:** drafted

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-04-29: drafted (revision 2). User flagged factual errors in the
  first_run hardware-onboarding section: the Hardware Configuration
  dialog is a *four-panel* horizontal layout (Loadouts | Configuration
  Overview | Hardware Browser | Configuration), not three panels with
  a separate Loadout-management strip. The Hardware Browser lists
  every hardware type Blackchirp *supports*, not "required by".
  Single-instance types (FtmwScope, AWG, LifLaser, LifScope per
  `HardwareRegistry::isMultiInstanceType`) need to be called out
  alongside multi-instance behaviour. Orchestrator rewrote the panel
  description, corrected the wording, added a Single-instance vs
  multi-instance paragraph, mentioned the validation status bar, and
  left a TODO marker pointing bundle 03 at the prose so it can wire
  in `:doc:` cross-references to `hardware_config.rst` once it
  exists. Sphinx build still clean against pre-existing baseline.
- 2026-04-29: in progress → drafted. Verifier (fresh-context Sonnet)
  graded all four acceptance criteria as PASS and flagged one
  load-bearing factual error: the drafter cited "Settings → Library
  Status" and "Settings → Hardware Configuration" as menu paths, but
  Library Status is only reachable as a tab inside the Hardware
  Selection dialog (opened via Hardware → Hardware Selection). The
  Application Settings dialog action is labelled "Application
  Settings" (not "Application Configuration"), and lives under the
  Settings toolbar button. Orchestrator corrected all three menu
  references directly via Edit (library_status.rst, first_run.rst,
  application_config.rst). Also fixed: forward-reference prose to a
  not-yet-existing hardware_config chapter (rewritten to cite only
  hardware_menu); ambiguous "left side of the configuration panel"
  in the Troubleshooting section. Three screenshot TODOs in place;
  Sphinx build clean (no new warnings beyond the three expected
  image-not-readable notices for those screenshots). All four
  acceptance criteria met. Awaiting user review and commit.
- 2026-04-29: not started → in progress. Drafter dispatched (Sonnet,
  isolated worktree). Bundle file scope verified: all five cited
  source classes (ApplicationConfigManager, ApplicationConfigDialog,
  LibraryStatusWidget, VendorLibrary, RuntimeHardwareConfigDialog)
  exist at the paths the bundle gives.
-->

Rewrites the first-run flow and introduces the user to the
runtime application-configuration and hardware-onboarding dialogs
that are new in 2.0.

## Scope

- Rewrite `doc/source/user_guide/first_run.rst` to describe the
  first-run sequence: data-storage location, application
  configuration (LIF runtime toggle, debug logging, etc.), hardware
  onboarding via `RuntimeHardwareConfigDialog` (a one-time guided
  walk through profile creation), library-status review.
- Add `doc/source/user_guide/application_config.rst` — a reference
  page for the Settings → Application Configuration dialog
  (`ApplicationConfigDialog`): what each option does, when changes
  take effect (immediate vs. requires restart), and the QSettings
  isolation by major version (2.0.0 uses
  `~/.config/CrabtreeLab/Blackchirp_v2.conf`-style storage).
- Add `doc/source/user_guide/library_status.rst` — explains the
  vendor-library status widget (which shows whether each
  dynamically-loaded library — LabJack UD/exodriver, Spectrum,
  vendor SDKs — is found at runtime). Document the user actions:
  Browse to a library file, Refresh status, what an "Unavailable"
  state means, and how to install platform-specific vendor libraries.
- Cross-link both new pages from the Hardware Configuration chapter
  (bundle 03).

## Out of scope

- Per-profile creation walkthrough (covered in bundle 03).
- Loadout creation (covered in bundle 03).
- Per-device communication setup (covered in bundle 04).

## Sources

- `dev-docs/labjack-cross-platform-support.md` — vendor-library
  search semantics.
- Source: `src/data/storage/applicationconfigmanager.{h,cpp}`,
  `src/gui/dialog/applicationconfigdialog.{h,cpp}` — read for the
  exact set of user-visible settings.
- Source: `src/gui/widget/librarystatuswidget.{h,cpp}`,
  `src/hardware/library/vendorlibrary.{h,cpp}` — read for the
  library-status user model.
- Source: `src/gui/dialog/runtimehardwareconfigdialog.{h,cpp}` —
  understand the first-run path and the onboarding wizard step.
- Commit `3b01a13e` — QSettings isolation by major version.

## Sphinx file deltas

**Modified:**
- `doc/source/user_guide/first_run.rst` — full rewrite.

**Created:**
- `doc/source/user_guide/application_config.rst`
- `doc/source/user_guide/library_status.rst`

## Toctree delta in `user_guide.rst`

Insert after `first_run`:

```
   user_guide/first_run
   user_guide/application_config
   user_guide/library_status
```

## Screenshots

- `_static/user_guide/first_run/savepathdialog.png` — already exists;
  re-shoot if the dialog changed.
- `_static/user_guide/first_run/onboarding-runtimeconfig.png` —
  RuntimeHardwareConfigDialog as it appears on first run.
- `_static/user_guide/application_config/dialog.png` — full Application
  Configuration dialog.
- `_static/user_guide/library_status/widget.png` — Library Status
  widget showing both available and unavailable libraries.

## Acceptance criteria

- A first-time user can follow `first_run.rst` from a clean install
  to a state where they have a data path, application configuration
  baseline, at least one hardware profile, and a clear understanding
  of which vendor libraries (if any) are missing.
- `application_config.rst` documents every user-facing option
  presented by `ApplicationConfigDialog`, classified as either
  "takes effect immediately" or "requires restart".
- `library_status.rst` lists every vendor library currently shipped
  via `VendorLibrary` subclasses (LabJack, Spectrum) with a brief
  install hint per platform.
- The QSettings isolation behaviour is explained in plain language
  in `application_config.rst` (one-paragraph: configurations from
  v1.x are not picked up automatically by v2.x; this is intentional).
