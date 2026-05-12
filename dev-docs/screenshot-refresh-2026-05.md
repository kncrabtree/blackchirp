# User-guide screenshot refresh — 2026-05

Baseline commit: `460f0ce2` (rename FtmwScope/LifScope to
FtmwDigitizer/LifDigitizer).

Three UI changes since the baseline invalidate one or more screenshots
under `doc/source/_static/user_guide/`:

- Rename: visible C++ class names (`FtmwScope`, `LifScope`,
  `VirtualFtmwScope`, `VirtualLifScope`, `PythonFtmwScope`,
  `PythonLifScope`) appear in dialog titles, tab headers, browser
  rows, and the Hardware menu. The on-disk hwType key for `LifScope`
  also changed to `LifDigitizer`; the FTMW namespace already used
  `"FtmwDigitizer"` in storage pre-rename.
- Manual FTMW backup (`6d6dbbbf`): new toolbar action on the CP-FTMW
  tab.
- Update check (`6d8517e6` + `9ca2afea`): new **Check for Updates**
  toggle in Application Settings; new **Help → Check for Updates...**
  action; conditional Help-menu badge when an update is available.

## Recapture these (confirmed stale by direct inspection)

### Rename surface

| File | Stale labels |
|---|---|
| `hardware_config/runtimedialog.png` | Right panel header "FtmwScope Configuration" / "FtmwScope Profiles"; browser rows "FtmwScope (1)", "LifScope (1)"; overview "FtmwScope: virtual (VirtualFtmwScope)", "LifScope: Default (PythonLifScope)"; Python Class "FtmwScopeDriver"; script path `my_ftmwscope.py`. |
| `first_run/onboarding-runtimeconfig.png` | Same dialog; FtmwScope/LifScope rows in browser and overview, right-panel header. |
| `hardware_config/addprofile.png` | Window title "Add FtmwScope Profile". Caption in `profiles.rst` already says "FtmwDigitizer profile" — image and caption are inconsistent. |
| `hardware_menu/menu.png` | Per-device entries "FtmwScope.virtual" and "LifScope.Default" at the bottom of the menu. |
| `hardware_menu/communication.png` | Device list rows "FtmwScope virtual (VirtualFtmwScope)" and "LifScope Default (PythonLifScope)"; right-panel header "Configuring: FtmwScope virtual (VirtualFtmwScope)". Caption already says "FtmwDigitizer device". |
| `python_hardware/profile_creation.png` | Hardware browser rows "FtmwScope (1)" and "LifScope (1)"; config overview lines for both. |

### Manual backup button

| File | Issue |
|---|---|
| `ui_overview/cp_ftmw.png` | CP-FTMW toolbar predates the new **Manual Backup** action (archive-arrow-down icon). `cp-ftmw.rst` now has a `Manual Backup` subsection describing a button not present in the image. |

### Update-check toggle

| File | Issue |
|---|---|
| `application_config/dialog.png` | Application Settings shows only LIF Module, Debug Logging, Application Font. The new **Check for Updates** toggle (default enabled, takes effect immediately) is missing. |
| `first_run/savepathdialog.png` | Same first-run "Welcome to Blackchirp - Initial Configuration" dialog; its Application Settings section is also missing the toggle. Referenced from both `first_run.rst` and `data_storage.rst`. |

## Worth a closer look (not confirmed stale at thumbnail scale)

- `lif/lif_config.png` — embeds the digitizer config widget; verify at
  full size whether a `LifScope`/`LifDigitizer` hwType key string is
  rendered anywhere (e.g., a settings-table headerKey column).
- `ui_overview/ui.png`, `viewer/main_window.png` — the Help toolbar
  tint and viewer Help-title star only render when an update is
  available. The default-state image is still accurate. Optional:
  capture a fresh shot with an active update if the docs should
  illustrate the badge.

## Verified unaffected

`ftmw_configuration/*.png` (tab labels "RF Config / Chirp Config /
Digitizer Config" predate the rename), `hwdialog/control_tab.png` and
`settings_tab.png` (Pulse Generator), `hardware_config/loadouts_menu.png`,
`drift_prompt.png`, `ftmw_presets_menu.png`,
`python_hardware/hwdialog_python.png`, `error_state.png`,
`template_copy_prompt.png` (PythonFlowController; no rename surface),
`lif/lif_tab.png`, `lif/lif_exp_setup.png`, all `experiment/*.png`,
`overlays/*.png`, `plot_controls/*.png`, `rolling_aux_data/*.png`,
`log_tab/*.png`, `library_status/widget.png`,
`ui_overview/peakfind.png`.
