.. index::
   single: Application Configuration
   single: Settings; application
   single: LIF Module; runtime toggle
   single: Debug Logging
   single: Application Font
   single: QSettings; version isolation
   single: Data Storage; path

.. _application-config:

Application Configuration
=========================

The Application Configuration dialog (**Settings → Application Settings**)
controls application-wide options that are independent of any particular
hardware profile or experiment. Options are grouped into two sections:
*Application Settings* and *Data Storage*.

To open the dialog: **Settings → Application Settings**.

.. TODO: capture screenshot — dialog.png: the full ApplicationConfigDialog
   (not first-run variant), showing the "Application Settings" group box with
   all three options (LIF Module checkbox, Debug Logging checkbox, Application
   Font row with Change button) and the "Data Storage" group box below. The
   dialog title should read "Application Settings". All three options should be
   visible with their default states: LIF Module checked, Debug Logging
   unchecked, Application Font showing the default font preview.

.. figure:: /_static/user_guide/application_config/dialog.png
   :alt: Application Configuration dialog

   The Application Configuration dialog, showing all available options.

Options marked **(requires restart)** in the dialog take effect only after
you close and reopen Blackchirp. All other options take effect immediately
when you click **OK**.

Application Settings
--------------------

LIF Module
^^^^^^^^^^

*Requires restart.*

Enables or disables the Laser-Induced Fluorescence hardware and associated
UI components. When disabled, all LIF-related widgets are hidden and LIF
hardware types are excluded from experiment configuration. When enabled, LIF
hardware types appear in the Hardware Selection dialog and the LIF tab
becomes accessible in the main window.

Default: **enabled**.

Enable this option only if your instrument includes a LIF channel. Disabling
it on a system without LIF hardware has no functional consequence but keeps
the interface uncluttered.

.. _app-config-debug-logging:

Debug Logging
^^^^^^^^^^^^^

*Takes effect immediately.*

When enabled, Blackchirp writes debug-level log messages to a dated file
(``debug_YYYYMM.csv``) inside the ``log`` subdirectory of the data storage
path. Normal informational and warning messages are written regardless of
this setting.

Default: **disabled**.

Enable debug logging temporarily when diagnosing communication failures or
unexpected behaviour. Disable it during normal operation to reduce disk usage.

Application Font
^^^^^^^^^^^^^^^^

*Takes effect immediately.*

Sets the font used throughout the application. Click **Change...** to open
the system font picker. The preview label in the dialog updates immediately
to reflect your selection.

Default: sans-serif, 8 pt.

Data Storage
------------

The *Data Storage* section contains the same save-path widget shown during
:ref:`first-run-data-path`. Use it to change the data directory or reset the
starting experiment number at any time after initial setup.

Click **Apply** within this section to validate the path before clicking
**OK**. Blackchirp does not overwrite existing data when you change the path;
it only creates the four standard subdirectories if they are absent.

.. _app-config-settings-isolation:

Configuration File and Version Isolation
-----------------------------------------

Blackchirp stores its settings using the Qt ``QSettings`` framework. On each
platform the settings are written to a system-standard location:

- **Linux** — ``~/.config/CrabtreeLab/Blackchirp2.conf``
- **macOS** — ``~/Library/Preferences/CrabtreeLab.Blackchirp2.plist``
- **Windows** — registry key ``HKCU\Software\CrabtreeLab\Blackchirp2``

The application name embedded in the path (``Blackchirp2``) includes the
major version number. This is intentional: settings written by one major
version of Blackchirp are not read by a different major version. If you have
configuration files from a previous major version (stored under a name such
as ``Blackchirp`` or ``Blackchirp1``), Blackchirp 2 will not pick them up
automatically. You must reconfigure hardware profiles, the data path, and
application settings from scratch when upgrading between major versions.
This isolation prevents stale or incompatible settings from silently affecting
behaviour after an upgrade.

Blackchirp records the current version numbers in the settings file on every
startup, which allows future tooling to detect the version that last wrote the
file if migration assistance is ever added.
