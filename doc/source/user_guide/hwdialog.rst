.. index::
   single: Hardware Dialog
   single: HwDialog
   single: Settings tab; hardware
   single: Control tab; hardware
   single: Required settings
   single: Important settings
   single: Advanced settings
   single: Test Connection; Hardware Dialog
   single: Communication Settings; Hardware Dialog link

Hardware Dialog
===============

Selecting any per-device entry from the :ref:`Hardware menu
<hardware-menu-device-entries>` opens the **Hardware Dialog** for that device.
The dialog is a tabbed window. The title bar shows the hardware key
(``Type.Label``) and the model name of the active driver.

The dialog provides two tabs: **Control** (when available) and **Settings**.
The tabs serve different purposes and have different save semantics — reading
both sections carefully before making changes is recommended.

Below the tabs, a row of convenience controls is always visible:

- **Test Connection** — runs the same connection check as the Communication
  Dialog's per-device test, against the currently active communication
  settings. An inline status label to the right of the button reports the
  outcome (a green *Connected* indicator on success, a red *Connection failed*
  indicator on failure with the error message available as a tooltip). The
  button is disabled while a test is in flight.
- **Communication Settings…** — opens the :ref:`Communication Settings dialog <user_guide/hardware_menu:Communication>` with this
  device pre-selected. The Communication Dialog is non-modal, so it can stay
  open alongside the Hardware Dialog while you adjust protocol or read-option
  settings; if it is already open, clicking the link raises it and switches
  the selection to this device. The Communication Dialog closes automatically
  when a real (non peak-up) experiment starts.

.. _hwdialog-control:

Control Tab
-----------

.. figure:: /_static/user_guide/hwdialog-control_tab.png
   :width: 800
   :alt: Hardware Dialog showing a Pulse Generator's Control tab with the
         channel configuration table and timing-diagram preview.

   The Pulse Generator's Control tab, illustrating the system-settings
   group, per-channel configuration table, and the timing-diagram pane that
   previews the resulting pulse train.

The **Control** tab is present for hardware types that expose live, interactive
controls. The screenshot above shows the Pulse Generator's Control tab; other
devices follow the same pattern but with controls appropriate to their
function. Examples include:

- **Pulse Generator** — a *System Settings* group with a master enable, the
  pulse mode (continuous or triggered), and the global repetition rate; a
  *Channel Configuration* table with one row per channel exposing the sync
  source, delay, width, mode, and enable state, plus a per-channel cog
  button that opens a configuration sub-dialog for advanced channel
  settings; and a timing-diagram pane on the right that previews the pulse
  train derived from the current channel settings.
- **Flow Controller** — gas channel setpoints, channel names, and pressure
  control mode. Changes are sent to the instrument as each value is modified.
- **Pressure Controller** — pressure setpoint, control mode toggle, and gate
  valve open/close actions. Each action is transmitted immediately.
- **Temperature Controller** — per-channel enable and name assignment; values
  are sent immediately upon change.

A label at the top of the Control tab reads: *"Changes made in this section
will be applied immediately."*

.. important::
   Commands issued through the Control tab are transmitted to the hardware
   as soon as you interact with the controls. Closing the dialog with the
   **Close** button or the window's **X** button does **not** undo control
   commands — those changes have already been sent to the instrument.

.. _hwdialog-settings:

Settings Tab
------------

.. figure:: /_static/user_guide/hwdialog-settings_tab.png
   :width: 800
   :alt: Hardware Dialog showing the Advanced sub-tab of a Pulse Generator's
         Settings tab.

   The Hardware Dialog open on the Pulse Generator's Settings tab. Inner
   sub-tabs separate the Settings group (Required + Important settings)
   from the Advanced group; the Advanced sub-tab is shown here.

A label at the top of the Settings tab reads: *"Changes made in this section
will only be applied when this dialog is closed with the Ok button. Editing
these settings incorrectly may result in unexpected behavior. Consider backing
up your config file before making changes."*

Each device can declare any number of persistent settings, organized by
priority. The Settings tab presents them through a nested set of sub-tabs:

- A **Settings** sub-tab containing the Required and Important groups (when
  any are declared).
- An **Advanced** sub-tab, shown only when the device declares advanced
  settings.

The screenshot above shows the Advanced sub-tab for the PulseGenerator. The
Settings sub-tab would render the Required and Important groups described
below.

**Required**
   Parameters that must be set correctly at profile creation time (for example,
   the number of digitizer input channels or the AWG output count). They appear
   in a *Required Settings* group on the **Settings** sub-tab and are displayed
   as read-only text — they cannot be changed once a profile has been created.
   To change a Required setting, create a new profile via
   :doc:`hardware_config/profiles`.

**Important**
   Settings with sensible defaults that you should review and confirm. They
   appear in an *Important Settings* group on the **Settings** sub-tab as a
   two-column table (Setting | Value) with editable widgets. Typical examples
   include communication-level defaults, calibration constants, and operating
   limits.

**Advanced**
   Settings that rarely need to change. They appear on the **Advanced**
   sub-tab as a Setting/Value table. The Advanced sub-tab is added only when
   advanced settings are declared. Array-type settings (for example, a list of
   available sample rates) are also edited here using a dedicated sub-dialog
   opened via an **Edit** button in the table row.

.. important::
   Settings tab changes are written to persistent storage **only** when you
   click **OK**. Closing the dialog with the **Close** button or the window's
   **X** button discards all unsaved Settings changes. This is intentional: the
   Settings tab modifies persistent configuration data, and the explicit
   confirmation step prevents accidental changes.

   This is distinct from the Control tab, where commands are sent immediately
   and cannot be recalled by closing the dialog.

.. warning::
   Incorrect or inappropriate settings may cause unexpected program behavior.
   Consider backing up your configuration file before making changes to
   hardware settings.

Array Settings
~~~~~~~~~~~~~~

Some devices declare array-type settings — for example, an ordered list of
supported sample rates. Each array setting appears as a table row showing the
current entry count and an **Edit** button. Clicking **Edit** opens the
array-edit sub-dialog, where you can add, remove, or reorder entries. Changes
made in the sub-dialog are staged in the Settings tab and written to storage
only when the parent dialog is accepted with **OK**.
