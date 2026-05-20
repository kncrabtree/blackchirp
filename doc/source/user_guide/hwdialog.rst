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
<hardware-menu-device-entries>` opens the **Hardware Dialog** for that
device. The title bar shows the hardware key (``Type.Label``) and the
model name of the active driver. The dialog provides two tabs:
**Control** (when available) and **Settings**, with different save
semantics — read both sections before making changes.

Below the tabs, a row of convenience controls is always visible:

- **Test Connection** — runs the same connection check as the
  Communication Dialog's per-device test. An inline status label to the
  right reports the outcome (green ``Connected`` on success, red
  ``Connection failed`` on failure, with the error message available as
  a tooltip). The button is disabled while a test is in flight.
- **Communication Settings…** — opens the :ref:`hardware-menu-communication`
  dialog with this device pre-selected. The Communication Dialog is
  non-modal; if it is already open, clicking the button raises it and
  switches the selection. It closes automatically when a real
  (non peak-up) experiment starts.

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

The **Control** tab is present for hardware types that expose live,
interactive controls. The screenshot above shows the Pulse Generator's
Control tab; other devices follow the same pattern with controls
appropriate to their function:

- **Pulse Generator** — a *System Settings* group (master enable, pulse
  mode, global repetition rate), a *Channel Configuration* table (one
  row per channel: sync source, delay, width, mode, enable, and a cog
  button for advanced settings), and a timing-diagram pane that previews
  the pulse train.
- **Flow Controller** — gas channel setpoints, channel names, and
  pressure control mode.
- **Pressure Controller** — pressure setpoint, control mode toggle, and
  gate-valve open/close actions.
- **Temperature Controller** — per-channel enable and name assignment.

.. important::
   Control-tab changes are sent to the instrument as each value is
   modified. Closing the dialog with **Close** or **X** does **not**
   undo those commands. A label at the top of the tab reads:
   *"Changes made in this section will be applied immediately."*

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

Each device can declare any number of persistent settings, organized by
priority. The Settings tab presents them through a nested set of
sub-tabs:

- A **Settings** sub-tab containing the Required and Important groups
  (when any are declared).
- An **Advanced** sub-tab, shown only when the device declares advanced
  settings.

The screenshot above shows the Advanced sub-tab for the PulseGenerator;
the Settings sub-tab would render the Required and Important groups
described below.

Each row's tooltip shows the registered description followed by the
native settings key on its own line (``Key: <key>``). The key is the
string passed to ``self.settings.get`` / ``self.settings.set`` from a
:doc:`Python hardware driver <python_hardware/writing_a_driver>` and
to ``SettingsStorage::get`` / ``set`` from C++.

**Required**
   Parameters that must be set correctly at profile creation time (for
   example, the number of digitizer input channels or the AWG output
   count). They appear in a *Required Settings* group on the
   **Settings** sub-tab as read-only text and cannot be changed once a
   profile has been created. To change a Required setting, create a new
   profile via :doc:`hardware_config/profiles`.

**Important**
   Settings with sensible defaults that should be reviewed and
   confirmed. They appear in an *Important Settings* group on the
   **Settings** sub-tab as a two-column table (Setting | Value) with
   editable widgets. Typical examples include communication-level
   defaults, calibration constants, and operating limits.

**Advanced**
   Settings that rarely need to change. They appear on the **Advanced**
   sub-tab as a Setting/Value table. Array-type settings (for example,
   a list of available sample rates) appear as a row with an **Edit**
   button that opens a sub-dialog for adding, removing, or reordering
   entries; the edits are staged until the parent dialog is accepted.

.. important::
   Settings-tab changes are written to persistent storage **only** when
   the dialog is closed with **OK**. Closing with **Close** or **X**
   discards all unsaved changes. A label at the top of the tab reads:
   *"Changes made in this section will only be applied when this dialog
   is closed with the Ok button. Editing these settings incorrectly may
   result in unexpected behavior. Consider backing up your config file
   before making changes."*
