.. index::
   single: User Interface
   single: Instrument Status
   single: Status Panel
   single: Status Box
   single: Clock Display Box
   single: Gas Flow Display Box
   single: Pressure Status Box
   single: Pulse Status Box
   single: Temperature Status Box
   single: LIF Laser Status Box
   single: Help Menu
   single: About Dialog
   single: Check for Updates; Help menu action

User Interface Overview
=======================

.. image:: /_static/user_guide/ui_overview-window.png
   :width: 800
   :alt: User interface screenshot

Main Toolbar
............

At the top of the window, the main toolbar contains most of the program controls. Depending on the current state of the program, not all of these controls are active.

- ``Acquire`` opens a menu with options for initiating a new experiment.

  - ``Start Experiment`` (hotkey: F2) opens a wizard for fully configuring a new experiment.
  - ``Quick Experiment`` (hotkey: F3) allows repeating a previous experiment or initializing the "Start Experiment" wizard with settings from a previous experiment.
  - ``Start Sequence`` performs a series of identical experiments with a time delay in between.

- ``Hardware`` opens a menu that provides access to hardware controls, communication settings, loadout and preset switching, and per-device dialogs. These are discussed in detail on the :doc:`Hardware Menu <hardware_menu>` page.
- ``Rolling Data`` and ``Aux Data`` are menus with settings pertaining to the :doc:`Rolling/Aux Data <rolling-aux-data>` tabs. Here you can control the number of plots on each tab and, for rolling data, the minimum amount of retained history.
- ``View Experiment`` loads and displays any previously-completed experiment in a new window.
- ``Settings`` contains miscellaneous program settings, including the program :ref:`Data Storage Location <first-run-data-path>`.
- ``Help`` opens a menu with links to online resources and the About dialog (see :ref:`ui-help-menu` below).

On the far right are additional controls that are most relevant during an acquisition:

- ``Pause`` suspends data processing during an active experiment. Any FIDs coming from the FTMW digitizer are discarded and no Aux Data is recorded while the acquisition is paused.
- ``Resume`` continues data processing as usual after an experiment has been paused.
- ``Abort`` terminates an ongoing acquisition. For "Peak Up" and "Forever" FTMW acquisition modes, pressing the abort button is the normal way to stop the experiment.
- ``Sleep`` places Blackchirp and its hardware into a standby state. If this button is pressed during an acquisition, Blackchirp will enter sleep mode when the experiment completes. Each piece of hardware can interpret sleep mode in its own way. At present, a PulseGenerator will stop generating pulses, and a FlowController will shut off all gas flows (but the actual flow rates will continue to be monitored). Other hardware objects do nothing.


.. _ui-instrument-status:

Instrument Status
.................

The left panel of the user interface (visible in the screenshot at the top of
this page) is the instrument status panel. Each item in the panel is a
collapsible **status box** with a title row and a body region. The title row
contains:

- A **collapse/expand** toggle button (chevron icon). Clicking it hides or
  reveals the body of the box, allowing you to reclaim screen space for boxes
  you rarely need to monitor.
- A **bold title label** showing the hardware key (``Type.Label``) of the
  associated device, or a fixed label for non-device boxes.
- A **configure** button (cog icon). For most status boxes, clicking it opens
  the :doc:`Hardware Dialog <hwdialog>` for the associated device, equivalent
  to selecting that device from the Hardware menu. The configure button is
  present on every hardware-backed status box; a few variants (notably the
  Clock Display Box) override its target — see the variant entries below.

Status boxes are added and removed dynamically when the active hardware map
changes (see :doc:`hardware_config/loadouts`). Only devices in the active
loadout have status boxes; devices not in the current loadout do not appear.
Status boxes are disabled (grayed out) when the corresponding device is
offline.

**Experiment info panel**

- ``Expt`` displays the last experiment number in the current save directory,
  or 0 if no experiments have been performed. The number increments upon
  successful initialization of an experiment.
- ``FTMW Progress`` shows the completion percentage of an ongoing FTMW
  acquisition.
- ``LIF Progress`` shows the completion percentage of an ongoing LIF
  acquisition (visible only when the LIF module is enabled).

**Status box variants**

The following status box types ship with Blackchirp. Which boxes appear in a
given session depends on the hardware types present in the active loadout.

*Clock Display Box*
   Shows the current logical clock frequencies (UpLO, DownLO, AwgRef,
   DRClock, DigRef, ComRef) configured in the RF chain. Each row names the
   clock role, its physical hardware assignment, and the most recent
   frequency reading. The title-bar configure button opens the
   **FTMW Configuration** dialog rather than a single
   device dialog, since clock roles are mapped to hardware there. A separate
   cog icon next to each row opens the Hardware Dialog for that row's
   physical clock device.

*Gas Flow Display Box* for flow controllers
   Displays the measured flow rate and setpoint for each gas channel, the
   channel enable state (LED), and the inlet pressure reading and pressure
   control mode. Present when a FlowController is in the active loadout.

*Pressure Status Box* for pressure controllers
   Shows the most recent chamber pressure reading and an LED indicating
   whether pressure control is active. Present when a PressureController is
   in the active loadout.

*Pulse Status Box* for pulse generators
   Shows one LED per pulse channel indicating the channel's current enabled
   state, plus the global pulse generator enable LED and repetition rate.
   Channel labels and tooltips reflect the channel names and timing parameters
   stored in settings. Present when a PulseGenerator is in the active loadout.

*Temperature Status Box* for temperature controllers
   Shows the most recent temperature reading for each enabled channel, labeled
   by channel name. Channels that are disabled in settings are hidden; a
   placeholder message appears when no channels are enabled. Present when a
   TemperatureController is in the active loadout.

*LIF Laser Status Box* for the LIF laser
   Shows the current laser position (wavelength or delay, depending on the
   driver) and a flashlamp enable LED. Present when the LIF module is
   enabled and a LifLaser is in the active loadout.


Display Tabs
............

The majority of important information is displayed in a tabbed interface in the center of the UI.

- ``CP-FTMW`` shows free-induction decay and Fourier transform data from an ongoing or just-completed experiment. More information about the plots and controls on this tab is available on the :doc:`cp-ftmw` page.
- ``LIF`` shows data from an ongoing or just-completed LIF experiment. More details can be found on the :doc:`lif/lif_tab` page.
- ``Rolling Data`` and ``Aux Data`` both show signals from hardware items recorded as a function of time. "Rolling" data is acquired continuously while Blackchirp is open, while "Aux" data is recorded only during an experiment. More details are provided on the :doc:`rolling-aux-data` page.
- ``Log`` shows program-related messages, including warnings and errors. The number of new messages shown since the last time the tab was viewed is displayed in parentheses. Any warnings are indicated with a yellow triangle icon on the tab, and errors are indicated with a red and white "X" icon. When an error occurs, additional information about the cause can be found here. All log messages are recorded to disk in a semicolon-delimited file format under the "log" folder in the current save path. A single log file contains all messages during a given month of program execution. Additionally, any log messages received during an experiment are stored in the same format as ``log.csv`` in the experiment's data folder.


.. _ui-help-menu:

Help Menu
.........

The **Help** toolbar button opens a menu with online resource links and
application information.

Online Resources
~~~~~~~~~~~~~~~~

Three links open the corresponding page in the system web browser:

- **Documentation** — the Blackchirp online user guide at
  ``https://blackchirp.readthedocs.io``.
- **GitHub Repository** — the Blackchirp source repository at
  ``https://github.com/kncrabtree/blackchirp``.
- **Discord Server** — the Blackchirp community Discord server.

Check for Updates
~~~~~~~~~~~~~~~~~

**Check for Updates...** issues an immediate query to the GitHub release
API and shows the result regardless of whether a previous version was
skipped or how recently the last check ran. When a newer release is
known to be available — whether from this manual action or the daily
startup check — the **Help** toolbar button is tinted with the
informational palette color and this action shows a sparkles icon and
the available version. See :ref:`app-config-update-check` for the
underlying behavior and the user-facing toggle that controls the
automatic startup check.

About Blackchirp
~~~~~~~~~~~~~~~~

**About Blackchirp** opens the About dialog. The dialog header shows the
application name, version string, and build commit hash. The body is a
tabbed view with three tabs:

- **Overview** — a short description of Blackchirp, copyright and license
  notice (MIT License), and an *Online Resources* group containing buttons
  for the Documentation, GitHub, and Discord links.
- **Third-Party Libraries** — a table listing the bundled open-source
  libraries (Qt, Qwt, GNU Scientific Library, Eigen3) with their versions
  and licenses.
- **Build Info** — Qt version, operating system, CPU architecture, and
  the enabled optional modules (CUDA, LIF).

**About Qt** opens Qt's standard About Qt dialog, which summarizes the Qt
version and license in use.
