.. index::
   single: First Run
   single: Data Storage; location
   single: Application Configuration; first run
   single: Hardware Onboarding
   single: Library Status; first run

.. _first-run:

First Run
=========

When Blackchirp starts for the first time on a new system (or with a
fresh configuration), it guides you through a short setup sequence before
opening the main window. The sequence covers four steps:

1. :ref:`first-run-data-path` — choose where experiment files are stored.
2. :ref:`first-run-app-config` — review and adjust application settings.
3. :ref:`first-run-hardware-onboarding` — assign hardware drivers to
   each required hardware type and create your first loadout.
4. :ref:`first-run-library-status` — check whether optional vendor libraries
   (LabJack, Spectrum) are available on this system.

Each step is presented as a dialog. You can revisit any of them later from
the application menus without repeating the full sequence.

.. _first-run-data-path:

Data Storage Location
---------------------

.. figure:: /_static/user_guide/first_run/savepathdialog.png
   :alt: Data storage path dialog

   The initial configuration dialog with the data storage section.

Use the **Browse** button to choose a directory, or type the path directly.
Click **Apply** once to create the directory structure. Blackchirp creates
four subdirectories:

``experiments``
    Experiment data files, one subdirectory per experiment number.

``log``
    Program log files. Debug logs are also written here when debug logging
    is enabled (see :ref:`app-config-debug-logging`).

``rollingdata``
    CSV files containing continuous monitoring (rolling / auxiliary) data.

``textexports``
    Default destination for XY text exports from plot views.

Experiment files are stored by number: experiment 1 is first, and the number
increments automatically. If you point Blackchirp at a directory that already
contains data, it detects the highest existing experiment number and sets the
next number accordingly. You can also set the starting number manually with
the spin box.

After clicking **Apply**, the **OK** button becomes active. Click **OK** to
proceed to the application settings step.

.. note::

   The data path and experiment number are stored in the Blackchirp
   configuration file. On Linux the default location is
   ``~/.config/CrabtreeLab/Blackchirp2.conf``. See
   :ref:`app-config-settings-isolation` for details on the versioned
   configuration namespace.

.. _first-run-app-config:

Application Configuration
--------------------------

After the data path is set, Blackchirp presents the Application Settings
dialog. On first run the dialog is titled *Welcome to Blackchirp - Initial
Configuration*; the same dialog is available later via
**Settings → Application Settings**.

See :doc:`application_config` for a complete reference. On first run, the
most relevant options are:

- **Check for Updates** — enabled by default. Blackchirp contacts the GitHub
  release API once per day at startup to check for newer stable releases.
  Uncheck this on air-gapped or policy-restricted systems where any
  outbound HTTPS traffic is unwanted. See :ref:`app-config-update-check`.
- **LIF Module** — enable or disable the Laser-Induced Fluorescence hardware
  and UI components. Change this only if your instrument has LIF, REMPI, or other laser-based detection capabilities. This setting requires a restart to take effect.
- **Debug Logging** — enable verbose debug messages written to a dated log
  file. Disabled by default; leave it off unless you are troubleshooting.
- **Application Font** — adjust the font and font size used throughout the UI.

Click **OK** to apply your choices and advance to hardware onboarding.

.. _first-run-hardware-onboarding:

Hardware Onboarding
--------------------

.. figure:: /_static/user_guide/first_run/onboarding-runtimeconfig.png
   :width: 800
   :alt: Runtime Hardware Configuration dialog on first run

   The Hardware Configuration dialog. Use this to assign hardware
   drivers and create your first loadout.

The Hardware Configuration dialog is where you tell Blackchirp which
physical or virtual instruments are connected and how they should be
configured. The dialog uses a four-panel horizontal layout:

**Loadout** (leftmost panel)
    Lists all saved loadouts. A *loadout* is a named collection of hardware
    assignments — one or more profiles per hardware type. Use the buttons
    below the list to **Activate**, **Save**, **Save As**, **Copy**, or
    **Delete** a loadout. The currently active loadout is highlighted.

**Configuration Overview**
    Previews the hardware that will be active if you accept the dialog.
    Hardware types and the profiles assigned to each are shown as a tree.
    This panel reflects pending edits made in the panels to its right.

**Hardware Browser**
    Lists every hardware type Blackchirp supports — for example, FTMW
    digitizer, AWG, pulse generator, flow controller, IO board. Select a
    type to configure profiles for it.

**Configuration** (rightmost panel)
    Displays the profile list for the type selected in the Hardware Browser
    and the settings for the currently selected profile. Use this panel
    to add, remove, enable, or disable profiles, and to edit profile
    settings. Fields vary by instrument type and driver.

Some hardware types are *single-instance*: only one profile of that type
may be active in a loadout at a time. Single-instance types are
**FTMW Digitizer**, **AWG**, **LIF Digitizer**, and **LIF Laser**. All other types
are *multi-instance*: a loadout may enable several profiles of the same
type simultaneously (for example, multiple flow controllers or
pulse generators).

The validation status bar at the bottom of the dialog reports whether the
preview configuration is valid. The dialog cannot be accepted while
validation is failing.

At minimum, create one loadout before closing this dialog. Blackchirp
requires a complete and valid loadout before an experiment can be started.

.. note::

   Detailed walkthroughs of profile creation, the Add Profile flow, loadout
   management, and FTMW presets are covered in
   :doc:`/user_guide/hardware_config`. The day-to-day Hardware menu
   navigation is covered in :doc:`/user_guide/hardware_menu`.

The **Library Status** tab within this dialog provides a quick view of
vendor-library availability. See :ref:`first-run-library-status` below and
the full reference at :doc:`library_status`.

.. _first-run-library-status:

Library Status
--------------

The Library Status tab (accessible inside the Hardware Selection dialog,
which is reopened any time from **Hardware → Hardware Selection**) lists
every optional vendor
library that Blackchirp supports. A library is *optional* in the sense that
Blackchirp can start and operate without it; however, any hardware that
depends on a missing library will be non-functional.

On first run, review this tab to confirm that libraries for your hardware are
detected. Libraries with status **Available** are loaded and ready. Libraries
with status **Not Found** or **Error** require attention before the
corresponding hardware can be used.

See :doc:`library_status` for full instructions on installing vendor
libraries and supplying custom search paths.

After closing the Hardware Configuration dialog, Blackchirp opens the main
window. You are ready to run experiments.
