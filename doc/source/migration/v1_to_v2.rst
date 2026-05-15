.. index::
   single: migration; v1 to v2
   single: upgrading; from v1
   single: configuration profiles; migration from v1
   single: hardware loadouts; migration from v1
   single: AWG markers; migration from v1
   single: QSettings; v1 isolation
   single: LIF Module; runtime toggle
   single: GPIB; runtime protocol selection
   single: Application Configuration; migration from v1
   single: FTMW Configuration; migration from v1
   single: Hardware Dialog; migration from v1
   single: Overlays; migration from v1
   single: hardware.csv; v1 format

Migrating from Blackchirp 1.x to 2.0
=====================================

Blackchirp 2.0 changes the build system, the configuration model, the
on-disk identifiers used to record which hardware ran each experiment,
and several user-interface entry points. This page is a checklist for
v1.x users upgrading an existing installation. Each section names the
v1.x starting condition, the 2.0 end state, and the steps to get from
one to the other. The :doc:`/changelog/2.0.0` release notes are the
authoritative summary of what changed; this page focuses on what you
need to *do* about the changes.

The page assumes you already use Blackchirp 1.x and have a working
configuration plus possibly a body of acquired data. If you are new
to Blackchirp, start with the :doc:`User Guide </user_guide/installation>` instead.

Pre-upgrade Checklist
---------------------

Do the following on your existing v1.x installation before installing
2.0:

#. **Note your current configuration.** Open Blackchirp 1.x and write
   down (or screenshot) the values you depend on: the hardware list
   compiled into the binary, your current ``config.pri`` settings, the
   per-device communication parameters (port, baud, IP address, GPIB
   address), and your AWG protection / gate timings. Blackchirp 2.0
   does not read your v1.x QSettings file (see
   :ref:`v1tov2-qsettings` below), so this list is your reference for
   recreating the configuration in 2.0. Your v1.x settings are not
   lost even if you skip this step: 2.0 writes its settings into a
   separate namespace and never overwrites the v1.x file. The v1.x
   configuration file (or registry key on Windows) remains on disk
   and can be inspected at any time; see :ref:`v1tov2-qsettings`
   below for the per-platform location.
#. **Back up your data directory** as a precaution. Existing
   experiment files remain readable in 2.0, but a backup protects
   against any operator error during the transition.
#. **Locate your v1.x experiment files.** Blackchirp 2.0 can continue
   to use the same data directory as v1.x: the on-disk layout is
   unchanged, the experiment numbering continues uninterrupted, and
   2.0 reads experiment files written by older versions. If you want
   to isolate v1.x and 2.0 data — for example, to keep a clean
   boundary while you validate the upgrade — you can instead point
   2.0 at a fresh directory at first-run time. Either way, have the
   path you intend to use on hand for the first-run dialog.

You do not need to uninstall Blackchirp 1.x before installing 2.0. The
two versions can coexist on the same machine; their QSettings
namespaces are distinct (see :ref:`v1tov2-qsettings`).

Installation
------------

Blackchirp 1.x was distributed as source only; you built it from a
``config.pri`` you edited by hand and the qmake build system. Blackchirp
2.0 replaces qmake with CMake and ships official binary packages.

For most users, **install the binary package** for your platform:
``.deb`` or ``.rpm`` on Linux, ``.AppImage`` for portable Linux use,
``.dmg`` on macOS, or the NSIS ``.exe`` installer on Windows. The
official binaries ship with every supported hardware implementation
compiled in and select between them at runtime through profiles, so
building from source is no longer required to access a particular
driver. See :ref:`installation-binary` for per-platform install
instructions.

A source build is appropriate if you intend to develop new features
or drivers, attach a debugger, or experiment with code changes. To
build from source, follow :ref:`installation-source`. ``config.pri``
is no longer used; build options are configured through CMake cache
variables instead.

First-Time Setup on 2.0
-----------------------

The first time you launch Blackchirp 2.0 it walks you through a
four-step first-run sequence covering the data path, application
settings, hardware onboarding, and library status. The sequence is
described in full on the :doc:`/user_guide/first_run` page; the notes
below highlight what a v1.x user should expect at each step.

#. **Data Storage Location.** Pointing Blackchirp at the same data
   directory you used in v1.x is the seamless path: the on-disk
   layout (``experiments/``, ``log/``, ``rollingdata/``,
   ``textexports/``) is unchanged, and Blackchirp detects the
   highest existing experiment number and continues from there. If
   you would rather keep v1.x and 2.0 data separate, choose a
   different directory at this step instead — there is no loss of
   functionality either way.
#. **Application Configuration.** Set the Application Font and
   confirm the **LIF Module** and **Debug Logging** toggles match
   your needs. Both controls are new (see :ref:`v1tov2-app-config`
   below); leave LIF disabled if your instrument does not have a LIF
   channel.
#. **Hardware Onboarding.** Build your hardware profiles in the
   :doc:`Runtime Hardware Configuration dialog
   </user_guide/hardware_config>` and create a loadout that groups
   them. The per-section guidance under
   :ref:`v1tov2-recreate-config` below describes how each v1.x
   subsystem maps onto the new model.
#. **Library Status.** If your installation depends on the LabJack
   exodriver or Spectrum digitizer SDK, the Library Status dialog
   reports whether Blackchirp can locate the library on your system
   and links to per-platform installation guidance. Use the dialog to
   confirm libraries are findable before continuing.

Each first-run dialog can be re-opened later from the application
menus (see :ref:`first-run-data-path`,
:ref:`first-run-app-config`, :ref:`first-run-hardware-onboarding`, and
:ref:`first-run-library-status`); nothing about the first-run
sequence is one-shot.

.. _v1tov2-recreate-config:

Recreating Your v1.x Configuration
----------------------------------

The bulk of the migration is rebuilding your hardware configuration
in the 2.0 runtime model. Each subsection below describes the v1.x
starting condition, the 2.0 end state, and the steps that bridge the
two.

.. _v1tov2-qsettings:

QSettings Isolation
~~~~~~~~~~~~~~~~~~~

**v1.x state.** All Blackchirp settings — the data path, the active
hardware list, every per-device setting, and the most recent FTMW
parameters — were stored in a single QSettings namespace
(``CrabtreeLab/Blackchirp``).

**2.0 state.** Blackchirp now embeds the major version number in the
QSettings application name, so 2.x stores its settings under
``CrabtreeLab/Blackchirp2`` (and a future 3.x release would write
``Blackchirp3``). Settings written by v1.x are not read by 2.0; the
two versions remain isolated even on the same machine. See
:ref:`app-config-settings-isolation` for the per-platform file paths.

**What to do.** No action is *required*; 2.0 simply does not see your
old settings file. Your v1.x settings are not lost — they remain in
the v1.x namespace, untouched — but you must reconfigure
hardware profiles, the data path, and application settings from
scratch on 2.0. Use the configuration notes you took during the
:ref:`pre-upgrade checklist <v1tov2-recreate-config>` as your
reference. Do not attempt to copy the v1.x QSettings file into the
2.0 namespace; the schema has changed in too many places for a
direct copy to be valid.

Hardware Selection: Compile-Time Lists to Runtime Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x state.** Hardware was selected at compile time. ``config.pri``
contained ``HARDWARE += …`` lines naming the C++ classes to compile
into the binary, and the resulting binary supported exactly that
hardware list. Adding or swapping hardware required editing
``config.pri`` and rebuilding.

**2.0 state.** A single binary supports any combination of the
compiled-in hardware. Hardware is described at runtime through three
layered concepts (see :doc:`/user_guide/hardware_config`):

- **Profiles** identify a single physical or virtual instrument by
  hardware type, label, and driver implementation.
- **Loadouts** group profiles into a complete hardware map for one
  experimental setup; you can define multiple loadouts and switch
  between them.
- **FTMW presets** are named operating points (RF chain, clocks,
  chirp, digitizer settings) saved within a loadout.

**Steps.**

#. Open **Hardware → Hardware Selection** to launch the Runtime
   Hardware Configuration dialog (the same dialog the first-run
   sequence presents). See
   :ref:`hardware-config-profiles-create`.
#. For each device in your v1.x ``HARDWARE +=`` list, create a
   profile by selecting the hardware type, choosing the driver
   implementation, and assigning a label. Labels are user-defined
   text — pick something that identifies the physical instrument
   (e.g. ``frontPanel`` or ``main``).
#. Group your profiles into a loadout
   (:ref:`hardware-config-loadouts-dialog`). Most installations need
   only one loadout; create additional loadouts only if you switch
   between distinct hardware setups.
#. Save the loadout and exit the dialog. The active loadout drives
   the menu entries, the Hardware Configuration display, and the
   experiment-info panel.

GPIB Controllers: Runtime Protocol Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x state.** GPIB-LAN and GPIB-USB / GPIB-RS232 controllers were
distinct compile-time hardware classes; supporting both required two
separate builds.

**2.0 state.** The Prologix GPIB-LAN and GPIB-USB controllers are
each their own profile implementation with its own communication
protocol. They can be created side-by-side in the same loadout: one
profile for the LAN controller (TCP protocol) and another for the
USB controller (RS232 protocol), each with its own label.

**Steps.**

#. In the Runtime Hardware Configuration dialog, create one
   ``GpibController`` profile per physical bridge. Choose
   ``Prologix GPIB-LAN`` or ``Prologix GPIB-RS232`` (USB) for each
   profile, and assign each a distinct label.
#. Add each ``GpibController`` profile to the loadout you intend to
   use it from.
#. Configure the per-controller communication parameters from
   **Hardware → Communication** (see
   :ref:`hardware-menu-communication`). Each GPIB controller exposes
   its own protocol fields (IP address for the LAN controller,
   serial port and baud rate for the USB controller).
#. For each GPIB instrument, set its ``GpibAddress`` setting in the
   per-device :doc:`Hardware Dialog </user_guide/hwdialog>`, and
   make sure its communication protocol points at the matching
   controller profile.

Marker Timing: Spinboxes to Marker Table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x state.** AWG protection and gate timing were configured by
four spinboxes on the chirp configuration page: **Pre-Chirp
Protection**, **Pre-Chirp Delay**, **Post-Chirp Delay**, and
**Post-Chirp Protection**. The defaults were 0.5 µs on each. The
"Delay" labels controlled the gate marker; "Protection" labels
controlled the protection marker. Marker channel assignment to the
AWG outputs was hard-coded.

**2.0 state.** A generalized marker model replaces the four
spinboxes. Each AWG output marker is described as a row in the
marker table on the **Markers** sub-tab of the Chirp Config tab;
each row carries a name, a role (Protection, Gate, Trigger, or
Custom), a start time relative to the chirp start, an end time
relative to the chirp end, and an enabled flag. Channel assignment
is explicit: the row index is the physical AWG marker output (0 for
output 1, 1 for output 2, and so on). See
:ref:`chirp-setup-markers` for the full reference and
:ref:`chirp-setup-marker-validation` for the safety checks the new
model performs.

**Steps.**

#. After creating a profile for your AWG, open
   **Hardware → FTMW Configuration** and switch to the
   :ref:`Markers sub-tab <chirp-setup-markers>` of the Chirp Config
   tab.
#. If your AWG reports two or more marker channels, Blackchirp
   pre-populates the table with **Protection** on channel 0 and
   **Gate** on channel 1, both timed at −0.5 µs / +0.5 µs. This is
   roughly equivalent to v1.x's 0.5 µs default on all four
   spinboxes; if you used the defaults, no further action is
   required.
#. If you customized the v1.x spinbox values, translate them into
   the new timings using the formulas below. Let ``preProt``,
   ``preGate``, ``postGate``, and ``postProt`` be your v1.x values
   (in microseconds):

   - **Protection row.** Set Start = ``-(preProt + preGate)`` and
     End = ``+postProt``.
   - **Gate row.** Set Start = ``-preGate`` and End = ``+postGate``.

   For example, v1.x defaults of 0.5 µs across all four spinboxes
   yield Protection = −1.0 / +0.5 µs and Gate = −0.5 / +0.5 µs in
   the new model — slightly wider than the 2.0 default but a
   faithful reproduction of v1.x behavior.
#. Confirm both rows are **Enabled** and the role column is set
   correctly. The validator emits a warning if the Protection
   window does not fully enclose the Gate window or the chirp
   itself.
#. Save the configuration as a named FTMW preset
   (:doc:`/user_guide/ftmw_configuration/presets`) so the timings
   are preserved across sessions.

.. warning::

   Marker timings drive the protection and gate signals that keep
   sensitive receiver amplifiers from being damaged by the chirp.
   Before running a live experiment with amplifiers connected,
   disconnect them from the AWG outputs, connect the AWG marker
   outputs directly to an oscilloscope, and verify on the scope
   that each marker pulse rises and falls exactly when you expect
   relative to the chirp. Confirm both polarity and timing for
   every enabled marker channel. Only reconnect the amplifiers
   once the marker behavior has been verified.

Hardware Identification on Disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x state.** Each experiment recorded a ``hardware.csv`` file
listing the hardware classes present in the build. Hardware keys
were numeric (``AWG.0``, ``Clock.0``, ``FtmwDigitizer.0``), and the
second column held the C++ implementation class string (for
example, ``awg70002a`` or ``valon5009``).

**2.0 state.** The same file is still written, with two columns:
``key`` and ``driver``. The ``key`` field now uses the user-assigned
label from the runtime hardware configuration in place of the
numeric index, so you might see entries like
``FlowController.frontPanel`` or ``FtmwDigitizer.virtual``. The
``driver`` field records the driver class identifier (the same
information the v1.x ``subKey`` column carried). The hardware-type
discriminator is recovered from the key prefix; an interim 2.0
development format also wrote a third ``hardwareType`` column
holding the integer enum value, and the loader silently ignores
that column when present so those transitional fixtures load
unchanged. Old experiments remain readable: the loader handles the
v1.x two-column layout, the very-old single-column layout, and
either header label (``subKey`` or ``driver``).

**What to do.** Nothing for existing experiment files; they remain
readable as-is. New experiments record using the label-based scheme.
If you maintain external scripts that parse ``hardware.csv``, expect
the two-column ``key;driver`` layout for new captures and accept
either ``subKey`` or ``driver`` in the header row when supporting
older experiments; the example block in
:doc:`/user_guide/data_storage` shows the current format.

.. note::

   Older versions of Blackchirp may not be able to read experiment
   files written by 2.0. If you take data on 2.0 and then need to
   roll back to v1.x for any reason, use the ``blackchirp-viewer``
   shipped with 2.0 to inspect the newer files in the meantime;
   the viewer is built alongside the main application and can be
   run independently of it. Once you upgrade back to 2.0 the files
   load directly into the main application again.

Quick Experiment Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x state.** Quick Experiment ("repeat this experiment") was
gated on whether the saved experiment's hardware list matched the
compile-time hardware list of the running binary.

**2.0 state.** Quick Experiment compares the saved experiment's
hardware map (per-key implementation strings) against the
**currently active** hardware map in the Runtime Hardware
Configuration — not against any named loadout's saved state. If
you have a named loadout selected but have edited the runtime
configuration without saving, the comparison uses the live (edited)
map. A v1.x experiment recorded under the numeric-key scheme will
not match a 2.0 hardware map directly; even two 2.0 experiments
will only match if their keys (``Type.label``) and implementation
strings are identical. The repeat is also gated on the major
version of Blackchirp that wrote the experiment.

**What to do.** To repeat a v1.x experiment in 2.0 you generally
need to recreate it from scratch using the experiment wizard rather
than the Quick Experiment shortcut. The original v1.x data is still
readable for analysis; only the one-click "repeat" path requires a
matching 2.0 hardware map.

LIF: Compile-Time Build Option to Runtime Toggle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x state.** The Laser-Induced Fluorescence module was a
compile-time build option (``BC_LIF`` in ``config.pri``).
Enabling LIF required a separate build.

**2.0 state.** The LIF module is built into every binary. Whether
its UI surfaces and hardware types are active is controlled at
runtime by the **LIF Module** checkbox in the Application
Configuration dialog (see :ref:`application-config`). The toggle
requires a restart to take effect.

**Steps.**

#. Open **Settings → Application Settings**.
#. Toggle **LIF Module** to match your instrument: enable it if you
   have a LIF or REMPI channel, disable it otherwise.
#. Click **OK** and restart Blackchirp.
#. After the restart, LIF hardware types appear in the Runtime
   Hardware Configuration dialog and the LIF tab is accessible from
   the main window. Build the LIF profiles into your loadout the
   same way as any other hardware. See
   :doc:`/user_guide/lif/experiment_setup`,
   :doc:`/user_guide/lif/configuration`, and
   :doc:`/user_guide/lif/lif_tab` for the rest of the LIF workflow.

.. _v1tov2-app-config:

Application Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x state.** Application-wide settings (font, save path, and a
small number of others) were spread across separate menu entries.

**2.0 state.** A single Application Configuration dialog
consolidates the application-wide settings into one place,
reachable from **Settings → Application Settings** or from the cog
button on the experiment-info panel. The dialog groups its options
into *Application Settings* (Font, LIF Module toggle, Debug Logging
toggle) and *Data Storage* (the same path widget the first-run
sequence presents). The full reference is on the
:doc:`/user_guide/application_config` page.

**What to do.** Each former menu entry has a destination in the new
dialog:

- **Font.** *Application Settings* group, **Application Font** row
  with a **Change…** button.
- **Data path / experiment number.** *Data Storage* group; same
  widget as the first-run *Data Storage Location* step.
- **LIF enabled.** *Application Settings* group, **LIF Module**
  checkbox (new in 2.0; replaces the v1.x build option).
- **Debug logging.** *Application Settings* group, **Debug
  Logging** checkbox (new in 2.0; writes ``debug_YYYYMM.csv``
  files into the ``log`` subdirectory while enabled).

If a v1.x menu entry is not listed above, it has either been
folded into the per-device :doc:`Hardware Dialog
</user_guide/hwdialog>` (per-device settings) or rendered obsolete
by the runtime configuration model.

FTMW Configuration Menu
~~~~~~~~~~~~~~~~~~~~~~~

**v1.x state.** RF-chain configuration, chirp configuration, and
digitizer configuration each had their own dialog or page,
typically reached through an **RF Configuration** entry on the
Hardware menu.

**2.0 state.** The three subsystems are consolidated into a single
:ref:`ftmw-configuration` dialog reachable from
**Hardware → FTMW Configuration**. The dialog has a preset bar at
the top and three tabs below: **RF Config**, **Chirp Config**, and
**Digitizer Config**. The contents of each tab match the equivalent
v1.x dialog page.

**What to do.** When you previously opened **Hardware → RF
Configuration** (or its v1.x equivalent), open
**Hardware → FTMW Configuration** instead and switch to the tab you
want. The :ref:`preset bar at the top of the dialog
<ftmw-preset-bar>` is also new (see
:doc:`/user_guide/ftmw_configuration/presets`); it lets you save
the full FTMW operating point under a name and switch between
operating points within the active loadout.

Hardware Dialog: Field Hiding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x state.** The per-device :doc:`Hardware Dialog
</user_guide/hwdialog>` exposed every hardware field, including
``commType`` (the communication protocol) and ``model`` (the driver
implementation), as editable controls in the dialog body.

**2.0 state.** The Hardware Dialog focuses on hardware-specific
*settings*. The driver implementation (``model``) is set when the
profile is created in the Runtime Hardware Configuration dialog and
is shown read-only at the top of the Hardware Dialog. The
communication protocol (``commType``) is moved out of the per-device
dialog into a dedicated **Communication Settings…** button (and its
companion **Hardware → Communication** menu entry); see
:ref:`hardware-menu-communication`.

**What to do.**

- To **switch a device to a different driver implementation**, you
  do not edit the existing profile — the implementation is part of
  the profile's identity and cannot be changed in place. Instead,
  create a new profile with the desired implementation in the
  Runtime Hardware Configuration dialog (see
  :ref:`hardware-config-profiles-create`), add it to the loadout
  you intend to use, and remove the old profile from that loadout
  if it is no longer needed.
- To **change the communication protocol or its parameters**
  (RS232 port, TCP address, GPIB address, etc.), open the
  Hardware Dialog for the device and click **Communication
  Settings…**, or open **Hardware → Communication** and select the
  device.
- To **change other per-device settings** (anything that v1.x
  exposed as a hardware setting), use the Settings tab of the
  Hardware Dialog as before. The set of settings each device
  exposes is now declared by a registry, so the dialog contents
  may differ from the v1.x layout for the same device.

Working with v1.x Data
----------------------

v1.x experiment files remain readable in 2.0 with no special
action: open the experiment by number from the **View Experiment**
dialog and the loader handles the older ``hardware.csv`` layout
transparently. Only the Quick Experiment workflow requires a
matching 2.0 hardware map (see above).

A few small format additions land with 2.0; the
:doc:`/user_guide/data_storage` page is the canonical reference and
covers each new file in detail.

- ``markers.csv`` — written under each new experiment directory,
  recording the AWG marker channel definitions for the run. v1.x
  experiments do not have this file; 2.0 reads only the markers
  the active configuration specified.
- ``version.csv`` — already present in v1.x; 2.0 records the new
  major version number.
- ``hardware.csv`` — keys now embed user-assigned labels rather than
  numeric indices and the second column has been renamed from
  ``subKey`` to ``driver``, both described in
  :ref:`v1tov2-recreate-config` above.

The :doc:`Overlays </user_guide/overlays>` subsystem was introduced
in v1.1; the on-disk schema has not changed in 2.0, so any
overlays attached to a 1.1 experiment continue to load without
modification. Experiments acquired on 1.0 simply do not have
overlays attached, and no migration action is required for either
generation of legacy data.

Where to Go Next
----------------

- The :doc:`/changelog/2.0.0` release notes list every user-visible
  change, with cross-links to the user-guide pages that document
  each new feature in depth.
- The :doc:`User Guide </user_guide/installation>` covers the 2.0 workflow end
  to end. New chapters that have no v1.x analogue include
  :doc:`/user_guide/python_hardware`,
  :doc:`/user_guide/overlays`, and
  :doc:`/user_guide/hardware_config/library_status`; chapters such as
  :doc:`/user_guide/hardware_config` and
  :doc:`/user_guide/hwdialog` describe the runtime model
  introduced in this release.
- If you run into a feature that you used in v1.x and cannot find
  in 2.0, check the :doc:`/changelog/2.0.0` page first; most
  former menu entries have been moved or renamed rather than
  removed.
