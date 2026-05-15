.. index::
   single: Profiles; hardware
   single: Hardware Profile
   single: HwSettingsWidget
   single: Required Settings
   single: Important Settings
   single: Optional Settings
   single: Advanced Settings
   single: Add Profile

.. _hardware-config-profiles:

Hardware Profiles
=================

A **hardware profile** is a persistent record that binds three pieces of
information:

- **Type** — the hardware role (e.g., ``FtmwDigitizer``, ``AWG``,
  ``PulseGenerator``, ``FlowController``). The type categorizes which kind
  of hardware this profile represents.
- **Label** — a short, user-chosen name that distinguishes profiles of the
  same type. Labels must be unique within a type. The type and label
  together form the profile's identity (``Type.label``) — a stable
  reference that loadouts use to name the profile.
- **Driver** — the driver class that communicates with the physical
  instrument (e.g., ``SpinCore PB24``, ``Spectrum M4i.44xx``,
  ``Virtual AWG``). The driver is fixed when the profile is
  created and determines which settings the profile exposes and which
  communication protocols are available; to use a different
  driver, create a new profile.

Profiles are created once and then reused across loadouts. Changing a
profile's settings affects every loadout that includes it.

.. figure:: /_static/user_guide/hardware_config-runtimedialog.png
   :width: 800
   :alt: Hardware Configuration dialog with Loadout, Configuration Overview, Hardware Browser, and per-profile Configuration panels
   :align: center

   The Hardware Configuration dialog. From left to right: the **Loadout**
   panel lists saved loadouts and offers loadout-level operations; the
   **Configuration Overview** summarizes the preview hardware map; the
   **Hardware Browser** shows the count of profiles for each supported
   hardware type; and the rightmost panel shows per-profile settings for
   the selection. The validation status bar across the bottom reports
   whether the preview is a valid configuration.

.. _hardware-config-profiles-system:

System Profiles
---------------

Blackchirp automatically creates a *system profile* for each required
hardware type when none exists. System profiles use the ``virtual`` label
and are backed by a virtual (no-op) driver. They satisfy Blackchirp's
requirement that every required hardware type has at least one profile,
and they allow you to start and configure the application before physical
instruments are connected.

System profiles cannot be deleted while they are the only profile of their
type. Add a real driver first, then remove the virtual profile from
any loadout that no longer needs it.

.. _hardware-config-profiles-create:

Creating a Profile
------------------

Open the Hardware Configuration dialog (**Hardware → Hardware Selection**),
select a hardware type in the **Hardware Browser**, then click **Add
Profile** in the Configuration panel on the right.

The **Add Profile** dialog collects:

1. A **label** for the new profile (required; must be unique within the type).
2. The **driver** to use, chosen from the list of available drivers
   for that hardware type.
3. For Python-backed drivers, the path to the Python script.
4. The **communication protocol** (RS232, TCP/IP, GPIB, Custom, or Virtual),
   where applicable.

After selecting a driver, the dialog shows a settings widget
organized into priority sections.

.. figure:: /_static/user_guide/hardware_config-addprofile.png
   :alt: Add Profile dialog showing the Driver, Label, and Required and Important Settings sections
   :align: center

   The Add Profile dialog (here shown for an FtmwDigitizer profile). The
   **Settings** tab contains the **Required Settings** and **Important
   Settings** sections; the **Advanced** tab (if present) hosts optional
   settings.

.. _hardware-config-profiles-priority:

Settings Priority Sections
~~~~~~~~~~~~~~~~~~~~~~~~~~

The settings shown during profile creation (and later, in the hardware
settings dialog) are grouped by priority. The grouping tells you how much
attention each setting deserves before you accept the dialog.

**Required Settings**
    Settings that must be correct before Blackchirp constructs the hardware
    object. Typical examples: the number of digitizer channels, or the
    number of pulse-generator channels. Required settings appear at the top
    of the **Settings** tab as an editable form. After the profile is
    created, these fields become **read-only**; changing them requires
    deleting and recreating the profile.

**Important Settings**
    Settings that have sensible defaults but which you should review for
    your specific instrument. Typical examples: sample rate tables, output
    voltage ranges. Important settings appear in a table below the Required
    section and remain editable after profile creation.

**Optional / Advanced Settings**
    Settings that rarely need changing. These appear under an **Advanced**
    tab. The tab is hidden if the driver has no optional settings.

Hovering over any setting row shows a tooltip with a description of the
setting and its effect.

When you accept the Add Profile dialog, all settings are written to the
Blackchirp configuration file before the hardware object is constructed.
This means the hardware object always starts with the values you chose,
even before it has been connected.

.. _hardware-config-profiles-edit:

Editing Profile Settings After Creation
---------------------------------------

To revisit profile settings after a profile has been created, open
**Hardware → [Device Name]** from the menu bar. The hardware dialog has a
**Settings** tab that hosts the same settings widget used at creation time,
except that Required settings are now shown read-only. Important and
Optional settings remain editable. Changes take effect when you click
**OK**.

.. seealso::

   :doc:`/user_guide/hardware_menu`

.. _hardware-config-profiles-enable:

Enabling and Disabling Profiles
--------------------------------

Within the Configuration panel of the Hardware Configuration dialog, each
profile in the list has an **Enable** checkbox. Disabling a profile removes
it from the active loadout's hardware map without deleting the profile. The
profile's settings are preserved and it can be re-enabled at any time.

Some hardware types are *single-instance*: only one profile of that type
may be active in a loadout simultaneously. These are **FTMW Digitizer**,
**AWG**, **LIF Digitizer**, and **LIF Laser**. Enabling a second profile of a
single-instance type automatically disables the first.

All other hardware types are *multi-instance*: you may enable several
profiles of the same type at the same time (for example, multiple flow
controllers or pulse generators).

.. _hardware-config-profiles-delete:

Deleting a Profile
------------------

Select the profile in the Configuration panel and click **Remove Profile**.
Blackchirp asks for confirmation before deleting. A profile that is the
sole profile of its type in any loadout cannot be deleted until an
alternative is added.

.. seealso::

   :doc:`/user_guide/hardware_config` — chapter overview

   :doc:`/user_guide/hardware_config/loadouts`

   :doc:`/user_guide/hardware_config/ftmw_presets`
