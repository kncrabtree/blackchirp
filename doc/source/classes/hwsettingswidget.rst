.. index::
   single: HwSettingsWidget
   single: HwSettingsMode
   single: hardware settings; profile creation
   single: hardware settings; edit dialog
   single: AddProfileDialog
   single: HWDialog

HwSettingsWidget
================

``HwSettingsWidget`` is an embeddable ``QWidget`` that renders the settings registered
in the :cpp:class:`HardwareRegistry` for a specific hardware type and driver.  It
appears in two places in the hardware workflow:

- **AddProfileDialog (Create mode)** — shown when the user creates a new hardware
  profile.  Required settings are presented as editable typed widgets so the user can
  supply construction-time values before the hardware object is instantiated.
- **HWDialog (Edit mode)** — shown when the user opens an existing profile for editing
  via the :doc:`/user_guide/hardware_menu` or the
  :doc:`/user_guide/hardware_config` pages.  Required settings are rendered as
  read-only text rows because they must not change after the hardware object has been
  constructed.

See the :doc:`/user_guide/hardware_config/profiles` page for the full profile creation
and editing workflow.

Required / Important / Optional tiers
--------------------------------------

Settings are organized into three priority tiers driven by the
:cpp:enum:`HwSettingPriority` values (documented on the
:doc:`HardwareRegistry page <hardwareregistry>`) and registered in the
:cpp:class:`HardwareRegistry`:

- **Required** — construction-time settings that identify the hardware instance (port,
  address, channel assignment, etc.).  Rendered in a ``QFormLayout`` at the top of the
  widget.  Editable in Create mode; read-only in Edit mode.
- **Important** — settings with sensible defaults that the user should review.  Rendered
  in an always-visible two-column table.
- **Optional** — rarely changed settings.  Rendered in a collapsible two-column table
  inside a ``QGroupBox`` labeled *Advanced Settings* in the UI.

Array settings
--------------

Settings declared as array types (``HwArraySettingDef``) appear as a table row showing
the current entry count and an **Edit** button.  Pressing the button opens
``HwArrayEditDialog``, which lets the user add, remove, and reorder the array entries.
The updated entries are stored in the widget and returned by ``arrayValues()``.

Extracting values
-----------------

After the user closes the parent dialog, the caller retrieves the entered data through
two methods:

- ``values()`` — returns all scalar settings as a ``QHash<QString, QVariant>`` keyed by
  :cpp:class:`SettingsStorage` key.  In Create mode this covers all tiers; in Edit mode
  it covers the Important and Optional tiers only (Required settings are read-only).
- ``arrayValues()`` — returns array settings as a
  ``QMap<QString, std::vector<SettingsStorage::SettingsMap>>``.

In Edit mode the caller can alternatively call ``saveToStorage(storageKey)`` to write
both scalar and array values directly to :cpp:class:`SettingsStorage`.

.. highlight:: cpp

API Reference
-------------

.. doxygenenum:: HwSettingsMode

.. doxygenclass:: HwSettingsWidget
   :members:
   :protected-members:
   :undoc-members:
