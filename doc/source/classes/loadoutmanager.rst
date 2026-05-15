.. index::
   single: LoadoutManager
   single: loadouts; persistence
   single: FTMW Presets; persistence
   single: __LastUsed__

LoadoutManager
==============

``LoadoutManager`` is the singleton that owns the persistent collection
of :cpp:struct:`HardwareLoadout` records and the FTMW presets they
contain. It is the sole writer of the ``Loadouts/`` QSettings subtree:
every other component in the application observes loadouts through this
manager, and every CRUD operation funnels through one of its public
methods. Loadout-level changes propagate to the rest of the application
through the manager's Qt signals; the user-facing model is described in
the :doc:`/user_guide/hardware_config/loadouts` and
:doc:`/user_guide/ftmw_configuration/presets` chapters.

The manager loads every known loadout into an in-memory ``QHash`` cache
during construction, then services subsequent reads against that cache.
Writes update both the cache and QSettings under the appropriate
sub-group, then emit the matching ``loadoutAdded``,
``loadoutChanged``, ``ftmwPresetChanged``, or ``currentFtmwPresetChanged``
signal so observing widgets can refresh. All public methods are
thread-safe via an internal ``QMutex``; signals are emitted on the
thread of the caller that triggered the change.

Loadouts and FTMW presets are paired, but the manager exposes them
through separate CRUD surfaces: ``putLoadout`` / ``removeLoadout`` /
``getLoadout`` for whole-loadout operations, and
``putFtmwPreset`` / ``removeFtmwPreset`` / ``renameFtmwPreset`` for
individual preset operations within a named loadout. The
``loadoutsMatchingHwKey`` helper supports the hardware-profile rename
flow by enumerating every loadout whose hardware map references a given
hardware key.

The ``__LastUsed__`` sentinel
-----------------------------

Each loadout reserves the preset name ``__LastUsed__``
(``BC::Store::LM::lastUsedFtmwPresetName``) for an automatic sentinel
preset that captures the most recent fully accepted FTMW
configuration. The sentinel is hidden from all user-facing dropdowns
and is never user-deletable; ``ftmwPresetNames`` excludes it unless its
``includeLastUsed`` flag is set.

The sentinel is updated only on two events:

- ``FtmwConfigDialog::accept`` — when the user dismisses the FTMW
  configuration dialog with **OK**.
- Experiment start — when an experiment begins with the current FTMW
  configuration.

It is not updated on **Apply**, on **Cancel**, or on incidental edits
to the live configuration. The semantics ensure that, after restoring a
loadout, the application can fall back to ``__LastUsed__`` to recover
the configuration the user most recently committed to, even if no named
preset matches.

QSettings storage layout
------------------------

The keys that name each field and sub-group below are declared in the
``BC::Store::LM`` namespace at the top of ``loadoutmanager.h``. The
RF-side keys used inside ``rfClocks/`` array entries belong to the
``BC::Store::RFC`` namespace; the loadout-specific extensions of that
namespace are declared in ``hardwareloadout.h`` and documented on
:doc:`hardwareloadout`.

.. code-block:: text

   Loadouts/
     currentLoadout = "<name>"
     defaultLoadout = "Default"
     names/                               # array of {name}
     <name>/
       name = "<name>"
       hardwareMap/                       # array of {key, value}
       currentFtmwPreset = "<presetName>" # may be "__LastUsed__"
       ftmwPresetNames/                   # array of {name}
       lastModified = <ISO timestamp>
       ftmwPresets/
         <presetName>/
           rfScalars/
           rfClocks/
           chirpScalars/
           chirpSegments/
           chirpMarkers/
           digiScalars/
           digiAnalog/
           digiDigital/
           digiHwKey = "<hwKey>"
           lastModified = <ISO timestamp>

The flattening from :cpp:struct:`HardwareLoadout` and
:cpp:struct:`FtmwPreset` into this layout is performed by the
``BC::Loadout`` free-function helpers; see :doc:`hardwareloadout` for
the helper signatures.

Relationship to other configuration singletons
----------------------------------------------

``LoadoutManager`` is one of three loosely coupled configuration
singletons. :cpp:class:`HardwareProfileManager` owns profile metadata
and persistence; :cpp:class:`RuntimeHardwareConfig` tracks which
profiles are active for the running session; ``LoadoutManager`` stores
named hardware maps and the FTMW presets that ride on top of them.
Switching the active loadout via the Hardware menu causes
``RuntimeHardwareConfigDialog`` to drive
:cpp:class:`RuntimeHardwareConfig` to the loadout's stored hardware
map, after the drift-detection prompt described in
:doc:`/user_guide/hardware_config/loadouts`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: LoadoutManager
   :members:
   :undoc-members:

.. doxygennamespace:: BC::Store::LM
   :members:
