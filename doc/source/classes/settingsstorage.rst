SettingsStorage
===============

``SettingsStorage`` is Blackchirp's owning, write-protected wrapper
around a ``QSettings`` group. It maintains an in-memory copy of every
key, array, group, and registered getter under a single ``QSettings``
group, exposes type-safe ``get`` / ``getArray`` / ``getGroupValue``
accessors for read-only consumers, and reserves the mutating ``set``
family to classes that inherit from it. This split is what lets any
code in the program look up a hardware setting (by constructing a
transient ``SettingsStorage`` over the appropriate group) while still
guaranteeing that only the owning :cpp:class:`HardwareObject` (or a
declared friend) can change it.

The user-facing surfaces of this storage layer are the profile creation
flow described in :doc:`/user_guide/hardware_config` and the hardware
settings dialog in :doc:`/user_guide/hwdialog`. All persistent settings
keys are declared statically in the ``BC::Key::`` namespace hierarchy
(see ``data/bcglobals.h`` and ``data/settings/hardwarekeys.h``); the
system-level role of ``SettingsStorage`` in Blackchirp's persistence
model is described in :doc:`/developer_guide/persistence`.

.. highlight:: cpp

Container shapes
----------------

Settings under a group may take any of four shapes; ``SettingsStorage``
keeps each in its own in-memory map:

- **Values** — single key/QVariant pairs. The default shape.
- **Arrays** — vectors of ``SettingsMap``; map directly to
  ``QSettings::beginWriteArray`` / ``beginReadArray``. Useful for
  repeated records like pulse-generator channels or AWG sample-rate
  tables.
- **Groups** — nested ``SettingsMap`` values. Each group is a flat
  key/value table under its own subgroup name; useful for
  protocol-specific or per-driver configuration blocks.
- **Getters** — ``std::function<QVariant()>`` callbacks bound to a
  key. When the key is read, the getter is called; when ``save()``
  runs, the getter's return value is what gets written. Owners use
  getters to keep a key in sync with a member variable or UI widget
  without having to call ``set()`` on every change. Getters apply to
  single-value keys only, never to arrays or groups.

Constructing
------------

``SettingsStorage`` opens its group by calling ``QSettings::beginGroup``
once for each entry in the constructor's ``keys`` list. An empty list
defaults to the top-level ``"Blackchirp"`` group:

.. code-block:: cpp

   SettingsStorage s;                            // top-level Blackchirp group
   SettingsStorage hw({"AWG.main"});             // hardware instance group
   SettingsStorage sub({"AWG.main", "virtual"}); // nested subgroup

Whenever possible, pass key constants from ``BC::Key::*`` (declared in
``bcglobals.h`` and ``hardwarekeys.h``) rather than literal strings so
the compiler catches typos.

The ``Type`` enum and the ``type`` constructor parameter are accepted
for source compatibility but have no effect on the opened group. Always
pass the full keys list; there is no per-type subkey lookup.

Owning a group
--------------

To gain write access, a class inherits from ``SettingsStorage`` and
initializes it with the keys that define its group. Multiple
inheritance with ``QObject`` is supported:

.. code-block:: cpp

   class MyClass : public QObject, public SettingsStorage
   {
   public:
       MyClass(QObject *parent = nullptr) :
           QObject(parent),
           SettingsStorage({BC::Key::MyClass::group})
       { ... }
   };

Inheriting from ``SettingsStorage`` deletes the copy and assignment
operators, so ``SettingsStorage`` owners must always be passed by
pointer or reference — never by value. Data classes that need value
semantics (e.g. :cpp:class:`Experiment`, :cpp:class:`FtmwConfig`) read
from ``SettingsStorage`` rather than inheriting from it.

The destructor calls ``save()``, so any pending changes land in
``QSettings`` when the owner goes away. Owners that bind getters to UI
widgets must call ``clearGetters()`` in their destructor *before* the
widgets are torn down, or ``save()`` will dereference deleted objects.

Setting and saving
------------------

The mutating API revolves around ``set()`` (single value),
``setArray()`` (array), ``setGroupValue()`` / ``setGroupValues()``
(group), and ``registerGetter()`` (callback). Each ``set`` /
``setArray`` takes an optional ``write`` flag: when ``true``, the value
is pushed to ``QSettings`` immediately; when ``false`` (the default), it
stays in memory until ``save()`` runs.

Defaults
--------

Two helpers seed values that should exist on first launch:

- ``setDefault()`` — write ``value`` if the key does not yet exist.
- ``getOrSetDefault()`` — same, but return the resulting value
  (existing or newly written) so the caller can use it inline.

For :cpp:class:`HardwareObject` subclasses, the hardware-settings
registry (see ``hardwareregistration.h`` and
:doc:`/developer_guide/hardware_configuration`) is the preferred way to
declare defaults: settings registered with ``REGISTER_HARDWARE_SETTINGS``
and friends are applied automatically by
:cpp:func:`HardwareObject::applyRegisteredSettings`. Reach for
``setDefault()`` directly only for non-hardware classes.

Reading and rereading
---------------------

The constructor reads the entire group up front. ``readAll()`` rereads
everything; useful when ``QSettings`` has been changed by another actor
(a hardware-settings dialog, a settings editor, an external helper).
Registered getters are skipped on reread, since reading them would
require interrogating the live source object — unregister or
``clearGetters()`` first if you really do want every key reloaded.

Discarding
----------

``discardChanges(true)`` tells the destructor (and the periodic
``save()`` paths) to skip writes. It is most useful when batching
writes inside a transient helper: discard, mutate, undiscard,
``save()`` exactly once. ``discardChanges()`` does not roll back values
that have already been written via the immediate-``write`` flag.

Removing
--------

- ``clearValue()`` removes a single key (across all containers) from
  memory and ``QSettings``.
- ``purge()`` wipes this object's entire group from ``QSettings``,
  clears the in-memory caches, and sets the discard flag so the
  destructor does not re-write anything. Used when permanently
  deleting a hardware profile.
- ``purgeGroup()`` and ``purgeGroupsBySuffix()`` are static helpers
  for removing groups when no live instance over that group exists
  (e.g. cleaning up widget-state groups after deleting a profile).

Friend-class write helper
-------------------------

The ``set`` / ``setArray`` family is protected, so a class that does
not own a group cannot write to it directly. The recommended escape
hatch is a tiny private friend subclass that exposes the protected API
to the trusted parent. :cpp:class:`LoadoutManager` uses exactly this
pattern:

.. code-block:: cpp

   class LoadoutManager
   {
   private:
       class LoadoutHelper : public SettingsStorage
       {
           friend class LoadoutManager;
       public:
           LoadoutHelper(const QStringList &keys) : SettingsStorage(keys) {}
       };

       void p_writeLoadout(const HardwareLoadout &loadout)
       {
           LoadoutHelper sub({key.toString(), loadout.name});
           sub.discardChanges(true);
           sub.setArray(hwMapKey, ...);
           sub.set(currentFtmwPresetKey, ...);
           sub.discardChanges(false);
           sub.save();        // batched write to QSettings
       }
   };

**Use this pattern with care.** The read-public / write-protected split
is the main guardrail against cross-class corruption of persisted
state, and a friend helper that routinely writes to groups owned by
other classes erodes that guarantee silently. The justified use cases
are short — manager classes that compose persistent state across many
owners (loadouts, presets, profile registries) and tests. If a helper
starts being reused from many places or for many groups, that is a
sign the data should move to a class that owns it directly.

Getter example
--------------

Getters allow the owner to expose a member variable as if it were a
stored key. The getter must be a const member function (or a
``std::function<T()>`` lambda) returning a ``QVariant``-compatible type:

.. code-block:: cpp

   class MyClass : public SettingsStorage
   {
   public:
       MyClass();
       int getInt() const { return d_int; }
   private:
       int d_int = 1;
   };

   MyClass::MyClass() : SettingsStorage()
   {
       registerGetter("myInt", this, &MyClass::getInt);
       int i = get<int>("myInt");      // 1
       d_int = 10;
       int j = get<int>("myInt");      // 10

       QVariant k = unRegisterGetter("myInt", false);
       // k == 10; key is now stored as a value, not a getter
       d_int = 20;
       int l = get<int>("myInt");      // still 10
   }

Custom return types must be registered with ``QVariant`` via
`Q_DECLARE_METATYPE <https://doc.qt.io/qt-6/qmetatype.html#Q_DECLARE_METATYPE>`_.

API Reference
-------------

.. doxygenclass:: SettingsStorage
   :members:
   :protected-members:
   :undoc-members:
