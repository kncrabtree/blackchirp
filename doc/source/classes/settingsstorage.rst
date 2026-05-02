SettingsStorage
===============

``SettingsStorage`` is Blackchirp's wrapper around ``QSettings``.
It maintains an in-memory copy of every key, array, group, and
registered getter under a given ``QSettings`` group, exposes
type-safe ``get`` / ``getArray`` / ``getGroupValue`` accessors for
read-only consumers, and reserves the mutating ``set`` family to
classes that inherit from it. This split is what lets any code in
the program look up a hardware setting (by constructing a transient
``SettingsStorage`` over the appropriate group) while still
guaranteeing that only the owning :cpp:class:`HardwareObject` (or a
declared friend) can change it.

Beyond plain key-value storage, ``SettingsStorage`` supports three
extensions that recur throughout Blackchirp: *array values* (a list
of maps, used for things like pulse generator channels and chirp
segments), *group values* (a nested map, used for protocol-specific
configuration blocks), and *getter registration* (binding a key to a
member function so that the value is computed on demand from the
owning object's live state and re-saved automatically on destruction).
Defaults are usually supplied by the hardware-settings registry —
see ``REGISTER_HARDWARE_SETTINGS`` and friends, applied from
:cpp:func:`HardwareObject::applyRegisteredSettings` — rather than by
ad-hoc ``setDefault`` calls in subclass constructors.

The user-facing surfaces of this storage layer are the profile
creation flow described in :doc:`/user_guide/hardware_config` and the
hardware settings dialog in :doc:`/user_guide/hwdialog`. All persistent
settings keys are declared statically in the ``BC::Key::`` namespace
hierarchy (see ``data/bcglobals.h`` and ``data/settings/hardwarekeys.h``).

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: SettingsStorage
   :members:
   :protected-members:
   :undoc-members:
