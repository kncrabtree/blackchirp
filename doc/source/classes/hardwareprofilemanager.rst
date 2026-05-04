.. index::
   single: HardwareProfileManager
   single: hardware; profiles
   single: profiles; hardware
   single: Python hardware; profile fields

HardwareProfileManager
======================

``HardwareProfileManager`` is the singleton that persists and manages hardware
profiles. A *profile* ties a user-chosen label (e.g., ``"frontPanel"``) to a
hardware-type/driver pair and stores whether it is active, along with
timestamps and an optional description. Profiles use human-readable keys of
the form ``<Type>.<label>`` (``"FlowController.frontPanel"``,
``"FlowController.backup"``) so that hardware identity survives reconfiguration.

Profiles are stored under the ``HardwareProfiles`` QSettings group, using
the path ``<Type>/<label>/<subkey>``. Every mutating operation is thread-safe
via an internal ``QReadWriteLock``. Profile data is loaded from QSettings
during construction and flushed on destruction; ``saveProfiles()`` and
``loadProfiles()`` allow explicit control when needed.

Labels must be non-empty, start with a letter, contain only letters, digits,
and hyphens, and not exceed 64 characters. ``validateLabel()`` and
``isValidLabel()`` enforce these rules; ``generateDefaultLabel()`` produces a
collision-free label drawn from the candidate list ``Default``, ``Main``,
``Primary``, ``Secondary``, ``Backup`` (falling back to ``<Type>1``,
``<Type>2``, … if all are taken). Mutating operations return explicit
success/failure values rather than substituting defaults, so callers must
handle errors.

Storage layout and usage
------------------------

Profiles live under the ``HardwareProfiles`` ``QSettings`` group, with the
path ``<Type>/<label>/<subkey>``:

.. code-block:: ini

   [HardwareProfiles]
   FlowController/frontPanel/implementation=mks647c
   FlowController/frontPanel/active=true
   FlowController/frontPanel/created=2024-01-15T10:30:00
   FlowController/frontPanel/description=Main flow controller
   FlowController/backup/implementation=virtual
   FlowController/backup/active=false

Typical mutation/query sequence:

.. code-block:: cpp

   HardwareProfileManager manager;

   // Create profiles with meaningful labels.
   QString label1 = manager.createHardwareProfile("FlowController", "mks647c", "frontPanel");
   QString label2 = manager.createHardwareProfile("FlowController", "virtual", "backup");

   // Query.
   QStringList active = manager.getActiveProfiles("FlowController");
   QString impl       = manager.getImplementation("FlowController", "frontPanel");

   // Manage state.
   manager.deactivateHardwareProfile("FlowController", "backup");
   manager.deleteHardwareProfile("FlowController", "backup");

Per-profile Python fields
-------------------------

Three fields stored per profile support Python-backed hardware drivers:

``pythonScriptPath``
   Absolute path to the ``.py`` script that implements the driver. Persisted
   under ``pythonScriptPath`` in QSettings. Read by the Python hardware
   trampoline when constructing the object.

``pythonClassName``
   Name of the Python class inside the script. Persisted under
   ``pythonClassName``. Must be a valid Python identifier that inherits from
   the appropriate Blackchirp Python base class.

``pythonEnvPath``
   Path to the virtual environment or conda environment directory
   (``<env>/bin/python3``). Persisted under ``pythonEnvPath``. An empty
   string means the system ``python3`` executable is used.

These fields are only meaningful for Python driver keys; non-Python
drivers leave them empty. Python hardware is described in
:doc:`/user_guide/python_hardware`.

System profiles
---------------

Some hardware types are required for Blackchirp to operate (e.g., the FTMW
clock and digitizer). For each required type, ``HardwareProfileManager``
guarantees a *system profile* — a profile with the label ``virtual`` backed
by the corresponding virtual driver. ``ensureSystemProfiles()`` creates
these on startup if they are missing, and ``isSystemProfile()`` flags them
so the UI can prevent deletion or relabeling.

Relationship to RuntimeHardwareConfig
-------------------------------------

``HardwareProfileManager`` owns all profile metadata. ``RuntimeHardwareConfig``
tracks which profiles are *active* — i.e., participating in the current
hardware configuration — and is refreshed to match the manager when profiles
are activated or deactivated. The profile creation and management UI is
described in :doc:`/user_guide/hardware_config`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: HardwareProfileManager
   :members:
   :undoc-members:

.. doxygenstruct:: HardwareProfileData
   :members:
   :undoc-members:
