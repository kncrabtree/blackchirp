.. index::
   single: HardwareRegistry
   single: HardwareRegistration
   single: HwSettingDef
   single: HwArraySettingDef
   single: CustomCommDef
   single: CustomCommType
   single: registration macros
   single: hardware; registration

HardwareRegistry
================

``HardwareRegistry`` is the singleton catalog that maps hardware-type and
implementation keys to factory functions and metadata. Every concrete hardware
driver calls one or more registration macros at static-initialization time —
before ``main()`` runs — to publish its factory, supported communication
protocols, setting definitions, and custom-connection parameters. The rest of
Blackchirp queries the registry without ever constructing a hardware object,
so dialogs such as ``AddProfileDialog`` and ``CommunicationDialog`` can
populate their widgets from registry data alone.

The registry is a *pure catalog*: it stores what is registered and creates
instances on demand. Availability checking, dependency resolution, and runtime
state are handled by :cpp:class:`RuntimeHardwareConfig` and
:cpp:class:`HardwareProfileManager`.

.. highlight:: cpp

Registration macros
-------------------

Hardware drivers register themselves by placing macro calls at file scope in
their ``.cpp`` files. All macros are declared in ``hardwareregistration.h``.
Registration runs once per process, at static-initialization time, before
``HardwareManager`` is constructed.

``REGISTER_HARDWARE_META(CLASS, DESC)``
   The primary registration macro. Uses Qt's ``staticMetaObject`` to derive
   the hardware-type key (direct child of ``HardwareObject``) and the
   implementation key (the class name itself). Registers a factory lambda and
   the full inheritance chain so that base-class settings are inherited
   automatically. Must appear before any other macro for the same class.

``REGISTER_HARDWARE_PROTOCOLS(CLASS, ...)``
   Registers the communication protocols the implementation supports
   (e.g., ``CommunicationProtocol::Rs232``, ``CommunicationProtocol::Tcp``).
   Must follow ``REGISTER_HARDWARE_META``.

``REGISTER_HARDWARE_SETTINGS(CLASS, ...)``
   Registers one or more :cpp:class:`HwSettingDef` descriptors for the
   implementation. Each descriptor carries a settings key, a user-facing label,
   a tooltip, a type-aware default value, optional bounds, and a
   ``HwSettingPriority`` that controls where the field appears in the UI.
   The ``defaultValue`` type drives the widget: ``int`` → ``QSpinBox``,
   ``double`` → ``ScientificSpinBox``, ``bool`` → ``QCheckBox``,
   ``QString`` → ``QLineEdit``.

``REGISTER_HARDWARE_BASE(CLASS, ...)``
   Registers setting definitions for a non-instantiable base class
   (e.g., ``Clock``, ``FtmwScope``). No prior ``REGISTER_HARDWARE_META``
   call is required. Settings are merged into any implementation whose
   inheritance chain contains the base class name. An implementation can
   override a base-class default by registering the same key with
   ``REGISTER_HARDWARE_SETTINGS``.

``REGISTER_HARDWARE_ARRAY(CLASS, ARRAY_KEY, LABEL, DESC, PRIORITY)``
   Declares an array-type setting with display metadata for a concrete
   implementation. Must precede any ``REGISTER_HARDWARE_ARRAY_ENTRY`` calls
   for the same key.

``REGISTER_HARDWARE_ARRAY_ENTRY(CLASS, ARRAY_KEY, ...)``
   Appends one entry (a ``SettingsStorage::SettingsMap``) to an array setting
   declared by a prior ``REGISTER_HARDWARE_ARRAY`` call.

``REGISTER_HARDWARE_BASE_ARRAY(CLASS, ARRAY_KEY, LABEL, DESC, PRIORITY)``
   Declares an array-type setting for a base class. Ensures the key always
   appears in the settings dialog even for implementations — such as
   Python-backed drivers — that do not supply their own array entries.

``REGISTER_HARDWARE_BASE_ARRAY_ENTRY(CLASS, ARRAY_KEY, ...)``
   Appends one default entry to a base-class array setting.

``REGISTER_LIBRARY(CLASS, LIBRARY_NAME)``
   Links a registered hardware implementation to a :cpp:class:`VendorLibrary`
   singleton. The registry records the dependency so
   ``HardwareRegistry::getLibraryDependencies`` can report which hardware
   must be torn down before a library reload.

``REGISTER_CUSTOM_COMM(CLASS, ...)``
   Registers one or more :cpp:class:`CustomCommDef` descriptors for a
   concrete implementation whose communication type is
   ``CommunicationProtocol::Custom``. Each descriptor specifies a settings
   key, label, description, ``CustomCommType`` (``String``, ``Int``, or
   ``FilePath``), and optional bounds. The UI reads these descriptors before
   construction to render the appropriate input widgets. Must follow
   ``REGISTER_HARDWARE_META``.

``REGISTER_CUSTOM_COMM_BASE(CLASS, ...)``
   Registers ``CustomCommDef`` descriptors for a non-instantiable base
   class. Merged into results for any implementation whose inheritance chain
   includes the base class. No prior ``REGISTER_HARDWARE_META`` call needed.

For Python-backed drivers, connection parameters are declared on the Python
side; ``CommunicationProtocol::Custom`` on such a driver is the explicit
signal that all connection handling is performed by the ``.py`` script.
Python-specific documentation is covered in the Python hardware guide.

The settings-registry developer guide (``dev-docs/settings-registry.md``)
gives worked examples of the full registration pattern. The user-facing
surfaces — profile creation and the hardware settings dialog — are described
in :doc:`/user_guide/hardware_config`.

API Reference
-------------

.. doxygenclass:: HardwareRegistry
   :members:
   :undoc-members:

.. doxygenstruct:: HardwareRegistration
   :members:
   :undoc-members:

.. doxygenclass:: HardwareAutoRegistration
   :members:
   :undoc-members:

.. doxygenstruct:: HwSettingDef
   :members:
   :undoc-members:

.. doxygenstruct:: HwArraySettingDef
   :members:
   :undoc-members:

.. doxygenstruct:: CustomCommDef
   :members:
   :undoc-members:

.. doxygenenum:: CustomCommType

.. doxygenenum:: HwSettingPriority
