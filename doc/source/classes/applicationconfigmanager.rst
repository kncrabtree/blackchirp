.. index::
   single: ApplicationConfigManager
   single: ApplicationConfig
   single: AppOption
   single: configuration; runtime toggles
   single: LIF; runtime enable toggle
   single: debug logging; runtime toggle
   single: BC::Key::AppConfig

ApplicationConfigManager
========================

``ApplicationConfigManager`` is the application-wide runtime configuration
singleton. It persists application-level settings via ``QSettings`` and exposes
them through a thread-safe API backed by a ``QMutex``. A declarative option
registry (``getOptions()``) describes every available option with its metadata,
enabling automatic UI generation and generic get/set access via
``getOptionValue()`` and ``setOptionValue()``.

The manager owns an ``ApplicationConfig`` value struct that holds the
in-memory state for the LIF module toggle, the CUDA module toggle, and the
debug-logging toggle. Signals notify connected components whenever any of these
values change.

Runtime toggles
---------------

**LIF module** — ``isLifEnabled()`` / ``setLifEnabled()`` control whether the
LIF hardware and UI components are active. The corresponding signal
``lifEnabledChanged(bool)`` is emitted on each change. The LIF-enabled state is
persisted under ``BC::Key::AppConfig::lifEnabled``.

**Debug logging** — ``isDebugLoggingEnabled()`` / ``setDebugLogging()`` control
whether ``Debug``-severity messages are written to the debug log file. Calling
``setDebugLogging()`` persists the setting and emits ``debugLoggingChanged(bool)``.
At application startup, ``MainWindow`` reads the persisted value, applies it to
:doc:`loghandler` via ``LogHandler::instance().setDebugLogging()``, and wires
the ``debugLoggingChanged`` signal to ``LogHandler::setDebugLogging`` so
subsequent changes propagate automatically.

**CUDA module** — ``isCudaEnabled()`` reflects whether the CUDA acceleration
path is active. The ``cudaEnabled`` field is present in the ``ApplicationConfig``
struct; its UI toggle is conditional on build-time CUDA availability.

Option registry
---------------

``getOptions()`` returns the full list of ``AppOption`` entries, each carrying a
``settingsKey``, a display ``label``, a ``description``, a type-aware
``defaultValue``, and a ``requiresRestart`` flag. The registry drives the
**Application Configuration** dialog described in
:doc:`/user_guide/application_config`. Keys are declared as ``constexpr``
``QLatin1StringView`` constants in the ``BC::Key::AppConfig`` namespace:
``appConfig`` (the QSettings group name), ``lifEnabled``, ``cudaEnabled``,
``debugLogging``, and ``appFont``.

.. highlight:: cpp

API Reference
-------------

.. doxygenstruct:: AppOption
   :members:
   :undoc-members:

.. doxygenclass:: ApplicationConfigManager
   :members:
   :undoc-members:
