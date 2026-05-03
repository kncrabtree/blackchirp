.. index::
   single: vendor libraries; developer guide
   single: VendorLibrary; developer guide
   single: LabjackLibrary; developer guide
   single: SpectrumLibrary; developer guide
   single: REGISTER_LIBRARY
   single: BC::Labjack facade
   single: dynamic library loading
   single: QLibrary
   single: staged configuration
   single: HardwareRegistry; library dependencies

Vendor Libraries
================

Blackchirp talks to several pieces of laboratory hardware through closed-source
vendor SDKs (LabJack's exodriver/UD, Spectrum Instrumentation's spcm). Linking
those SDKs at compile time would tie each binary to the set of libraries
present on the build machine and would force a separate rebuild for every
deployment. The :cpp:class:`VendorLibrary` family lets Blackchirp ship as a
single binary that resolves vendor SDKs at runtime through ``QLibrary``: if a
library is present, the dependent hardware comes online; if it is absent,
the dependent hardware reports itself as unavailable and the rest of the
application starts normally.

This page documents the contract :cpp:class:`VendorLibrary` imposes on its
subclasses, the staged-configuration model the
:cpp:class:`LibraryStatusWidget` uses to edit library paths without disturbing
running hardware, the ``REGISTER_LIBRARY`` linkage between hardware
implementations and libraries, the two concrete subclasses, the LabJack
``exo``/``UD`` cross-platform split as a worked example, and a recipe for
adding a new ``VendorLibrary`` subclass. Per-class API detail lives on
:doc:`/classes/vendorlibrary` and :doc:`/classes/hardwareregistry`; the
user-facing surface is :doc:`/user_guide/library_status`.

Why dynamic loading
-------------------

The constraint that drives the design is that Blackchirp must run on machines
that lack any given vendor SDK. A laboratory installation may have only the
LabJack driver, only the Spectrum driver, both, or neither. Statically
linking against either SDK would force at least four binary variants and
would prevent a binary distribution from supporting hardware whose driver is
not installed on the build machine.

Each :cpp:class:`VendorLibrary` subclass uses ``QLibrary`` to locate and load
its vendor library at runtime. If the library is absent or fails to load,
``isAvailable()`` returns ``false`` and ``errorString()`` carries the reason;
the dependent hardware implementations then surface the failure when the user
tries to connect them, but the application itself starts normally and other
hardware is unaffected. The result is one binary, runtime-discovered hardware
support.

VendorLibrary contract
----------------------

:cpp:class:`VendorLibrary` is an abstract :cpp:class:`QObject` that also
inherits from :cpp:class:`SettingsStorage` so each library can persist its
own search paths under
``vendorLibraries/<libraryKey>/`` in the application settings. Concrete
subclasses are singletons: each exposes a static ``instance()`` that returns
a reference to the per-process instance, ensuring a single :cpp:class:`QLibrary`
is responsible for every load attempt and every function-pointer cache.

A subclass implements four pure virtuals that drive the load:

- ``libraryName()`` returns the human-readable display name shown in the
  Library Status widget.
- ``platformLibraryNames()`` returns the candidate filenames to try, ordered
  most-likely first. A platform-specific subclass may compile in a different
  list per ``Q_OS_*`` block.
- ``defaultSearchPaths()`` returns the conventional install locations for
  the platform (``/usr/local/lib`` and ``/opt/spectrum/lib`` on Linux,
  ``C:/Windows/System32`` and the LabJack/Spectrum install directories on
  Windows, and so on).
- ``loadFunctions()`` is invoked by the base class once :cpp:class:`QLibrary`
  reports a successful load. The subclass calls ``resolveFunction()`` for
  each symbol it needs, stores typed function pointers as data members, and
  flips ``d_libraryLoaded`` to ``true`` once every required symbol has
  resolved.

The base class drives the actual load. Calling ``loadLibrary()`` (or
``reloadLibrary()`` after a settings change) walks an ordered candidate list
built from:

1. The active user-provided path, if set.
2. The last path that loaded successfully, if different from the user path.
3. The active user-specified search directories.
4. The platform default search paths, if automatic discovery is enabled.

For each candidate directory the base class tries each platform library name,
then falls back on bare names (which lets the system dynamic linker take over)
and the explicit system paths from :cpp:func:`QCoreApplication::libraryPaths`.
Once :cpp:class:`QLibrary` reports a load, the base class delegates to the
subclass's ``loadFunctions()``. If essential symbols are missing the subclass
sets ``d_libraryLoaded = false`` and updates ``d_errorString``; the base
class then unloads the library and tries the next candidate. The first
candidate that produces both a loaded library and a complete symbol set
wins, and its path is persisted as ``lastWorkingPath`` so the next process
start tries it first.

Loaded function pointers live as public (or near-public) typed members on
the subclass:

.. code-block:: cpp

   auto &lib = SpectrumLibrary::instance();
   if (!lib.isAvailable())
       return false;
   void *hDevice = lib.spcm_hOpen("/dev/spcm0");

Hardware code calls those pointers exactly as it would call a statically
linked function. There is no per-call overhead beyond the indirect-call cost
of a function-pointer dereference.

Lifecycle is straightforward: the singleton's constructor calls
``loadLibrary()`` at first use, the library stays loaded for the life of the
process, and Qt's ``destroyed`` signal on the embedded :cpp:class:`QLibrary`
clears the cached state at shutdown. ``reloadLibrary()`` unloads, re-resolves
search paths from the active configuration, and re-attempts the load — this
is the path the staged-configuration UI exercises when the user changes
search paths from the Library Status widget.

Staged configuration
--------------------

The Library Status widget in the Hardware Configuration dialog must let the
user edit a library's search paths without yanking the library out from
under hardware that may already be using it. :cpp:class:`VendorLibrary`
solves this by carrying *two* configurations side by side:

- An **active** configuration (``d_activeUserPath``, ``d_activeSearchPaths``,
  ``d_activeAutoDiscovery``) that is what the most recent successful load
  used and is what hardware code sees through ``getActiveUserProvidedPath()``,
  ``getActiveSearchPaths()``, and ``isActiveAutoDiscoveryEnabled()``.
- A **staged** configuration (``d_stagedUserPath``, ``d_stagedSearchPaths``,
  ``d_stagedAutoDiscovery``) that the UI mutates freely through
  ``setStagedUserProvidedPath()``, ``setStagedSearchPaths()``, and
  ``setStagedAutoDiscoveryEnabled()``.

UI code reads ``hasUnstagedChanges()`` to decide whether to enable an
**Apply** button or mark the row as dirty. When the user accepts the
changes, the surrounding workflow calls ``applyChanges()``, which writes
the staged values to :cpp:class:`SettingsStorage`, promotes them to the
active configuration, and calls ``reloadLibrary()``. ``revertChanges()``
discards staged edits and resets staged state to match active state. Until
``applyChanges()`` runs, hardware that is already holding function-pointer
references continues to call into the previously loaded library.

Reloading a vendor library is not, however, safe to do while hardware is
holding references into it: a stale function pointer becomes a crash. The
:cpp:class:`HardwareManager` coordinates this through three phases inside
``syncWithRuntimeConfig()``:

1. Before tearing anything down, the manager calls
   :cpp:func:`HardwareRegistry::getLibrariesWithChanges` to find any libraries
   whose staged configuration differs from their active configuration, then
   calls :cpp:func:`HardwareRegistry::getLibraryDependencies` to find all
   currently loaded hardware that depends on those libraries. Those hardware
   keys are added to the replacement list alongside whatever the user-driven
   loadout change already requested.
2. With every dependent hardware object destroyed, the manager calls
   ``applyVendorLibraryChanges()``, which walks each :cpp:class:`VendorLibrary`
   singleton and calls ``applyChanges()`` on the ones with unstaged changes.
   This is the only point in the lifecycle where a vendor library can be
   reloaded without risk of dangling pointers.
3. The manager then re-creates the dependent hardware in the third phase,
   which re-binds the freshly resolved function pointers.

The user-facing flow — opening the dialog, browsing for a path, clicking
Apply — is documented in :doc:`/user_guide/library_status`. The
:cpp:class:`LibraryStatusWidget` source under ``src/gui/widget/`` is a
worked example of a UI consumer of the staging API; it also exposes a
**Test Load** action that temporarily applies staged changes, reports
success or failure, and restores the staged state for further editing.

REGISTER_LIBRARY linkage
------------------------

A hardware implementation declares its dependency on a vendor library with
a single macro call after ``REGISTER_HARDWARE_META``:

.. code-block:: cpp

   REGISTER_HARDWARE_META(M4i2220x8, "Spectrum Instrumentation M4i.2220-x8 ...")
   REGISTER_HARDWARE_PROTOCOLS(M4i2220x8, CommunicationProtocol::Custom)
   REGISTER_LIBRARY(M4i2220x8, SpectrumLibrary)

``REGISTER_LIBRARY(CLASS, LIBRARY_NAME)`` (defined in
``hardware/core/hardwareregistration.h``) records two facts in
:cpp:class:`HardwareRegistry` at static-initialization time:

- The dependency itself, so that
  :cpp:func:`HardwareRegistry::getLibraryDependencies` can answer "which
  vendor libraries does *this* implementation need" and
  :cpp:func:`HardwareRegistry::getHardwareDependingOnLibrary` can answer
  the inverse question for the reload coordination above.
- A ``std::function<VendorLibrary*()>`` that returns the library's singleton
  instance. The registry stores these getters in
  ``d_libraryGetters`` so :cpp:func:`HardwareRegistry::getLibrariesWithChanges`
  can poll every registered library for ``hasUnstagedChanges()`` without
  the registry having to know about every concrete subclass at compile
  time. New libraries plug into this mechanism for free as long as they
  are registered through the macro.

The Hardware Configuration dialog uses ``getLibraryDependencies()`` to label
implementations whose required library is missing, so users can see at a
glance which entries cannot be selected with the current driver state. See
:doc:`/classes/hardwareregistry` for the full registry API.

Concrete subclasses
-------------------

Two :cpp:class:`VendorLibrary` subclasses ship with Blackchirp.

LabjackLibrary
^^^^^^^^^^^^^^

:cpp:class:`LabjackLibrary` wraps the LabJack U3 driver. It is unusual among
vendor libraries because the vendor's library and ABI differ between
platforms — the open-source ``exodriver`` exposes a low-level USB transport
on Linux/macOS, while the proprietary UD driver on Windows exposes a
higher-level "easy-functions" API. The subclass uses ``#ifdef Q_OS_WIN`` to
compile a different symbol set on each platform; the LabJack hardware
implementations never see those symbols directly because a thin facade
(``BC::Labjack``) hides the platform difference. The arrangement is the
worked example in *Case study: LabJack exo/UD split*, below. Cross-link to
:doc:`/classes/vendorlibrary` for member-level detail.

SpectrumLibrary
^^^^^^^^^^^^^^^

:cpp:class:`SpectrumLibrary` wraps the Spectrum Instrumentation SDK
(``spcm_linux`` on Linux, ``spcm64.dll`` on Windows). It is used by the
M4i family of FTMW digitizers (see :cpp:class:`M4i2220x8`). The Spectrum
SDK exposes the same symbol set on every platform, so the subclass needs
only a per-platform ``platformLibraryNames()`` and ``defaultSearchPaths()``
implementation; ``loadFunctions()`` is platform-agnostic. The singleton
constraint is load-bearing here in a way it is not for LabJack: the
Spectrum library maintains global state (driver-level handles, kernel
objects) and cannot be loaded twice in one process. See
:doc:`/classes/vendorlibrary` for member-level detail.

Case study: LabJack exo/UD split
--------------------------------

The LabJack integration illustrates the pattern future cross-platform vendor
libraries can follow when the vendor exposes different ABIs on different
operating systems.

Three layers
^^^^^^^^^^^^

Three translation units sit between :cpp:class:`LabjackU3` and the vendor
SDK. The hardware class talks to a thin facade; the facade is implemented
twice (once per platform) and each implementation talks to the dynamic
loader. The dynamic loader is :cpp:class:`LabjackLibrary` and exposes a
different symbol set per platform.

.. mermaid::

   flowchart TB
       hw["LabjackU3 (hardware)<br/>hardware/optional/ioboard/labjacku3.cpp"] --> facade
       subgraph facade [BC::Labjack facade]
           direction TB
           h["labjackdriver.h<br/>(public interface)"]
       end
       facade -- "NOT WIN32" --> exo["labjackdriver_exo.cpp<br/>wraps u3.cpp helpers"]
       facade -- "WIN32" --> ud["labjackdriver_ud.cpp<br/>calls UD easy-functions"]
       exo --> lib["LabjackLibrary<br/>hardware/library/labjacklibrary.{cpp,h}"]
       ud --> lib
       lib -- "NOT WIN32" --> exodriver["liblabjackusb.so / .dylib<br/>(LJUSB_* transport)"]
       lib -- "WIN32" --> uddll["LabJackUD.dll<br/>(OpenLabJack, eAIN, ...)"]

The three layers are:

1. :cpp:class:`LabjackLibrary`
   (``hardware/library/labjacklibrary.{cpp,h}``) — the dynamic loader. The
   symbol set is conditionally compiled. On Linux/macOS the loader resolves
   the LJUSB transport symbols (``LJUSB_OpenDevice``, ``LJUSB_CloseDevice``,
   ``LJUSB_Read``, ``LJUSB_Write``, …) from ``liblabjackusb.so`` /
   ``.dylib`` — the vendor's open-source *exodriver*. On Windows the loader
   resolves the high-level UD symbols (``OpenLabJack``, ``Close``, ``eAIN``,
   ``eDAC``, ``eDI``, ``eDO``, ``eTCConfig``, ``eTCValues``,
   ``ErrorToString``, ``GetDriverVersion``) from ``LabJackUD.dll``. The
   Windows ``__stdcall`` decoration is a no-op on x86-64, so bare names
   resolve through ``QLibrary::resolve``.

2. The ``BC::Labjack`` facade
   (``hardware/library/labjackdriver.h``) — a thin, platform-neutral
   namespace API. It declares ``isAvailable()``, ``errorString()``,
   ``openU3(serialOrLocalId)`` (returning an opaque ``HandlePtr``), and the
   per-operation functions that the hardware class actually calls:
   ``readAnalog``, ``writeAnalog``, ``readDigital``, ``writeDigital``,
   ``configureTimers``, ``readTimers``. The header also carries a
   ``BC::Labjack::Const`` namespace of timer-clock and device-type
   constants so that callers do not need to ``#include`` either backend's
   private headers.

3. **Backend translation units** — implement the facade. Exactly one is
   compiled per build, selected by CMake:

   - ``labjackdriver_exo.cpp`` (Linux/macOS, gated on ``NOT WIN32``) wraps
     the vendored ``u3.cpp`` helper. Its ``DeviceHandle`` carries a
     ``void* h`` (an LJUSB handle returned by ``openUSBConnection``) and a
     ``u3CalibrationInfo`` union member populated at open time.
   - ``labjackdriver_ud.cpp`` (Windows, gated on ``WIN32``) calls UD easy
     functions directly through :cpp:class:`LabjackLibrary` and reports
     errors through ``ErrorToString``. Its ``DeviceHandle`` carries a
     ``long h`` (``LJ_HANDLE``) and no calibration cache, because the UD
     library handles calibration internally.

The vendored ``u3.cpp`` helper is also gated to ``NOT WIN32`` in CMake and
is the sole consumer of the LJUSB transport symbols on the exo backend.

Caller pattern
^^^^^^^^^^^^^^

The hardware class uses the ``BC::Labjack::*`` facade exclusively and never
sees a raw ``LJUSB_*`` symbol or a raw UD function:

.. code-block:: cpp

   // labjacku3.cpp
   d_handle = BC::Labjack::openU3(d_serialNo);     // -1 → first found
   BC::Labjack::configureTimers(d_handle.get(),
                                {0L, 0L}, {0L, 0L}, 4L,
                                BC::Labjack::Const::tc48MHZ, 0L,
                                {0L, 0L}, {0.0, 0.0});
   BC::Labjack::readAnalog(d_handle.get(), channel, voltage);

``HandlePtr`` is ``std::unique_ptr<DeviceHandle, void(*)(DeviceHandle*)>``;
the deleter calls the appropriate close function (``LJUSB_CloseDevice`` on
the exo backend, UD ``Close`` on the UD backend) and frees the struct. A
null ``HandlePtr`` (pointer and deleter both null) is safe because
``unique_ptr`` does not invoke the deleter on a null managed pointer, so
failure paths can simply ``return HandlePtr(nullptr, destroyHandle);``.

Why the split exists
^^^^^^^^^^^^^^^^^^^^

The split is not stylistic: the LJUSB transport library and the UD
high-level library have *different ABIs* and *different vendor licenses*.
LJUSB is open-source and exposes raw USB I/O; the host code (vendored
``u3.cpp``) implements the U3 wire protocol on top of that transport. UD
is closed-source, distributed as a compiled DLL, and presents already-cooked
"easy" functions that perform analog/digital reads in a single call.
Wrapping both behind a facade lets the hardware class stay platform-agnostic
while keeping each backend's platform-specific logic confined to one
translation unit chosen at CMake configure time. Adding a per-backend
optimization, switching one backend to a newer SDK, or even replacing one
backend entirely is a change in one ``.cpp`` file with no impact on the
hardware class or the dynamic loader.

Adding a new LabJack model
^^^^^^^^^^^^^^^^^^^^^^^^^^

The facade is shaped so that adding a new LabJack device (the U6 is the
canonical next candidate) does not require restructuring:

1. Declare ``openU6(int serialOrLocalId)`` in ``BC::Labjack`` and add a
   ``Kind::U6`` enumerator to the backend ``DeviceHandle`` structs.
2. On the exo backend, vendor ``u6.cpp`` (analogous to ``u3.cpp``) and add
   a ``u6CalibrationInfo`` union member to the ``DeviceHandle``. Add a
   ``Kind::U6`` arm to each ``switch`` in ``labjackdriver_exo.cpp``. Gate
   ``u6.cpp`` to ``NOT WIN32`` in :file:`cmake/BlackchirpHardware.cmake`.
3. On the UD backend, add a ``Kind::U6`` arm to ``openU6`` that passes
   ``DeviceType = LJ_dtU6`` to ``OpenLabJack``. The other operational
   functions (``eAIN``, ``eDAC``, ``eDI``, ``eDO``, …) are device-agnostic
   in the UD library and need no per-model changes.
4. Add a :cpp:class:`LabjackU6` hardware class under
   ``hardware/optional/ioboard/`` modelled on :cpp:class:`LabjackU3`,
   register it with ``REGISTER_HARDWARE_META`` and ``REGISTER_LIBRARY``,
   and pick up the new entry through the CMake glob.

:cpp:class:`LabjackLibrary` itself does not change, and the operational
facade signatures do not change — only the open call grows a new entry
point.

Recipe: adding a new VendorLibrary subclass
-------------------------------------------

When integrating a new closed-source SDK, follow these eight steps:

1. **Create the source files.** Put ``hardware/library/<name>library.{cpp,h}``
   alongside the existing ``vendorlibrary.h``,
   ``labjacklibrary.{cpp,h}``, and ``spectrumlibrary.{cpp,h}``. Define a
   singleton subclass of :cpp:class:`VendorLibrary` with a private
   constructor, a ``static <Name>Library& instance()``, and a
   ``static <Name>Library *s_instance``. Pass a unique settings key (e.g.,
   ``BC::Key::<Vendor>::yourLibrary``) to the base constructor; the base
   class will persist the user-provided path, search paths, and
   auto-discovery flag under ``vendorLibraries/<key>/``.

2. **Override** ``loadFunctions()``. Resolve every vendor symbol the
   library needs through ``resolveFunction()`` (or ``d_library.resolve``)
   and store the results in typed function-pointer members. Mark which
   symbols are required and which are optional: if any required symbol
   resolves to ``nullptr``, set ``d_libraryLoaded = false`` and populate
   ``d_errorString`` with a message naming the missing symbols. The base
   class will unload the library and fall back to the next candidate path.

3. **Define typed accessors.** Hardware code calls vendor functions through
   the singleton, so make the resolved function pointers reachable —
   either as public data members (the LabJack and Spectrum approach) or
   through inline accessors that wrap the call. Either way, prefer typed
   wrappers over raw ``void*`` so the hardware code reads like a normal
   function call.

4. **Override** ``platformLibraryNames()`` **and** ``defaultSearchPaths()``.
   Return the list of candidate filenames to try (most-specific first) and
   the conventional install locations for each platform. The base class
   walks these in order; the first match wins.

5. **Register the source files in CMake.** Add the new
   ``<name>library.cpp`` to the ``HARDWARE_SYSTEM_SOURCES`` list in
   :file:`cmake/BlackchirpHardware.cmake`. The hardware-implementation
   glob does **not** pick up files under ``hardware/library/`` — those
   are enumerated explicitly in ``HARDWARE_SYSTEM_SOURCES`` so the
   library layer is unconditionally part of every build.

6. **Wire dependent hardware with** ``REGISTER_LIBRARY``. In each
   hardware implementation that needs the library, add
   ``REGISTER_LIBRARY(YourHwClass, YourLibraryClass)`` after
   ``REGISTER_HARDWARE_META`` in the implementation's ``.cpp`` file.
   The :cpp:class:`HardwareRegistry` will then know which hardware to
   tear down before a library reload.

7. **Consider a facade if the SDK is multi-platform.** If the vendor SDK
   has a per-platform ABI split (different driver name, different calling
   convention, different functional decomposition) on the order of the
   LabJack ``exo``/``UD`` divergence, do not bake the conditionals into
   the hardware class. Provide a thin platform-neutral facade header and
   select the backend ``.cpp`` at CMake configure time, following the
   :cpp:class:`LabjackLibrary` pattern above.

8. **Document singleton constraints.** If the vendor library has global
   mutable state and cannot be reloaded safely while other code is using
   it, follow the :cpp:class:`SpectrumLibrary` pattern (one process-wide
   instance, no copy/assignment) and call out the restriction in the
   class-level Doxygen ``\brief`` so a future contributor does not try to
   stand up a second instance.

Once the new library is in place, the Library Status widget picks it up
automatically: it iterates the registered :cpp:class:`VendorLibrary`
singletons through :cpp:class:`HardwareRegistry`, so no UI changes are
required to make the new library configurable.
