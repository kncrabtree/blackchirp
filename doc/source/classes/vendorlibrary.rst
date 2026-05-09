.. index::
   single: VendorLibrary
   single: LabjackLibrary
   single: SpectrumLibrary
   single: vendor libraries; dynamic loading
   single: LibraryStatusWidget

VendorLibrary
=============

``VendorLibrary`` is the abstract base class for all dynamically loaded vendor
driver wrappers. Blackchirp is built and distributed without compile-time
dependencies on vendor SDKs; instead, each ``VendorLibrary`` subclass uses
``QLibrary`` to locate and load the vendor library at runtime. If the library
is absent or cannot be loaded, Blackchirp starts normally and the dependent
hardware drivers report themselves as unavailable.

The loading sequence is:

1. Try the user-provided path (``setUserProvidedPath``), if set.
2. Try user-specified search directories (``setUserSearchPaths``), then
   platform-specific default paths (``defaultSearchPaths``), if automatic
   discovery is enabled.
3. Resolve function pointers by calling the subclass ``loadFunctions()``
   implementation.
4. Validate that all required symbols were found.

The staged-configuration API separates UI interaction from live hardware use.
UI code calls ``setStagedUserProvidedPath``, ``setStagedSearchPaths``, and
``setStagedAutoDiscoveryEnabled`` to accumulate changes without affecting
running hardware, then calls ``applyChanges()`` to promote the staged
settings, persist them, and reload the library. ``revertChanges()`` discards
any staged edits. ``hasUnstagedChanges()`` allows the surrounding dialog to
enable or disable its Apply button. The ``LibraryStatusWidget`` in the
Application Settings dialog uses this API to manage library paths safely
without requiring a hardware restart until the user explicitly applies the
changes.

Hardware drivers that depend on a vendor library register the
dependency with ``REGISTER_LIBRARY`` (see :doc:`/classes/hardwareregistry`).
The ``HardwareRegistry`` tracks these dependencies so ``HardwareManager``
can destroy and recreate affected hardware objects around a library reload.

Concrete subclasses
--------------------

``LabjackLibrary``
   Loads the LabJack U3 driver. On Linux and macOS the driver is the
   LJUSB transport library (``liblabjackusb.so`` / ``.dylib``); on Windows
   it is the UD high-level library (``LabJackUD.dll``, 64-bit). The
   ``BC::Labjack`` facade in ``labjackdriver.h`` abstracts the platform
   difference and is the preferred entry point for hardware code. The
   cross-platform architecture is described in
   :doc:`/developer_guide/vendor_libraries`.

``SpectrumLibrary``
   Loads the Spectrum Instrumentation driver (``spcm_linux`` / ``spcm64.dll``).
   Used by M4i digitizer drivers for FTMW acquisition. Because the
   Spectrum library maintains global state, a single instance is enforced via
   the singleton pattern. M4i driver code reaches the resolved symbols through
   the singleton accessor:

   .. code-block:: cpp

      SpectrumLibrary &lib = SpectrumLibrary::instance();
      if (lib.isAvailable()) {
          void *handle = lib.spcm_hOpen("/dev/spcm0");
          // ... use other Spectrum functions
      }

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: VendorLibrary
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: LabjackLibrary
   :members:
   :undoc-members:

.. doxygenclass:: SpectrumLibrary
   :members:
   :undoc-members:
