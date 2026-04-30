.. index::
   single: Library Status
   single: Vendor Libraries
   single: LabJack; driver installation
   single: Spectrum Instrumentation; driver installation
   single: Dynamic Libraries

.. _library-status:

Library Status
==============

Blackchirp is distributed as a single binary that does not require vendor
SDKs to be present at compile time. Optional vendor libraries are loaded
dynamically at runtime. If a library is absent, Blackchirp starts normally
and all hardware that does *not* depend on that library continues to function.
Hardware types that do require an absent library will report an error when
you attempt to connect them.

The Library Status widget lists every vendor library that Blackchirp
supports, shows the current load status for each, and lets you supply
custom search paths.

**To open the Library Status widget:**

- During :ref:`first-run-hardware-onboarding`, select the **Library Status**
  tab inside the Hardware Selection dialog.
- At any later time, choose **Hardware → Hardware Selection** from the menu
  bar and select the **Library Status** tab.

.. TODO: capture screenshot — widget.png: the LibraryStatusWidget showing the
   full widget with both the LabJack U3 Driver and spcm rows in the overview
   table. Ideally one library shows "Available" (green) and the other shows
   "Not Found" (red) so both status colours are visible. The right-hand
   configuration panel should be visible with the Library Path field,
   Additional Paths field, and Auto-Discovery checkbox. The installation
   guidance text area at the bottom-right should be partially visible.

.. figure:: /_static/user_guide/library_status/widget.png
   :width: 700
   :alt: Library Status widget

   The Library Status widget. The overview table shows load status for each
   vendor library. Select a row to configure search paths and view
   installation guidance.

Overview Table
--------------

The table at the top lists every registered vendor library with four columns:

**Library**
    The library base name.

**Status**
    One of three values:

    - *Available* — the library was found and all required symbols were
      resolved successfully. Shown in green.
    - *Not Found* — loading was not attempted or the library was not located
      on any search path. Shown in red.
    - *Error* — loading was attempted but failed (missing symbols, wrong
      architecture, or other loading error). Shown in red.

**Version**
    The library version string if the library exposes a version query function,
    or *Unknown* if not available.

**Load Path**
    The full filesystem path from which the library was loaded, or
    *Not loaded* if it is unavailable.

Click **Refresh** (in the configuration panel) to re-attempt library loading
with the current search path settings.

Configuration Panel
-------------------

Select a row in the overview table to display and edit the search path
configuration for that library.

**Library Path**
    A specific file path or directory to try first. Use **Browse...** to
    open a directory picker. Leave blank to rely on automatic discovery and
    additional paths only.

**Additional Paths**
    A semicolon-separated list of directories to search, in addition to the
    automatic discovery paths. Useful when the library is installed in a
    non-standard location.

**Auto-Discovery**
    When checked (the default), Blackchirp searches the platform default
    paths for the library in addition to any paths you supply. Uncheck this
    to restrict loading to the paths you specify explicitly.

Changes in the configuration panel are *staged*: they do not take effect
until you click **OK** in the enclosing Hardware Configuration dialog. While
you have unsaved changes, modified controls are highlighted and the panel
label shows an asterisk.

**Test Load**
    Attempts to load the library with the current settings (staged, if you
    have staged changes; active otherwise) and reports success or failure
    in a popup. This is a safe way to validate a new path before committing.

The lower portion of the configuration panel shows platform-specific
installation guidance for the selected library.

Vendor Libraries
----------------

LabJack U3 Driver
^^^^^^^^^^^^^^^^^

Used by: LabJack U3 flow-controller and IO-board hardware.

The LabJack library name and install method differ by platform:

**Linux and macOS**
    Blackchirp loads ``liblabjackusb.so`` (Linux) or ``liblabjackusb.dylib``
    (macOS) from the LabJack open-source *exodriver*. Install it with:

    .. code-block:: bash

       # Ubuntu / Debian
       sudo apt install libusb-1.0-0-dev
       git clone https://github.com/labjack/exodriver.git
       cd exodriver && sudo ./install.sh

    After installation the library is placed in ``/usr/local/lib/``, which
    is searched automatically. Ensure your user is in the ``plugdev`` group
    (or equivalent) to access the USB device:

    .. code-block:: bash

       sudo usermod -aG plugdev $USER

    Log out and back in for the group change to take effect.

**Windows**
    Blackchirp loads ``LabJackUD.dll`` from the LabJack UD driver package
    (64-bit). Download and run the LabJack Windows installer from the
    `LabJack website <https://labjack.com/support/software/installers/ud>`_.
    The installer places the DLL in the system directory, which is searched
    automatically. A system restart is typically required after installation.

Spectrum Instrumentation (spcm)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by: Spectrum Instrumentation M4i/M2p digitizer hardware.

The library base name is ``spcm``. Platform-specific names tried include
``spcm_linux`` (Linux), ``spcm64.dll`` (Windows 64-bit), and ``spcm.dll``
(Windows 32-bit).

**Linux**
    Install the Spectrum kernel module and driver from the Spectrum
    Instrumentation `download page <https://spectrum-instrumentation.com/en/drivers-and-examples>`_.
    The installer places ``libspcm_linux.so`` in the system library path
    (typically ``/usr/local/lib/`` or ``/opt/spectrum/lib/``), both of which
    are searched automatically. Run ``sudo ldconfig`` after installation to
    update the linker cache.

**Windows**
    Run the Spectrum driver installer as Administrator. The DLL is registered
    in the Windows system directory and is found automatically. A system
    restart is required after installation.

**macOS**
    Spectrum hardware is not officially supported on macOS at this time. If
    you have a development build of the Spectrum library for macOS, supply the
    path manually via the **Library Path** field.

Troubleshooting
---------------

If a library shows *Error* status after installation:

1. Click **Refresh** to re-attempt loading without restarting Blackchirp.
2. Check the *Details* pane to the left of the configuration controls for
   the specific error message and the paths that were tried.
3. If the library is installed in a non-standard location, enter the directory
   path in **Library Path** or **Additional Paths** and click **Test Load**.
4. On Linux, run ``sudo ldconfig`` to refresh the dynamic linker cache, then
   click **Refresh**.
5. Verify that the installed library architecture (32-bit vs. 64-bit) matches
   the Blackchirp binary you are running.
