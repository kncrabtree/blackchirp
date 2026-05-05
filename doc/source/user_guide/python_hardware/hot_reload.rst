.. index::
   single: Python hardware; reload
   single: Reload Script
   single: Open in Editor
   single: HwDialog; Python controls
   single: Script error feedback

.. _python-hardware-hot-reload:

Reloading and Editing a Script
==============================

When a Python driver is in use, the per-device hardware
dialog gains a **Python Script** group at the top of its control
panel. This group lets you open the script in an external editor
and reload it after editing, without restarting Blackchirp.

Open the hardware dialog from the device's entry on the
**Hardware** menu, or from its row in the **Status Panel**, the
same way as for any other driver (see :doc:`/user_guide/hwdialog`). The Python
controls appear above the type-specific control widget; for
example, a profile backed by ``PythonFlowController`` shows the
Python controls and the standard gas-control widget on the same
dialog.

.. figure:: /_static/user_guide/python_hardware/hwdialog_python.png
   :alt: HwDialog for a PythonFlowController device. The Python Script group at the top of the Control tab shows the script path, Open in Editor and Reload Script buttons, and a Running status label. The standard gas-control widget appears below.
   :align: center

   HwDialog for a ``PythonFlowController`` profile. The **Python
   Script** group sits at the top of the **Control** tab and shows
   the current script path, the **Open in Editor** and
   **Reload Script** buttons, and a status label (``Running``)
   reporting the most recent reload result. The standard
   gas-control widget appears below it on the same tab.

Open in Editor
--------------

**Open in Editor** hands the script path to the operating system's
default handler for ``.py`` files via Qt's
``QDesktopServices``. On most desktops this opens the user's
preferred Python editor or IDE. Blackchirp does not provide an
in-app editor; pick whichever external tool you prefer.

The button is disabled when the profile has no script path set.
Configure the path from the Hardware Configuration dialog (see
:doc:`selecting`) before you can edit the script from here.

Reload Script
-------------

**Reload Script** stops the current Python subprocess and launches
a fresh one with the same script path, class name, and
environment. The status label changes to ``Reloading...`` while
the new process starts and switches to ``Running`` once
``test_connection`` succeeds.

What survives a reload
~~~~~~~~~~~~~~~~~~~~~~

Reload is intentionally lightweight. It tears down the Python
subprocess but leaves the C++ ``HardwareObject`` intact, including:

- All persistent settings stored through the settings registry.
- The communication-protocol object (``self.comm`` reaches the same
  serial port, TCP socket, GPIB instrument, or virtual backend
  before and after the reload).
- The hardware object's worker thread and its signal/slot
  connections, so any other widget bound to this device continues
  to receive updates after the reload.
- The active loadout, current FTMW preset, and any in-progress
  experiment configuration.

Anything held in memory by the Python script itself does **not**
survive. The subprocess is killed and a new interpreter is
launched, so module-level variables, instance attributes set in
``initialize`` or accumulated during runtime, and any cached
hardware state on the Python side are gone. ``initialize`` is
called again on the new process, exactly as if Blackchirp had
just connected to the device.

Persistent state should therefore be stored through
``self.settings`` (which round-trips to Blackchirp's settings
storage and survives a reload) rather than in plain Python
variables.

Error feedback
~~~~~~~~~~~~~~

If the new subprocess fails to start, the status label switches
to the error theme color, shows ``Error: <message>``, and stays
that way until the next successful reload. The full message is
also available as a tooltip on the label, which is useful when a
long Python traceback wraps off the visible area. Common failure
modes include:

- **Missing or unreadable script path.** The label reports that
  the script path is empty or that the file cannot be opened.
  Use the Hardware Configuration dialog to set or correct the
  path (see :doc:`selecting`).
- **Syntax errors in the script.** Python's traceback is
  forwarded from the subprocess and surfaced in the status
  label, with the full traceback also written to the hardware
  log panel. Fix the syntax error in your editor and click
  **Reload Script** again.
- **Missing or misnamed class.** If the configured class name is
  not defined in the script, the label reports that the class
  could not be loaded. Either rename the class in the script or
  update the **Python Class** field on the profile to match.
- **Import errors.** A failure to import a module the script
  depends on is reported the same way as a syntax error. Check
  that the package is installed in the environment configured
  for the profile (see :doc:`selecting`).

.. figure:: /_static/user_guide/python_hardware/error_state.png
   :width: 800
   :alt: Composite view of the Blackchirp main window with HwDialog open showing a red Error status label, the Log tab showing the full Python traceback, and an external editor displaying the script with the offending line highlighted.
   :align: center

   The error state after a failed reload, captured alongside the
   tools the user typically reaches for next. The HwDialog
   **Python Script** group shows ``Error:`` followed by the
   exception message in red; the Log tab below it carries the
   full Python traceback (also available as a tooltip on the
   status label); and the external editor on the right shows
   the script with the offending line highlighted.

Once the underlying problem is fixed, click **Reload Script**
again. There is no need to close the dialog or restart
Blackchirp; the reload is the only action required to pick up
script changes.
