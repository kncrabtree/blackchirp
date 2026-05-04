.. index::
   single: Python hardware; selecting
   single: Add Profile; Python driver
   single: Python Script (field)
   single: Python Class (field)
   single: Python Environment (field)
   single: Template script
   single: pythonEnvPath

.. _python-hardware-selecting:

Selecting a Python Driver
=========================

A Python driver is chosen the same way as any other driver:
through the **Hardware Configuration** dialog (**Hardware → Hardware
Selection**). Pick a Python-backed driver when adding a
profile, then point Blackchirp at the script that contains your
driver class. Three settings on the profile control the binding:
the script path, the class name within the script, and an optional
Python environment.

For an introduction to profiles in general, see
:ref:`hardware-config-profiles`. The notes below cover the
Python-specific parts of that flow.

Choosing a Python driver
------------------------

In the **Add Profile** dialog, the **Driver** dropdown lists
every driver class registered for the selected hardware type. Python
drivers are named with a ``Python`` prefix (for example,
``PythonAwg``, ``PythonFlowController``). The trampoline classes that
ship in the build are listed in :ref:`python-hardware-overview`;
the same selection is also available from the **Driver**
dropdown shown on the existing profile in the Hardware Configuration
dialog.

.. figure:: /_static/user_guide/python_hardware/profile_creation.png
   :width: 800
   :target: ../../_images/profile_creation.png
   :alt: Hardware Configuration dialog with a PythonAwg profile selected. The Advanced section shows the Python Script, Python Class, and Python Environment fields, with the resolved interpreter version displayed below.
   :align: center

   Hardware Configuration dialog with a Python-backed AWG profile
   (``PyAWG (PythonAwg)``) selected. The **Advanced** section of the
   per-profile panel exposes the **Python Script**, **Python Class**,
   and **Python Environment** fields; the resolved interpreter
   version (``System: Python 3.9.18``) is shown below the
   environment row.

Template-script copy
--------------------

When you save a new profile that uses a Python driver,
Blackchirp offers to seed it with the bundled template:

.. figure:: /_static/user_guide/python_hardware/template_copy_prompt.png
   :alt: Dialog asking whether to copy the Python template script, with the security warning and Yes/No buttons.
   :align: center

   The template-copy prompt shown after a Python profile is created.
   The bold text at the top is the security warning quoted in
   :ref:`python-hardware-security`; the question below it offers
   to copy the bundled template into a location of the user's
   choice.

The prompt repeats the security warning from
:ref:`python-hardware-security` and asks "Would you like to create
a copy of the template script to customize?". Choosing **Yes**
opens a file-save dialog with a suggested filename based on the
hardware type (for example, ``my_flow_controller.py``). The
template that matches the chosen driver is copied to the
location you pick, and the saved path is filled in on the new
profile automatically.

When you switch a Python profile to a real protocol after creating
it, the **Communication Settings** dialog (Hardware → Communication)
exposes the same RS-232 / TCP / GPIB / Custom / Virtual options the
driver registered. Picking **Custom** is appropriate when the
driver bypasses ``self.comm`` entirely and talks to its hardware
through a vendor Python package; the panel shows a note pointing the
user back to the script in that case (see
:ref:`python-hardware-custom-protocol`).

The starter templates ship with working Virtual-protocol behavior
and inline docstrings for every method, so a freshly copied script
runs end-to-end before you change anything. Each template uses the
default class name listed in :ref:`python-hardware-overview`.

Choosing **No** leaves the script path empty; you can fill it in
later from the Hardware Configuration dialog by browsing to your
own ``.py`` file.

Python script path
------------------

The **Python Script** field in the **Advanced** section of the
profile holds the absolute path to the ``.py`` file that defines
your driver class. The accompanying **Browse...** button opens a
file picker filtered to ``*.py``. The field is shown only when the
selected driver is Python-backed; switching to a
native C++ driver hides it.

The script may live anywhere readable by the user account that runs
Blackchirp. A common pattern is to keep all of a site's Python
drivers in a single directory under the data path so they are
backed up alongside experimental data.

Class name
----------

The **Python Class** field is an editable dropdown. Whenever the
script path changes, Blackchirp scans the file for ``class``
definitions at the top of a line and populates the dropdown with
the class names it finds. Pick the one that contains your driver
methods.

The placeholder text on the dropdown is the default class name
expected by the trampoline (``AwgDriver``, ``FlowControllerDriver``,
and so on; see :ref:`python-hardware-overview` for the full list).
You can use any class name, but the script must contain exactly
one class with that name and that class must implement the methods
the trampoline calls. The capabilities each trampoline expects of
its driver class are listed in :doc:`per_type_capabilities`.

Python environment
------------------

The **Python Environment** field selects the interpreter used to
launch the script. Leave it empty to use the system ``python3``
that is on ``PATH``. To use a virtual environment or conda
environment, set the field to the environment's directory. The
**Browse...** button next to it opens a directory picker.

Blackchirp resolves the interpreter by looking for, in order:

- ``<env>/bin/python3``
- ``<env>/bin/python``
- ``<env>/Scripts/python.exe``

The status line below the field shows the version reported by the
resolved interpreter (for example, ``Python 3.12.4``) so you can
confirm the environment is wired up correctly. If the directory
exists but no interpreter is found inside it, Blackchirp falls
back to the system ``python3`` and shows a warning. If neither the
environment nor the system has a working ``python3``, the status
line reports an error and the profile will not be able to launch
its driver.

The environment field is per profile. Two Python profiles may use
different environments, which is useful when one driver depends
on a vendor SDK that conflicts with packages required by another.

In the persistent profile data, this setting is stored as
``pythonEnvPath`` alongside the script path and class name.

Where the settings are saved
----------------------------

The script path, class name, and environment path are stored
on the profile by ``HardwareProfileManager`` and reloaded when
Blackchirp starts. They are independent of the per-driver
settings shown in the Required Settings, Important Settings, and
Advanced sections, so changing them does not affect any other
profile that uses the same Python driver.
