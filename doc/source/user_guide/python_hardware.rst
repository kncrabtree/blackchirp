.. index::
   single: Python hardware
   single: Python driver
   single: hardware driver; Python

Python Hardware
===============

.. toctree::
   :hidden:

   python_hardware/overview
   python_hardware/selecting
   python_hardware/writing_a_driver
   python_hardware/hot_reload
   python_hardware/per_type_capabilities

Blackchirp can drive a hardware device through a user-supplied Python
script instead of a compiled C++ driver. The Python driver
runs in a separate subprocess and communicates with Blackchirp over a
JSON-lines pipe, so adding a new device or customizing an existing
one does not require rebuilding the application.

A Python driver is selected per-profile, the same way a built-in
driver is. Each Python-backed hardware type has a template
script that ships with Blackchirp and works out of the box against
the Virtual communication protocol; users start from that template
and adapt it to their hardware. The :doc:`hardware_config/profiles`
page covers profile creation generally; this chapter covers the
parts that are specific to Python drivers.

.. rubric:: In this chapter

- :doc:`python_hardware/overview` — what Python hardware is, the
  architecture in one paragraph, and the supported hardware types
- :doc:`python_hardware/selecting` — choosing a Python driver
  in the Hardware Configuration dialog and configuring the script,
  class name, and Python environment for a profile
- :doc:`python_hardware/writing_a_driver` — the Python API contract:
  lifecycle methods, injected proxies, and return-type expectations
- :doc:`python_hardware/hot_reload` — editing and reloading a script
  from the per-device hardware dialog without restarting Blackchirp
- :doc:`python_hardware/per_type_capabilities` — what each trampoline
  type expects the driver to implement
