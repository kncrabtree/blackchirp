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
script instead of a compiled C++ driver. A Python driver is selected
per profile, the same way a built-in driver is, and each Python-backed
hardware type ships with a template script that runs out of the box
against the Virtual communication protocol. The
:doc:`hardware_config/profiles` page covers profile creation in
general; this chapter covers the parts specific to Python drivers.

.. rubric:: In this chapter

- :doc:`python_hardware/overview` — architecture and the list of
  supported hardware types
- :doc:`python_hardware/selecting` — wiring a script, class name, and
  Python environment to a profile
- :doc:`python_hardware/writing_a_driver` — lifecycle methods, injected
  proxies, and return-type expectations
- :doc:`python_hardware/hot_reload` — editing and reloading a script
  from the per-device hardware dialog
- :doc:`python_hardware/per_type_capabilities` — methods each
  trampoline expects from the driver class
