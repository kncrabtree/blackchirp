.. index::
   single: ExperimentConfigPage
   single: experiment setup; wizard page contract
   single: initialize; ExperimentConfigPage
   single: validate; ExperimentConfigPage
   single: apply; ExperimentConfigPage

ExperimentConfigPage
====================

``ExperimentConfigPage`` is the abstract base class for every page in the
experiment-setup dialog.  It inherits both ``QWidget`` (for the visual representation)
and :cpp:class:`SettingsStorage` (for persistent per-page configuration keyed by
``d_key``).  Each subclass represents one conceptual step in the setup sequence and
holds a non-owning pointer to the in-flight :cpp:class:`Experiment` it is helping to
configure.

The wizard routes user-facing diagnostic messages through the ``warning(QString)`` and
``error(QString)`` signals so that individual pages remain decoupled from the surrounding
dialog layout.

Validate / apply lifecycle
--------------------------

The wizard drives each page through a three-slot lifecycle:

1. **initialize()** — called by the wizard after all pages are constructed and before
   the page is first shown.  Use this slot for any value that depends on choices made
   on a preceding page.  Values that can be determined purely from a previous experiment
   or from ``SettingsStorage`` should be set in the constructor instead, so they are
   available before the wizard opens.

2. **validate()** — called when the user attempts to advance past the page.  Return
   ``true`` if the page's settings are consistent; return ``false`` and emit ``error()``
   or ``warning()`` to block the advance and display a diagnostic.

3. **apply()** — called after ``validate()`` returns ``true``.  Write the page's
   settings into ``p_exp`` here.

Concrete subclasses
-------------------

Representative implementations include:

- ``ExperimentFtmwConfigPage`` — configures the FTMW acquisition parameters
  (averaging mode, shot count, segment settings).
- ``ExperimentLifConfigPage`` — configures the LIF delay grid and laser settings
  when the LIF module is active.
- ``ExperimentTypePage`` — selects the top-level experiment type (FTMW, LIF, etc.)
  and controls which subsequent pages the wizard shows.

Additional subclasses handle pulse-generator, flow-controller, IO-board, pressure-
controller, temperature-controller, and validator configuration.  The experiment-setup
workflow is described in :doc:`/user_guide/experiment_setup`.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: ExperimentConfigPage
   :members:
   :protected-members:
   :undoc-members:
