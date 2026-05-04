.. index::
   single: Optional Hardware
   single: Pulse Generator
   single: Flow Controller
   single: Pressure Controller
   single: IO Board
   single: Temperature Controller
   single: Experiment Wizard

Optional Hardware Pages
=======================

After the FTMW chirp and digitizer pages, the experiment wizard inserts a dedicated configuration page for each optional hardware device that is present in the active :doc:`hardware loadout <../hardware_config/loadouts>`.
Each page is labeled with the display name of the hardware item so that it is unambiguous when multiple devices of the same type are configured.

The pages that may appear correspond to the following device types; each is documented in detail on its :doc:`Hardware Details <../hardware_details>` page, including the controls that appear on the wizard page:

- :doc:`Pulse Generator <../hw/pulsegenerator>`
- :doc:`Flow Controller <../hw/flowcontroller>`
- :doc:`Pressure Controller <../hw/pressurecontroller>`
- :doc:`IO Board <../hw/ioboard>`
- :doc:`Temperature Controller <../hw/temperaturecontroller>`

Initialization from live settings
---------------------------------

By default each optional hardware page is pre-populated with the **live settings** read from the running device at the time the wizard opens.
This means the values shown reflect the state of the hardware rather than any stored default, so day-to-day parameters (flow setpoints, pulse timings, etc.) require no manual adjustment for a routine experiment.

To override live settings with saved values — for example, when repeating a specific historical experimental configuration — use the :doc:`Quick Experiment <quick_experiment>` dialog and uncheck the ``Use Current Settings`` box for the relevant device before opening the wizard.

Changes made on an optional hardware wizard page are applied to the device when the experiment starts; they do not take effect on the running hardware until that point.

.. note::
   Optional hardware pages only appear for devices that are enabled in the active loadout.
   If a device type is not listed in the active loadout, its wizard page is omitted entirely.
   To add or remove optional hardware devices, use the :doc:`Hardware Configuration <../hardware_config>` dialog before opening the experiment wizard.
