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

After the FTMW chirp and digitizer pages, the experiment wizard
inserts one configuration page per optional hardware device in the
active :doc:`loadout <../hardware_config/loadouts>`. Each page is
labeled with the device's display name so that multiple devices of
the same type are unambiguous. To add or remove optional devices,
edit the loadout in the :doc:`Hardware Configuration <../hardware_config>`
dialog before opening the wizard.

The pages that may appear correspond to the device types below; the
controls on each page are documented on the device's
:doc:`Hardware Details <../hardware_details>` page:

- :doc:`Pulse Generator <../hw/pulsegenerator>`
- :doc:`Flow Controller <../hw/flowcontroller>`
- :doc:`Pressure Controller <../hw/pressurecontroller>`
- :doc:`IO Board <../hw/ioboard>`
- :doc:`Temperature Controller <../hw/temperaturecontroller>`

Initialization from live settings
---------------------------------

Each page is pre-populated with the **live settings** read from the
running device when the wizard opens, so day-to-day parameters (flow
setpoints, pulse timings, etc.) require no manual adjustment for a
routine experiment. To override the live values with the settings
saved alongside a previous experiment, open the
:doc:`Quick Experiment <quick_experiment>` dialog and uncheck
``Use Current Settings`` for the relevant device before launching the
wizard.

Changes made on an optional hardware page are applied to the device
when the experiment starts, not while the wizard is open.
