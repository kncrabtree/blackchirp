LIF Laser
=========

* Overview_
* Settings_
* Drivers_

Overview
--------

The LIF laser is the laser whose frequency is stepped during an LIF acquisition. It is assumed to be a pulsed, tunable laser, optionally with a software-controlled flashlamp. The position is interpreted in whatever units the driver reports (wavelength, frequency, motor steps, etc.); Blackchirp uses the configured units only for display and for labeling text output.

Settings
--------

Most LIF laser settings are exposed in the :doc:`hardware dialog </user_guide/hwdialog>` with inline labels and tooltips, so they need no additional explanation here. A few items are worth highlighting:

* ``minPos`` / ``maxPos`` define the allowable position range. Blackchirp refuses values outside this window, so the values must reflect the actual hardware capability and the chosen ``units``.
* ``units`` is a free-form string used only for display and text-file headers; it is not parsed by Blackchirp and does not trigger any unit conversion.
* ``decimals`` controls the number of decimal places shown on UI controls and in error messages.
* ``hasFlashlampControl`` advertises whether the laser exposes a software-controlled flashlamp. When true, the flashlamp is enabled at the start of an acquisition and is disabled at the end if the LIF configuration's *Disable Flashlamp* option is set.

Drivers
-------

Virtual
.................

A dummy driver.

Opolette
...................

The OPOTek Opolette is a tunable pulsed OPO laser. Operation requires a Windows-based SDK; Blackchirp connects to a custom TCP server that runs on a Windows machine and wraps the SDK. The socket-server code is not in the Blackchirp repository, but it can be added in the unlikely event another group needs the same laser.

Sirah Cobra
........................

.. warning::
   The Sirah Cobra driver is under active development and is not ready for general use.

The Sirah Cobra is one of a series of tunable pulsed dye lasers. For maximum cross-platform compatibility, Blackchirp talks directly to the internal stepper motor through a serial port, emulating the behavior of the Sirah drivers (including the backlash correction for wavelength tuning). The Sirah drivers themselves are not required. The Sirah Cobra has no integrated flashlamp control; use a :doc:`/user_guide/hw/pulsegenerator` channel to drive the flashlamp instead.

The driver controls only the fundamental laser position; frequency-conversion units are not supported, though the driver leaves room to add them. Operating the laser requires several parameters per "stage" to be entered in the LIF Laser hardware dialog. These parameters are supplied with the laser's datasheets and should also be available in an ini file shipped with the Sirah library. A ``stages`` array contains entries for the resonator (stage 0) and any attached frequency-conversion units (stages 1+); for each stage, the following entries should be defined:

- ``stageStartFreqHz``: starting motor frequency (Hz). Defaults to 3000.
- ``stageHighFreqHz``: final motor frequency (Hz). Defaults to 12000.
- ``stageRampLength``: steps for motor acceleration. Defaults to 2400.
- ``stageMax``: maximum motor position. Defaults to 3300000.
- ``stageBacklashSteps``: number of steps for backlash correction.
- ``stageWavelengthDataCsv``: path to a csv file containing motor calibration data.
- ``stagePolyOrder``: order of the polynomial fit to motor/wavelength position data.

Wavelength calibration uses an interpolated lookup table. A csv file containing motor positions and the corresponding wavelengths across the laser's operating range must be provided, with all wavelengths given with respect to the resonator (not any multiples thereof). Blackchirp fits the supplied data to a polynomial of the indicated order in both directions (motor vs. wavelength and wavelength vs. motor) and uses the derived polynomials for conversion. A ``multFactor`` key sets the multiplication afforded by any FCU stages and is used to convert the position entered in the user interface to the resonator wavelength: with ``multFactor`` set to 2, an entry of 300 nm corresponds to a resonator wavelength of 600 nm.
