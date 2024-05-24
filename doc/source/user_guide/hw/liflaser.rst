LIF Laser
=========

* Overview_
* Settings_
* Implementations_

Overview
--------

The LIF Laser is the laser whose frequency is stepped during an LIF acquisition. It is assumed to be a pulsed, tunable laser with the ability to toggle on and off a flashlamp (though this behavior is optional).

Settings
--------

- ``decimals`` (int): Number of decimal places to display on UI.
- ``maxPos`` (double): Maximum laser position setting. Interpretation of the units is up to the implementation; may be wavelength, frequency, stepper motor position, etc.
- ``maxPos`` (double): Minimum laser position setting.
- ``units`` (string): Units for laser position. Only used for display on UI and in text output files.


Implementations
---------------

Virtual (virtual)
.................

A dummy implementation.

Opolette (opolette)
...................

The OPOTek Opolette is a tunable pulsed OPO laser. Operation of the laser requires a Windows-based SDK. Blackchirp communicates using a custom-written TCP server which runs on a Windows machine, and communicates with Blackchirp through a TCP socket connection. At the time of writing, the socket server code is not in the Blackchirp repository, though it can be added in the unlikely event anyone else uses the same laser.

Sirah Cobra (sirahcobra)
........................

The Sirah Cobra is one of a series of tunable pulsed dye lasers. For maximum
cross-platform compatability, Blackchirp communicates directly with the
internal motor though a serial port connection. The Sirah drivers are not
required for use with Blackchirp. Blackchirp emulates the behavior of the Sirah
drivers, including the backlash correction for wavelength tuning. The Sirah
Cobra does not have integrated flashlamp control; use a delay generator channel
to control the flashlamp instead.

Currently, the only implemented functionality is controlling the fundamental
laser position; control of any frequency conversion units is not yet supported
but it is possible to do so in the future. Control of the system requires
several parameters to be defined for each "stage"; these parameters are
supplied with the datasheets for each laser but should also be available in an
ini file located in the Sirah library. These parameters must be defined for
your system before correct operation is possible; the settings are made in the
LIF Laser hardware menu. A dropdown menu named "stages" contains entries for
the resonator (stage 0) and any attached frequency conversion units (stages
1+). For each stage, the following entries should be defined:

- `stageStartFreqHz`: Starting motor frequency (Hz). Defaults to 3000.
- `stageHighFreqHz`: Final motor frequency (Hz). Defaults to 12000.
- `stageRampLength`: Steps for motor acceleration. Defaults to 2400.
- `stageMax`: Maximum motor position. Defaults to 3300000.
- `stageBacklashSteps`: Number of steps for backlash correction.
- `stageWavelengthDataCsv`: Path to csv file containing motor calibration data.
- `stagePolyOrder`: Order of polynomial fit to motor/wavelength position data.

Wavelength calibration is implemented via an interpolated lookup table. To this
end, a csv file containing motor positions and corresponding wavelengths across
the laser operation range must be provided. All wavelength readings must be
with respect to the resonator wavelength, not any multiples thereof. Blackchirp
will fit the provided data to a polynomial of the indicated order (both motor
vs wavelength and wavelength vs motor) and use the derived polynomials for
conversion between motor position and wavelength. Finally, a `multFactor` key
is provided to indicate the multiplication afforded by any FCU stages. This
will be used to convert the laser position entered in the user interface to a
resonater laser wavelength. For instance, if `multFactor` is 2, then a user
entry of 300 nm would be converted to a resonator wavelength of 600 nm.
