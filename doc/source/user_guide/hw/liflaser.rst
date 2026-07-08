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
* ``units`` selects the display unit from a fixed set — ``Cm1``, ``Nm``, ``GHz``, or ``eV`` — shown under those literal names in the Position Units combo box. Position values elsewhere in the UI (spin boxes, previews) are converted to and from the internal cm⁻¹ representation and labeled with the corresponding pretty unit (cm⁻¹, nm, GHz, eV).
* ``decimals`` controls the number of decimal places shown on UI controls and in error messages.
* ``hasFlashlampControl`` advertises whether the laser exposes a software-controlled flashlamp. When true, the flashlamp is enabled at the start of an acquisition and is disabled at the end if the LIF configuration's *Disable Flashlamp* option is set.

Drivers
-------

Virtual
.................

A dummy driver.

Opolette
...................

The OPOTek Opolette is a tunable pulsed OPO laser. Operation requires a Windows-based SDK; Blackchirp connects to a custom TCP server that runs on a Windows machine and wraps the SDK. The socket-server code is not in the Blackchirp repository.

Sirah Cobra
........................

.. warning::
   The Sirah Cobra driver is incomplete; do not rely on it for production acquisitions.

The Sirah Cobra is one of a series of tunable pulsed dye lasers. For maximum cross-platform compatibility, Blackchirp talks directly to the internal stepper motor through a serial port, emulating the behavior of the Sirah drivers (including the backlash correction for wavelength tuning). The Sirah drivers themselves are not required. The Sirah Cobra has no integrated flashlamp control; use a :doc:`/user_guide/hw/pulsegenerator` channel to drive the flashlamp instead.

The driver controls only the fundamental laser (resonator) position; it does not itself handle frequency conversion. If the laser's output passes through a doubling or mixing stage before reaching the sample, add a :doc:`LIF Conversion Stage <liffreqconversionstage>` device (for example, the Sirah FCU driver for a Sirah frequency-conversion unit) and wire it into the :doc:`frequency-conversion chain </user_guide/lif/conversion>`; the fundamental position reported by this driver is then the input to that chain rather than the sample-facing wavelength.

Operating the laser requires several parameters describing the resonator's stepper motor to be entered in the LIF Laser hardware dialog. These parameters are supplied with the laser's datasheets and should also be available in an ini file shipped with the Sirah library:

- ``stageStartFreqHz``: starting motor frequency (Hz). Defaults to 3000.
- ``stageHighFreqHz``: final motor frequency (Hz). Defaults to 12000.
- ``stageRampLength``: steps for motor acceleration. Defaults to 2400.
- ``stageMaxPos``: maximum motor position. Defaults to 3300000.
- ``stageBacklashSteps``: number of steps for backlash correction.
- ``stageLeverLengthMm``, ``stageLinearOffsetMm``, ``stageAngleOffsetDeg``, ``stageGrazingAngleDeg``, ``stageGratingGroovesPerMm``, ``stageScrewPitchmmPerRev``, ``stageMotorResolutionStepsPerRev``: the sine-bar tuning geometry for the grating motor stage.

Wavelength calibration is computed directly from the sine-bar geometry parameters via the grating equation, rather than from a lookup table; the geometry values above must reflect the specific unit's as-built mechanics for the conversion to be accurate.
