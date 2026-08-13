Laser Frequency Conversion Stage
================================

* Overview_
* Settings_
* Drivers_
* `Sirah FCU Calibration`_

Overview
--------

A Laser Frequency Conversion Stage represents one physical optical element — a doubling crystal, a mixing crystal — in the LIF frequency-conversion chain between the tunable laser and the sample. Each configured stage appears as a row in the Conversion tab's table (see :doc:`/user_guide/lif/conversion`), where its inputs are wired and, for at most one stage, its output is marked as the excitation beam. The device itself only moves to and reports a local input-beam position (in cm⁻¹); the chain's topology — which stage feeds which, and which stage is the excitation beam — is per-experiment state configured on the Conversion tab, not part of the device profile.

Settings
--------

Most Laser Frequency Conversion Stage settings are exposed in the :doc:`hardware dialog </user_guide/hwdialog>` with inline labels and tooltips, so they need no additional explanation here. A few items are worth highlighting:

* ``Conversion Operation`` sets the stage's operation — NHG (N-th harmonic), SFG, or DFG. Concrete drivers usually pin this to a fixed value that matches the physical device (a doubler is always NHG, for example), so it is not normally something you choose yourself.
* ``Harmonic Order`` sets the harmonic order N for an NHG stage; it is ignored for SFG/DFG stages. Some drivers fix this at profile creation rather than leaving it freely editable (see Sirah FCU below).
* ``Verify Move`` controls whether a move is confirmed against the stage's read-back position after every move. When enabled (the default), a mismatch beyond ``Verify Tolerance`` fails the move; when disabled, a mismatch is only logged as a warning and the move is treated as successful.
* ``Verify Tolerance (cm-1)`` sets the read-back window, in cm⁻¹, used by ``Verify Move``. Defaults to 1.0.

Drivers
-------

Virtual
.................

A dummy driver.

Fixed
.................

Represents a conversion-topology node that is not under Blackchirp's control — a crystal or compensator tuned by hand, or one you simply want represented in the chain's math without automating its motion. Moves always report success at the commanded position, so the stage participates fully in the chain with no separate place to enter its state.

Sirah FCU
.................

The Sirah Frequency Conversion Unit is a doubling-stage driver for a Sirah frequency-conversion unit — a separate Sirah instrument from the :doc:`Sirah Cobra <liflaser>` grating controller, connected on its own serial port with its own communication settings. Harmonic order defaults to 2, the common lone-doubler case; the default is set prominently at profile creation, but it remains changeable afterward through the Conversion tab's **Change harmonic…** action (see :ref:`lif-conversion-harmonic`). The doubling crystal's angle-to-wavelength tuning curve is evaluated through a selectable :ref:`calibration scheme <sirah-fcu-calibration>` rather than a fixed formula; see that section for the schemes, their settings, and the offline calibration workflow that produces the numbers you enter here.

.. _sirah-fcu-calibration:

Sirah FCU Calibration
-----------------------

The Sirah FCU's doubling crystal is angle-tuned: the fundamental
wavelength that phase-matches at a given crystal angle is set by the
crystal's own dispersion, not by a diffraction law. The driver
evaluates this fundamental-wavelength-to-motor-position tuning curve
through a user-selectable **Calibration Scheme** rather than a single
built-in formula. Blackchirp only *evaluates* a scheme's already-fitted
curve — it does not fit one itself; the fit is produced offline against
a bench-recorded calibration run, as described in `Offline Calibration
Workflow`_ below. Since the compensator is geared directly to the
crystal stage, one calibration curve covers the whole device.

``Calibration Scheme`` selects between the three schemes below.
Changing it immediately shows only the fields relevant to the newly
selected scheme in the hardware dialog and hides the rest — including
the *Crystal Stage Geometry* table's motor-motion fields
(``stageStartFreqHz``, ``stageHighFreqHz``, ``stageRampLength``,
``stageMaxPos``, ``stageBacklashSteps``), which are only reachable from
the dialog while **Physical** is selected. Hiding is a dialog display
convenience, not a reset: a field's stored value stays in effect even
while it is not shown.

**Physical**
   A best-effort Type-I second-harmonic phase-matching model for a BBO
   or KDP doubling crystal, combined with the sine-bar drive mechanics
   that turn a phase-match angle into a motor position. It needs only a
   handful of fitted numbers, extrapolates gracefully beyond the
   measured wavelength range because it follows the crystal's actual
   dispersion, and is the best choice unless the crystal is tuned close
   to a 90° phase-match angle, where the physical curve turns very
   steep and a coarse fit through it is unforgiving.

   Its settings, on the *Crystal Stage Geometry* table and alongside
   ``Calibration Scheme``:

   * ``Crystal Type`` — ``BBO`` or ``KDP``; selects which crystal's
     dispersion the model uses.
   * ``Invert Phase Match`` — selects the alternate (−) branch of the
     phase-match relation; toggle this if a fitted curve runs the
     wrong direction.
   * ``stageCutAngleDeg`` and ``stageTemperature`` — the crystal's
     optic-axis cut angle and an arbitrary-units dispersion fit knob
     (not a controlled physical temperature).
   * ``stageLeverLengthMm``, ``stageLinearOffsetMm``,
     ``stageAngleOffsetDeg``, ``stageScrewPitchmmPerRev``, and
     ``stageMotorResolutionStepsPerRev`` — the sine-bar drive's
     mechanical constants, the same geometry documented for the
     :doc:`Sirah Cobra <liflaser>` grating stage.

   These are the same numbers that ``fcu_fit.py`` prints after fitting
   a bench calibration run; type them directly into the corresponding
   fields rather than importing a CSV. By default the script fits only
   three of them — temperature, linear offset, and angle offset — and
   holds the cut angle and screw pitch fixed at the values you supply,
   fitting those two only when ``--fit-cut-angle`` or
   ``--fit-screw-pitch`` is passed.

**Polynomial**
   Imported forward (wavelength → position) and inverse (position →
   wavelength) coefficient lists, one row per polynomial order. It
   makes no assumption about the underlying physics, so it fits
   whatever curve shape the data shows — but, unlike Physical, it only
   interpolates the fitted curve and needs denser, well-spread
   measurements across the whole tuning range to be reliable. A
   low-order polynomial is also a poor proxy for a steep near-90°
   curve.

   Its only setting is the *Polynomial Coefficients* table (columns
   ``order`` / ``forward`` / ``inverse``), populated by importing the
   CSV ``fcu_fit.py`` exports (see below) rather than typed in by hand.

**Spline**
   An imported table of ``(wavelength, position)`` points; Blackchirp
   builds a monotone interpolating spline through them in each
   direction. Like Polynomial, it is model-agnostic and needs denser
   measurements than Physical, but a spline tracks a steep near-90°
   curve locally without the ringing a high-order polynomial fit is
   prone to, making it the more robust choice there.

   Its only setting is the *Spline Points* table (columns
   ``wavelengthNm`` / ``positionSteps``), also populated by CSV
   import.

Offline Calibration Workflow
.............................

Fitting parameters or curves for any of the three schemes happens
outside Blackchirp, using the standalone scripts under ``python/tools/``
in the Blackchirp source repository on GitHub. These are a
specialty offline utility: they are **not** bundled with the Blackchirp
application download and **not** part of the installable ``blackchirp``
Python package — obtain them by cloning or downloading the repository.
See that directory's ``README.md`` for the full command-line reference:

#. **Record a calibration run** with ``fcu_measure.py``. At each
   fundamental wavelength, hand-tune the doubling crystal at the bench
   (peak the doubled output, following the unit's own calibration
   procedure), then let the script query the FCU's motor position over
   its serial port and append a ``wavelengthNm;positionSteps`` row to a
   measurements CSV::

       python fcu_measure.py --port /dev/ttyUSB1 --output measurements.csv

   A ``--simulate`` mode records a synthetic curve instead, so the tool
   is exercisable without hardware. Hardware mode needs the
   ``pyserial`` package.

#. **Fit and visualize** the run with ``fcu_fit.py``, which fits all
   three schemes from the same measurements CSV in one pass and plots
   the resulting tuning curves and residuals (needs ``numpy``,
   ``scipy``, and ``matplotlib``)::

       python fcu_fit.py measurements.csv --crystal bbo --cut-angle 32.3 \
           --save calibration.png

   It prints the fitted Physical scheme's parameters to the console
   (fitting temperature, linear offset, and angle offset by default,
   with cut angle and screw pitch held fixed unless ``--fit-cut-angle``
   or ``--fit-screw-pitch`` is passed), and writes an
   ``order;forward;inverse`` CSV for the Polynomial scheme and a
   ``wavelengthNm;positionSteps`` CSV for the Spline scheme alongside
   the measurements file.

#. **Bring the result into Blackchirp.** Open the Sirah FCU's hardware
   dialog, select the matching ``Calibration Scheme``, and either type
   the printed Physical parameters directly into their fields, or open
   the *Polynomial Coefficients* / *Spline Points* row's **Edit…**
   button and, inside that sub-dialog, click **Import CSV…** to load
   the corresponding exported file (see :ref:`hwdialog-settings` for
   how array settings and CSV import work generally). The importer
   matches header columns to the table's fields by name, so either
   exported CSV loads without further preparation.

.. note::
   The Physical scheme is a best-effort BBO/KDP phase-matching model
   fit to *your* measured data — its functional form follows real
   crystal dispersion, but ``fcu_fit.py`` fits it independently for
   each unit and is not guaranteed to reproduce any vendor-internal
   calibration curve exactly. Record enough calibration points across
   your working range, and check the plotted residuals, before trusting
   a fit (Physical, Polynomial, or Spline) for production tuning.
