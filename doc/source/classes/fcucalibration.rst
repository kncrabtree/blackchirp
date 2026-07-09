.. index::
   single: FcuCalibration
   single: Sirah FCU; calibration scheme
   single: Calibration Scheme; FcuCalibration
   single: Physical calibration scheme
   single: Polynomial calibration scheme
   single: Spline calibration scheme

FcuCalibration
================

``FcuCalibration`` is the assembled, validated form of a Sirah
FCU-style doubling-crystal tuning curve: a pure value type, hardware-free
like :cpp:class:`LifConversion` (its sibling in ``data/lif/``), that
maps a fundamental wavelength (nm) to and from a motor position (steps)
under one of three interchangeable schemes. One of the static factories
— ``physical()``, ``polynomial()``, or ``spline()`` — is the sole
validating construction path; a default-constructed instance is the
unconfigured, always-invalid case, since (unlike an empty
``LifConversion`` topology) there is no meaningful identity calibration
for a doubling crystal. ``wavelengthToPos()``/``posToWavelength()``
delegate to the active scheme and are defined for any finite input
regardless of ``isValid()`` — a malformed calibration evaluates to
mathematically well-defined (if meaningless) numbers, and an evaluation
that cannot be carried out for a structural reason (an unbracketed
``Physical`` root find, an out-of-domain ``Spline`` query) returns NaN
rather than aborting. ``isValid()``/``errorString()`` report whether the
calibration was assembled from well-formed input, not whether a
particular evaluation succeeded.

``SirahFcu::hwReadSettings()`` is the only current caller: it builds
and caches a ``FcuCalibration`` from the driver's registered
``calibrationScheme`` setting and the active scheme's settings, and
``setPos()``/``readPos()`` delegate to it. See
:doc:`/user_guide/hw/liffreqconversionstage` for the schemes from a
user's point of view — when to choose each, their user-visible
settings, and the offline ``python/tools/`` workflow (``fcu_measure.py``
/ ``fcu_fit.py``) that fits the parameters or curves Blackchirp then
imports or has typed in directly.

Schemes
-------

``Scheme::Physical`` evaluates a best-effort Type-I second-harmonic
phase-match law for a BBO or KDP crystal (``CrystalType``) — the Eimerl
(1987) and Zernike (1964) Sellmeier equations respectively, with a
linear dispersion knob against an arbitrary-units ``temperature``
parameter — combined with the sine-bar drive's mechanics and a
crystal-face refraction term. ``phaseMatchAngleDeg()`` is exposed
publicly so the phase-match law can be validated in isolation (for
example, against the Sirah Autotracker service manual's Table 6-1 cut
angles) independent of the sine-bar geometry. The inverse direction
(position → wavelength) is a bracketed 1-D root find over the crystal's
phase-matchable fundamental band, since the forward map has no
closed-form inverse.

``Scheme::Polynomial`` and ``Scheme::Spline`` both evaluate imported,
already-fitted data rather than a physical law: ``Polynomial`` is
Horner evaluation of separately-imported forward and inverse
coefficient lists (ascending order, so neither direction needs a root
find); ``Spline`` builds two monotone (``gsl_interp_steffen``)
interpolating splines from an imported ``(wavelength, position)`` point
table — one keyed by wavelength, one by position — since the mapping
must be invertible in both directions and a single spline object is
keyed by one axis only.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: FcuCalibration
   :members:
   :undoc-members:

.. doxygenenum:: BC::FcuCal::Scheme

.. doxygenenum:: BC::FcuCal::CrystalType
