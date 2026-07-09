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

The physical model
------------------

The ``Physical`` scheme composes three pieces: crystal dispersion
(Sellmeier equations), the Type-I phase-match condition, and the
sine-bar drive geometry. The equations below are best-effort literature
values, fit to the user's own calibration data rather than matched
bit-for-bit to any vendor curve.

**Sellmeier equations.** The ordinary and extraordinary refractive
indices come from the crystal's Sellmeier equation (wavelength
:math:`\lambda` in micrometres) with a linear thermo-optic term applied
to :math:`n` about a 293-unit reference:

.. math::
   n(\lambda, T) = \sqrt{n^2(\lambda)} + \frac{dn}{dT}\,(T - 293)

For BBO (Eimerl, 1987), with
:math:`dn_o/dT = -16.6\times10^{-6}` and
:math:`dn_e/dT = -9.3\times10^{-6}`:

.. math::
   n_o^2 = 2.7405 + \frac{0.0184}{\lambda^2 - 0.0179} - 0.0155\,\lambda^2

.. math::
   n_e^2 = 2.3730 + \frac{0.0128}{\lambda^2 - 0.0156} - 0.0044\,\lambda^2

For KDP (Zernike, 1964), with
:math:`dn_o/dT = -3.4\times10^{-5}` and
:math:`dn_e/dT = -2.4\times10^{-5}`:

.. math::
   n_o^2 = 2.259276 + \frac{0.01008956}{\lambda^2 - 0.012942625}
           + \frac{13.00522\,\lambda^2}{\lambda^2 - 400}

.. math::
   n_e^2 = 2.132668 + \frac{0.008637494}{\lambda^2 - 0.012281043}
           + \frac{3.2279924\,\lambda^2}{\lambda^2 - 400}

The ``temperature`` parameter is in arbitrary units — a dispersion fit
knob rather than a controlled physical temperature — and enters only
through the thermo-optic term above.

**Phase-match condition.** For Type-I second-harmonic generation in a
negative uniaxial crystal, the phase-match angle :math:`\theta_{pm}`
(between the beam and the crystal optic axis) satisfies

.. math::
   \sin^2\theta_{pm} =
   \frac{n_o(\lambda)^{-2} - n_o(\lambda/2)^{-2}}
        {n_e(\lambda/2)^{-2} - n_o(\lambda/2)^{-2}}

with the indices evaluated at the fundamental :math:`\lambda` and its
second harmonic :math:`\lambda/2`. ``phaseMatchAngleDeg()`` returns this
angle (clamped to :math:`[0, 90]` outside the phase-matchable band).

**Sine-bar geometry.** A lead screw drives a sine bar that rotates the
crystal. With cut angle :math:`\theta_c`, linear offset :math:`L_0`,
angle offset :math:`\alpha_0`, lever length :math:`\ell`, screw pitch
:math:`s`, motor resolution :math:`r`, and sign :math:`\sigma = \pm 1`
from the ``invert`` flag, the forward map from fundamental wavelength to
motor position :math:`p` is

.. math::
   \alpha_\mathrm{int} &= \sigma\,(\theta_{pm} - \theta_c) \\
   \alpha_\mathrm{ext} &= \arcsin\!\big(n_o(\lambda)\,\sin\alpha_\mathrm{int}\big) \\
   x &= L_0 - \ell\,\sin(\alpha_0 - \alpha_\mathrm{ext}) \\
   p &= \frac{r}{s}\,x

The :math:`\alpha_\mathrm{int}\!\to\!\alpha_\mathrm{ext}` step is Snell
refraction at the crystal face (using the ordinary index at the
fundamental), which makes the cut angle and angle offset independently
identifiable rather than degenerate. The inverse direction (position to
wavelength) has no closed form and is a bracketed 1-D root find over the
crystal's phase-matchable band.

.. highlight:: cpp

API Reference
-------------

.. doxygenclass:: FcuCalibration
   :members:
   :undoc-members:

.. doxygenenum:: BC::FcuCal::Scheme

.. doxygenenum:: BC::FcuCal::CrystalType
