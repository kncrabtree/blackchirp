#ifndef FCUCALIBRATION_H
#define FCUCALIBRATION_H

#include <memory>
#include <utility>
#include <vector>

#include <QObject>
#include <QString>

#include <gsl/gsl_spline.h>

/*!
 * \file fcucalibration.h
 * \brief \c FcuCalibration — the doubling-crystal tuning law for a Sirah
 *        FCU-style frequency-conversion unit, evaluated in fundamental
 *        wavelength (nm) <-> motor position (steps).
 */
namespace BC::FcuCal {
Q_NAMESPACE

/*!
 * \brief Which tuning-curve model a \c FcuCalibration was assembled from.
 */
enum class Scheme {
    Physical,   ///< Best-effort Type-I SHG phase-match + sine-bar mechanics.
    Polynomial, ///< Imported forward/inverse coefficient lists (Horner evaluation).
    Spline      ///< Imported (wavelength, position) point table (GSL monotone splines).
};
Q_ENUM_NS(Scheme)

/*!
 * \brief Doubling-crystal species supported by the \c Physical scheme; picks
 *        the Sellmeier equations used for the phase-match calculation.
 */
enum class CrystalType {
    BBO, ///< Beta barium borate (Eimerl 1987 Sellmeier).
    KDP  ///< Potassium dihydrogen phosphate (Zernike 1964 Sellmeier).
};
Q_ENUM_NS(CrystalType)

}

/*!
 * \brief Pure value type representing an assembled FCU tuning curve: the
 *        doubling crystal's fundamental wavelength (nm) <-> motor position
 *        (steps) mapping, in either direction.
 *
 * Hardware-free and copyable, like \c LifConversion (data/lif/lifconversion.h),
 * its sibling in this directory: callers build a \c FcuCalibration from a
 * settings snapshot (or, in tests, directly from the static factories below)
 * and hand it to a driver without that driver touching the tuning math
 * itself. One of \c physical(), \c polynomial(), or \c spline() is the only
 * validating construction path; a default-constructed instance is the
 * unconfigured, always-invalid case (there is no meaningful identity
 * calibration the way an empty \c LifConversion topology is the identity
 * conversion).
 *
 * \c wavelengthToPos() / \c posToWavelength() are defined for any finite
 * input regardless of \c isValid() — malformed coefficients evaluate to
 * mathematically well-defined (if meaningless) numbers, and an evaluation
 * that cannot be carried out for a structural reason (an unbracketed
 * \c Physical root find, an out-of-domain \c Spline query) returns NaN
 * rather than aborting or throwing. \c isValid() reports whether the
 * calibration was assembled from well-formed input, not whether a
 * particular evaluation succeeded.
 */
class FcuCalibration
{
public:
    /*!
     * \brief Construct the unconfigured, invalid calibration. Use one of the
     *        static factories to build a usable instance.
     */
    FcuCalibration();

    /*!
     * \brief Assemble the \c Physical scheme: a Type-I SHG phase-match law
     *        for \a crystal (see \c phaseMatchAngleDeg()) combined with the
     *        sine-bar drive mechanics and a crystal-face refraction term.
     *
     * \a cutAngleDeg is the crystal's optic-axis-to-face cut angle (Table
     * 6-1 of the Sirah Autotracker service manual, per crystal/band).
     * \a temperature is an arbitrary-units dispersion fit knob (see the
     * Sellmeier comment in the .cpp), not a controlled physical temperature;
     * it defaults to 293 in \c phaseMatchAngleDeg() but must be supplied
     * explicitly here. \a linearOffsetMm, \a angleOffsetDeg, \a screwPitchMm,
     * \a leverLengthMm, and \a motorResolution are the sine-bar drive's
     * mechanical constants. \a invert selects the phase-match relation's ±
     * branch. Invalid (non-finite parameters, or a zero screw pitch) yields
     * \c isValid()==false with \c errorString() explaining why.
     */
    static FcuCalibration physical(BC::FcuCal::CrystalType crystal,
                                    double cutAngleDeg, double temperature,
                                    double linearOffsetMm, double angleOffsetDeg,
                                    double screwPitchMm, double leverLengthMm,
                                    double motorResolution, bool invert);

    /*!
     * \brief Assemble the \c Polynomial scheme from imported forward
     *        (wavelength (nm) -> position) and inverse (position ->
     *        wavelength (nm)) coefficient lists, ascending order
     *        (\c c0 + c1*x + c2*x^2 + ...). Evaluated by Horner's method.
     *        Invalid (either list empty) yields \c isValid()==false.
     */
    static FcuCalibration polynomial(std::vector<double> forwardCoeffs,
                                      std::vector<double> inverseCoeffs);

    /*!
     * \brief Assemble the \c Spline scheme from an imported \c (wavelength
     *        (nm), position) point table. Builds two monotone
     *        (\c gsl_interp_steffen) interpolating splines internally — one
     *        keyed by wavelength for \c wavelengthToPos(), one keyed by
     *        position for \c posToWavelength() — since the mapping must be
     *        invertible in both directions.
     *
     * \a points need not be pre-sorted. Invalid (fewer than GSL's minimum
     * point count for a Steffen spline, duplicate wavelengths, or a
     * position sequence that is not strictly monotone in wavelength order
     * and therefore not invertible) yields \c isValid()==false.
     */
    static FcuCalibration spline(std::vector<std::pair<double,double>> points);

    /*!
     * \brief Fundamental wavelength (nm) -> motor position (steps), per the
     *        active scheme. See the class comment for the no-\c isValid()
     *        per-call convention.
     */
    double wavelengthToPos(double lamNm) const;

    /*!
     * \brief Motor position (steps) -> fundamental wavelength (nm), per the
     *        active scheme. See the class comment for the no-\c isValid()
     *        per-call convention.
     *
     * \c Physical inverts by a bracketed 1-D root find over the crystal's
     * phase-matchable fundamental band; \c Polynomial is Horner evaluation
     * of the inverse coefficient list; \c Spline evaluates the
     * position-keyed interpolating spline. Returns NaN if \a pos cannot be
     * inverted (no bracket found, or outside the spline's domain).
     */
    double posToWavelength(double pos) const;

    /*!
     * \brief \c true iff this instance was assembled from well-formed input
     *        by one of the static factories.
     */
    bool isValid() const;

    /*!
     * \brief Populated iff \c !isValid(), explaining why assembly failed.
     */
    QString errorString() const;

    /*!
     * \brief Type-I second-harmonic phase-match angle (degrees) for
     *        \a crystal at fundamental wavelength \a lamFundNm (nm) and
     *        dispersion-fit \a temperature (arbitrary units; see the
     *        \c Physical factory comment).
     *
     * Exposed publicly so it can be validated in isolation against the
     * Sirah Autotracker service manual's Table 6-1 cut angles. The
     * underlying \c sin^2(theta_pm) expression is clamped to [0,1] before
     * the \c asin(), so this always returns a value in [0,90] even outside
     * the crystal's physically phase-matchable band (where it saturates at
     * the boundary rather than reporting a domain error).
     */
    static double phaseMatchAngleDeg(BC::FcuCal::CrystalType crystal,
                                      double lamFundNm, double temperature = 293.0);

private:
    /*!
     * \brief \c Physical scheme parameters, named and grouped exactly as the
     *        forward-map formula (see the .cpp) consumes them.
     */
    struct PhysicalParams {
        BC::FcuCal::CrystalType crystal{BC::FcuCal::CrystalType::BBO};
        double cutAngleDeg{0.0};
        double temperature{293.0};
        double linearOffsetMm{0.0};
        double angleOffsetDeg{0.0};
        double screwPitchMm{1.0};
        double leverLengthMm{0.0};
        double motorResolution{1.0};
        bool invert{false};
    };

    /// Fundamental wavelength (nm) -> motor position (steps), \c Physical
    /// scheme. The building block \c posToWavelength()'s root find brackets.
    double physicalForward(double lamNm) const;

    /// Motor position (steps) -> fundamental wavelength (nm), \c Physical
    /// scheme, by bracketed bisection of \c physicalForward().
    double physicalInverse(double pos) const;

    /// Evaluate \a spline at \a x, returning NaN if \a spline is null (an
    /// invalidly-assembled instance) or \a x falls outside
    /// [\a domainMin, \a domainMax]. GSL's own domain check on
    /// \c gsl_spline_eval() routes through the global error handler, which
    /// aborts the process unless the application has disabled it (as
    /// \c main.cpp does); a value type usable from a hardware-free unit
    /// test cannot rely on that, so this guards the domain itself and never
    /// calls into GSL out of range.
    double splineEval(const std::shared_ptr<gsl_spline> &spline,
                       double domainMin, double domainMax, double x) const;

    BC::FcuCal::Scheme d_scheme{BC::FcuCal::Scheme::Physical};
    bool d_valid{false};
    QString d_errorString;

    PhysicalParams d_physical;

    std::vector<double> d_forwardCoeffs;  ///< \c Polynomial: wavelength (nm) -> position.
    std::vector<double> d_inverseCoeffs;  ///< \c Polynomial: position -> wavelength (nm).

    /// \c Spline: canonical point table, sorted ascending by wavelength.
    std::vector<std::pair<double,double>> d_splinePoints;
    /// \c Spline: wavelength (nm) -> position, keyed by wavelength.
    /// \c shared_ptr (custom deleter \c gsl_spline_free) rather than
    /// \c unique_ptr so \c FcuCalibration stays a cheaply-copyable value
    /// type; \c gsl_spline itself is not copyable and evaluation is not a
    /// hot loop (once per acquisition point), so sharing the underlying
    /// spline across copies rather than deep-copying it is the simpler and
    /// sufficient choice.
    std::shared_ptr<gsl_spline> ps_wavelengthToPosSpline;
    /// \c Spline: position -> wavelength (nm), keyed by position.
    std::shared_ptr<gsl_spline> ps_posToWavelengthSpline;
    double d_splineWavelengthMin{0.0};
    double d_splineWavelengthMax{0.0};
    double d_splinePosMin{0.0};
    double d_splinePosMax{0.0};
};

#endif // FCUCALIBRATION_H
