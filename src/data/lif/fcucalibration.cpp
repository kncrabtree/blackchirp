#include <data/lif/fcucalibration.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>

#ifndef M_PI
#define M_PI 3.1415926535897323846
#endif

using namespace Qt::Literals::StringLiterals;
using namespace BC::FcuCal;

namespace {

constexpr double kDegToRad = M_PI/180.0;
constexpr double kRadToDeg = 180.0/M_PI;
constexpr double kReferenceTemperature = 293.0; ///< Sellmeier dn/dT zero point.

// Best-effort Sellmeier equations (wavelength in µm) and linear dn/dT
// dispersion-fit knobs. BBO: Eimerl, "Optical, mechanical, and thermal
// properties of barium borate," IEEE J. Quantum Electron. 23, 575 (1987).
// KDP: Zernike, "Refractive indices of ammonium dihydrogen phosphate and
// potassium dihydrogen phosphate between 2000 Å and 1.5 µ," J. Opt. Soc.
// Am. 54, 1215 (1964).
constexpr double kBboDnoDT = -16.6e-6;
constexpr double kBboDneDT = -9.3e-6;
constexpr double kKdpDnoDT = -3.4e-5;
constexpr double kKdpDneDT = -2.4e-5;

double bboNo2(double lamUm)
{
    const double lam2 = lamUm*lamUm;
    return 2.7405 + 0.0184/(lam2 - 0.0179) - 0.0155*lam2;
}

double bboNe2(double lamUm)
{
    const double lam2 = lamUm*lamUm;
    return 2.3730 + 0.0128/(lam2 - 0.0156) - 0.0044*lam2;
}

double kdpNo2(double lamUm)
{
    const double lam2 = lamUm*lamUm;
    return 2.259276 + 0.01008956/(lam2 - 0.012942625) + 13.00522*lam2/(lam2 - 400.0);
}

double kdpNe2(double lamUm)
{
    const double lam2 = lamUm*lamUm;
    return 2.132668 + 0.008637494/(lam2 - 0.012281043) + 3.2279924*lam2/(lam2 - 400.0);
}

/// Ordinary index n_o(lamUm) at the given dispersion-fit \a temperature
/// (arbitrary units; see the \c FcuCalibration::physical() comment), from
/// the crystal's best-effort Sellmeier equation with a linear dn/dT applied
/// to \c n itself (not \c n^2) about the 293-unit reference point.
double sellmeierNo(CrystalType crystal, double lamUm, double temperature)
{
    const double n2 = (crystal == CrystalType::BBO) ? bboNo2(lamUm) : kdpNo2(lamUm);
    const double dndT = (crystal == CrystalType::BBO) ? kBboDnoDT : kKdpDnoDT;
    return std::sqrt(n2) + dndT*(temperature - kReferenceTemperature);
}

/// Extraordinary index n_e(lamUm); see \c sellmeierNo().
double sellmeierNe(CrystalType crystal, double lamUm, double temperature)
{
    const double n2 = (crystal == CrystalType::BBO) ? bboNe2(lamUm) : kdpNe2(lamUm);
    const double dndT = (crystal == CrystalType::BBO) ? kBboDneDT : kKdpDneDT;
    return std::sqrt(n2) + dndT*(temperature - kReferenceTemperature);
}

/// Unclamped Type-I SHG phase-match sin^2(theta_pm) for fundamental
/// \a lamFundNm (nm). Outside [0,1] the crystal cannot phase-match this
/// fundamental at all; \c FcuCalibration::phaseMatchAngleDeg() clamps this
/// to report a boundary angle instead of a domain error, but the physical
/// forward map's root find (see \c FcuCalibration::physicalInverse()) needs
/// the unclamped value to restrict its search to the band where the mapping
/// is actually invertible.
double rawPhaseMatchS2(CrystalType crystal, double lamFundNm, double temperature)
{
    const double lamUm = lamFundNm/1000.0;
    const double lam2Um = lamUm/2.0;
    const double noFund = sellmeierNo(crystal, lamUm, temperature);
    const double noSh = sellmeierNo(crystal, lam2Um, temperature);
    const double neSh = sellmeierNe(crystal, lam2Um, temperature);
    return (1.0/(noFund*noFund) - 1.0/(noSh*noSh)) / (1.0/(neSh*neSh) - 1.0/(noSh*noSh));
}

/// Horner evaluation of \a coeffs (ascending order, c0 + c1*x + c2*x^2 +
/// ...) at \a x. Empty \a coeffs evaluates to 0.0 (isValid() is the guard
/// against this being a meaningless calibration, not a crash here).
double horner(const std::vector<double> &coeffs, double x)
{
    double result = 0.0;
    for(auto it = coeffs.rbegin(); it != coeffs.rend(); ++it)
        result = result*x + *it;
    return result;
}

/// Allocate and initialize a \c gsl_interp_steffen spline over \a x/\a y
/// (already validated strictly monotone in \a x by the caller), returning
/// it in a \c shared_ptr with \c gsl_spline_free as the deleter. Returns
/// null on allocation or GSL-reported initialization failure.
std::shared_ptr<gsl_spline> makeSteffenSpline(const std::vector<double> &x, const std::vector<double> &y)
{
    gsl_spline *raw = gsl_spline_alloc(gsl_interp_steffen, x.size());
    if(!raw)
        return nullptr;
    if(gsl_spline_init(raw, x.data(), y.data(), x.size()) != GSL_SUCCESS)
    {
        gsl_spline_free(raw);
        return nullptr;
    }
    return std::shared_ptr<gsl_spline>(raw, gsl_spline_free);
}

}

FcuCalibration::FcuCalibration() :
    d_errorString{u"Calibration not configured."_s}
{
}

FcuCalibration FcuCalibration::physical(CrystalType crystal, double cutAngleDeg, double temperature,
                                         double linearOffsetMm, double angleOffsetDeg, double screwPitchMm,
                                         double leverLengthMm, double motorResolution, bool invert)
{
    FcuCalibration cal;
    cal.d_scheme = Scheme::Physical;
    cal.d_physical = PhysicalParams{crystal, cutAngleDeg, temperature, linearOffsetMm, angleOffsetDeg,
                                     screwPitchMm, leverLengthMm, motorResolution, invert};

    const bool finite = std::isfinite(cutAngleDeg) && std::isfinite(temperature)
            && std::isfinite(linearOffsetMm) && std::isfinite(angleOffsetDeg)
            && std::isfinite(screwPitchMm) && std::isfinite(leverLengthMm)
            && std::isfinite(motorResolution);
    if(!finite)
    {
        cal.d_valid = false;
        cal.d_errorString = u"Physical calibration parameters must all be finite."_s;
        return cal;
    }
    if(screwPitchMm == 0.0)
    {
        cal.d_valid = false;
        cal.d_errorString = u"Physical calibration requires a non-zero screw pitch."_s;
        return cal;
    }

    cal.d_valid = true;
    cal.d_errorString.clear();
    return cal;
}

FcuCalibration FcuCalibration::polynomial(std::vector<double> forwardCoeffs, std::vector<double> inverseCoeffs)
{
    FcuCalibration cal;
    cal.d_scheme = Scheme::Polynomial;
    cal.d_forwardCoeffs = std::move(forwardCoeffs);
    cal.d_inverseCoeffs = std::move(inverseCoeffs);

    if(cal.d_forwardCoeffs.empty() || cal.d_inverseCoeffs.empty())
    {
        cal.d_valid = false;
        cal.d_errorString = u"Polynomial calibration requires non-empty forward and inverse coefficient lists."_s;
        return cal;
    }

    cal.d_valid = true;
    cal.d_errorString.clear();
    return cal;
}

FcuCalibration FcuCalibration::spline(std::vector<std::pair<double,double>> points)
{
    FcuCalibration cal;
    cal.d_scheme = Scheme::Spline;

    std::sort(points.begin(), points.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    const unsigned int minSize = gsl_interp_type_min_size(gsl_interp_steffen);
    if(points.size() < minSize)
    {
        cal.d_valid = false;
        cal.d_errorString = u"Spline calibration requires at least %1 points."_s.arg(minSize);
        return cal;
    }

    for(std::size_t i = 1; i < points.size(); ++i)
    {
        if(!(points[i-1].first < points[i].first))
        {
            cal.d_valid = false;
            cal.d_errorString = u"Spline calibration points must have distinct wavelengths."_s;
            return cal;
        }
    }

    // Invertibility requires the position sequence to be strictly monotone
    // in wavelength order (either direction; doubling reverses it).
    bool increasing = true, decreasing = true;
    for(std::size_t i = 1; i < points.size(); ++i)
    {
        if(!(points[i].second > points[i-1].second))
            increasing = false;
        if(!(points[i].second < points[i-1].second))
            decreasing = false;
    }
    if(!increasing && !decreasing)
    {
        cal.d_valid = false;
        cal.d_errorString = u"Spline calibration positions must be strictly monotone in wavelength order."_s;
        return cal;
    }

    cal.d_splinePoints = points;

    std::vector<double> lamX, lamY;
    lamX.reserve(points.size());
    lamY.reserve(points.size());
    for(const auto &p : points)
    {
        lamX.push_back(p.first);
        lamY.push_back(p.second);
    }
    cal.ps_wavelengthToPosSpline = makeSteffenSpline(lamX, lamY);

    // Positions are already known strictly monotone in wavelength order
    // above, so sorting ascending by position is just the same points in
    // (possibly) reverse order; distinctness carries over.
    auto byPos = points;
    std::sort(byPos.begin(), byPos.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
    std::vector<double> posX, posY;
    posX.reserve(byPos.size());
    posY.reserve(byPos.size());
    for(const auto &p : byPos)
    {
        posX.push_back(p.second);
        posY.push_back(p.first);
    }
    cal.ps_posToWavelengthSpline = makeSteffenSpline(posX, posY);

    if(!cal.ps_wavelengthToPosSpline || !cal.ps_posToWavelengthSpline)
    {
        cal.d_valid = false;
        cal.d_errorString = u"Failed to construct GSL interpolation spline."_s;
        return cal;
    }

    cal.d_splineWavelengthMin = lamX.front();
    cal.d_splineWavelengthMax = lamX.back();
    cal.d_splinePosMin = posX.front();
    cal.d_splinePosMax = posX.back();

    cal.d_valid = true;
    cal.d_errorString.clear();
    return cal;
}

double FcuCalibration::wavelengthToPos(double lamNm) const
{
    switch(d_scheme)
    {
    case Scheme::Physical:
        return physicalForward(lamNm);
    case Scheme::Polynomial:
        return horner(d_forwardCoeffs, lamNm);
    case Scheme::Spline:
        return splineEval(ps_wavelengthToPosSpline, d_splineWavelengthMin, d_splineWavelengthMax, lamNm);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double FcuCalibration::posToWavelength(double pos) const
{
    switch(d_scheme)
    {
    case Scheme::Physical:
        return physicalInverse(pos);
    case Scheme::Polynomial:
        return horner(d_inverseCoeffs, pos);
    case Scheme::Spline:
        return splineEval(ps_posToWavelengthSpline, d_splinePosMin, d_splinePosMax, pos);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

bool FcuCalibration::isValid() const
{
    return d_valid;
}

QString FcuCalibration::errorString() const
{
    return d_errorString;
}

double FcuCalibration::phaseMatchAngleDeg(CrystalType crystal, double lamFundNm, double temperature)
{
    const double s2 = std::clamp(rawPhaseMatchS2(crystal, lamFundNm, temperature), 0.0, 1.0);
    return std::asin(std::sqrt(s2)) * kRadToDeg;
}

double FcuCalibration::physicalForward(double lamNm) const
{
    const double thetaPmRad = phaseMatchAngleDeg(d_physical.crystal, lamNm, d_physical.temperature) * kDegToRad;
    const double sign = d_physical.invert ? -1.0 : 1.0;
    const double alphaInt = sign * (thetaPmRad - d_physical.cutAngleDeg*kDegToRad);

    // Snell refraction at the crystal face, using the ordinary index at the
    // fundamental (not the second harmonic) as the refraction index.
    const double n = sellmeierNo(d_physical.crystal, lamNm/1000.0, d_physical.temperature);
    const double alphaExt = std::asin(std::clamp(n*std::sin(alphaInt), -1.0, 1.0));

    const double x = d_physical.linearOffsetMm
            - d_physical.leverLengthMm*std::sin(d_physical.angleOffsetDeg*kDegToRad - alphaExt);
    return (d_physical.motorResolution/d_physical.screwPitchMm) * x;
}

double FcuCalibration::physicalInverse(double pos) const
{
    // Coarse grid over a plausible fundamental range, restricted to where
    // the crystal can actually phase-match (rawPhaseMatchS2 in [0,1]) so the
    // bracket search never straddles the clamped, non-monotone region
    // outside the crystal's tuning band.
    constexpr double lamMin = 380.0;
    constexpr double lamMax = 1000.0;
    constexpr int steps = 4000;

    bool havePrev = false;
    double prevLam = 0.0, prevF = 0.0;
    double bracketLo = 0.0, bracketHi = 0.0;
    bool found = false;

    for(int i = 0; i <= steps && !found; ++i)
    {
        const double lam = lamMin + (lamMax - lamMin)*static_cast<double>(i)/steps;
        const double s2 = rawPhaseMatchS2(d_physical.crystal, lam, d_physical.temperature);
        if(s2 < 0.0 || s2 > 1.0)
        {
            havePrev = false;
            continue;
        }

        const double f = physicalForward(lam) - pos;
        if(havePrev && ((prevF <= 0.0 && f >= 0.0) || (prevF >= 0.0 && f <= 0.0)))
        {
            bracketLo = prevLam;
            bracketHi = lam;
            found = true;
            break;
        }
        prevLam = lam;
        prevF = f;
        havePrev = true;
    }

    if(!found)
        return std::numeric_limits<double>::quiet_NaN();

    double lo = bracketLo, hi = bracketHi;
    double flo = physicalForward(lo) - pos;
    for(int iter = 0; iter < 100; ++iter)
    {
        const double mid = 0.5*(lo + hi);
        const double fmid = physicalForward(mid) - pos;
        if((flo <= 0.0 && fmid >= 0.0) || (flo >= 0.0 && fmid <= 0.0))
            hi = mid;
        else
        {
            lo = mid;
            flo = fmid;
        }
    }
    return 0.5*(lo + hi);
}

double FcuCalibration::splineEval(const std::shared_ptr<gsl_spline> &spline, double domainMin, double domainMax,
                                   double x) const
{
    if(!spline || x < domainMin || x > domainMax)
        return std::numeric_limits<double>::quiet_NaN();
    return gsl_spline_eval(spline.get(), x, nullptr);
}
