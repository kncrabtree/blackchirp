#include <data/lif/lifunits.h>

using namespace Qt::Literals::StringLiterals;

namespace {

/// Vacuum speed of light, cm/s (CODATA).
constexpr double kSpeedOfLightCmPerS{2.99792458e10};

/// f_GHz = kGHzPerCm1 * cm1; cm1 = f_GHz / kGHzPerCm1.
constexpr double kGHzPerCm1{kSpeedOfLightCmPerS / 1.0e9};

/// E_eV = kEvPerCm1 * cm1 (CODATA hc in eV*cm).
constexpr double kEvPerCm1{1.239841984e-4};

/// cm1 = kCm1PerEv * E_eV (CODATA, reciprocal of kEvPerCm1).
constexpr double kCm1PerEv{8065.543937};

/// Vacuum wavelength conversion factor: lambda_nm = kNmCm1 / cm1.
constexpr double kNmCm1{1.0e7};

/// Returned in place of a value that would require dividing by a
/// non-positive input (guards the Nm reciprocal-in-wavelength relation).
constexpr double kInvalid{-1.0};

}

namespace BC::LifConv {

double toCm1(double value, LaserUnit u)
{
    switch(u)
    {
    case LaserUnit::Nm:
        if(value <= 0.0)
            return kInvalid;
        return kNmCm1/value;
    case LaserUnit::GHz:
        return value/kGHzPerCm1;
    case LaserUnit::eV:
        return value*kCm1PerEv;
    case LaserUnit::Cm1:
    default:
        return value;
    }
}

double fromCm1(double cm1, LaserUnit u)
{
    switch(u)
    {
    case LaserUnit::Nm:
        if(cm1 <= 0.0)
            return kInvalid;
        return kNmCm1/cm1;
    case LaserUnit::GHz:
        return kGHzPerCm1*cm1;
    case LaserUnit::eV:
        return kEvPerCm1*cm1;
    case LaserUnit::Cm1:
    default:
        return cm1;
    }
}

QString unitLabel(LaserUnit u)
{
    switch(u)
    {
    case LaserUnit::Nm:
        return "nm"_L1;
    case LaserUnit::GHz:
        return "GHz"_L1;
    case LaserUnit::eV:
        return "eV"_L1;
    case LaserUnit::Cm1:
    default:
        return u"cm⁻¹"_s;
    }
}

}
