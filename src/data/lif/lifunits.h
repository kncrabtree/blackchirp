#ifndef LIFUNITS_H
#define LIFUNITS_H

#include <QObject>
#include <QString>

/*!
 * \file lifunits.h
 * \brief User-facing display units for the LIF laser/axis pipeline and the
 *        conversion utility between them and the internal vacuum
 *        wavenumber (cm⁻¹) representation.
 *
 * The entire LIF laser/axis pipeline works internally in vacuum wavenumber
 * (cm⁻¹); \c LaserUnit and this conversion utility exist only at the two
 * unit boundaries — the driver (hardware-native unit ↔ cm⁻¹) and the
 * display (cm⁻¹ ↔ the user-selected unit).
 */
namespace BC::LifConv {
Q_NAMESPACE

/*!
 * \brief User-facing display units. Internal representation is always \c Cm1.
 */
enum class LaserUnit {
    Cm1, ///< Vacuum wavenumber (cm⁻¹) — the internal representation.
    Nm,  ///< Vacuum wavelength (nm).
    GHz, ///< Frequency (GHz).
    eV   ///< Photon energy (eV).
};
Q_ENUM_NS(LaserUnit)

/*!
 * \brief Convert \a value expressed in unit \a u to internal vacuum
 *        wavenumber (cm⁻¹).
 *
 * \c Nm is reciprocal in wavelength; a non-positive \a value returns a
 * negative sentinel rather than dividing by zero.
 */
double toCm1(double value, LaserUnit u);

/*!
 * \brief Convert internal vacuum wavenumber (cm⁻¹) \a cm1 to a value in
 *        unit \a u.
 *
 * \c Nm is reciprocal in wavelength; a non-positive \a cm1 returns a
 * negative sentinel rather than dividing by zero.
 */
double fromCm1(double cm1, LaserUnit u);

/*!
 * \brief Short display suffix for a unit ("cm⁻¹", "nm", "GHz", "eV").
 */
QString unitLabel(LaserUnit u);

/*!
 * \brief Conversion-node operation kind (\c LifConversion, data/lif/lifconversion.h).
 *
 * N-th harmonic generation, sum-frequency generation, difference-frequency
 * generation. Registered here rather than in lifconversion.h because a
 * \c Q_NAMESPACE-tagged namespace may have exactly one moc-generated
 * metaobject definition; splitting \c Q_ENUM_NS declarations for the same
 * namespace across two headers either leaves the second header's enums
 * unregistered (AUTOMOC refuses to moc a header that lacks the \c
 * Q_NAMESPACE tag) or, if \c Q_NAMESPACE is repeated, duplicates the
 * \c BC::LifConv::staticMetaObject symbol at link time. All \c BC::LifConv
 * enums therefore live in this one file.
 */
enum class Op {
    NHG, ///< N-th harmonic generation: \c n * (single input).
    SFG, ///< Sum-frequency generation: primary + secondary input.
    DFG  ///< Difference-frequency generation: primary - secondary input.
};
Q_ENUM_NS(Op)

/*!
 * \brief Kind of a conversion node's input reference (\c LifConversion,
 *        data/lif/lifconversion.h). See \c Op for why this lives here
 *        rather than in lifconversion.h.
 */
enum class RefType {
    Laser, ///< The single tunable source (the active LifLaser's fundamental).
    Stage, ///< Another conversion node's output, named by its stage hwKey.
    Fixed  ///< A constant mixing beam (cm⁻¹).
};
Q_ENUM_NS(RefType)

}

#endif // LIFUNITS_H
