#ifndef LIFFREQCONVERSIONSTAGE_H
#define LIFFREQCONVERSIONSTAGE_H

#include <hardware/core/hardwareobject.h>
#include <data/lif/lifconversion.h>

namespace BC::Key::LifConvStage {
inline constexpr QLatin1StringView op{"conversionOp"};          ///< BC::LifConv::Op key name.
inline constexpr QLatin1StringView harmonic{"harmonicOrder"};   ///< int N, NHG only.
inline constexpr QLatin1StringView isFinal{"finalBeam"};        ///< bool.
inline constexpr QLatin1StringView verify{"verifyMove"};        ///< bool, default true.
inline constexpr QLatin1StringView tolerance{"verifyToleranceCm1"}; ///< double cm⁻¹, move-verification window.
inline constexpr QLatin1StringView inputs{"conversionInputs"};  ///< array.
// inputs[] entry subkeys:
inline constexpr QLatin1StringView refType{"refType"};          ///< RefType key name.
inline constexpr QLatin1StringView refKey{"refStageKey"};       ///< stage hwKey when Stage.
inline constexpr QLatin1StringView refFixedCm1{"fixedCm1"};     ///< when Fixed.
}

/*!
 * \brief Base class for a LIF frequency-conversion stage (FCU): a crystal or
 *        compensator node in the optical conversion topology between the
 *        tunable LifLaser fundamental and the FINAL output beam.
 *
 * A direct child of HardwareObject (sibling of LifLaser), so it earns its
 * own hwType. The base owns only the generic contract shared by every
 * conversion node: a registered node descriptor (conversionNode(), read into
 * a BC::LifConv::Node — op, harmonic order, ordered input refs, FINAL
 * marker), a per-device verify flag, and the setPosition()/readPosition()
 * dispatch slots that move to and confirm a local input-beam wavenumber
 * (cm⁻¹). Structured calibration settings (crystal/compensator angle-vs-
 * wavelength polynomials, mount addresses) differ by driver and are not
 * declared here.
 *
 * A stage emits no output-position update to the display; only
 * LifLaser::laserPosUpdate drives the axis.
 */
class LifFreqConversionStage : public HardwareObject
{
    Q_OBJECT
public:
    LifFreqConversionStage(const QString& impl, const QString& label, QObject *parent = nullptr);
    ~LifFreqConversionStage() override;

    //! Read this stage's registered node descriptor into a BC::LifConv::Node (stageKey == d_key).
    BC::LifConv::Node conversionNode() const;

public slots:
    /*!
     * \brief Dispatch target: move to this stage's PRIMARY input-beam
     *        wavenumber (cm⁻¹).
     *
     * \a localCm1 is already computed by the caller from the assembled
     * conversion topology (LifConversion::stageInput); the stage itself
     * needs no topology. Maps the requested wavenumber to a phase-match
     * motor position via the driver's calibration, moves, then verifies via
     * readPos(). When the verify flag (BC::Key::LifConvStage::verify) is
     * off, a verification mismatch is logged as a warning and the call
     * still returns true (best-effort); when it is on, a mismatch or a
     * negative readPos() emits hardwareFailure() and returns false.
     *
     * \return Whether the move (and, if enabled, its verification) succeeded.
     */
    bool setPosition(double localCm1);

    //! Verify hook: this stage's achieved local input-beam wavenumber (cm⁻¹), or <0 on error.
    double readPosition();

private:
    //! Driver hook: move to the given local input-beam wavenumber (cm⁻¹).
    virtual void setPos(double localCm1) = 0;
    //! Driver hook: read the achieved local input-beam wavenumber (cm⁻¹), or a negative sentinel on error.
    virtual double readPos() = 0;
};

#endif // LIFFREQCONVERSIONSTAGE_H
