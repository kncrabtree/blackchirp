#ifndef SIRAHFCU_H
#define SIRAHFCU_H

#include <hardware/core/liflaser/liffreqconversionstage.h>
#include <hardware/core/liflaser/sirahprotocol.h>
#include <data/lif/fcucalibration.h>

namespace BC::Key::SirahFcu {
inline constexpr QLatin1StringView stages{"stages"};
inline constexpr QLatin1StringView sStart{"stageStartFreqHz"};
inline constexpr QLatin1StringView sHigh{"stageHighFreqHz"};
inline constexpr QLatin1StringView sRamp{"stageRampLength"};
inline constexpr QLatin1StringView sMax{"stageMaxPos"};
inline constexpr QLatin1StringView sbls{"stageBacklashSteps"};
inline constexpr QLatin1StringView sLeverLength{"stageLeverLengthMm"};
inline constexpr QLatin1StringView sLinearOffset{"stageLinearOffsetMm"};
inline constexpr QLatin1StringView sAngleOffset{"stageAngleOffsetDeg"};
inline constexpr QLatin1StringView sCutAngle{"stageCutAngleDeg"};
inline constexpr QLatin1StringView sTemperature{"stageTemperature"};
inline constexpr QLatin1StringView sPitch{"stageScrewPitchmmPerRev"};
inline constexpr QLatin1StringView sMotorResolution{"stageMotorResolutionStepsPerRev"};
inline constexpr QLatin1StringView calScheme{"calibrationScheme"};
inline constexpr QLatin1StringView calCrystal{"crystalType"};
inline constexpr QLatin1StringView calInvert{"invertPhaseMatch"};
inline constexpr QLatin1StringView polyCoeffs{"polyCoeffs"};
inline constexpr QLatin1StringView pcOrder{"order"};
inline constexpr QLatin1StringView pcForward{"forward"};
inline constexpr QLatin1StringView pcInverse{"inverse"};
inline constexpr QLatin1StringView splinePoints{"splinePoints"};
inline constexpr QLatin1StringView spWavelength{"wavelengthNm"};
inline constexpr QLatin1StringView spPosition{"positionSteps"};
}

/*!
 * \brief Sirah Frequency Conversion Unit (FCU) driver: a doubling-crystal
 *        stage in the LIF conversion topology.
 *
 * A separate Sirah unit from the Cobra grating controller, on its own
 * RS232 port, but (to the best current knowledge, pending hardware
 * confirmation) sharing the Cobra's sine-bar tuning mechanism and binary
 * command/status protocol (BC::Sirah::buildCommand()/parseStatus()). The
 * comm-driving loops are duplicated from SirahCobra rather than shared,
 * since that hardware equivalence has not been bench-verified (see
 * sirahprotocol.h). Unlike the Cobra's grating, the doubling crystal's
 * angle <-> wavelength law is not diffraction: it is evaluated by a
 * FcuCalibration assembled in hwReadSettings() from the registered
 * calibration scheme (see data/lif/fcucalibration.h).
 *
 * The registered harmonic-order default is overridden for the common
 * lone-doubler case: N defaults to 2 and is Required (set once at profile
 * creation). This is an NHG device by identity — conversionOp() is pinned
 * to Op::NHG — though the base op setting (already defaulting to NHG)
 * stays registered so it remains snapshot-visible.
 */
class SirahFcu : public LifFreqConversionStage
{
    Q_OBJECT
public:
    explicit SirahFcu(const QString& label, QObject *parent = nullptr);

    // LifFreqConversionStage interface
    BC::LifConv::Op conversionOp() const override;

    // HardwareObject interface
protected:
    void initialize() override;
    bool testConnection() override;
    void hwReadSettings() override;

    // LifFreqConversionStage interface
private:
    void setPos(double localCm1) override;
    double readPos() override;

    BC::Sirah::Status d_status;
    FcuCalibration d_calibration;

    bool prompt();
    void moveRelative(qint32 steps);
    bool moveAbsolute(qint32 targetPos);
};

#endif // SIRAHFCU_H
