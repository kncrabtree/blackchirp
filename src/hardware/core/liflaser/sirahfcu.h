#ifndef SIRAHFCU_H
#define SIRAHFCU_H

#include <hardware/core/liflaser/liffreqconversionstage.h>
#include <hardware/core/liflaser/sirahprotocol.h>

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
inline constexpr QLatin1StringView sGrazingAngle{"stageGrazingAngleDeg"};
inline constexpr QLatin1StringView sGrooves{"stageGratingGroovesPerMm"};
inline constexpr QLatin1StringView sPitch{"stageScrewPitchmmPerRev"};
inline constexpr QLatin1StringView sMotorResolution{"stageMotorResolutionStepsPerRev"};
}

/*!
 * \brief Sirah Frequency Conversion Unit (FCU) driver: a doubling-crystal
 *        stage in the LIF conversion topology.
 *
 * A separate Sirah unit from the Cobra grating controller, on its own
 * RS232 port, but (to the best current knowledge, pending hardware
 * confirmation) sharing the Cobra's sine-bar tuning mechanism and binary
 * command/status protocol (BC::Sirah::buildCommand()/parseStatus()). The
 * comm-driving loops and tuning math are duplicated from SirahCobra rather
 * than shared, since that hardware equivalence has not been bench-verified
 * (see sirahprotocol.h).
 *
 * The node-descriptor defaults are overridden for the common lone-doubler
 * case: harmonic order N defaults to 2 and is Required (set once at
 * profile creation), and isFinal defaults to true (a single FCU with no
 * further downstream stage is the FINAL beam). op stays the base default
 * (NHG) and conversionInputs stays the base default (one Laser input).
 *
 * The sine-bar geometry defaults below are placeholders copied from
 * SirahCobra's grating stage (the actual doubler-crystal geometry is
 * unknown until the unit is calibrated on the bench).
 */
class SirahFcu : public LifFreqConversionStage
{
    Q_OBJECT
public:
    struct TuningParameters {
        double lLen;
        double linOff;
        double angOff;
        double grazAng;
        double grooves;
        double pitch;
        double mRes;
    };

    explicit SirahFcu(const QString& label, QObject *parent = nullptr);

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
    std::vector<TuningParameters> d_params;

    bool prompt();
    double posToWavelength(qint32 pos, uint stage=0);
    qint32 wavelengthToPos(double wl, uint stage=0);
    void moveRelative(qint32 steps);
    bool moveAbsolute(qint32 targetPos);
};

#endif // SIRAHFCU_H
