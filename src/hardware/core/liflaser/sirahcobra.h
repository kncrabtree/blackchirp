#ifndef SIRAHCOBRA_H
#define SIRAHCOBRA_H

#include "liflaser.h"
#include "sirahprotocol.h"

namespace BC::Key::LifLaser {
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
 * \brief Sirah Cobra dye-laser grating driver.
 *
 * Speaks the binary command/status protocol (BC::Sirah::buildCommand()/
 * parseStatus()) on its own RS232 port to drive the grating's sine-bar
 * tuning mechanism. minPos/maxPos/setPos()/readPos() work in the grating
 * fundamental (vacuum wavenumber, cm-1, per the LifLaser base contract);
 * the sine-bar geometry itself (posToWavelength()/wavelengthToPos()) is
 * evaluated in nm, so those two functions are the driver's nm<->cm-1
 * boundary.
 */
class SirahCobra : public LifLaser
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

    explicit SirahCobra(const QString& label, QObject *parent = nullptr);

    // HardwareObject interface
protected:
    void initialize() override;
    bool testConnection() override;

    // LifLaser interface
private:
    double readPos() override;
    void setPos(double pos) override;
    bool readFl() override;
    bool setFl(bool en) override;

    BC::Sirah::Status d_status;
    std::vector<TuningParameters> d_params;

    bool prompt();
    double posToWavelength(qint32 pos, uint stage=0);
    qint32 wavelengthToPos(double wl, uint stage=0);
    void moveRelative(qint32 steps);
    bool moveAbsolute(qint32 targetPos);

    // LifLaser interface
private:
    void lifLaserReadSettings() override;
};

#endif // SIRAHCOBRA_H
