#ifndef SIRAHCOBRA_H
#define SIRAHCOBRA_H

#include "liflaser.h"

class QSerialPort;

namespace BC::Key::LifLaser {
inline constexpr QLatin1StringView mFactor{"multFactor"};
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
inline constexpr QLatin1StringView hasExtStage{"hasExternalStage"};
inline constexpr QLatin1StringView extStagePort{"externalStagePort"};
inline constexpr QLatin1StringView extStageBaud{"externalStageBaudRate"};
inline constexpr QLatin1StringView extStageCrystalAddress{"externalStageCrystalAddress"};
inline constexpr QLatin1StringView extStageCompAddress{"externalStageCompensatorAddress"};
inline constexpr QLatin1StringView extStageCrystalTheta0{"externalStageCrystalAngle0deg"};
inline constexpr QLatin1StringView extStageCrystalSlope{"externalStageCrystalSlopeDegPerNm"};
inline constexpr QLatin1StringView extStageCompTheta0{"externalStageCompensatorAngle0deg"};
inline constexpr QLatin1StringView extStageCompSlope{"externalStageCompensatorSlopeDegPerNm"};
inline constexpr QLatin1StringView extStageCrystalPoly{"externalStageCrystalPolynomial"};
inline constexpr QLatin1StringView extStageCompPoly{"externalStageCompPolynomial"};
inline constexpr QLatin1StringView polyOrder{"order"};
inline constexpr QLatin1StringView polyValue{"value"};
}

class SirahCobra : public LifLaser
{
    Q_OBJECT
public:
    struct SirahStatus {
        quint8 err;
        quint8 cStatus;
        quint8 m1Status;
        qint32 m1Pos;
        quint8 m2Status;
        qint32 m2Pos;
        int lastMoveDir{0};
    };

    struct TuningParameters {
        double lLen;
        double linOff;
        double angOff;
        double grazAng;
        double grooves;
        double pitch;
        double mRes;
    };

    struct StageStatus {
        double pos{0.0};
        double stepsPerDeg{100};
        double theta0{0.0};
        double slope{1.0};
    };
    
    struct PolyStageStatus {
        double pos{0.0};
        double stepsPerDeg{100};
        std::map<double,double> coefs;
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

    SirahStatus d_status;
    std::vector<TuningParameters> d_params;

    QByteArray buildCommand(char cmd, QByteArray args = {});
    bool prompt();
    double posToWavelength(qint32 pos, uint stage=0);
    qint32 wavelengthToPos(double wl, uint stage=0);
    void moveRelative(qint32 steps);
    bool moveAbsolute(qint32 targetPos);

    // HardwareObject interface
private:
    void readSettings() override;

    Rs232Instrument *p_extStagePort{ nullptr };
    PolyStageStatus d_crystalStatus, d_compStatus;


};

#endif // SIRAHCOBRA_H
