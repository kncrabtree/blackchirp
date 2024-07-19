#ifndef SIRAHCOBRA_H
#define SIRAHCOBRA_H

#include "liflaser.h"

class QSerialPort;

namespace BC::Key::LifLaser {
static const QString sCobra{"sirahCobra"};
static const QString sCobraName{"Sirah Cobra Dye Laser"};
static const QString mFactor{"multFactor"};
static const QString stages{"stages"};
static const QString sStart{"stageStartFreqHz"};
static const QString sHigh{"stageHighFreqHz"};
static const QString sRamp{"stageRampLength"};
static const QString sMax{"stageMaxPos"};
static const QString sbls{"stageBacklashSteps"};
static const QString sLeverLength{"stageLeverLengthMm"};
static const QString sLinearOffset{"stageLinearOffsetMm"};
static const QString sAngleOffset{"stageAngleOffsetDeg"};
static const QString sGrazingAngle{"stageGrazingAngleDeg"};
static const QString sGrooves{"stageGratingGroovesPerMm"};
static const QString sPitch{"stageScrewPitchmmPerRev"};
static const QString sMotorResolution{"stageMotorResolutionStepsPerRev"};
static const QString hasExtStage{"hasExternalStage"};
static const QString extStagePort{"externalStagePort"};
static const QString extStageBaud{"externalStageBaudRate"};
static const QString extStageCrystalAddress{"externalStageCrystalAddress"};
static const QString extStageCompAddress{"externalStageCompensatorAddress"};
static const QString extStageCrystalTheta0{"externalStageCrystalAngle0deg"};
static const QString extStageCrystalSlope{"externalStageCrystalSlopeDegPerNm"};
static const QString extStageCompTheta0{"externalStageCompensatorAngle0deg"};
static const QString extStageCompSlope{"externalStageCompensatorSlopeDegPerNm"};
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

    explicit SirahCobra(QObject *parent = nullptr);

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
    StageStatus d_crystalStatus, d_compStatus;


};

#endif // SIRAHCOBRA_H
