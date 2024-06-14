#ifndef SIRAHCOBRA_H
#define SIRAHCOBRA_H

#include "liflaser.h"

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
static const QString sDat{"stageWavelengthDataCsv"};
static const QString sPolyOrder{"stagePolyOrder"};
}

class SirahCobra : public LifLaser
{
    Q_OBJECT
public:
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

    // HardwareObject interface
private:
    void readSettings() override;
};

#endif // SIRAHCOBRA_H
