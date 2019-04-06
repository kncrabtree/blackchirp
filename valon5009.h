#ifndef VALON5009_H
#define VALON5009_H

#include "clock.h"

class Valon5009 : public Clock
{
public:
    explicit Valon5009(int clockNum, QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings() override;

    // Clock interface
public:
    QStringList channelNames() override;

protected:
    bool testConnection() override;
    void initializeClock() override;
    bool setHwFrequency(double freqMHz, int outputIndex) override;
    double readHwFrequency(int outputIndex) override;
    Experiment prepareClock(Experiment exp) override;

private:
    bool valonWriteCmd(QString cmd);
    QByteArray valonQueryCmd(QString cmd);

    bool d_lockToExt10MHz;
};

#endif // VALON5009_H
