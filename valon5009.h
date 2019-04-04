#ifndef VALON5009_H
#define VALON5009_H

#include "clock.h"

class Valon5009 : public Clock
{
public:
    explicit Valon5009(int clockNum, QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings();
    bool testConnection();
    void initialize();
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    // Clock interface
public:
    QStringList channelNames();

protected:
    bool setHwFrequency(double freqMHz, int outputIndex);
    double readHwFrequency(int outputIndex);

private:
    bool valonWriteCmd(QString cmd);
    QByteArray valonQueryCmd(QString cmd);

    bool d_lockToExt10MHz;


    // HardwareObject interface
public slots:
    Experiment prepareForExperiment(Experiment exp);
};

#endif // VALON5009_H
