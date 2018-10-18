#ifndef VALON5015_H
#define VALON5015_H

#include "clock.h"


class Valon5015 : public Clock
{
    Q_OBJECT
public:
    explicit Valon5015(int clockNum, QObject* parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();
    QStringList channelNames();

    Experiment prepareForExperiment(Experiment exp);

    // Clock interface
protected:
    bool setHwFrequency(double freqMHz, int outputIndex);
    double readHwFrequency(int outputIndex);

private:
    bool valonWriteCmd(QString cmd);
    QByteArray valonQueryCmd(QString cmd);

    bool d_lockToExt10MHz;
};

#endif // VALON5015_H
