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
    void readSettings();
    void beginAcquisition();
    void endAcquisition();
    QStringList channelNames();

    Experiment prepareForExperiment(Experiment exp);

    // Clock interface
protected:
    bool testConnection();
    void initialize();
    bool setHwFrequency(double freqMHz, int outputIndex);
    double readHwFrequency(int outputIndex);

private:
    bool valonWriteCmd(QString cmd);
    QByteArray valonQueryCmd(QString cmd);

    bool d_lockToExt10MHz;
};

#endif // VALON5015_H
