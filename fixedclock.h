#ifndef FIXEDCLOCK_H
#define FIXEDCLOCK_H

#include "clock.h"

class FixedClock : public Clock
{
    Q_OBJECT
public:
    FixedClock(int clockNum, QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    // Clock interface
protected:
    bool setHwFrequency(double freqMHz, int outputIndex);
    double readHwFrequency(int outputIndex);

private:
    double d_currentFrequency;
};

#endif // FIXEDCLOCK_H
