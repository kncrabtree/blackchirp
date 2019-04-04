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
    void readSettings();
    bool testConnection();
    void initialize();
    void beginAcquisition();
    void endAcquisition();

    // Clock interface
protected:
    bool setHwFrequency(double freqMHz, int outputIndex);
    double readHwFrequency(int outputIndex);

private:
    QList<double> d_currentFrequencyList;
};

#endif // FIXEDCLOCK_H
