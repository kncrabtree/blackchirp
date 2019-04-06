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
    void beginAcquisition();
    void endAcquisition();

    // Clock interface
protected:
    bool testConnection();
    void initialize();
    bool setHwFrequency(double freqMHz, int outputIndex);
    double readHwFrequency(int outputIndex);

private:
    QList<double> d_currentFrequencyList;
};

#endif // FIXEDCLOCK_H
