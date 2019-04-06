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
    void readSettings() override;

    // Clock interface
protected:
    bool testConnection() override;
    void initializeClock() override;
    bool setHwFrequency(double freqMHz, int outputIndex) override;
    double readHwFrequency(int outputIndex) override;

private:
    QList<double> d_currentFrequencyList;
};

#endif // FIXEDCLOCK_H
