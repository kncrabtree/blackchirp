#ifndef FIXEDCLOCK_H
#define FIXEDCLOCK_H

#include <hardware/core/clock/clock.h>

namespace BC::Key::Clock {
static const QString fixed{"fixed"};
static const QString fixedName("Fixed Clock");
static const QString ch{"fixedOutputs"};
static const QString freq{"lastFreqMHz"};
}

class FixedClock : public Clock
{
    Q_OBJECT
public:
    FixedClock(QObject *parent = nullptr);
    ~FixedClock();

    // Clock interface
protected:
    bool testClockConnection() override;
    void initializeClock() override;
    bool setHwFrequency(double freqMHz, int outputIndex) override;
    double readHwFrequency(int outputIndex) override;

private:
    QList<double> d_currentFrequencyList;

    // HardwareObject interface
public slots:
    QStringList forbiddenKeys() const override;
};

#endif // FIXEDCLOCK_H
