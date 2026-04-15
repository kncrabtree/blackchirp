#ifndef FIXEDCLOCK_H
#define FIXEDCLOCK_H

#include <hardware/core/clock/clock.h>

namespace BC::Key::Clock {
inline constexpr QLatin1StringView ch{"fixedOutputs"};
inline constexpr QLatin1StringView freq{"lastFreqMHz"};
}

class FixedClock : public Clock
{
    Q_OBJECT
public:
    FixedClock(const QString& label, QObject *parent = nullptr);
    ~FixedClock();

    // Clock interface
protected:
    bool testClockConnection() override;
    void initializeClock() override;
    bool setHwFrequency(double freqMHz, int outputIndex) override;
    double readHwFrequency(int outputIndex) override;

private:
    QList<double> d_currentFrequencyList;

};

#endif // FIXEDCLOCK_H
