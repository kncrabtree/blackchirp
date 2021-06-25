#ifndef FIXEDCLOCK_H
#define FIXEDCLOCK_H

#include <hardware/core/clock/clock.h>

namespace BC {
namespace Key {
static const QString fixed("fixed");
static const QString fixedName("Fixed Clock (#%1)");
}
}

class FixedClock : public Clock
{
    Q_OBJECT
public:
    FixedClock(int clockNum, QObject *parent = nullptr);

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
