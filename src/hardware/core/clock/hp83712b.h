#ifndef HP83712B_H
#define HP83712B_H

#include "clock.h"

namespace BC::Key {
static const QString hp83712b{"hp83712b"};
static const QString hp83712bName{"HP 83712B Synthesizer"};
}

class HP83712B : public Clock
{
    Q_OBJECT
public:
    explicit HP83712B(QObject *parent = nullptr);

    // Clock interface
protected:
    void initializeClock() override;
    bool testClockConnection() override;
    bool setHwFrequency(double freqMHz, int outputIndex) override;
    double readHwFrequency(int outputIndex) override;
};

#endif // HP83712B_H
