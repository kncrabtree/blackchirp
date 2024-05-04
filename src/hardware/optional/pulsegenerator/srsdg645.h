#ifndef SRSDG645_H
#define SRSDG645_H

#include "pulsegenerator.h"

namespace BC::Key::PGen {
static const QString dg645{"dg645"};
static const QString dg645Name{"Pulse Generator SRS DG645"};
}

class SRSDG645 : public PulseGenerator
{
    Q_OBJECT
public:
    explicit SRSDG645(QObject *parent = nullptr);

    // HardwareObject interface
protected:
    bool testConnection() override;

    // PulseGenerator interface
protected:
    void initializePGen() override;
    bool setChWidth(const int index, const double width) override;
    bool setChDelay(const int index, const double delay) override;
    bool setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level) override;
    bool setChEnabled(const int index, const bool en) override;
    bool setChSyncCh(const int index, const int syncCh) override;
    bool setChMode(const int index, const PulseGenConfig::ChannelMode mode) override;
    bool setChDutyOn(const int index, const int pulses) override;
    bool setChDutyOff(const int index, const int pulses) override;
    bool setHwPulseMode(PulseGenConfig::PGenMode mode) override;
    bool setHwRepRate(double rr) override;
    bool setHwPulseEnabled(bool en) override;
    double readChWidth(const int index) override;
    double readChDelay(const int index) override;
    PulseGenConfig::ActiveLevel readChActiveLevel(const int index) override;
    bool readChEnabled(const int index) override;
    int readChSynchCh(const int index) override;
    PulseGenConfig::ChannelMode readChMode(const int index) override;
    int readChDutyOn(const int index) override;
    int readChDutyOff(const int index) override;
    PulseGenConfig::PGenMode readHwPulseMode() override;
    double readHwRepRate() override;
    bool readHwPulseEnabled() override;
};

#endif // SRSDG645_H
