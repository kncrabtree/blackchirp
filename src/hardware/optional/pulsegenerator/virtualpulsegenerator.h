#ifndef VIRTUALPULSEGENERATOR_H
#define VIRTUALPULSEGENERATOR_H

#include <hardware/optional/pulsegenerator/pulsegenerator.h>

namespace BC::Key {
static const QString vpGen("Virtual Pulse Generator");
}

class VirtualPulseGenerator : public PulseGenerator
{
    Q_OBJECT
public:
    explicit VirtualPulseGenerator(QObject *parent = nullptr);
    ~VirtualPulseGenerator();

    // PulseGenerator interface
protected:
    bool testConnection() override;
    void initializePGen() override {}
    bool setChWidth(const int index, const double width) override;
    bool setChDelay(const int index, const double delay) override;
    bool setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level) override;
    bool setChEnabled(const int index, const bool en) override;
    bool setHwRepRate(double rr) override;
    bool setChSyncCh(const int index, const int syncCh) override;
    bool setChMode(const int index, const PulseGenConfig::ChannelMode mode) override;
    bool setChDutyOn(const int index, const int pulses) override;
    bool setChDutyOff(const int index, const int pulses) override;
    bool setHwPulseMode(PulseGenConfig::PGenMode mode) override;
    bool setHwPulseEnabled(bool en) override;


    double readChWidth(const int index) override;
    double readChDelay(const int index) override;
    PulseGenConfig::ActiveLevel readChActiveLevel(const int index) override;
    bool readChEnabled(const int index) override;
    double readHwRepRate() override;
    int readChSynchCh(const int index) override;
    PulseGenConfig::ChannelMode readChMode(const int index) override;
    int readChDutyOn(const int index) override;
    int readChDutyOff(const int index) override;
    PulseGenConfig::PGenMode readHwPulseMode() override;
    bool readHwPulseEnabled() override;


};

#endif // VIRTUALPULSEGENERATOR_H
