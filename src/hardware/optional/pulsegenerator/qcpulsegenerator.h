#ifndef QCPULSEGENERATOR_H
#define QCPULSEGENERATOR_H

#include "pulsegenerator.h"

class QCPulseGenerator : public PulseGenerator
{
    Q_OBJECT
public:
    explicit QCPulseGenerator(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int numChannels, QObject *parent = nullptr, bool threaded = false, bool critical = false);
    virtual ~QCPulseGenerator();



    // HardwareObject interface
protected:
    bool testConnection() override final;
    void sleep(bool b) override final;

    // PulseGenerator interface
protected:
    bool setChWidth(const int index, const double width) override final;
    bool setChDelay(const int index, const double delay) override final;
    bool setChActiveLevel(const int index, const ActiveLevel level) override final;
    bool setChEnabled(const int index, const bool en) override final;
    bool setChSyncCh(const int index, const int syncCh) override final;
    bool setChMode(const int index, const ChannelMode mode) override final;
    bool setChDutyOn(const int index, const int pulses) override final;
    bool setChDutyOff(const int index, const int pulses) override final;
    bool setHwPulseMode(PGenMode mode) override final;
    bool setHwRepRate(double rr) override final;
    bool setHwPulseEnabled(bool en) override final;
    double readChWidth(const int index) override final;
    double readChDelay(const int index) override final;
    PulseGenConfig::ActiveLevel readChActiveLevel(const int index) override final;
    bool readChEnabled(const int index) override final;
    int readChSynchCh(const int index) override final;
    PulseGenConfig::ChannelMode readChMode(const int index) override final;
    int readChDutyOn(const int index) override final;
    int readChDutyOff(const int index) override final;
    PulseGenConfig::PGenMode readHwPulseMode() override final;
    double readHwRepRate() override final;
    bool readHwPulseEnabled() override final;

protected:
    void lockKeys(bool lock);
    virtual bool pGenWriteCmd(QString cmd) =0;
    virtual QByteArray pGenQueryCmd(QString cmd) =0;


    virtual QString idResponse() =0;
    virtual QString sysStr() =0;
    virtual QString clock10MHzStr() =0;
    virtual QString trigBase() =0;

private:
    const QStringList d_channels{"T0","CHA","CHB","CHC","CHD","CHE","CHF","CHG","CHH"};
};

#endif // QCPULSEGENERATOR_H
