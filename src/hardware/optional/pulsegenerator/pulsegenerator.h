#ifndef PULSEGENERATOR_H
#define PULSEGENERATOR_H

#include <hardware/core/hardwareobject.h>
#include <data/settings/hardwarekeys.h>

#include <data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.h>

class PulseGenerator : public HardwareObject
{
    Q_OBJECT
public:
    PulseGenerator(const QString& impl, const QString& label, int numChannels, QObject *parent = nullptr);
    virtual ~PulseGenerator();

public slots:
    void initialize() override final;
    bool prepareForExperiment(Experiment &exp) override final;
    void storeOptHwConfig(Experiment *exp) override { exp->addOptHwConfig(config()); }

    PulseGenConfig config() { readAll(); return d_config; }
    void readChannel(const int index);
    double readRepRate();
    PulseGenConfig::PGenMode readPulseMode();
    bool readPulseEnabled();


    bool setPGenSetting(const int index, const PulseGenConfig::Setting s, const QVariant val);
    bool setChannel(const int index, const PulseGenConfig::ChannelConfig &cc);
    bool setAll(const PulseGenConfig &cc);
    bool setRepRate(double d);
    bool setPulseMode(PulseGenConfig::PGenMode mode);
    bool setPulseEnabled(bool en);
    bool hasRole(PulseGenConfig::Role r);


    virtual bool setLifDelay(double d);

signals:
    void settingUpdate(int,PulseGenConfig::Setting,QVariant,QPrivateSignal);
    void configUpdate(PulseGenConfig,QPrivateSignal);

    // void modeUpdate(PulseGenConfig::PGenMode,QPrivateSignal);
    // void repRateUpdate(double,QPrivateSignal);
    // void pulseEnabledUpdate(bool,QPrivateSignal);

protected:
    void readSettings() override;
    void readAll();

    const PulseGenConfig &getConfig() const { return d_config; }

    virtual void initializePGen() =0;
    void sleep(bool b) override final;

    virtual bool setChWidth(const int index, const double width) =0;
    virtual bool setChDelay(const int index, const double delay) =0;
    virtual bool setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level) =0;
    virtual bool setChEnabled(const int index, const bool en) =0;
    virtual bool setChSyncCh(const int index, const int syncCh) =0;
    virtual bool setChMode(const int index, const PulseGenConfig::ChannelMode mode) =0;
    virtual bool setChDutyOn(const int index, const int pulses) =0;
    virtual bool setChDutyOff(const int index, const int pulses) =0;
    virtual bool setHwPulseMode(PulseGenConfig::PGenMode mode) =0;
    virtual bool setHwRepRate(double rr) =0;
    virtual bool setHwPulseEnabled(bool en) =0;

    virtual double readChWidth(const int index) =0;
    virtual double readChDelay(const int index) =0;
    virtual PulseGenConfig::ActiveLevel readChActiveLevel(const int index) =0;
    virtual bool readChEnabled(const int index) =0;
    virtual int readChSynchCh(const int index) =0;
    virtual PulseGenConfig::ChannelMode readChMode(const int index) =0;
    virtual int readChDutyOn(const int index) =0;
    virtual int readChDutyOff(const int index) =0;
    virtual PulseGenConfig::PGenMode readHwPulseMode() =0;
    virtual double readHwRepRate() =0;
    virtual bool readHwPulseEnabled() =0;

    int d_numChannels;

private:
    PulseGenConfig d_config;

    friend class VirtualPulseGenerator;
};


#endif // PULSEGENERATOR_H
