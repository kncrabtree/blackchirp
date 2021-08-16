#ifndef PULSEGENERATOR_H
#define PULSEGENERATOR_H

#include <hardware/core/hardwareobject.h>

#include <hardware/optional/pulsegenerator/pulsegenconfig.h>

namespace BC::Key::PGen {
static const QString key("PulseGenerator");
static const QString numChannels("numChannels");
static const QString minWidth("minWidth");
static const QString maxWidth("maxWidth");
static const QString minDelay("minDelay");
static const QString maxDelay("maxDelay");
static const QString minRepRate("minRepRateHz");
static const QString maxRepRate("maxRepRateHz");
static const QString lockExternal("lockExternal");
}

class PulseGenerator : public HardwareObject
{
    Q_OBJECT
public:
    PulseGenerator(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int numChannels, QObject *parent = nullptr,bool threaded = false, bool critical = true);
    virtual ~PulseGenerator();

public slots:
    void initialize() override final;
    bool prepareForExperiment(Experiment &exp) override final;

    PulseGenConfig config() { readAll(); return d_config; }
    void readChannel(const int index);
    double readRepRate();


    bool setPGenSetting(const int index, const PulseGenConfig::Setting s, const QVariant val);
    bool setChannel(const int index, const PulseGenConfig::ChannelConfig &cc);
    bool setAll(const PulseGenConfig &cc);
    bool setRepRate(double d);


#ifdef BC_LIF
    virtual bool setLifDelay(double d);
#endif

signals:
    void settingUpdate(int,PulseGenConfig::Setting,QVariant,QPrivateSignal);
    void configUpdate(PulseGenConfig,QPrivateSignal);
    void repRateUpdate(double,QPrivateSignal);

private:
    PulseGenConfig d_config;

protected:
    void readAll();

    virtual void initializePGen() =0;

    virtual bool setChWidth(const int index, const double width) =0;
    virtual bool setChDelay(const int index, const double delay) =0;
    virtual bool setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level) =0;
    virtual bool setChEnabled(const int index, const bool en) =0;
    virtual bool setHwRepRate(double rr) =0;

    virtual double readChWidth(const int index) =0;
    virtual double readChDelay(const int index) =0;
    virtual PulseGenConfig::ActiveLevel readChActiveLevel(const int index) =0;
    virtual bool readChEnabled(const int index) =0;
    virtual double readHwRepRate() =0;

    const int d_numChannels;

#if BC_PGEN==0
    friend class VirtualPulseGenerator;
#endif

    // HardwareObject interface
public slots:
    QStringList forbiddenKeys() const override;
};


#ifdef BC_PGEN
#if BC_PGEN == 0
#include "virtualpulsegenerator.h"
class VirtualPulseGenerator;
typedef VirtualPulseGenerator PulseGeneratorHardware;
#elif BC_PGEN==1
#include "qc9528.h"
class Qc9528;
typedef Qc9528 PulseGeneratorHardware;
#elif BC_PGEN==2
#include "qc9518.h"
class Qc9518;
typedef Qc9518 PulseGeneratorHardware;
#endif
#endif


#endif // PULSEGENERATOR_H
