#ifndef PULSEGENERATOR_H
#define PULSEGENERATOR_H

#include <src/hardware/core/hardwareobject.h>

#include <src/data/experiment/pulsegenconfig.h>

namespace BC::Key::PGen {
static const QString key("pGen");
static const QString numChannels("numChannels");
static const QString minWidth("minWidth");
static const QString maxWidth("maxWidth");
static const QString minDelay("minDelay");
static const QString maxDelay("maxDelay");
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
    void readSettings() override;

    PulseGenConfig config() const { return d_config; }
    virtual QVariant read(const int index, const PulseGenConfig::Setting s) =0;
    virtual double readRepRate() =0;

    virtual PulseGenConfig::ChannelConfig read(const int index);

    virtual bool set(const int index, const PulseGenConfig::Setting s, const QVariant val) =0;
    virtual bool setChannel(const int index, const PulseGenConfig::ChannelConfig &cc);
    virtual bool setAll(const PulseGenConfig cc);

    virtual bool setRepRate(double d) =0;

#ifdef BC_LIF
    virtual bool setLifDelay(double d);
#endif

signals:
    void settingUpdate(int,PulseGenConfig::Setting,QVariant);
    void configUpdate(const PulseGenConfig);
    void repRateUpdate(double);

protected:
    PulseGenConfig d_config;
    virtual void readAll();
    virtual void initializePGen() =0;

    double d_minWidth;
    double d_maxWidth;
    double d_minDelay;
    double d_maxDelay;
    const int d_numChannels;
};


#if BC_PGEN==1
#include "qc9528.h"
class Qc9528;
typedef Qc9528 PulseGeneratorHardware;
#elif BC_PGEN==2
#include "qc9518.h"
class Qc9518;
typedef Qc9518 PulseGeneratorHardware;
#else
#include "virtualpulsegenerator.h"
class VirtualPulseGenerator;
typedef VirtualPulseGenerator PulseGeneratorHardware;
#endif


#endif // PULSEGENERATOR_H
