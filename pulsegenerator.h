#ifndef PULSEGENERATOR_H
#define PULSEGENERATOR_H

#include "hardwareobject.h"

#include "pulsegenconfig.h"

class PulseGenerator : public HardwareObject
{
    Q_OBJECT
public:
    PulseGenerator(QObject *parent = nullptr);
    ~PulseGenerator();

public slots:
    PulseGenConfig config() const { return d_config; }
    virtual QVariant read(const int index, const BlackChirp::PulseSetting s) =0;
    virtual double readRepRate() =0;

    virtual BlackChirp::PulseChannelConfig read(const int index);

    virtual bool set(const int index, const BlackChirp::PulseSetting s, const QVariant val) =0;
    virtual bool setChannel(const int index, const BlackChirp::PulseChannelConfig cc);
    virtual bool setAll(const PulseGenConfig cc);

    virtual bool setRepRate(double d) =0;
    virtual bool setLifDelay(double d);

signals:
    void settingUpdate(int,BlackChirp::PulseSetting,QVariant);
    void configUpdate(const PulseGenConfig);
    void repRateUpdate(double);

protected:
    PulseGenConfig d_config;
    virtual void readAll();

    double d_minWidth;
    double d_maxWidth;
    double d_minDelay;
    double d_maxDelay;
};


#if BC_PGEN==1
#include "qc9528.h"
class Qc9528;
typedef Qc9528 PulseGeneratorHardware;
#else
#include "virtualpulsegenerator.h"
class VirtualPulseGenerator;
typedef VirtualPulseGenerator PulseGeneratorHardware;
#endif


#endif // PULSEGENERATOR_H
