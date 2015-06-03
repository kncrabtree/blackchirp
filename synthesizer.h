#ifndef SYNTHESIZER_H
#define SYNTHESIZER_H

#include "hardwareobject.h"

class Synthesizer : public HardwareObject
{
    Q_OBJECT
public:
    Synthesizer(QObject *parent = nullptr);
    ~Synthesizer();

signals:
    void txFreqRead(double);
    void rxFreqRead(double);

public slots:
    virtual double readTxFreq() =0;
    virtual double readRxFreq() =0;
    virtual double setTxFreq(const double f) =0;
    virtual double setRxFreq(const double f) =0;

protected:
    double d_txFreq, d_rxFreq;
};

#ifdef BC_SYNTH
#if BC_SYNTH == 1
#include "valon5009.h"
class Valon5009;
typedef Valon5009 SynthesizerHardware;
#else
#include "virtualvalonsynth.h"
class VirtualValonSynth;
typedef VirtualValonSynth SynthesizerHardware;
#endif
#endif

#endif // SYNTHESIZER_H
