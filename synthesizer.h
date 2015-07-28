#ifndef SYNTHESIZER_H
#define SYNTHESIZER_H

#include "hardwareobject.h"

class Synthesizer : public HardwareObject
{
    Q_OBJECT
public:
    Synthesizer(QObject *parent = nullptr);
    virtual ~Synthesizer();

signals:
    void txFreqRead(double);
    void rxFreqRead(double);

public slots:
    virtual double readTxFreq() =0;
    virtual double readRxFreq() =0;
    double setTxFreq(const double f);
    double setRxFreq(const double f);
    virtual double setSynthTxFreq(const double f) =0;
    virtual double setSynthRxFreq(const double f) =0;

protected:
    double d_txFreq, d_rxFreq;
    double d_minFreq, d_maxFreq;
};

#ifdef BC_SYNTH
#if BC_SYNTH == 1
#include "valon5009.h"
class Valon5009;
typedef Valon5009 SynthesizerHardware;
#elif BC_SYNTH == 2
#include "pldrogroup.h"
class PldroGroup;
typedef PldroGroup SynthesizerHardware;
#else
#include "virtualvalonsynth.h"
class VirtualValonSynth;
typedef VirtualValonSynth SynthesizerHardware;
#endif
#endif

#endif // SYNTHESIZER_H
