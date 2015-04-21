#ifndef SYNTHESIZER_H
#define SYNTHESIZER_H

#include "hardwareobject.h"

#if BC_SYNTH == 1
class Valon5009;
typedef Valon5009 SynthesizerHardware;
#else
class VirtualValonSynth;
typedef VirtualValonSynth SynthesizerHardware;
#endif

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

#endif // SYNTHESIZER_H
