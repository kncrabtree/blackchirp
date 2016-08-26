#ifndef VIRTUALSYNTH_H
#define VIRTUALSYNTH_H

#include "synthesizer.h"

class VirtualSynth : public Synthesizer
{
    Q_OBJECT
public:
    explicit VirtualSynth(QObject *parent = nullptr);
    ~VirtualSynth();

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    // Synthesizer interface
public slots:
    double readSynthTxFreq();
    double readSynthRxFreq();
    double setSynthTxFreq(const double f);
    double setSynthRxFreq(const double f);
};

#endif // VIRTUALSYNTH_H
