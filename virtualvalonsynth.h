#ifndef VIRTUALVALONSYNTH_H
#define VIRTUALVALONSYNTH_H

#include "synthesizer.h"

class VirtualValonSynth : public Synthesizer
{
    Q_OBJECT
public:
    explicit VirtualValonSynth(QObject *parent = nullptr);
    ~VirtualValonSynth();

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

#endif // VIRTUALVALONSYNTH_H
