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
    double readTxFreq();
    double readRxFreq();
    double setTxFreq(const double f);
    double setRxFreq(const double f);
};

#endif // VIRTUALVALONSYNTH_H
