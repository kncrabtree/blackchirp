#ifndef VALON5009_H
#define VALON5009_H

#include "synthesizer.h"

#define BC_VALONTXCHANNEL 1
#define BC_VALONRXCHANNEL 2

class Valon5009 : public Synthesizer
{
public:
    explicit Valon5009(QObject *parent = nullptr);

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

private:
    bool valonWriteCmd(QString cmd);
    QByteArray valonQueryCmd(QString cmd);

    bool setSynth(int channel, double f);
    bool readSynth(int channel);
};

#endif // VALON5009_H
