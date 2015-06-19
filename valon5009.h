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
    double readTxFreq();
    double readRxFreq();
    double setTxFreq(const double f);
    double setRxFreq(const double f);

private:
    bool valonWriteCmd(QString cmd);
    QByteArray valonQueryCmd(QString cmd);
    double d_minFreq, d_maxFreq;
};

#endif // VALON5009_H
