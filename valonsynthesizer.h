#ifndef VALONSYNTHESIZER_H
#define VALONSYNTHESIZER_H

#include "rs232instrument.h"


class ValonSynthesizer : public Rs232Instrument
{
    Q_OBJECT
public:
    ValonSynthesizer();
    ~ValonSynthesizer();

signals:
    void txFreqRead(double);
    void rxFreqRead(double);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    double readTxFreq();
    double readRxFreq();
    double setTxFreq(const double f);
    double setRxFreq(const double f);

private:
    double d_txFreq, d_rxFreq;

    // HardwareObject interface
public slots:

};

#endif // VALONSYNTHESIZER_H
