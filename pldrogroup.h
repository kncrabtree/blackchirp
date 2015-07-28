#ifndef PLDROGROUP_H
#define PLDROGROUP_H

#include "synthesizer.h"

class PldroGroup : public Synthesizer
{
    Q_OBJECT
public:
    explicit PldroGroup(QObject *parent = nullptr);

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
    double setSynthTxFreq(const double f);
    double setSynthRxFreq(const double f);
};

#endif // PLDROGROUP_H
