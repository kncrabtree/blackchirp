#ifndef AWG_H
#define AWG_H

#include "tcpinstrument.h"

class AWG : public TcpInstrument
{
public:
    AWG();
    ~AWG();

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();
};

#endif // AWG_H
