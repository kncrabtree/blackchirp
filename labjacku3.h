#ifndef LABJACKU3_H
#define LABJACKU3_H

#include "ioboard.h"

#include "u3.h"

class LabjackU3 : public IOBoard
{
    Q_OBJECT
public:
    explicit LabjackU3(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

private:
    HANDLE d_handle;
    u3CalibrationInfo d_calInfo;

    int d_serialNo;

    void configure();
    void closeConnection();
};

#endif // LABJACKU3_H
