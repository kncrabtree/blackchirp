#ifndef VIRTUALPRESSURECONTROLLER_H
#define VIRTUALPRESSURECONTROLLER_H

#include "pressurecontroller.h"

class VirtualPressureController : public PressureController
{
   Q_OBJECT
public:
    VirtualPressureController(QObject *parent =nullptr);
    ~VirtualPressureController();

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

    // PressureController interface
public slots:
    double readPressure();
    double setPressureSetpoint(const double val);
    double readPressureSetpoint();
    void setPressureControlMode(bool enabled);
    bool readPressureControlMode();
};

#endif // VIRTUALPRESSURECONTROLLER_H
