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
    void readSettings();
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
    void openGateValve();
    void closeGateValve();

protected:
    bool testConnection();
    void pcInitialize();


private:
    QTimer *p_readTimer;
    double randPressure;
};

#endif // VIRTUALPRESSURECONTROLLER_H
