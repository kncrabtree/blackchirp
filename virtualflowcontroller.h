#ifndef VIRTUALFLOWCONTROLLER_H
#define VIRTUALFLOWCONTROLLER_H

#include "flowcontroller.h"

class VirtualFlowController : public FlowController
{
    Q_OBJECT
public:
    explicit VirtualFlowController(QObject *parent = nullptr);
    ~VirtualFlowController();

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();


    // FlowController interface
    double setFlowSetpoint(const int ch, const double val);
    double setPressureSetpoint(const double val);
    double readFlowSetpoint(const int ch);
    double readPressureSetpoint();
    double readFlow(const int ch);
    double readPressure();
    void setPressureControlMode(bool enabled);
    bool readPressureControlMode();
};

#endif // VIRTUALFLOWCONTROLLER_H
