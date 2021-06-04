#ifndef VIRTUALFLOWCONTROLLER_H
#define VIRTUALFLOWCONTROLLER_H

#include <src/hardware/optional/flowcontroller/flowcontroller.h>

class VirtualFlowController : public FlowController
{
    Q_OBJECT
public:
    explicit VirtualFlowController(QObject *parent = nullptr);
    ~VirtualFlowController();

public slots:
    // FlowController interface
    void hwSetFlowSetpoint(const int ch, const double val) override;
    void hwSetPressureSetpoint(const double val) override;
    double hwReadFlowSetpoint(const int ch) override;
    double hwReadPressureSetpoint() override;
    double hwReadFlow(const int ch) override;
    double hwReadPressure() override;
    void hwSetPressureControlMode(bool enabled) override;
    int hwReadPressureControlMode() override;
    void poll() override;

protected:
    bool fcTestConnection() override;
    void fcInitialize() override;
};

#endif // VIRTUALFLOWCONTROLLER_H
