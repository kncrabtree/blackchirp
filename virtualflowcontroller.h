#ifndef VIRTUALFLOWCONTROLLER_H
#define VIRTUALFLOWCONTROLLER_H

#include "flowcontroller.h"

class VirtualFlowController : public FlowController
{
    Q_OBJECT
public:
    explicit VirtualFlowController(QObject *parent = nullptr);
    ~VirtualFlowController();

public slots:
    // FlowController interface
    double setFlowSetpoint(const int ch, const double val) override;
    double setPressureSetpoint(const double val) override;
    double readFlowSetpoint(const int ch) override;
    double readPressureSetpoint() override;
    double readFlow(const int ch) override;
    double readPressure() override;
    void setPressureControlMode(bool enabled) override;
    bool readPressureControlMode() override;

protected:
    bool testConnection() override;
    void fcInitialize() override;
};

#endif // VIRTUALFLOWCONTROLLER_H
