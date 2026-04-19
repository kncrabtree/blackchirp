#ifndef VIRTUALFLOWCONTROLLER_H
#define VIRTUALFLOWCONTROLLER_H

#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <random>

class VirtualFlowController : public FlowController
{
    Q_OBJECT
public:
    explicit VirtualFlowController(const QString& label, QObject *parent = nullptr);
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

protected:
    bool fcTestConnection() override;
    void fcInitialize() override;

private:
    std::mt19937 d_rng{std::random_device{}()};
};

#endif // VIRTUALFLOWCONTROLLER_H
