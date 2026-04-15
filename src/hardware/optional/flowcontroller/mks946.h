#ifndef MKS947_H
#define MKS947_H

#include <hardware/optional/flowcontroller/flowcontroller.h>

namespace BC::Key::Flow {
// Implementation-specific keys for MKS946
inline constexpr QLatin1StringView address{"mksaddress"};
inline constexpr QLatin1StringView pressureChannel{"pressureChannel"};
inline constexpr QLatin1StringView offset{"channelOffset"};
}

class Mks946 : public FlowController
{
    Q_OBJECT
public:
    Mks946(const QString& label, QObject *parent = nullptr);

    // HardwareObject interface
protected:
    bool fcTestConnection() override;

    // FlowController interface
public slots:
    void hwSetFlowSetpoint(const int ch, const double val) override;
    void hwSetPressureSetpoint(const double val) override;
    double hwReadFlowSetpoint(const int ch) override;
    double hwReadPressureSetpoint() override;
    double hwReadFlow(const int ch) override;
    double hwReadPressure() override;
    void hwSetPressureControlMode(bool enabled) override;
    int hwReadPressureControlMode() override;

protected:
    void fcInitialize() override;
    bool mksWrite(QString cmd);
    QByteArray mksQuery(QString cmd);

    // HardwareObject interface
public slots:
    void sleep(bool b) override;
};

#endif // MKS947_H
