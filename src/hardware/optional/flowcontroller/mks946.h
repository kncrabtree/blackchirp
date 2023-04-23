#ifndef MKS947_H
#define MKS947_H

#include <hardware/optional/flowcontroller/flowcontroller.h>

namespace BC::Key::Flow {
static const QString mks947{"mks947"};
static const QString mks947Name("MKS 946 Flow Controller");
static const QString address{"mksaddress"};
static const QString pressureChannel{"pressureChannel"};
static const QString offset{"channelOffset"};
}

class Mks946 : public FlowController
{
    Q_OBJECT
public:
    Mks946(QObject *parent = nullptr);

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
    void poll() override;

protected:
    void fcInitialize() override;
    bool mksWrite(QString cmd);
    QByteArray mksQuery(QString cmd);

    // HardwareObject interface
public slots:
    void sleep(bool b) override;

private:
    int d_nextRead;
};

#endif // MKS947_H
