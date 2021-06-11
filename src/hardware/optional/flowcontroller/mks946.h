#ifndef MKS947_H
#define MKS947_H

#include <src/hardware/optional/flowcontroller/flowcontroller.h>

namespace BC {
namespace Key {
static const QString mks947("mks947");
static const QString mks947Name("MKS 946 Flow Controller");
}
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

    int d_address;
    int d_pressureChannel;
    int d_channelOffset; //Some models may have a pressure sensor module on channels 1 and 2, so flow channels would start at 3. This should contain the offset needed to convert logical channels (e.g., 0-3 in BlackChirp) to actual channel number on the device (must be 1-6).

    // HardwareObject interface
public slots:
    void readSettings() override;
    void sleep(bool b) override;

private:
    int d_nextRead;
};

#endif // MKS947_H
