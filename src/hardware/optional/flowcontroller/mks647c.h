#ifndef MKS647C_H
#define MKS647C_H

#include <src/hardware/optional/flowcontroller/flowcontroller.h>

namespace BC {
namespace Key {
static const QString mks647c("mks647c");
static const QString mks647cName("MKS 647C Flow Control Unit");
}
}

class Mks647c : public FlowController
{
public:
    explicit Mks647c(QObject *parent = nullptr);

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

    // HardwareObject interface
    void sleep(bool b) override;

protected:
    bool fcTestConnection() override;

    // FlowController interface
    virtual void fcInitialize() override;



private:
    QList<double> d_gasRangeList;
    QList<double> d_pressureRangeList;
    int d_pressureRangeIndex;
    QList<int> d_rangeIndexList;
    QList<double> d_gcfList;

    QByteArray mksQueryCmd(QString cmd, int respLength);
    int d_maxTries;
    int d_nextRead;
};

#endif // MKS647C_H
